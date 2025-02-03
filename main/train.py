import flwr as fl
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import utils

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
print(f"Flower {fl.__version__} / PyTorch {torch.__version__}")

'''''''''''''''''''''''''''''''''' Feature Groups '''''''''''''''''''''''''''''''''''''''

global_feature = ['Laterality', 'Age', 'Gender', 'SepNodule', 'PleuInva', 'Tumorsz', 'LYMND', 'AJCC', 'Radiation', 
                 'Chemotherapy', 'Surgery']

global_feature_en = ['Age_6', 'Tumorsz_1', 'Tumorsz_4', 'LYMND_3', 'Chemotherapy_1', 'AJCC_1', 'Surgery_2', 
                     'SepNodule_2', 'Laterality_2', 'PleuInva_1', 'Tumorsz_2', 'AJCC_3', 'Laterality_1', 'Age_4',
                     'Chemotherapy_2', 'LYMND_9', 'Gender_2', 'Tumorsz_9', 'Age_7', 'Age_9', 'Gender_1', 'AJCC_2',
                     'Laterality_3', 'Radiation_1', 'Laterality_9', 'LYMND_5', 'Age_3', 'PleuInva_9', 'Radiation_2',
                     'Tumorsz_3', 'LYMND_1', 'LYMND_4', 'Age_2', 'AJCC_5', 'Age_8', 'AJCC_9', 'AJCC_4', 'PleuInva_2',
                     'LYMND_2', 'Surgery_1', 'Age_5', 'SepNodule_9', 'SepNodule_1']

taiwan_feature = ['PleuEffu', 'EGFR', 'ALK', 'MAGN', 'DIFF', 'BMI_label', 'CIG', 'BN', 'ALC']
seer_feature = ['Income', 'Area', 'Race']

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class Net(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.LayerNorm(12)
        self.fc3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(12, 6)
        self.fc5 = nn.LayerNorm(6) 
        self.fc6 = nn.Linear(6, output_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.hardshrink(self.fc1(x)) 
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.hardshrink(self.fc4(x))
        x = self.fc5(x)
        x = F.softmax(self.fc6(x), dim=1)
        return x

    def penultimateLayerOutput(self, x: torch.Tensor) -> torch.Tensor:
        x = F.hardshrink(self.fc1(x)) 
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.hardshrink(self.fc4(x))
        x = self.fc5(x)
        return x


def train(net, x_train, y_train, epochs: int, class_weights, verbose=True):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) 
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    net.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net.forward(x_train)

        loss = criterion(outputs.float().to(DEVICE), y_train.float()) 
        loss.backward()
        
        scheduler.step(loss)
        optimizer.step()
        
        # Metrics
        if verbose:
            auroc = roc_auc_score(y_true=y_train[:, 1].detach().cpu().numpy() , y_score=outputs[:, 1].detach().cpu().numpy())
            print(f"Epoch {epoch+1}: train loss {loss.item():.4f}, auc {auroc:.4f}")


# This test function (evaluate function) will return loss and auc score 
def testLossAUC(net, x_test, y_test):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()

    net.eval()
    with torch.no_grad():
        outputs = net.forward(x_test)
    
        loss = criterion(outputs.to(DEVICE), y_test.float()).item()
        auroc = roc_auc_score(y_true=y_test[:, 1].cpu().numpy(), y_score=outputs[:, 1].cpu().numpy())
    
    return loss, auroc


# Evaluate Function & Extract the Penultimate Output 
def evalPenultimate(net, x_test, y_test):
    """Evaluate the network on the entire test set."""

    net.eval()
    # Forward pass
    with torch.no_grad():
        outputs = net(x_test)
        penultimate = net.penultimateLayerOutput(x_test)
        auroc = roc_auc_score(y_true=y_test[:, 1].cpu().numpy(), y_score=outputs[:, 1].cpu().numpy())
        auprc = average_precision_score(y_true=y_test[:, 1].cpu().numpy(), y_score=outputs[:, 1].cpu().numpy())
        y_pred = [0 if row[0] > row[1] else 1 for row in outputs]
        f1 = f1_score(y_true=y_test[:, 1].cpu().numpy(), y_pred=y_pred)

    return auroc, auprc, f1, penultimate, y_test


def lateFusion(FederatedNet, LocalizedNet, x_test_fed, x_test_loc, y_test):
    FederatedNet.eval()
    with torch.no_grad():
        fed_outputs = FederatedNet.forward(x_test_fed)

    LocalizedNet.eval()
    with torch.no_grad():
        loc_outputs = LocalizedNet.forward(x_test_loc)

    avg_outputs = 0.5 * fed_outputs + 0.5 * loc_outputs
    auroc = roc_auc_score(y_true=y_test[:, 1].cpu().numpy(), y_score=avg_outputs[:, 1].cpu().numpy())
    auprc = average_precision_score(y_true=y_test[:, 1].cpu().numpy(), y_score=avg_outputs[:, 1].cpu().numpy())
    y_pred = [0 if row[0] > row[1] else 1 for row in avg_outputs]
    f1 = f1_score(y_true=y_test[:, 1].cpu().numpy(), y_pred=y_pred)


    return auroc, auprc, f1 


# Model definition
def federatedLearning(x_train, y_train, x_test, y_test, institution, class_weights, seed):

    x_train = x_train[global_feature]
    x_test = x_test[global_feature]

    # One-hot encoding
    columns_exclude = ['Radiation', 'Chemotherapy', 'Surgery']

    x_train = pd.get_dummies(x_train, drop_first=False, columns=[col for col in global_feature if col not in columns_exclude])
    x_test = pd.get_dummies(x_test, drop_first=False, columns=[col for col in global_feature if col not in columns_exclude])

    for col in global_feature_en: 
        if col not in x_train.columns:
            x_train[col] = False
        if col not in x_test.columns:
            x_test[col] = False

    # Ensure columns order is the same
    x_train = x_train[global_feature_en]
    x_test = x_test[global_feature_en]
    
    FederatedNet = Net(input_size=len(x_train.columns), output_size=2).to(DEVICE)
    x_train, y_train = torch.from_numpy(x_train.values).float().to(DEVICE), torch.from_numpy(y_train).float().to(DEVICE)
    x_test, y_test = torch.from_numpy(x_test.values).float().to(DEVICE), torch.from_numpy(y_test).float().to(DEVICE)

    train(FederatedNet, x_train, y_train, epochs = 400,  class_weights = class_weights)

    # Start Flower client
    client_hospital = utils.SpcancerClient(FederatedNet, x_train, y_train, x_test, y_test, class_weights)
    fl.client.start_client(server_address="127.0.0.1:6000", client=client_hospital)


    # Passing seed from main is only used in here
    # utils.featureInterpreter('Localized Learning', model, x_train.astype(np.int32), institution, 'baseline', seed)

    return FederatedNet, x_test, y_test


def localizedLearning(x_train, y_train, x_test, y_test, institution, class_weights, seed):

    col_exclude_tw = ['Radiation', 'Chemotherapy', 'Surgery']
    col_exclude_seer = ['Radiation', 'Chemotherapy', 'Surgery']

    local_feature = list(global_feature)
    local_feature += (list(taiwan_feature) if institution == 1 else list(seer_feature))

    columns_exclude = list(col_exclude_tw) if institution == 1 else list(col_exclude_seer)

    x_train = x_train[local_feature]
    x_test = x_test[local_feature]


    # One hot encoding 
    x_train = pd.get_dummies(x_train, drop_first=False, columns=[col for col in local_feature if col not in columns_exclude])
    x_test = pd.get_dummies(x_test, drop_first=False, columns=[col for col in local_feature if col not in columns_exclude])


    # Make excluded features into boolean datatype same as other features 
    for col in columns_exclude: 
        x_train[col] = x_train[col].apply(lambda val: True if val != 0 else False)
        x_test[col] = x_test[col].apply(lambda val: True if val != 0 else False)

    LocalizedNet = Net(input_size=len(x_train.columns), output_size=2).to(DEVICE)
    x_train, y_train = torch.from_numpy(x_train.values).float().to(DEVICE), torch.from_numpy(y_train).float().to(DEVICE)
    x_test, y_test = torch.from_numpy(x_test.values).float().to(DEVICE), torch.from_numpy(y_test).float().to(DEVICE)

    train(LocalizedNet, x_train, y_train, epochs=600, class_weights = class_weights)

    # Passing seed from main is only used in here
    # utils.featureInterpreter('Localized Learning', model, x_train.astype(np.int32), institution, 'baseline', seed)

    return LocalizedNet, x_test, y_test


def main() -> None:  
    '''
    If you use the script to run this program, where you can test multiple seeds per time. You need to comment 
    LINE: institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): ")), 42
    Otherwise, you need to comment the following line, so you can only test for one seed.
    LINE: seed, institution = utils.parse_argument_for_running_script()
    '''
    # institution, seed = utils.parse_argument_for_running_script()
    institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): ")), 42

    columns = list(global_feature)

    if institution == 1:
        columns.extend(taiwan_feature)
        df = pd.read_csv('Data_folder/Taiwan_en.csv')
    else: 
        columns.extend(seer_feature)
        df = pd.read_csv('Data_folder/SEER_en.csv')

    columns.append('Target')
    df = df[columns]

    trainset, testset = train_test_split(df, test_size=0.2, stratify=df['Target'], random_state=seed)

    x_train, y_train = trainset.drop(columns=['Target']), trainset['Target']
    x_test, y_test = testset.drop(columns=['Target']), testset['Target']
    y_train_one_hot = utils.to_categorical(y_train, num_classes=2)
    y_test_one_hot = utils.to_categorical(y_test, num_classes=2)


    print(f'------------------------{f"Name of your Institution: {institution}"}------------------------')
    # class weights
    beta = (len(trainset)-1)/len(trainset)
    class_weights = utils.get_class_balanced_weights(y_train, beta)


    FederatedNet, x_test_fed, y_test_fed = federatedLearning(x_train, y_train_one_hot, x_test, y_test_one_hot, institution, class_weights, seed)
    LocalizedNet, x_test_loc, y_test_loc = localizedLearning(x_train, y_train_one_hot, x_test, y_test_one_hot, institution, class_weights, seed)


    # different = 0
    # for i in range(len(y_test_fed)):
    #     if(y_test_fed[i, 0] != y_test_loc[i, 0]):
    #         different+=1
    # print(f"does the data be in same order {different}")


    # Extract Penultimate Layer Output
    auroc_global, auprc_global, f1_global, x_pen_fed, y_pen_fed = evalPenultimate(FederatedNet, x_test_fed, y_test_fed)
    auroc_local, auprc_local, f1_local, x_pen_loc, y_pen_loc = evalPenultimate(LocalizedNet, x_test_loc, y_test_loc)


    print(f"Global AUC {auroc_global}")
    print(f"Local AUC {auroc_local}")

    # different = 0
    # for i in range(len(y_pen_fed)):
    #     if(y_pen_fed[i, 0] != y_test.iloc[i]):
    #         different+=1
    # print(f"does the data be in same order {different}")

    x_pen_fed, y_pen_fed = x_pen_fed.cpu().numpy(), y_pen_fed.cpu().numpy()
    x_pen_loc, y_pen_loc = x_pen_loc.cpu().numpy(), y_pen_loc.cpu().numpy()

    penultimate_layer_output = [(list(fed_row), list(loc_row), y_test.iloc[i]) for i, (fed_row, loc_row) in enumerate(zip(x_pen_fed, x_pen_loc))]

    df = pd.DataFrame(penultimate_layer_output)
    df.rename(columns={0: 'Federate', 1: 'Localize', 2: 'Target'}, inplace=True)
    df.to_csv(f"middle_{institution}.csv", index=False)


    # Late Fusion (average results from two models)
    auroc_late_fusion, auprc_late_fusion, f1_late_fusion = lateFusion(FederatedNet, LocalizedNet, x_test_fed, x_test_loc, y_test_fed)
    print(f"Late Fusion AUC {auroc_late_fusion}")


    # Saving Baseline Models Results 
    hospital = 'Taiwan' if institution == 1 else 'USA'
    baseline = {
        f'Model | {hospital} | seed={seed}': ['Federated Learning', 'Localized Learning', 'Late Fusion'],
        'auroc': [np.array(auroc_global).astype(float), np.array(auroc_local).astype(float), np.array(auroc_late_fusion).astype(float)], 
        'auprc': [np.array(auprc_global).astype(float), np.array(auprc_local).astype(float), np.array(auprc_late_fusion).astype(float)], 
        'F1-score': [np.array(f1_global).astype(float), np.array(f1_local).astype(float), np.array(f1_late_fusion).astype(float)]
    }

    baseline_results = pd.DataFrame(baseline)
    baseline_results.to_csv('Results/Results_Baseline.csv', mode='a', index=False)
    print("Results saved to Results_Baseline.csv")



if __name__ == "__main__":
    main()
