import ast 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import utils

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
print(f"PyTorch {torch.__version__}")

def dataPreprocess(dataframe, seed):
    trainset, testset = train_test_split(dataframe, test_size=0.25, stratify=dataframe['Target'], random_state=seed)

    x_train = torch.from_numpy(trainset.drop(columns=['Target']).values).float().to(DEVICE)
    x_test = torch.from_numpy(testset.drop(columns=['Target']).values).float().to(DEVICE)
    y_train = torch.from_numpy(utils.to_categorical(trainset['Target'], num_classes=2)).float().to(DEVICE)
    y_test = torch.from_numpy(utils.to_categorical(testset['Target'], num_classes=2)).float().to(DEVICE)

    return x_train, y_train, x_test, y_test


class Net(nn.Module):
    def __init__(self, epochs:int, learning_rate:float, dataframe, seed) -> None:
        self.x_train, self.y_train, self.x_test, self.y_test = dataPreprocess(dataframe, seed)
        self.epochs = epochs
        self.learning_rate = learning_rate 

        super(Net, self).__init__()
        self.fc1 = nn.Linear(self.x_train.shape[1], 12)
        self.fc2 = nn.LayerNorm(12)
        self.fc3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(12, 6)
        self.fc5 = nn.LayerNorm(6) 
        self.fc6 = nn.Linear(6, 2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.hardshrink(self.fc1(x)) 
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.hardshrink(self.fc4(x))
        x = self.fc5(x)
        x = F.softmax(self.fc6(x), dim=1)
        return x

def train(net, class_weights, verbose=True):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) 
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    net.train()
    for epoch in range(net.epochs):
        optimizer.zero_grad()
        outputs = net.forward(net.x_train)

        loss = criterion(outputs.float().to(DEVICE), net.y_train.float())  # Ensure labels are float for BCE
        loss.backward()
        
        scheduler.step(loss)
        optimizer.step()
        
        # Metrics
        if verbose:
            auroc = roc_auc_score(y_true=net.y_train[:, 1].detach().cpu().numpy() , y_score=outputs[:, 1].detach().cpu().numpy())
            print(f"Epoch {epoch+1}: train loss {loss.item():.4f}, auc {auroc:.4f}")



def evaluate_model(net):

    net.eval()
    with torch.no_grad():
        outputs = net.forward(net.x_test)
    
    probability = outputs[:, 1].cpu().numpy()
    predictions = [1 if row[1] > row[0] else 0 for row in outputs]
    
    y_test = net.y_test[:, 1].cpu().numpy()

    if (net.y_test[:, 1]).sum():
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        auprc = average_precision_score(y_true=y_test, y_score=probability)
        auroc = roc_auc_score(y_true=y_test, y_score=probability)
    else:
        accuracy, f1, precision, recall, auprc, auroc = None, None, None, None, None, None

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auprc': auprc, 
        'auroc': auroc
    }


def main():
    '''
    If you use the script to run this program, where you can test multiple seeds per time. You need to comment 
    LINE: institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): "))
    Otherwise, you need to comment, where you can only test for one seed.
    LINE: institution, seed = utils.parse_argument_for_running_script()
    '''
    # institution, seed = utils.parse_argument_for_running_script()
    institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): ")), 42
    
    df = pd.read_csv(f'middle_{institution}.csv')

    df['Federate'], df['Localize'] = df['Federate'].apply(ast.literal_eval), df['Localize'].apply(ast.literal_eval)
    df['Concatenate'] = (df['Federate'] + df['Localize']).apply(np.array)

    df['Federate'], df['Localize'] = df['Federate'].apply(np.array), df['Localize'].apply(np.array)
    df['Add'] = (df['Federate'] + df['Localize']).apply(np.array)

    # concatenate dataframe 
    df_concatenate = pd.DataFrame(df['Concatenate'].tolist(), index=df.index)
    df_concatenate.columns = [f'Concatenate_{i}' for i in range(df_concatenate.shape[1])]
    df_concatenate['Target'] = df['Target']

    # add dataframe 
    df_add = pd.DataFrame(df['Add'].tolist(), index=df.index)
    df_add.columns = [f'Add_{i}' for i in range(df_add.shape[1])]
    df_add['Target'] = df['Target']


    # class weights
    beta = (len(df['Target'])-1)/len(df['Target'])
    class_weights = utils.get_class_balanced_weights(df['Target'], beta)

    models = {
        'CCN': Net(epochs = 400, learning_rate = 0.003, dataframe = df_concatenate, seed=seed).to(DEVICE),
        'ADD': Net(epochs = 400, learning_rate = 0.003, dataframe = df_add, seed=seed).to(DEVICE),
    }

    all_results = []

    for name, model in models.items():
        train(model, class_weights=class_weights)
        result = evaluate_model(model)
        result['model'] = name
        all_results.append(result)


    all_results = pd.DataFrame(all_results)
    all_results = all_results[['model', 'accuracy', 'f1', 'precision', 'recall', 'auprc', 'auroc']]


    print("Cross-validation results:")
    # print(all_results)

    # Saving NSC Models Results 
    hospital = 'Taiwan' if institution == 1 else 'USA'
    all_results = all_results[['model', 'auroc', 'auprc', 'f1']]
    all_results.rename(columns={'model': f'Model | {hospital} | seed={seed}'}, inplace=True)
    print(all_results)

    # all_results.to_csv('Results/Results_NSC.csv', mode='a', index=False)
    # print("results saved to Results_NSC.csv")

if __name__ == "__main__":
    main()
