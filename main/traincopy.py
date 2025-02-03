import os
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import utils

# tf.debugging.set_log_device_placement(False)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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

def federated_learning(x_train, y_train, x_test, y_test, institution, class_weights, seed):

    x_train = x_train[global_feature]
    x_test = x_test[global_feature]

    # One hot encoding 
    columns_exclude = ['Radiation', 'Chemotherapy', 'Surgery']

    x_train = pd.get_dummies(x_train, drop_first=False, columns=[col for col in global_feature if col not in columns_exclude])
    x_test = pd.get_dummies(x_test, drop_first=False, columns=[col for col in global_feature if col not in columns_exclude])

    for col in global_feature_en: 
        if col not in x_train.columns:
            x_train[col] = 0
        if col not in x_test.columns:
            x_test[col] = 0

    # Ensure columns order is the same
    x_train = x_train[global_feature_en]
    x_test = x_test[global_feature_en]


    # Load and compile Keras model
    opt_adam = Adam(learning_rate = 0.003)
    model = Sequential() 
    model.add(Dense(12, activation = 'relu', input_shape = (x_train.shape[1],))) 
    model.add(BatchNormalization())
    model.add(Dense(6, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(optimizer = opt_adam, loss = "categorical_crossentropy", metrics = ['accuracy'])

    # Start Flower client
    client_hospital = utils.SpcancerClient(model, x_train, y_train, x_test, y_test, class_weights)
    fl.client.start_numpy_client("127.0.0.1:6001", client=client_hospital)

    pred_prob = model.predict(x_test.astype(float))
    auc = roc_auc_score(y_test, pred_prob[:, 1])
    print(f"Global model auc score: {auc}")

    # Passing seed from main is only used in here
    utils.featureInterpreter('Federated Learning', model, x_train.astype(np.int32), institution, 'baseline', seed)

    return auc, pred_prob


def localized_learning(x_train, y_train, x_test, y_test, institution, class_weights, seed):
 
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


    # Load and compile Keras model
    opt_adam = Adam(learning_rate = 0.003)
    model = Sequential() 
    model.add(Dense(12, activation = 'relu', input_shape = (x_train.shape[1],))) 
    model.add(BatchNormalization())
    model.add(Dense(6, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(optimizer = opt_adam, loss = "categorical_crossentropy", metrics = ['accuracy'])

    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.000005)
    history = model.fit(x_train.astype(float), y_train, epochs = 300, class_weight = class_weights, callbacks=[lr_scheduler])

    # utils.draw_loss_function(history=history, name="localized learning")

    pred_prob = model.predict(x_test.astype(float))
    auc = roc_auc_score(y_test, pred_prob[:, 1])
    print(f"Local model AUC score: {auc}")

    # Passing seed from main is only used in here
    utils.featureInterpreter('Localized Learning', model, x_train.astype(np.int32), institution, 'baseline', seed)

    return auc, pred_prob


def main() -> None:  
    '''
    If you use the script to run this program, where you can test multiple seeds per time. You need to comment 
    LINE: institution = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): "))
    Otherwise, you need to comment, where you can only test for one seed.
    LINE: seed, institution = utils.parse_argument_for_running_script()
    '''
    seed, institution = utils.parse_argument_for_running_script()
    # institution = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): "))

    columns = list(global_feature)

    if institution == 1:
        columns.extend(taiwan_feature)
        df = pd.read_csv('Data_folder/Taiwan_en.csv')
    else: 
        columns.extend(seer_feature)
        df = pd.read_csv('Data_folder/SEER_en.csv')

    columns.append('Target')
    df = df[columns]

    trainset, testset = train_test_split(df, test_size=0.6, stratify=df['Target'], random_state=seed)

    x_train, y_train = trainset.drop(columns=['Target']), trainset['Target']
    x_test, y_test = testset.drop(columns=['Target']), testset['Target']
    y_train_one_hot = to_categorical(y_train, num_classes=2)

    print(f'------------------------{f"Name of your Institution: {institution}"}------------------------')
    print(f"x_train (data number, feature number): {x_train.shape}")
    print(f"x_test (data number, feature number): {x_test.shape}")
    print(f'The number of true cases in y_train:  {(y_train == 1).sum()}')
    print(f'The number of true cases in y_test:  {(y_test == 1).sum()}')

    # class weights
    beta = (len(x_train)-1)/len(x_train)
    print(f"Beta{beta}")
    class_weights = utils.get_class_balanced_weights(y_train, beta)
    print(f"class weights: {class_weights}")

    auc_global, fed_prob = federated_learning(x_train, y_train_one_hot, x_test, y_test, institution, class_weights, seed)
    auc_local, cen_prob = localized_learning(x_train, y_train_one_hot, x_test, y_test, institution, class_weights, seed)

    auc = {
        'global auc': np.array(auc_global).astype(float),
        'local auc': np.array(auc_local).astype(float)
    }

    auc_csv = pd.DataFrame(auc, index=[0])
    print(auc_csv)
    auc_csv.to_csv(f"init_{institution}.csv",index=False)

    result = {
        'global model predict yes prob': fed_prob[:,1],
        'global model predict no prob': fed_prob[:,0],
        'local model predict yes prob': cen_prob[:,1],
        'local model predict no prob': cen_prob[:,0],
        'Outcome': y_test
    }

    middle = pd.DataFrame(result)
    middle.to_csv(f"middle_{institution}.csv",index=False)

    # Saving Baseline Models Results 
    hospital = 'Taiwan' if institution == 1 else 'USA'
    baseline = {
        f'Model | {hospital} | seed={seed}': ['Federated Learning', 'Localized Learning'],
        'auc': [np.array(auc_global).astype(float), np.array(auc_local).astype(float)]
    }
    baseline_results = pd.DataFrame(baseline)
    baseline_results.to_csv('Results/Results_Baseline.csv', mode='a', index=False)
    print("Results saved to Results_Baseline.csv")



if __name__ == "__main__":
    main()
