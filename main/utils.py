import pandas as pd
import argparse
import numpy as np
import flwr as fl
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from collections import OrderedDict
from train import train, testLossAUC
from sklearn.metrics import roc_auc_score
import shap
from typing import List, Tuple
import torch 

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(DEVICE)


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

    print("-------------------------------------------------")
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")
    print("-------------------------------------------------")
    
    net.load_state_dict(state_dict, strict=False)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class SpcancerClient(fl.client.NumPyClient):
    def __init__(self, net, x_train, y_train, x_test, y_test, class_weights):
        self.net = net
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_weights = class_weights

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.x_train, self.y_train, epochs=100,  class_weights = self.class_weights)
        return get_parameters(self.net), len(self.x_train), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, auc = testLossAUC(self.net, self.x_test, self.y_test)
        return float(loss), len(self.y_test), {"accuracy": float(auc)}




# class SpcancerClient(fl.client.NumPyClient):
    # def __init__(self, model, x_train, y_train, x_test, y_test, class_weights):
    #     self.model = model
    #     self.x_train, self.y_train = x_train.astype(float), y_train.astype(float)
    #     self.x_test, self.y_test = x_test.astype(float), y_test.astype(float)
    #     self.class_weights = class_weights

    # def get_parameters(self):
    #     return self.model.get_weights()


    # # config is the information which is sent by the server every round.
    # # The content of the config will change every round
    # def fit(self, parameters, config):
    #     self.model.set_weights(parameters)

    #     print(f"Round: {config['round']}")
    #     epochs: int = config["local_epochs"]

    #     # lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.000005)
    #     lr_scheduler = 0.005

    #     history = self.model.fit(self.x_train, self.y_train, epochs=epochs, class_weight=self.class_weights, callbacks=[lr_scheduler])

    #     # draw_loss_function(history=history, name="federated learning")

    #     # Return updated model parameters and results
    #     results = {
    #         "loss": history.history["loss"][0],
    #         "accuracy": history.history["accuracy"][0],
    #     }

    #     return self.model.get_weights(), len(self.x_train), results


    # def evaluate(self, parameters, config):
    #     self.model.set_weights(parameters)

    #     pred_prob = self.model.predict(self.x_test)

    #     loss, accuracy = self.model.evaluate(self.x_test, to_categorical(self.y_test, num_classes=2), steps = config['val_steps'])
    #     auc = roc_auc_score(self.y_test, pred_prob[:, 1])

    #     results = {
    #         "accuracy": accuracy,
    #         "auc": auc,
    #     }

    #     return loss, len(self.x_test), results



def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

# def scheduler():

def get_class_balanced_weights(y_train, beta):
    # Count the number of samples for each class
    class_counts = Counter(y_train)

    # Calculate the effective number for each class
    effective_num = {}
    for class_label, count in class_counts.items():
        effective_num[class_label] = (1 - beta**count) / (1 - beta)

    # Calculate the class-balanced weight 
    scaling = 10000
    class_weights = [(1 / effective_num[0]) * scaling, (1 / effective_num[1]) * scaling]


    print(f"SPC False: {class_weights[0]}   SPC True: {class_weights[1]}")
    return torch.FloatTensor(class_weights).cuda().to(DEVICE)


def draw_loss_function(history, name):
    try:
        plt.plot(history.history['loss'])
    except:
        plt.plot(history[0], history[1]) #adding this for seesawing weights 

    plt.title(f'Model loss -- {name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc = 'upper left')
    plt.show()


def parse_argument_for_running_script():
    parser = argparse.ArgumentParser(description="Training Script for a Federated Learning Model")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--hospital', type=int, default=1, help='Hospital Data for training')
    args = parser.parse_args()
    return args.hospital, args.seed


def featureInterpreter(name, model, x_train, institution, method, seed):
    hospital = 'Taiwan' if institution == 1 else 'USA'

    background_data = shap.sample(x_train, 100)
    explainer = shap.KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(x_train.iloc[299:399, :])

    # shap summary plot 
    # shap.summary_plot(shap_values, x_train.iloc[299:399, :], show=False)
    
    # shap summary beeswarm plot (yes class)
    shap.summary_plot(shap_values[1], x_train.iloc[299:399, :], show=False)

    plt.subplots_adjust(top=0.85) 
    plt.title(f'{name} | {hospital} | summary | seed = {seed}')
    plt.savefig(f'Results/shap/{name}_{hospital}_{seed}.png')
    plt.close('all')



def featureInterpreter_SSW(ser_weight, loc_weight, institution, seed):
    hospital = 'Taiwan' if institution == 1 else 'USA'

    feature_result = pd.DataFrame({
        'feature': ['global model predict no prob', 'global model predict yes prob', 
                    'local model predict no prob', 'local model predict yes prob'],
        'weight': [ser_weight, ser_weight, loc_weight, loc_weight]
    })
    
    plt.subplots_adjust(left=0.35)
    sns.barplot(x='weight', y='feature', data=feature_result)
    plt.xlabel("Weight")

    plt.title(f'SSW | {hospital} | summary | seed = {seed}')
    plt.savefig(f'Results/shap/SSW_{hospital}_{seed}.png')
    plt.close('all')