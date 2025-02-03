import os
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Global settings
min_client = 2
rounds = 5
total_feature_number = 43

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define PyTorch Model
class Net(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.LayerNorm(12)
        self.fc3 = nn.Linear(12, 6)
        self.fc4 = nn.LayerNorm(6) 
        self.fc5 = nn.Dropout(0.2)
        self.fc6 = nn.Linear(6, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.fc5(x)
        x = F.softmax(self.fc6(x), dim=1)
        return x
    

# Function to get model parameters
def get_model_parameters(model: nn.Module):
    """Extracts model parameters as a list of NumPy arrays."""
    return [param.detach().cpu().numpy() for param in model.state_dict().values()]

def fit_config(rnd: int):
    """Defines the configuration for training in each round."""
    config = {
        "round": rnd,
        "local_epochs": 60 if rnd < 4 else 80
    }
    return config

def evaluate_config(rnd: int):
    """Defines the configuration for evaluation in each round."""
    config = {
        "val_steps": 5 if rnd < 3 else 10
    }
    return config 

# Main Function to Start Flower Server
def main() -> None:
    model = Net(input_size=total_feature_number, output_size=2)
    model.to(DEVICE)  # Ensure model is on the correct device

    strategy = fl.server.strategy.FedAdam(
        min_fit_clients=min_client,
        min_evaluate_clients=min_client,
        min_available_clients=min_client,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters(model))
    )

    fl.server.start_server(
        server_address="127.0.0.1:6000",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy
    )

# Run the Flower Server
if __name__ == "__main__":
    main()
