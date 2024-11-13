import torch
import torch.nn as nn

# Configuration for hyperparameters
config = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 100,
    'camera_ids': ['50591643', '58860488', '60457274', '65906101'],
    'num_joints_2d': 17,
    'num_joints_3d': 25,
    'train_split': 0.7,
    'val_split': 0.2,
    'test_split': 0.1,
    'model_name': 'CNN',  # 'FCNN', 'CNN', 'RNN'
    'hidden_sizes': [16384, 8192, 8192, 4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 32, 16, 8],  # For FCNN
    'dropout': 0.3,
    'cnn_extra_layers': [32, 64, 128, 256, 512, 1024],
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fully Connected Neural Network (FCNN)
class FCNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout):
        super(FCNNModel, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Convolutional Neural Network (CNN)
class CNNModel(nn.Module):
    def __init__(self, output_size, extra_layers):
        super(CNNModel, self).__init__()
        layers = []
        # Initial Conv Layer
        layers.append(nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        # Add extra Conv layers as specified
        in_channels = 16
        for out_channels in extra_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.conv_layers = nn.Sequential(*layers)
        # Calculate the flattened size after convolution
        self.num_cameras = len(config['camera_ids'])
        self.num_joints_2d = config['num_joints_2d']
        flattened_size = in_channels * self.num_cameras * self.num_joints_2d
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, output_size)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Recurrent Neural Network (RNN)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the output from the last time step
        out = self.fc(out)
        return out

def print_model_info():
    models = ['FCNN', 'CNN', 'RNN']
    for model_name in models:
        print(f"Model: {model_name}")
        if model_name == 'FCNN':
            input_size = len(config['camera_ids']) * config['num_joints_2d'] * 2  # x and y coordinates
            output_size = config['num_joints_3d'] * 3  # x, y, z coordinates
            model = FCNNModel(input_size, output_size, config['hidden_sizes'], config['dropout'])
        elif model_name == 'CNN':
            output_size = config['num_joints_3d'] * 3
            extra_layers = config['cnn_extra_layers']
            model = CNNModel(output_size, extra_layers)
        elif model_name == 'RNN':
            input_size = config['num_joints_2d'] * 2  # x and y coordinates per camera
            hidden_size = 256  # Adjust as needed
            num_layers = 2     # Adjust as needed
            output_size = config['num_joints_3d'] * 3
            model = RNNModel(input_size, hidden_size, num_layers, output_size, config['dropout'])
        else:
            raise ValueError(f"Model {model_name} not implemented.")

        # Move model to device (if necessary)
        model.to(device)

        # Print model structure
        print(model)
        # Calculate number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params}")
        print("========================================\n")

if __name__ == '__main__':
    print_model_info()