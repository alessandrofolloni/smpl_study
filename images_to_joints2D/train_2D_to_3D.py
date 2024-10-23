#import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb

# Configuration for hyperparameters
config = {
    'batch_size': 64,  # Adjust batch size
    'learning_rate': 1e-4,  # Adjust learning rate
    'epochs': 50,  # Adjust the number of epochs
    'hidden_sizes': [1024, 512, 256],  # Change model architecture
    'dropout': 0.3,  # Adjust dropout rate
    'camera_ids': ['50591643', '58860488', '60457274', '65906101'],  # Include or exclude cameras
    'num_joints_2d': 17,  # Number of joints per camera
    'num_joints_3d': 25,  # Number of joints in 3D data
    'train_split': 0.7,  # Adjust training data ratio
    'val_split': 0.15,  # Adjust validation data ratio
    'test_split': 0.15,  # Adjust test data ratio
}
# Initialize wandb
wandb.init(project='2D_to_3D_Joints', entity='alessandrofolloni',
           name=f'train_epochs{config["epochs"]}_bs{config["batch_size"]}')
wandb.config.update(config)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset Class
class JointsDataset(Dataset):
    def __init__(self, mega_dict, camera_ids):
        self.samples = []
        self.camera_ids = camera_ids
        for exercise_key, data in mega_dict.items():
            joints2d = data["joints2d"]
            joints3d = data["gt"]
            for frame_key in joints2d.keys():
                sample = {
                    'joints2d': joints2d[frame_key],
                    'joints3d': joints3d[frame_key]
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        joints2d_list = []
        missing_cameras = []
        for cam_id in self.camera_ids:
            cam_joints = sample['joints2d'].get(cam_id, None)
            if cam_joints is not None:
                joints2d_list.extend(cam_joints)
            else:
                # Fill missing camera data with zeros
                missing_cameras.append(cam_id)
                joints2d_list.extend([[0.0, 0.0]] * config['num_joints_2d'])
            if missing_cameras:
                print(f"Added zeros for missing cameras {missing_cameras} in frame {idx}")
        joints2d_array = np.array(joints2d_list).flatten()
        joints3d_array = np.array(sample['joints3d']).flatten()

        # Convert to tensors
        joints2d_tensor = torch.tensor(joints2d_array, dtype=torch.float32)
        joints3d_tensor = torch.tensor(joints3d_array, dtype=torch.float32)

        return joints2d_tensor, joints3d_tensor

# Model Definition
class JointsModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout):
        super(JointsModel, self).__init__()
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

# Mean Per Joint Position Error (MPJPE)
def mpjpe(predicted, target):
    predicted = predicted.view(-1, config['num_joints_3d'], 3)
    target = target.view(-1, config['num_joints_3d'], 3)
    error = torch.norm(predicted - target, dim=2).mean()
    return error

# Load Data
def load_data(mega_dict_path):
    with open(mega_dict_path, 'r') as f:
        mega_dict = json.load(f)
    return mega_dict

# Training Function
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_mpjpe = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        mpjpe_value = mpjpe(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_mpjpe += mpjpe_value.item() * inputs.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_mpjpe = running_mpjpe / len(loader.dataset)
    return epoch_loss, epoch_mpjpe

# Validation Function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_mpjpe = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mpjpe_value = mpjpe(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            running_mpjpe += mpjpe_value.item() * inputs.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_mpjpe = running_mpjpe / len(loader.dataset)
    return epoch_loss, epoch_mpjpe

# Test Function
def test(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_mpjpe = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mpjpe_value = mpjpe(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            running_mpjpe += mpjpe_value.item() * inputs.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_mpjpe = running_mpjpe / len(loader.dataset)
    return epoch_loss, epoch_mpjpe

# Main Training Loop
def main():
    mega_dict_path = '/public.hpc/alessandro.folloni2/smpl_study' \
                     '/datasets/FIT3D/train/mega_dict.json'
    mega_dict = load_data(mega_dict_path)

    # Prepare Dataset and DataLoader
    dataset = JointsDataset(mega_dict, config['camera_ids'])

    # Split the dataset into train, val, and test sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_split = int(np.floor(config['train_split'] * dataset_size))
    val_split = int(np.floor(config['val_split'] * dataset_size))

    train_indices = indices[:train_split]
    val_indices = indices[train_split:train_split + val_split]
    test_indices = indices[train_split + val_split:]

    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize Model
    input_size = len(config['camera_ids']) * config['num_joints_2d'] * 2  # x and y coordinates
    output_size = config['num_joints_3d'] * 3  # x, y, z coordinates
    model = JointsModel(input_size, output_size, config['hidden_sizes'], config['dropout'])
    model.to(device)

    # Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training Loop
    for epoch in range(config['epochs']):
        train_loss, train_mpjpe = train(model, train_loader, criterion, optimizer)
        val_loss, val_mpjpe = validate(model, val_loader, criterion)

        # Logging
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_mpjpe': train_mpjpe,
            'val_loss': val_loss,
            'val_mpjpe': val_mpjpe
        })

        print(f"Epoch [{epoch + 1}/{config['epochs']}], "
              f"Train Loss: {train_loss:.4f}, Train MPJPE: {train_mpjpe:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val MPJPE: {val_mpjpe:.4f}")

    # Test the model
    test_loss, test_mpjpe = test(model, test_loader, criterion)
    wandb.log({
        'test_loss': test_loss,
        'test_mpjpe': test_mpjpe
    })
    print(f"Test Loss: {test_loss:.4f}, Test MPJPE: {test_mpjpe:.4f}")

    # Save the trained model
    model_path = 'joints_model.pth'
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)
    print("Model saved.")

if __name__ == '__main__':
    main()