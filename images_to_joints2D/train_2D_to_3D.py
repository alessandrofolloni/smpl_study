import json
import os

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    'hidden_sizes': [512, 256, 128, 64, 32, 16, 8],  # For FCNN
    'dropout': 0.3,
    'cnn_extra_layers': [32, 64, 128, 256, 512, 1024],
}

wandb.init(
    project='2D_to_3D_Joints_filtered',
    entity='alessandrofolloni',
    name=f"{config['model_name']}{len(config['cnn_extra_layers'])}_epochs{config['epochs']}_bs{config['batch_size']}_lr{config['learning_rate']}_logs",
)
wandb.config.update(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Plotting Function to plot both predicted and ground truth joints
def plot_pred_and_gt_joints(pred_joints, gt_joints, exercise_key, frame_idx, output_dir='3d_plots', save=True, return_figure=False):
    """
    Plot both predicted and ground truth 3D joints in the same plot.

    Parameters:
        pred_joints (np.ndarray): A 25x3 array containing the predicted 3D coordinates of joints.
        gt_joints (np.ndarray): A 25x3 array containing the ground truth 3D coordinates of joints.
        exercise_key (str): A string identifier for the exercise (used in filename).
        frame_idx (int): The frame index to use in the output filename.
        output_dir (str): The directory to save the output plot (default is '3d_plots').
        save (bool): Whether to save the plot to a file.
        return_figure (bool): Whether to return the figure object.

    Returns:
        fig (matplotlib.figure.Figure): The figure object (if return_figure is True).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if pred_joints.shape != (25, 3):
        raise ValueError(f"Expected shape (25, 3) for pred_joints, but got {pred_joints.shape}")
    if gt_joints.shape != (25, 3):
        raise ValueError(f"Expected shape (25, 3) for gt_joints, but got {gt_joints.shape}")

    # Extract coordinates
    x_pred, y_pred, z_pred = pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2]
    x_gt, y_gt, z_gt = gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the predicted joints
    ax.scatter(x_pred, y_pred, z_pred, c='r', marker='o', label='Predicted')

    # Plot the ground truth joints
    ax.scatter(x_gt, y_gt, z_gt, c='g', marker='^', label='Ground Truth')

    # Define the connections between joints based on the joint mapping
    connections = [
        # Head and Face
        (10, 9),  # Head to Nose
        (9, 8),   # Nose to Neck

        # Torso
        (8, 7),   # Neck to Stomach
        (7, 0),   # Stomach to Central Hip
        (0, 1),   # Central Hip to Left Hip
        (0, 4),   # Central Hip to Right Hip

        # Right Arm
        (8, 14), (14, 15), (15, 16),  # Neck to Right Shoulder to Right Elbow to Right Wrist
        (16, 23), (16, 24),           # Right Wrist to Right Palm and Fingers

        # Left Arm
        (8, 11), (11, 12), (12, 13),  # Neck to Left Shoulder to Left Elbow to Left Wrist
        (13, 21), (13, 22),           # Left Wrist to Left Palm and Fingers

        # Right Leg
        (4, 5), (5, 6), (6, 19), (6, 20),  # Right Hip to Right Knee to Right Ankle to Right Toe

        # Left Leg
        (1, 2), (2, 3), (3, 17), (3, 18)   # Left Hip to Left Knee to Left Ankle to Left Toe
    ]

    # Plot the lines connecting the joints for predicted joints
    for idx1, idx2 in connections:
        ax.plot(
            [x_pred[idx1], x_pred[idx2]],
            [y_pred[idx1], y_pred[idx2]],
            [z_pred[idx1], z_pred[idx2]],
            c='r'
        )

    # Plot the lines connecting the joints for ground truth joints
    for idx1, idx2 in connections:
        ax.plot(
            [x_gt[idx1], x_gt[idx2]],
            [y_gt[idx1], y_gt[idx2]],
            [z_gt[idx1], z_gt[idx2]],
            c='g'
        )

    # Set equal axis scaling for better visualization
    max_range = np.array([x_gt.max() - x_gt.min(), y_gt.max() - y_gt.min(), z_gt.max() - z_gt.min()]).max() / 2.0
    mid_x = (x_gt.max() + x_gt.min()) * 0.5
    mid_y = (y_gt.max() + y_gt.min()) * 0.5
    mid_z = (z_gt.max() + z_gt.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set labels and adjust the view angle
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=60)  # Adjust viewing angle for better understanding

    # Add a legend
    ax.legend()

    # Improve layout
    plt.tight_layout()

    if save:
        output_path = os.path.join(output_dir, f'{exercise_key}_frame_{frame_idx}_comparison.jpg')
        plt.savefig(output_path, format='jpg')
        print(f"Saved comparison plot for {exercise_key} frame {frame_idx}")

    # Return the figure for Wandb logging if required
    if return_figure:
        return fig

    # Close the figure to free memory
    plt.close(fig)

# Custom Dataset Class
class JointsDataset(Dataset):
    def __init__(self, mega_dict, camera_ids, model_name):
        self.samples = []
        self.camera_ids = camera_ids
        self.model_name = model_name
        for exercise_key, data in mega_dict.items():
            joints2d = data["joints2d"]
            joints3d = data["gt"]
            for frame_key in joints2d.keys():
                sample = {
                    'joints2d': joints2d[frame_key],
                    'joints3d': joints3d[frame_key],
                    'exercise_key': exercise_key,
                    'frame_idx': frame_key
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        joints2d_list = []
        for cam_id in self.camera_ids:
            cam_joints = sample['joints2d'].get(cam_id, None)
            if cam_joints is not None:
                joints2d_list.append(cam_joints)  # Shape: [num_joints_2d, 2]
            else:
                # Fill missing camera data with zeros
                joints2d_list.append([[0.0, 0.0]] * config['num_joints_2d'])

        # Prepare input based on model type
        if self.model_name == 'CNN':
            # For CNN, we can reshape the data to resemble an image
            joints2d_array = np.array(joints2d_list)  # Shape: [num_cameras, num_joints_2d, 2]
            # Expand dimensions to [channels, height, width]
            joints2d_array = joints2d_array.transpose(2, 0, 1)  # Shape: [2, num_cameras, num_joints_2d]
            joints2d_tensor = torch.tensor(joints2d_array, dtype=torch.float32)
        elif self.model_name == 'RNN':
            # For RNN, we can treat cameras as sequence
            joints2d_array = np.array(joints2d_list)  # Shape: [num_cameras, num_joints_2d, 2]
            joints2d_array = joints2d_array.reshape(len(self.camera_ids), -1)  # Shape: [num_cameras, num_joints_2d*2]
            joints2d_tensor = torch.tensor(joints2d_array, dtype=torch.float32)
        else:
            # For FCNN
            joints2d_array = np.array(joints2d_list).flatten()
            joints2d_tensor = torch.tensor(joints2d_array, dtype=torch.float32)

        joints3d_array = np.array(sample['joints3d']).flatten()
        joints3d_tensor = torch.tensor(joints3d_array, dtype=torch.float32)

        return joints2d_tensor, joints3d_tensor, sample['exercise_key'], sample['frame_idx']

# Models

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
        # Assuming input shape [batch_size, 2, num_cameras, num_joints_2d]
        # Output shape will be [batch_size, in_channels * num_cameras * num_joints_2d]
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

# Mean Per Joint Position Error (MPJPE)
def mpjpe(predicted, target):
    predicted = predicted.view(-1, config['num_joints_3d'], 3)
    target = target.view(-1, config['num_joints_3d'], 3)
    error = torch.norm(predicted - target, dim=2).mean()
    return error

def load_data(mega_dict_path):
    with open(mega_dict_path, 'r') as f:
        mega_dict = json.load(f)
    return mega_dict

def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_mpjpe = 0.0
    for batch in loader:
        inputs, targets, _, _ = batch  # Unpack all returned values
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

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_mpjpe = 0.0
    images = []
    max_images = 5  # Number of samples to plot
    image_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets, exercise_keys, frame_idxs = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mpjpe_value = mpjpe(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            running_mpjpe += mpjpe_value.item() * inputs.size(0)

            # For plotting
            batch_size = inputs.size(0)
            for i in range(batch_size):
                if image_count >= max_images:
                    break
                # Get the predicted and ground truth joints
                pred_joints = outputs[i].cpu().numpy().reshape(config['num_joints_3d'], 3)
                gt_joints = targets[i].cpu().numpy().reshape(config['num_joints_3d'], 3)
                exercise_key = exercise_keys[i]
                frame_idx = frame_idxs[i]

                # Plot both predicted and ground truth joints and save to folder
                fig_comparison = plot_pred_and_gt_joints(pred_joints, gt_joints, exercise_key, frame_idx, save=True, return_figure=True)
                comparison_image = wandb.Image(fig_comparison, caption=f"Comparison {exercise_key} frame {frame_idx}")
                plt.close(fig_comparison)  # Ensure the figure is closed

                images.append({
                    "Comparison": comparison_image,
                    "exercise_key": exercise_key,
                    "frame_idx": frame_idx
                })

                image_count += 1
            if image_count >= max_images:
                break

    # Log images to wandb
    if images:
        for img_dict in images:
            exercise_key = img_dict['exercise_key']
            frame_idx = img_dict['frame_idx']
            wandb.log({
                f"Validation/{exercise_key}/frame_{frame_idx}/Comparison": img_dict["Comparison"],
            })

    epoch_loss = running_loss / len(loader.dataset)
    epoch_mpjpe = running_mpjpe / len(loader.dataset)
    return epoch_loss, epoch_mpjpe

def test(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_mpjpe = 0.0
    images = []
    max_images = 5  # Number of samples to plot
    image_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets, exercise_keys, frame_idxs = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mpjpe_value = mpjpe(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            running_mpjpe += mpjpe_value.item() * inputs.size(0)

            # For plotting
            batch_size = inputs.size(0)
            for i in range(batch_size):
                if image_count >= max_images:
                    break
                # Get the predicted and ground truth joints
                pred_joints = outputs[i].cpu().numpy().reshape(config['num_joints_3d'], 3)
                gt_joints = targets[i].cpu().numpy().reshape(config['num_joints_3d'], 3)
                exercise_key = exercise_keys[i]
                frame_idx = frame_idxs[i]

                # Plot both predicted and ground truth joints and save to folder
                fig_comparison = plot_pred_and_gt_joints(pred_joints, gt_joints, exercise_key, frame_idx, save=True, return_figure=True)
                comparison_image = wandb.Image(fig_comparison, caption=f"Comparison {exercise_key} frame {frame_idx}")
                plt.close(fig_comparison)  # Ensure the figure is closed

                images.append({
                    "Comparison": comparison_image,
                    "exercise_key": exercise_key,
                    "frame_idx": frame_idx
                })

                image_count += 1
            if image_count >= max_images:
                break

    # Log images to wandb
    if images:
        for img_dict in images:
            exercise_key = img_dict['exercise_key']
            frame_idx = img_dict['frame_idx']
            wandb.log({
                f"Test/{exercise_key}/frame_{frame_idx}/Comparison": img_dict["Comparison"],
            })

    epoch_loss = running_loss / len(loader.dataset)
    epoch_mpjpe = running_mpjpe / len(loader.dataset)
    return epoch_loss, epoch_mpjpe

# Main Training Loop
def main():
    """ SELECT MEGA DICT OR THE FILTERED ONE """
    # Use the original dataset
    mega_dict_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/' \
                     'mega_dict_filtered.json'
    mega_dict = load_data(mega_dict_path)

    # Prepare Dataset and DataLoader
    dataset = JointsDataset(mega_dict, config['camera_ids'], config['model_name'])

    # Split the dataset into train, val, and test sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(42)
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
    if config['model_name'] == 'FCNN':
        input_size = len(config['camera_ids']) * config['num_joints_2d'] * 2  # x and y coordinates
        output_size = config['num_joints_3d'] * 3  # x, y, z coordinates
        model = FCNNModel(input_size, output_size, config['hidden_sizes'], config['dropout'])
    elif config['model_name'] == 'CNN':
        output_size = config['num_joints_3d'] * 3
        extra_layers = config['cnn_extra_layers']
        model = CNNModel(output_size, extra_layers)
    elif config['model_name'] == 'RNN':
        input_size = config['num_joints_2d'] * 2  # x and y coordinates per camera
        hidden_size = 256  # You can adjust this
        num_layers = 2  # You can adjust this
        output_size = config['num_joints_3d'] * 3
        model = RNNModel(input_size, hidden_size, num_layers, output_size, config['dropout'])
    else:
        raise ValueError(f"Model {config['model_name']} not implemented.")

    model.to(device)

    # Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training Loop
    for epoch in range(config['epochs']):
        train_loss, train_mpjpe = train(model, train_loader, criterion, optimizer)
        val_loss, val_mpjpe = validate(model, val_loader, criterion)

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

    test_loss, test_mpjpe = test(model, test_loader, criterion)
    wandb.log({
        'test_loss': test_loss,
        'test_mpjpe': test_mpjpe
    })
    print(f"Test Loss: {test_loss:.4f}, Test MPJPE: {test_mpjpe:.4f}")

    # Save the model
    paths_dir = 'paths'
    os.makedirs(paths_dir, exist_ok=True)
    model_filename = f'{config["model_name"]}{len(config["cnn_extra_layers"])}_bs{config["batch_size"]}_ep{config["epochs"]}_joints_model.pth'
    model_path = os.path.join(paths_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)
    print("Model saved.")

if __name__ == '__main__':
    main()