import os
import json
import numpy as np
import torch
import torch.nn as nn
import wandb
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader

config = {
    'model_type': 'FCNN',
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 60,
    'num_joints_3d': 25,
    'train_split': 0.75,
    'val_split': 0.2,
    'test_split': 0.05,
    'hidden_sizes': [8192, 4096, 2048, 1024, 512],
    'cnn_channels': [],  # Non usato per FCNN
    'cnn_kernel_size': 0,
    'pool_every': 0,
    'transformer_d_model': 0,
    'transformer_nhead': 0,
    'transformer_num_layers': 0,
    'dropout': 0.2,
    'output_size': 188,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# For transformers try [256, 8, 4, 0.1, bs=32], [512, 16, 6, 0.2, bs=16], [128, 4, 2, 0.1, bs=64]


# Generate a unique output folder based on model parameters
model_params = ''
if config['model_type'] == 'FCNN':
    model_params = f"hidden_sizes_{'_'.join(map(str, config['hidden_sizes']))}_dropout_{config['dropout']}"
elif config['model_type'] == 'CNN':
    model_params = f"channels_{'_'.join(map(str, config['cnn_channels']))}_kernel_{config['cnn_kernel_size']}_dropout_{config['dropout']}"
elif config['model_type'] == 'Transformer':
    model_params = f"dmodel_{config['transformer_d_model']}_nhead_{config['transformer_nhead']}_layers_{config['transformer_num_layers']}_dropout_{config['dropout']}"
else:
    model_params = 'unknown_model'

run_name = f"{config['model_type']}_bs{config['batch_size']}_lr{config['learning_rate']}_epochs{config['epochs']}_{model_params}"

wandb.init(
    project='3D_Joints_to_SMPLX',
    entity='alessandrofolloni',
    config=config,
    name=run_name,
)

device = torch.device(config['device'])

# Create an output directory with the run name
output_folder = os.path.join('runs', run_name)
os.makedirs(output_folder, exist_ok=True)
print(f"Outputs will be saved to: {output_folder}")

class JointsToSMPLXDataset(Dataset):
    def __init__(self, mega_dict):
        self.samples = []
        required_keys = ['transl', 'global_orient', 'body_pose', 'betas', 'left_hand_pose',
                         'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']

        for exercise_key, frames in mega_dict.items():
            for frame_key, frame_data in frames.items():
                gt_params = frame_data['gt']
                if all(key in gt_params for key in required_keys):
                    sample = {
                        'joints3d': frame_data['joints3d'],
                        'gt': gt_params,
                        'exercise_key': exercise_key,
                        'frame_idx': frame_key
                    }
                    self.samples.append(sample)
                else:
                    print(f"Skipping sample {exercise_key} frame {frame_key} due to missing SMPL-X parameters.")
        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def rotation_matrices_to_axis_angles(self, rotation_matrices):
        """
        Converts rotation matrices to axis-angle representations.

        Parameters:
            rotation_matrices (numpy.ndarray): Rotation matrices of shape (..., 3, 3)

        Returns:
            axis_angles (numpy.ndarray): Axis-angle vectors of shape (..., 3)
        """
        rot_obj = R.from_matrix(rotation_matrices)
        axis_angles = rot_obj.as_rotvec()
        return axis_angles

    def __getitem__(self, idx):
        sample = self.samples[idx]
        joints3d = np.array(sample['joints3d'])  # Shape: (25, 3)
        gt_params = sample['gt']

        try:
            # Process SMPL-X parameters
            # Convert rotation matrices to axis-angle representations
            global_orient_mat = np.array(gt_params['global_orient'])  # Shape: (1, 3, 3)
            body_pose_mat = np.array(gt_params['body_pose'])          # Shape: (21, 3, 3)
            left_hand_pose_mat = np.array(gt_params['left_hand_pose'])  # Shape: (15, 3, 3)
            right_hand_pose_mat = np.array(gt_params['right_hand_pose'])  # Shape: (15, 3, 3)
            jaw_pose_mat = np.array(gt_params['jaw_pose'])            # Shape: (1, 3, 3)
            leye_pose_mat = np.array(gt_params['leye_pose'])          # Shape: (1, 3, 3)
            reye_pose_mat = np.array(gt_params['reye_pose'])          # Shape: (1, 3, 3)

            # Convert rotation matrices to axis-angle
            global_orient_axis_angle = self.rotation_matrices_to_axis_angles(global_orient_mat)   # Shape: (1, 3)
            body_pose_axis_angle = self.rotation_matrices_to_axis_angles(body_pose_mat)           # Shape: (21, 3)
            left_hand_pose_axis_angle = self.rotation_matrices_to_axis_angles(left_hand_pose_mat) # Shape: (15, 3)
            right_hand_pose_axis_angle = self.rotation_matrices_to_axis_angles(right_hand_pose_mat)  # Shape: (15, 3)
            jaw_pose_axis_angle = self.rotation_matrices_to_axis_angles(jaw_pose_mat)             # Shape: (1, 3)
            leye_pose_axis_angle = self.rotation_matrices_to_axis_angles(leye_pose_mat)           # Shape: (1, 3)
            reye_pose_axis_angle = self.rotation_matrices_to_axis_angles(reye_pose_mat)           # Shape: (1, 3)

            # Collect all parameters
            transl = np.array(gt_params['transl'])               # Shape: (3,)
            betas = np.array(gt_params['betas'])                 # Shape: (10,)
            expression = np.array(gt_params['expression'])       # Shape: (10,)

            # Flatten all parameters into a single vector
            gt_vector = np.concatenate([
                transl,
                global_orient_axis_angle.flatten(),
                body_pose_axis_angle.flatten(),
                betas,
                left_hand_pose_axis_angle.flatten(),
                right_hand_pose_axis_angle.flatten(),
                jaw_pose_axis_angle.flatten(),
                leye_pose_axis_angle.flatten(),
                reye_pose_axis_angle.flatten(),
                expression
            ])

            # Convert to tensors
            # For CNN and Transformer, we need to reshape joints3d to (sequence_length, features)
            joints3d_tensor = torch.tensor(joints3d, dtype=torch.float32)  # Shape: (25, 3)
            gt_tensor = torch.tensor(gt_vector, dtype=torch.float32)

            return joints3d_tensor, gt_tensor, sample['exercise_key'], sample['frame_idx']
        except Exception as e:
            print(f"Error processing sample index {idx}: {e}")
            # Skip this sample
            return self.__getitem__((idx + 1) % self.num_samples)  # Get the next sample

# Define the model classes
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
        x = x.view(x.size(0), -1)
        return self.network(x)

class CNNModel(nn.Module):
    def __init__(self, input_channels, output_size, cnn_channels, kernel_size, dropout, pool_every):
        super(CNNModel, self).__init__()
        layers = []
        in_channels = input_channels
        sequence_length = 25  # Starting sequence length
        layer_idx = 0

        for out_channels in cnn_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            in_channels = out_channels
            layer_idx += 1

            # Apply MaxPool1d conditionally based on pool_every
            if pool_every > 0 and layer_idx % pool_every == 0 and sequence_length > 1:
                layers.append(nn.MaxPool1d(kernel_size=2, ceil_mode=True))
                sequence_length = int(np.ceil(sequence_length / 2))

            layers.append(nn.Dropout(dropout))

        self.cnn = nn.Sequential(*layers)
        conv_output_size = in_channels * sequence_length
        self.fc = nn.Linear(conv_output_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch_size, features, sequence_length)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * config['num_joints_3d'], output_size)

    def forward(self, x):
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch in loader:
        inputs, targets, _, _ = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            inputs, targets, _, _ = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def test(model, loader, output_folder):
    model.eval()
    running_loss = 0.0
    predicted_params_list = []
    gt_params_list = []
    exercise_keys = []
    frame_idxs = []
    with torch.no_grad():
        for batch in loader:
            inputs, targets, ex_keys, fr_idxs = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            # Save predictions and ground truths
            outputs_cpu = outputs.cpu().numpy()
            targets_cpu = targets.cpu().numpy()

            predicted_params_list.extend(outputs_cpu)
            gt_params_list.extend(targets_cpu)
            exercise_keys.extend(ex_keys)
            frame_idxs.extend(fr_idxs)

    epoch_loss = running_loss / len(loader.dataset)

    # Save the results to a file in the output folder
    predicted_params_path = os.path.join(output_folder, 'predicted_parameters.npz')
    np.savez(predicted_params_path,
             predicted_params=predicted_params_list,
             gt_params=gt_params_list,
             exercise_keys=exercise_keys,
             frame_idxs=frame_idxs)

    print(f"Predicted parameters saved to '{predicted_params_path}'.")

    return epoch_loss

def main():
    # Load mega_dict
    mega_dict_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/mega_dict_3dSMPLX.json'  # Adjust the path
    with open(mega_dict_path, 'r') as f:
        mega_dict = json.load(f)

    # Prepare Dataset and DataLoader
    dataset = JointsToSMPLXDataset(mega_dict)

    # Check if the dataset is empty
    if len(dataset) == 0:
        print("Dataset is empty after filtering. Please check the data.")
        return

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
    model_type = config['model_type']
    if model_type == 'FCNN':
        input_size = config['num_joints_3d'] * 3
        output_size = config['output_size']
        model = FCNNModel(
            input_size, output_size,
            config['hidden_sizes'],
            config['dropout']
        )
    elif model_type == 'CNN':
        input_channels = 3
        output_size = config['output_size']
        model = CNNModel(
            input_channels, output_size,
            config['cnn_channels'],
            config['cnn_kernel_size'],
            config['dropout'],
            config['pool_every']
        )
    elif model_type == 'Transformer':
        input_size = 3
        output_size = config['output_size']
        model = TransformerModel(
            input_size, output_size,
            config['transformer_d_model'],
            config['transformer_nhead'],
            config['transformer_num_layers'],
            config['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)

    # Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training Loop
    for epoch in range(config['epochs']):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        print(f"Epoch [{epoch + 1}/{config['epochs']}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")

    # Test the model and save predictions
    test_loss = test(model, test_loader, output_folder)
    wandb.log({'test_loss': test_loss})
    print(f"Test Loss: {test_loss:.4f}")

    # Save the model to the output folder
    model_save_path = os.path.join(output_folder, f'joints3d_to_smplx_model_{model_type}.pth')
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model_save_path)
    print(f"Model saved at '{model_save_path}'.")

if __name__ == '__main__':
    main()