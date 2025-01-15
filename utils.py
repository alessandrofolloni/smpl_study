import os
import json
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
from tqdm import tqdm


def set_seed(seed=42):
    """
    Set the random seed for Python's random, NumPy, and PyTorch to ensure reproducibility.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_mean_std(dataset_root):
    """
    Load the mean and standard deviation values used for normalization from a JSON file.
    """
    mean_std_path = os.path.join(dataset_root, 'joints2d_mean_std.json')
    if not os.path.exists(mean_std_path):
        raise FileNotFoundError(f"Mean and std file not found at {mean_std_path}.")
    with open(mean_std_path, 'r') as f:
        mean_std_data = json.load(f)
    mean2d = mean_std_data.get('mean2d', None)
    std2d = mean_std_data.get('std2d', None)
    if mean2d is None or std2d is None:
        raise ValueError("Mean2d or std2d not found in the mean and std file.")
    return mean2d, std2d


def extract_2d_keypoints(video_path, model, device, mean2d, std2d, num_joints_2d=17, conf=0.5, imgsz=640):
    """
    Extract and normalize 2D keypoints from a video using YOLO pose model.
    """
    mean_x, mean_y = mean2d
    std_x, std_y = std2d
    std_x = std_x if std_x != 0 else 1
    std_y = std_y if std_y != 0 else 1

    all_keypoints = []
    results = model(
        source=video_path,
        stream=True,
        device=device,
        conf=conf,
        imgsz=imgsz,
        verbose=False
    )

    for result in tqdm(results, desc=f"Extracting keypoints from {os.path.basename(video_path)}"):
        if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints) > 0:
            kp = result.keypoints[0]
            kp_array = kp.xy.cpu().numpy().reshape(-1, 2)
            # Ensure the correct number of joints
            if kp_array.shape[0] != num_joints_2d:
                if kp_array.shape[0] > num_joints_2d:
                    kp_array = kp_array[:num_joints_2d, :]
                else:
                    padding = np.zeros((num_joints_2d - kp_array.shape[0], 2))
                    kp_array = np.vstack((kp_array, padding))

            # Normalize
            normalized_x = (kp_array[:, 0] - mean_x) / std_x
            normalized_y = (kp_array[:, 1] - mean_y) / std_y
            normalized_keypoints = np.stack((normalized_x, normalized_y), axis=1)
            kp_list = normalized_keypoints.tolist()
        else:
            # No detections: fill with zeros
            kp_list = [[0.0, 0.0] for _ in range(num_joints_2d)]

        all_keypoints.append(kp_list)

    return all_keypoints


def load_joints3d(joints3d_file_path):
    """
    Load 3D joints data from a JSON file. The JSON should contain a 'joints3d_25' key.
    """
    with open(joints3d_file_path, 'r') as f:
        data = json.load(f)
    if "joints3d_25" not in data:
        raise ValueError(f"Key 'joints3d_25' not found in {joints3d_file_path}.")
    joints3d_array = np.array(data["joints3d_25"])
    if joints3d_array.ndim != 3 or joints3d_array.shape[1:] != (25, 3):
        raise ValueError(f"Unexpected shape for 3D joints in {joints3d_file_path}: {joints3d_array.shape}")
    return joints3d_array


def plot_pred_and_gt_joints(pred_joints, gt_joints, output_path):
    """
    Plot predicted and ground truth 3D joints in the same 3D scatter plot and save as a JPG image.
    """
    if pred_joints.shape[1] != 3:
        raise ValueError(f"Predicted joints must have shape Nx3, but have {pred_joints.shape}")
    if gt_joints.shape[1] != 3:
        raise ValueError(f"GT joints must have shape Nx3, but have {gt_joints.shape}")

    x_pred, y_pred, z_pred = pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2]
    x_gt, y_gt, z_gt = gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot predicted joints
    ax.scatter(x_pred, y_pred, z_pred, c='r', marker='o', label='Predicted')

    # Plot ground truth joints
    ax.scatter(x_gt, y_gt, z_gt, c='g', marker='^', label='GT')

    # Connections for SMPL skeleton (example)
    connections = [
        (10, 9), (9, 8),
        (8, 7), (7, 0), (0, 1), (0, 4),
        (8, 14), (14, 15), (15, 16), (16, 23), (16, 24),
        (8, 11), (11, 12), (12, 13), (13, 21), (13, 22),
        (4, 5), (5, 6), (6, 19), (6, 20),
        (1, 2), (2, 3), (3, 17), (3, 18)
    ]

    for idx1, idx2 in connections:
        ax.plot([x_pred[idx1], x_pred[idx2]],
                [y_pred[idx1], y_pred[idx2]],
                [z_pred[idx1], z_pred[idx2]], c='r')
        ax.plot([x_gt[idx1], x_gt[idx2]],
                [y_gt[idx1], y_gt[idx2]],
                [z_gt[idx1], z_gt[idx2]], c='g')

    max_range = np.array([x_gt.max() - x_gt.min(), y_gt.max() - y_gt.min(), z_gt.max() - z_gt.min()]).max() / 2.0
    mid_x = (x_gt.max() + x_gt.min()) * 0.5
    mid_y = (y_gt.max() + y_gt.min()) * 0.5
    mid_z = (z_gt.max() + z_gt.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=60)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, format='jpg')
    plt.close(fig)
    print(f"Comparison plot saved at {output_path}")


def init_yolo_model(yolo_weights_path, device):
    """
    Initialize the YOLO model from ultralytics.
    """
    if not os.path.exists(yolo_weights_path):
        print(f"Model file not found at {yolo_weights_path}")
        return None
    print(f"Loading YOLO model from {yolo_weights_path}")
    pose_model = YOLO(yolo_weights_path)
    if pose_model is None:
        print("Failed to load YOLO model.")
        return None
    pose_model.to(device)
    return pose_model


# ====================
# Model Definitions
# ====================

class FCNNModel(nn.Module):
    """
    Fully Connected Neural Network (FCNN) Model
    """

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


class CNNModel(nn.Module):
    """
    Convolutional Neural Network (CNN) Model for 2D keypoints input.
    """

    def __init__(self, output_size, extra_layers, num_cameras, num_joints_2d, dropout):
        super(CNNModel, self).__init__()
        layers = []
        # Initial Conv layer: input is (2, num_cameras, num_joints_2d)
        layers.append(nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_channels = 16

        # Additional Conv layers
        for out_channels in extra_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        layers.append(nn.Flatten())
        self.conv_layers = nn.Sequential(*layers)

        # Flattened size = in_channels * num_cameras * num_joints_2d
        flattened_size = in_channels * num_cameras * num_joints_2d
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, output_size)
        )

    def forward(self, x):
        # x expected shape: (batch_size, 2, num_cameras, num_joints_2d)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class RNNModel(nn.Module):
    """
    Recurrent Neural Network (RNN) model, specifically LSTM-based.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # output from the last time step
        out = self.fc(out)
        return out


class TransformerModel(nn.Module):
    """
    Transformer-based model for sequence data.
    """

    def __init__(self, input_size, output_size, sequence_length, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * sequence_length, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = self.input_projection(x)
        x = self.transformer_encoder(x)  # (batch_size, sequence_length, d_model)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


def load_model(model_path, model_type='Transformer', num_joints_3d=25, output_size=188):
    """
    Load a trained model from a .pth file.

    Parameters:
    - model_path (str): Path to the model weights (.pth).
    - model_type (str): One of ['FCNN', 'CNN', 'RNN', 'Transformer'].
    - num_joints_3d (int): Number of 3D joints (used for input size in some models).
    - output_size (int): Size of the model's output vector.

    Returns:
    - model (nn.Module): The loaded model in eval mode.
    - device (torch.device): The device the model is on.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # For simplicity, only showing Transformer case as per previous example
    # Adjust as needed for other model types
    if model_type == 'Transformer':
        input_size = 3  # x, y, z per joint
        d_model = 256
        nhead = 8
        num_layers = 6
        dropout = 0.1
        sequence_length = num_joints_3d
        model = TransformerModel(
            input_size=input_size,
            output_size=output_size,
            sequence_length=sequence_length,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError("Please implement loading logic for your chosen model_type.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def smplx_parameters_from_vector(params_vector):
    """
    Given a predicted params_vector of shape (188,),
    split it into the individual SMPL-X parameters.

    The parameter order is expected to be:
    transl (3),
    global_orient (3),
    body_pose (21*3 = 63),
    betas (10),
    left_hand_pose (15*3 = 45),
    right_hand_pose (15*3 = 45),
    jaw_pose (3),
    leye_pose (3),
    reye_pose (3),
    expression (10)

    Returns a dictionary of parameter names to arrays.
    """
    transl = params_vector[0:3]
    global_orient = params_vector[3:6]
    body_pose = params_vector[6:69]
    betas = params_vector[69:79]
    left_hand_pose = params_vector[79:124]
    right_hand_pose = params_vector[124:169]
    jaw_pose = params_vector[169:172]
    leye_pose = params_vector[172:175]
    reye_pose = params_vector[175:178]
    expression = params_vector[178:188]

    return {
        'transl': transl,
        'global_orient': global_orient,
        'body_pose': body_pose,
        'betas': betas,
        'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,
        'jaw_pose': jaw_pose,
        'leye_pose': leye_pose,
        'reye_pose': reye_pose,
        'expression': expression
    }