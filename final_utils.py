import os
import json
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from smplx import create as smplx_create
import trimesh
import pyrender
import imageio
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_mean_std(dataset_root):
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
            if kp_array.shape[0] != num_joints_2d:
                if kp_array.shape[0] > num_joints_2d:
                    kp_array = kp_array[:num_joints_2d, :]
                else:
                    padding = np.zeros((num_joints_2d - kp_array.shape[0], 2))
                    kp_array = np.vstack((kp_array, padding))

            normalized_x = (kp_array[:, 0] - mean_x) / std_x
            normalized_y = (kp_array[:, 1] - mean_y) / std_y
            normalized_keypoints = np.stack((normalized_x, normalized_y), axis=1)
            kp_list = normalized_keypoints.tolist()
        else:
            kp_list = [[0.0, 0.0] for _ in range(num_joints_2d)]

        all_keypoints.append(kp_list)

    return all_keypoints


def load_joints3d(joints3d_file_path):
    with open(joints3d_file_path, 'r') as f:
        data = json.load(f)
    if "joints3d_25" not in data:
        raise ValueError(f"Key 'joints3d_25' not found in {joints3d_file_path}.")
    joints3d_array = np.array(data["joints3d_25"])
    if joints3d_array.ndim != 3 or joints3d_array.shape[1:] != (25, 3):
        raise ValueError(f"Unexpected shape for 3D joints in {joints3d_file_path}: {joints3d_array.shape}")
    return joints3d_array


def plot_pred_and_gt_joints(pred_joints, gt_joints, output_path):
    if pred_joints.shape[1] != 3 or gt_joints.shape[1] != 3:
        raise ValueError("Predicted and GT joints must have shape Nx3.")

    x_pred, y_pred, z_pred = pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2]
    x_gt, y_gt, z_gt = gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_pred, y_pred, z_pred, c='r', marker='o', label='Predicted')
    ax.scatter(x_gt, y_gt, z_gt, c='g', marker='^', label='GT')

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


class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * 25, output_size)  # Assuming 25 joints (sequence length)

    def forward(self, x):
        # x shape: (batch_size, 25, 3)
        x = self.input_projection(x)
        x = self.transformer_encoder(x)  # (batch_size, 25, d_model)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


def load_model(model, model_path, device):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}.")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def axis_angle_to_rotation_matrices(axis_angles):
    rot = R.from_rotvec(axis_angles)
    return rot.as_matrix()


def save_smplx_json(smplx_data, output_path):
    # Convert the numpy array to a list for JSON serialization.
    smplx_list = smplx_data.tolist()
    with open(output_path, 'w') as f:
        json.dump(smplx_list, f, indent=4)
    print(f"Saved SMPLX JSON to {output_path}")




def load_smplx_json(json_path):
    with open(json_path, 'r') as f:
        smplx_data = json.load(f)
    return smplx_data


def save_mesh(mesh, output_path):
    mesh.export(output_path)
    print(f"Mesh saved to {output_path}")


def run_inference(model, input_data, device):
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)
    return output.cpu().numpy().squeeze()