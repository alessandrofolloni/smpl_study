import os
import json
import random
from datetime import datetime

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

from utils import (
    set_seed,
    load_mean_std,
    extract_2d_keypoints,
    load_joints3d,
    plot_pred_and_gt_joints,
    init_yolo_model
)

# ----------------------------
# Set Seed and Device
# ----------------------------
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Define Dataset and Constants
# ----------------------------
dataset_root = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train'
num_joints_2d = 17
num_joints_3d = 25

# ----------------------------
# Common Configuration
# ----------------------------
common_config = {
    'model_name': 'Transformer',
    'num_joints_3d': num_joints_3d,
    'mega_dict_file': os.path.join(dataset_root, "mega_dict_3dSMPLX.json")
}


# ----------------------------
# Utility Functions
# ----------------------------
def axis_angle_to_rotation_matrices(axis_angles):
    r = R.from_rotvec(axis_angles)
    return r.as_matrix()


def rotation_matrices_to_axis_angle(rotation_matrices):
    r = R.from_matrix(rotation_matrices)
    return r.as_rotvec()


# ----------------------------
# Define Transformer Model for 2D-to-3D
# ----------------------------
class TransformerJointsModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dropout, seq_length):
        super(TransformerJointsModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * seq_length, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_length, input_dim)
        batch_size, seq_length, _ = x.shape
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1).reshape(batch_size, -1)
        x = self.fc(x)
        return x


# ----------------------------
# 2D-to-3D Model Configuration (Transformer)
# ----------------------------
td2d_model_path = '/public.hpc/alessandro.folloni2/smpl_study/models/Transformer_bs64_ep200_joints_model.pth'
# Using all available cameras (4) for 2D keypoints extraction.
all_cameras = ['50591643', '58860488', '60457274', '65906101']
# You can choose a subset if needed, e.g., cameras = ['58860488', '60457274', '65906101']
cameras = all_cameras
seq_length = len(cameras) * num_joints_2d  # 4*17 = 68

td2d_config = {
    'd_model': 256,
    'nhead': 8,
    'num_layers': 4,
    'dropout': 0.1,
    'input_dim': 2,
    'seq_length': seq_length,
    'output_dim': num_joints_3d * 3  # 75
}

td2d_model = TransformerJointsModel(
    input_dim=td2d_config['input_dim'],
    output_dim=td2d_config['output_dim'],
    d_model=td2d_config['d_model'],
    nhead=td2d_config['nhead'],
    num_layers=td2d_config['num_layers'],
    dropout=td2d_config['dropout'],
    seq_length=td2d_config['seq_length']
).to(device)
from utils import load_model  # Ensure load_model is imported

td2d_model = load_model(td2d_model, td2d_model_path, device)
td2d_model.eval()
print("2D-to-3D Transformer model loaded.")

# ----------------------------
# Other Paths and Model Initialization
# ----------------------------
yolo_weights_path = '/public.hpc/alessandro.folloni2/smpl_study/pretrained_yolos/yolov8s-pose.pt'

mean2d, std2d = load_mean_std(dataset_root)
pose_model = init_yolo_model(yolo_weights_path, device)
if pose_model is None:
    raise RuntimeError("Failed to load YOLO model.")

# ----------------------------
# Subject and Exercise Selection
# ----------------------------
subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
subjects.sort()
subject = random.choice(subjects)
subject_path = os.path.join(dataset_root, subject)
print(f"Selected subject: {subject}")

joints3d_folder = os.path.join(subject_path, 'joints3d_25')
exercises = [f for f in os.listdir(joints3d_folder) if f.endswith('.json')]
if not exercises:
    raise RuntimeError("No exercise files found for the selected subject.")
exercise_file = random.choice(exercises)
exercise_name = os.path.splitext(exercise_file)[0]
print(f"Selected exercise: {exercise_name}")

# ----------------------------
# Prepare Video Paths for All Cameras
# ----------------------------
videos_dir = os.path.join(subject_path, 'videos')
cam_videos = {}
for cam_id in all_cameras:
    cam_path = os.path.join(videos_dir, cam_id)
    if os.path.isdir(cam_path):
        video_candidates = [v for v in os.listdir(cam_path) if v.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        matched = [v for v in video_candidates if os.path.splitext(v)[0] == exercise_name]
        if matched:
            cam_videos[cam_id] = os.path.join(cam_path, matched[0])
if len(cam_videos) < len(cameras):
    raise RuntimeError("Not all required cameras have matching videos.")
print("Using videos from cameras:", cam_videos)

# ----------------------------
# Create Output Folder with Timestamp
# ----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join("/public.hpc/alessandro.folloni2/smpl_study/results",
                             f"{subject}_{exercise_name}_{timestamp}")
os.makedirs(output_folder, exist_ok=True)
print(f"Saving results to: {output_folder}")

log_file = os.path.join(output_folder, "experiment_log.txt")
log = open(log_file, "w")
log.write(f"Experiment log for subject {subject}, exercise {exercise_name}\n")
log.write(f"Timestamp: {timestamp}\n\n")

# ----------------------------
# Extract 2D Keypoints from Each Camera
# ----------------------------
joints2d_data = {}
for cam_id in cameras:
    v_path = cam_videos[cam_id]
    log.write(f"Extracting 2D keypoints from camera {cam_id}: {v_path}\n")
    keypoints = extract_2d_keypoints(v_path, pose_model, device, mean2d, std2d, num_joints_2d)
    if len(keypoints) == 0:
        log.write(f"Error: No keypoints found in camera {cam_id}.\n")
        raise RuntimeError(f"No keypoints found in camera {cam_id}.")
    joints2d_data[cam_id] = keypoints
log.write("\n2D keypoints extraction complete.\n")

# ----------------------------
# Synchronize Frames Across Cameras
# ----------------------------
lengths = [len(joints2d_data[cam_id]) for cam_id in cameras]
min_length = min(lengths)
log.write(f"Frames available per camera: {lengths}, using minimum: {min_length}\n")
for cam_id in cameras:
    joints2d_data[cam_id] = joints2d_data[cam_id][:min_length]

# ----------------------------
# Load 3D Ground Truth Joints
# ----------------------------
joints3d_file_path = os.path.join(joints3d_folder, exercise_file)
joints3d_array = load_joints3d(joints3d_file_path)
joints3d_array = joints3d_array[:min_length]
log.write(f"Loaded 3D ground truth with shape: {joints3d_array.shape}\n\n")

# ----------------------------
# Select 3 Random Frame Indices
# ----------------------------
random_frames = random.sample(range(min_length), 3)
log.write(f"Selected frame indices: {random_frames}\n")

tracking_info = {
    "subject": subject,
    "exercise": exercise_name,
    "selected_frames": random_frames,
    "video_paths": cam_videos
}
tracking_path = os.path.join(output_folder, f"{subject}_{exercise_name}_tracking.json")
with open(tracking_path, "w") as f:
    json.dump(tracking_info, f, indent=4)
log.write(f"Saved tracking info to: {tracking_path}\n\n")

# ----------------------------
# Process Each Random Frame
# ----------------------------
for frame_idx in random_frames:
    log.write(f"Processing frame index: {frame_idx}\n")

    # For each camera, get the 2D keypoints for the selected frame
    keypoints_list = []
    for cam_id in cameras:
        keypoints_cam = np.array(joints2d_data[cam_id][frame_idx])  # shape: (17,2)
        keypoints_list.append(keypoints_cam)
    # Concatenate keypoints -> shape: (68,2)
    frame_2d = np.concatenate(keypoints_list, axis=0)
    log.write(f"2D keypoints for frame {frame_idx} shape: {frame_2d.shape}\n")

    # ----------------------------
    # 2D-to-3D Prediction using Transformer
    # ----------------------------
    td2d_input = torch.tensor(frame_2d, dtype=torch.float32).unsqueeze(0).to(device)  # (1,68,2)
    with torch.no_grad():
        pred_3d_flat = td2d_model(td2d_input).cpu().numpy().squeeze()  # (75,)
    if pred_3d_flat.shape[0] != num_joints_3d * 3:
        raise RuntimeError(f"Unexpected predicted 3D joints shape: {pred_3d_flat.shape}")
    pred_3d = pred_3d_flat.reshape(num_joints_3d, 3)
    gt_3d = joints3d_array[frame_idx]
    log.write(f"Predicted 3D joints shape: {pred_3d.shape}\n")
    log.write(f"Ground truth 3D joints shape: {gt_3d.shape}\n")

    # Save 3D joints comparison plot
    plot_path = os.path.join(output_folder, f"{subject}_{exercise_name}_frame_{frame_idx:04d}_joints.jpg")
    fig = plot_pred_and_gt_joints(pred_3d, gt_3d, plot_path)
    plt.close(fig)
    log.write(f"Saved 3D joints comparison plot to: {plot_path}\n")

    # (The script stops here, as requested)
    log.write(f"2D-to-3D prediction complete for frame {frame_idx}.\n\n")

print("Processing complete for the selected frames.")
log.close()