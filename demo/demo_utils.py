import os
import random
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
from scipy.spatial.transform import Rotation as R
import trimesh
from ultralytics import YOLO

# ---------- Selection and logging functions ----------
def get_subjects(dataset_root):
    subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    if not subjects:
        raise ValueError("No subjects found in the dataset root!")
    return subjects

def select_random_subject(subjects, dataset_root):
    subject = random.choice(subjects)
    subject_path = os.path.join(dataset_root, subject)
    return subject, subject_path

def get_exercise_files(subject_path, folder_name='joints3d_25'):
    folder_path = os.path.join(subject_path, folder_name)
    if not os.path.exists(folder_path):
        raise ValueError(f"No '{folder_name}' folder found for subject {subject_path}")
    exercise_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    if not exercise_files:
        raise ValueError(f"No exercise files found in {folder_path}")
    return exercise_files, folder_path

def select_random_exercise(exercise_files):
    exercise_file = random.choice(exercise_files)
    exercise_name = os.path.splitext(exercise_file)[0]
    return exercise_name, exercise_file

def get_videos_for_exercise(subject_path, exercise_name, all_cameras, videos_folder_name='videos'):
    videos_info = {}
    videos_root = os.path.join(subject_path, videos_folder_name)
    log_info = ""
    for cam_id in all_cameras:
        cam_path = os.path.join(videos_root, cam_id)
        if os.path.isdir(cam_path):
            video_candidates = [f for f in os.listdir(cam_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            matched = [v for v in video_candidates if os.path.splitext(v)[0] == exercise_name]
            if matched:
                video_path = os.path.join(cam_path, matched[0])
                videos_info[cam_id] = video_path
                log_info += f"Camera {cam_id}: Found video at {video_path}\n"
            else:
                log_info += f"Camera {cam_id}: No video found for exercise {exercise_name}\n"
        else:
            log_info += f"Camera {cam_id}: Folder not found\n"
    return videos_info, log_info

def save_text_log(output_dir, filename, text):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w') as f:
        f.write(text)
    print(f"Log saved to: {file_path}")

def save_json_data(output_dir, filename, data):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON data saved to: {file_path}")

def get_min_total_frames(videos_info):
    min_frames = float('inf')
    for cam_id, video_path in videos_info.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        min_frames = min(min_frames, total_frames)
        cap.release()
    if min_frames <= 0 or min_frames == float('inf'):
        raise ValueError("No frames found in one or more videos.")
    return min_frames

def select_n_frames_from_videos(videos_info, output_dir, exercise_name, n=5):
    # Select n random frame indices (sorted for order)
    min_frames = get_min_total_frames(videos_info)
    frame_indices = sorted(random.sample(range(min_frames), n))
    frames_info_all = {}
    for frame_idx in frame_indices:
        frames_info = {}
        for cam_id, video_path in videos_info.items():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read frame {frame_idx} from video {video_path}")
            cam_output_dir = os.path.join(output_dir, "frames", f"frame_{frame_idx}")
            os.makedirs(cam_output_dir, exist_ok=True)
            image_filename = f"{exercise_name}_{cam_id}_frame_{frame_idx}.jpg"
            image_path = os.path.join(cam_output_dir, image_filename)
            cv2.imwrite(image_path, frame)
            cap.release()
            frames_info[cam_id] = {"frame_index": frame_idx, "image_path": image_path}
        frames_info_all[frame_idx] = frames_info
    return frames_info_all

# ---------- Ground Truth Loading ----------
def load_joints3d(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    if "joints3d_25" not in data:
        print(f"Errore: Chiave 'joints3d_25' non trovata in {file_path}.")
        return {}, 0
    joints3d_array = np.array(data["joints3d_25"])
    if joints3d_array.ndim != 3 or joints3d_array.shape[1:] != (25, 3):
        print(f"Errore: Forma inattesa per joints3d in {file_path}: {joints3d_array.shape}.")
        return {}, 0
    num_frames = joints3d_array.shape[0]
    joints3d = {}
    for frame_idx in range(num_frames):
        frame_key = f"frame_{frame_idx}"
        joints3d[frame_key] = joints3d_array[frame_idx].tolist()
    return joints3d, num_frames

# ---------- Multiview Processing and Transformer Models ----------
def process_multiview_frames(frames_info, mean2d, std2d, num_joints_2d=17):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO('pretrained_yolos/yolov8s-pose.pt')
    model.to(device)
    normalized_keypoints = {}
    for cam_id, info in frames_info.items():
        image_path = info["image_path"]
        results = model.predict(source=image_path, device=device, conf=0.5, imgsz=640)
        if results and hasattr(results[0], 'keypoints') and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kp = results[0].keypoints[0]
            kp_array = kp.xy.cpu().numpy().reshape(-1, 2)
            if kp_array.shape[0] != num_joints_2d:
                if kp_array.shape[0] > num_joints_2d:
                    kp_array = kp_array[:num_joints_2d, :]
                else:
                    padding = np.zeros((num_joints_2d - kp_array.shape[0], 2))
                    kp_array = np.vstack((kp_array, padding))
            mean_x, mean_y = mean2d
            std_x, std_y = std2d
            std_x = std_x if std_x != 0 else 1
            std_y = std_y if std_y != 0 else 1
            normalized_x = (kp_array[:, 0] - mean_x) / std_x
            normalized_y = (kp_array[:, 1] - mean_y) / std_y
            normalized_kp = np.stack((normalized_x, normalized_y), axis=1)
            normalized_keypoints[cam_id] = normalized_kp.tolist()
        else:
            normalized_keypoints[cam_id] = [[0.0, 0.0] for _ in range(num_joints_2d)]
    return normalized_keypoints

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, sequence_length, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * sequence_length, output_size)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

def load_transformer_model_joints3d(weights_path, device):
    input_size = 2; output_size = 75; sequence_length = 34
    d_model = 256; nhead = 2; num_layers = 12; dropout = 0.2
    model = TransformerModel(input_size, output_size, sequence_length, d_model, nhead, num_layers, dropout)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_joints3d(normalized_keypoints, transformer_model, camera_ids, num_joints_2d=17):
    keypoints_list = []
    for cam in camera_ids:
        kp = np.array(normalized_keypoints[cam])
        keypoints_list.append(kp)
    keypoints_stack = np.stack(keypoints_list, axis=0)
    input_tensor = keypoints_stack.reshape(1, -1, 2)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    device = next(transformer_model.parameters()).device
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = transformer_model(input_tensor)
    joints3d = output.cpu().numpy().reshape(25, 3)
    return joints3d

def load_transformer_model_smplx(weights_path, device):
    input_size = 3; output_size = 188; sequence_length = 25
    d_model = 384; nhead = 6; num_layers = 4; dropout = 0.2
    model = TransformerModel(input_size, output_size, sequence_length, d_model, nhead, num_layers, dropout)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def predict_smplx_params(joints3d, transformer_model):
    input_tensor = torch.tensor(joints3d, dtype=torch.float32).unsqueeze(0)
    device = next(transformer_model.parameters()).device
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = transformer_model(input_tensor)
    params = output.cpu().numpy().reshape(-1)
    return params

def plot_pred_and_gt_joints(pred_joints, gt_joints, exercise_key, frame_idx, output_dir='3d_plots', save=True, return_figure=False):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    if pred_joints.shape[1] != 3 or gt_joints.shape[1] != 3:
        raise ValueError("I joints devono avere forma Nx3.")
    x_pred, y_pred, z_pred = pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2]
    x_gt, y_gt, z_gt = gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_pred, y_pred, z_pred, c='r', marker='o', label='Predicted')
    ax.scatter(x_gt, y_gt, z_gt, c='g', marker='^', label='Ground Truth')
    connections = [
        (10, 9), (9, 8), (8, 7), (7, 0), (0, 1), (0, 4),
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
    max_range = np.array([x_gt.max()-x_gt.min(), y_gt.max()-y_gt.min(), z_gt.max()-z_gt.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (x_gt.max()+x_gt.min())*0.5, (y_gt.max()+y_gt.min())*0.5, (z_gt.max()+z_gt.min())*0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=60)
    ax.legend()
    plt.tight_layout()
    if save:
        output_path = os.path.join(output_dir, f'{exercise_key}_frame_{frame_idx}_comparison.jpg')
        plt.savefig(output_path, format='jpg')
        print(f"Saved comparison plot for {exercise_key} frame {frame_idx} at {output_path}")
    if return_figure:
        return fig
    plt.close(fig)

def generate_smplx_mesh(predicted_params, smplx_model, device):
    predicted_params = predicted_params.squeeze()  # (188,)
    transl = predicted_params[0:3]
    global_orient = predicted_params[3:6].reshape(1, 3)
    body_pose = predicted_params[6:69].reshape(1, 63)
    jaw_pose = predicted_params[69:72].reshape(1, 3)
    leye_pose = predicted_params[72:75].reshape(1, 3)
    reye_pose = predicted_params[75:78].reshape(1, 3)
    left_hand_pose = predicted_params[78:123].reshape(1, 45)
    right_hand_pose = predicted_params[123:168].reshape(1, 45)
    betas = predicted_params[168:178].reshape(1, 10)
    expression = predicted_params[178:188].reshape(1, 10)

    transl_t = torch.tensor(transl, dtype=torch.float32).unsqueeze(0).to(device)
    global_orient_t = torch.tensor(global_orient, dtype=torch.float32).to(device)
    body_pose_t = torch.tensor(body_pose, dtype=torch.float32).to(device)
    jaw_pose_t = torch.tensor(jaw_pose, dtype=torch.float32).to(device)
    leye_pose_t = torch.tensor(leye_pose, dtype=torch.float32).to(device)
    reye_pose_t = torch.tensor(reye_pose, dtype=torch.float32).to(device)
    left_hand_pose_t = torch.tensor(left_hand_pose, dtype=torch.float32).to(device)
    right_hand_pose_t = torch.tensor(right_hand_pose, dtype=torch.float32).to(device)
    betas_t = torch.tensor(betas, dtype=torch.float32).to(device)
    expression_t = torch.tensor(expression, dtype=torch.float32).to(device)

    with torch.no_grad():
        smplx_output = smplx_model(
            betas=betas_t,
            expression=expression_t,
            global_orient=global_orient_t,
            body_pose=body_pose_t,
            jaw_pose=jaw_pose_t,
            leye_pose=leye_pose_t,
            reye_pose=reye_pose_t,
            left_hand_pose=left_hand_pose_t,
            right_hand_pose=right_hand_pose_t,
            transl=transl_t,
            return_verts=True
        )
    vertices = smplx_output.vertices[0].cpu().numpy()
    faces = smplx_model.faces
    mesh = trimesh.Trimesh(vertices, faces)
    return mesh