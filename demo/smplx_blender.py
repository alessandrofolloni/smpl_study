import os
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from smplx import create as smplx_create
import trimesh


# ----------------------------
# Utility Functions
# ----------------------------

def axis_angle_to_rotation_matrices(axis_angles):
    """
    Convert axis-angle vectors to rotation matrices.
    :param axis_angles: numpy array of shape (N, 3) or (1, 3)
    :return: numpy array of shape (N, 3, 3) or (1, 3, 3)
    """
    r = R.from_rotvec(axis_angles)
    return r.as_matrix()


def rotation_matrices_to_axis_angle(rotation_matrices):
    """
    Convert rotation matrices to axis-angle vectors.
    :param rotation_matrices: numpy array of shape (N, 3, 3) or (1, 3, 3)
    :return: numpy array of shape (N, 3) or (1, 3)
    """
    r = R.from_matrix(rotation_matrices)
    return r.as_rotvec()


def save_smplx_json(smplx_data, output_path):
    with open(output_path, 'w') as f:
        json.dump(smplx_data, f, indent=4)


def save_mesh(mesh, output_path):
    mesh.export(output_path)


def load_mega_dict(json_path):
    """Load the mega dictionary JSON and create a list of samples.
       Each sample is a dictionary with keys: 'joints3d', 'gt', 'exercise_key', 'frame_idx'
    """
    with open(json_path, 'r') as f:
        mega_dict = json.load(f)
    samples = []
    # Assume mega_dict is structured as: {exercise_key: {frame_key: frame_data, ...}, ...}
    for ex_key, frames in mega_dict.items():
        for fr_key, frame_data in frames.items():
            if 'joints3d' in frame_data:
                sample = {
                    'joints3d': frame_data['joints3d'],
                    'gt': frame_data.get('gt', None),
                    'exercise_key': ex_key,
                    'frame_idx': fr_key
                }
                samples.append(sample)
    return samples


# ----------------------------
# Model Configurations (Best Models)
# ----------------------------
# Configuration 34: d=384, n=6, L=4, dp=0.2
# Configuration 35: d=256, n=8, L=3, dp=0.15
# Configuration 41: d=128, n=2, L=2, dp=0.1
model_configs = [
    {
        'run_name': 'Transformer_bs32_lr0.0002_epochs70_dmodel_384_nhead_6_layers_4_dropout_0.2',
        'model_file': 'model_Transformer.pth',
        'transformer_d_model': 384,
        'transformer_nhead': 6,
        'transformer_num_layers': 4,
        'dropout': 0.2
    },
    {
        'run_name': 'Transformer_bs24_lr0.00011_epochs60_dmodel_256_nhead_8_layers_3_dropout_0.15',
        'model_file': 'model_Transformer.pth',
        'transformer_d_model': 256,
        'transformer_nhead': 8,
        'transformer_num_layers': 3,
        'dropout': 0.15
    },
    {
        'run_name': 'Transformer_bs16_lr1e-05_epochs100_dmodel_128_nhead_2_layers_2_dropout_0.1',
        'model_file': 'model_Transformer.pth',
        'transformer_d_model': 128,
        'transformer_nhead': 2,
        'transformer_num_layers': 2,
        'dropout': 0.1
    }
]

# Common configuration for dataset and SMPL-X
common_config = {
    'model_name': 'Transformer',
    'num_joints_3d': 25,
    'smplx_model_file': "/public.hpc/alessandro.folloni2/smpl_study/smplx_usage/files/body_models/SMPLX_NEUTRAL.npz",
    'mega_dict_file': "/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/mega_dict_3dSMPLX.json"
}

# ----------------------------
# Paths Setup
# ----------------------------
base_runs_folder = "/public.hpc/alessandro.folloni2/smpl_study/runs"
# Create a new folder for comparison outputs
comparison_output_folder = os.path.join("/public.hpc/alessandro.folloni2/smpl_study", "comparison_output")
os.makedirs(comparison_output_folder, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------
# Load Mega Dict and Select 10 Random Samples
# ----------------------------
samples = load_mega_dict(common_config['mega_dict_file'])
if len(samples) < 10:
    raise ValueError("Mega dict does not contain enough samples.")
np.random.seed(42)
random_indices = np.random.choice(len(samples), 10, replace=False)
print(f"Selected random sample indices: {random_indices}")

# ----------------------------
# Define a Minimal Transformer Model
# ----------------------------
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Assuming a fixed sequence length (25 joints)
        self.fc = nn.Linear(d_model * 25, output_size)

    def forward(self, x):
        # x shape: [batch, 25, 3]
        x = self.input_projection(x)  # -> [batch, 25, d_model]
        x = x.permute(1, 0, 2)  # -> [25, batch, d_model]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # -> [batch, 25, d_model]
        x = x.reshape(x.size(0), -1)  # Flatten -> [batch, 25*d_model]
        x = self.fc(x)
        return x


# ----------------------------
# Load SMPL-X Model (for Mesh Generation)
# ----------------------------
smplx_model = smplx_create(
    common_config['smplx_model_file'],
    model_type='smplx',
    gender='neutral',
    ext='npz',
    use_pca=False
).to(device)
print("SMPL-X model loaded.")

# ----------------------------
# Loop Over Each Model Configuration
# ----------------------------
for mconfig in model_configs:
    run_name = mconfig['run_name']
    model_folder = os.path.join(base_runs_folder, run_name)
    model_path = os.path.join(model_folder, mconfig['model_file'])
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Create a dedicated output folder for this model inside the comparison folder
    model_output_folder = os.path.join(comparison_output_folder, run_name)
    os.makedirs(model_output_folder, exist_ok=True)

    # Initialize and load the Transformer model for this configuration
    model = TransformerModel(
        input_size=3,
        output_size=188,
        d_model=mconfig['transformer_d_model'],
        nhead=mconfig['transformer_nhead'],
        num_layers=mconfig['transformer_num_layers'],
        dropout=mconfig['dropout']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model {run_name} loaded and set to evaluation mode.")

    # Process the same 3 random samples for this model
    for i, idx in enumerate(random_indices):
        sample = samples[idx]
        joints3d = np.array(sample['joints3d'])
        if joints3d.shape != (common_config['num_joints_3d'], 3):
            print(f"Sample index {idx} has invalid joints3d shape {joints3d.shape}. Skipping.")
            continue

        # Prepare input tensor for inference
        joints_input = torch.tensor(joints3d, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_params = model(joints_input)
        predicted_params = predicted_params.squeeze(0).cpu().numpy()

        # Define parameter slices for predicted parameters
        transl = predicted_params[0:3]
        global_orient = predicted_params[3:6].reshape(1, 3)
        body_pose = predicted_params[6:69].reshape(21, 3)
        betas = predicted_params[69:79].reshape(1, 10)
        left_hand_pose = predicted_params[79:124].reshape(15, 3)
        right_hand_pose = predicted_params[124:169].reshape(15, 3)
        jaw_pose = predicted_params[169:172].reshape(1, 3)
        leye_pose = predicted_params[172:175].reshape(1, 3)
        reye_pose = predicted_params[175:178].reshape(1, 3)
        expression = predicted_params[178:188].reshape(1, 10)

        # Convert predicted axis-angle to rotation matrices
        global_orient_mat = axis_angle_to_rotation_matrices(global_orient)
        body_pose_mat = axis_angle_to_rotation_matrices(body_pose)
        left_hand_pose_mat = axis_angle_to_rotation_matrices(left_hand_pose)
        right_hand_pose_mat = axis_angle_to_rotation_matrices(right_hand_pose)
        jaw_pose_mat = axis_angle_to_rotation_matrices(jaw_pose)
        leye_pose_mat = axis_angle_to_rotation_matrices(leye_pose)
        reye_pose_mat = axis_angle_to_rotation_matrices(reye_pose)

        # Save predicted SMPL-X parameters as JSON (optional)
        predicted_json = {
            "transl": transl.tolist(),
            "global_orient": global_orient_mat.tolist(),
            "body_pose": body_pose_mat.tolist(),
            "betas": betas.tolist(),
            "left_hand_pose": left_hand_pose_mat.tolist(),
            "right_hand_pose": right_hand_pose_mat.tolist(),
            "jaw_pose": jaw_pose_mat.tolist(),
            "leye_pose": leye_pose_mat.tolist(),
            "reye_pose": reye_pose_mat.tolist(),
            "expression": expression.tolist()
        }
        pred_json_filename = os.path.join(model_output_folder, f"sample_{i}_predicted_smplx.json")
        with open(pred_json_filename, 'w') as pf:
            json.dump(predicted_json, pf, indent=4)
        print(f"[{run_name}] Predicted SMPL-X JSON saved to {pred_json_filename}")

        # Convert predicted parameters to torch tensors for SMPL-X inference
        transl_t = torch.tensor(transl, dtype=torch.float32).unsqueeze(0).to(device)
        global_orient_t = torch.tensor(global_orient, dtype=torch.float32).to(device)
        body_pose_t = torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0).to(device)
        betas_t = torch.tensor(betas, dtype=torch.float32).to(device)
        left_hand_pose_t = torch.tensor(left_hand_pose, dtype=torch.float32).unsqueeze(0).to(device)
        right_hand_pose_t = torch.tensor(right_hand_pose, dtype=torch.float32).unsqueeze(0).to(device)
        jaw_pose_t = torch.tensor(jaw_pose, dtype=torch.float32).to(device)
        leye_pose_t = torch.tensor(leye_pose, dtype=torch.float32).to(device)
        reye_pose_t = torch.tensor(reye_pose, dtype=torch.float32).to(device)
        expression_t = torch.tensor(expression, dtype=torch.float32).to(device)

        # Convert predicted rotation matrices back to axis-angle for SMPL-X input
        global_orient_aa = torch.tensor(R.from_matrix(global_orient_mat).as_rotvec(), dtype=torch.float32).unsqueeze(
            0).to(device)
        body_pose_aa = torch.tensor(R.from_matrix(body_pose_mat).as_rotvec(), dtype=torch.float32).view(1, -1).to(
            device)
        left_hand_pose_aa = torch.tensor(R.from_matrix(left_hand_pose_mat).as_rotvec(), dtype=torch.float32).view(1,
                                                                                                                  -1).to(
            device)
        right_hand_pose_aa = torch.tensor(R.from_matrix(right_hand_pose_mat).as_rotvec(), dtype=torch.float32).view(1,
                                                                                                                    -1).to(
            device)
        jaw_pose_aa = torch.tensor(R.from_matrix(jaw_pose_mat).as_rotvec(), dtype=torch.float32).unsqueeze(0).to(device)
        leye_pose_aa = torch.tensor(R.from_matrix(leye_pose_mat).as_rotvec(), dtype=torch.float32).unsqueeze(0).to(
            device)
        reye_pose_aa = torch.tensor(R.from_matrix(reye_pose_mat).as_rotvec(), dtype=torch.float32).unsqueeze(0).to(
            device)

        # Generate predicted mesh
        with torch.no_grad():
            pred_output_mesh = smplx_model(
                betas=betas_t,
                expression=expression_t,
                global_orient=global_orient_aa,
                body_pose=body_pose_aa,
                left_hand_pose=left_hand_pose_aa,
                right_hand_pose=right_hand_pose_aa,
                jaw_pose=jaw_pose_aa,
                leye_pose=leye_pose_aa,
                reye_pose=reye_pose_aa,
                transl=transl_t,
                return_verts=True
            )
        pred_vertices = pred_output_mesh.vertices[0].cpu().numpy()
        faces = smplx_model.faces

        pred_mesh = trimesh.Trimesh(pred_vertices, faces)
        pred_obj_filename = os.path.join(model_output_folder, f"sample_{i}_predicted_smplx.obj")
        pred_mesh.export(pred_obj_filename)
        print(f"[{run_name}] Predicted mesh for sample {i} saved to {pred_obj_filename}")

        # ---- Ground Truth Mesh ----
        if sample['gt'] is not None:
            gt = sample['gt']
            transl_gt = np.array(gt["transl"])
            if transl_gt.ndim == 1:
                transl_gt = transl_gt.reshape(1, -1)
            global_orient_gt = np.array(gt["global_orient"])
            body_pose_gt = np.array(gt["body_pose"])
            betas_gt = np.array(gt["betas"])
            if betas_gt.ndim == 1:
                betas_gt = betas_gt.reshape(1, -1)
            left_hand_pose_gt = np.array(gt["left_hand_pose"])
            right_hand_pose_gt = np.array(gt["right_hand_pose"])
            jaw_pose_gt = np.array(gt["jaw_pose"])
            leye_pose_gt = np.array(gt["leye_pose"])
            reye_pose_gt = np.array(gt["reye_pose"])
            expression_gt = np.array(gt["expression"])
            if expression_gt.ndim == 1:
                expression_gt = expression_gt.reshape(1, -1)

            global_orient_gt_aa = torch.tensor(rotation_matrices_to_axis_angle(global_orient_gt),
                                               dtype=torch.float32).to(device)
            body_pose_gt_aa = torch.tensor(rotation_matrices_to_axis_angle(body_pose_gt),
                                           dtype=torch.float32).view(1, -1).to(device)
            left_hand_pose_gt_aa = torch.tensor(rotation_matrices_to_axis_angle(left_hand_pose_gt),
                                                dtype=torch.float32).view(1, -1).to(device)
            right_hand_pose_gt_aa = torch.tensor(rotation_matrices_to_axis_angle(right_hand_pose_gt),
                                                 dtype=torch.float32).view(1, -1).to(device)
            jaw_pose_gt_aa = torch.tensor(R.from_matrix(jaw_pose_gt).as_rotvec(), dtype=torch.float32).unsqueeze(0).to(
                device)
            leye_pose_gt_aa = torch.tensor(R.from_matrix(leye_pose_gt).as_rotvec(), dtype=torch.float32).unsqueeze(
                0).to(device)
            reye_pose_gt_aa = torch.tensor(R.from_matrix(reye_pose_gt).as_rotvec(), dtype=torch.float32).unsqueeze(
                0).to(device)

            transl_gt_t = torch.tensor(transl_gt, dtype=torch.float32).to(device)
            betas_gt_t = torch.tensor(betas_gt, dtype=torch.float32).to(device)
            expression_gt_t = torch.tensor(expression_gt, dtype=torch.float32).to(device)

            with torch.no_grad():
                gt_output_mesh = smplx_model(
                    betas=betas_gt_t,
                    expression=expression_gt_t,
                    global_orient=global_orient_gt_aa.unsqueeze(0),
                    body_pose=body_pose_gt_aa,
                    left_hand_pose=left_hand_pose_gt_aa,
                    right_hand_pose=right_hand_pose_gt_aa,
                    jaw_pose=jaw_pose_gt_aa,
                    leye_pose=leye_pose_gt_aa,
                    reye_pose=reye_pose_gt_aa,
                    transl=transl_gt_t,
                    return_verts=True
                )
            gt_vertices = gt_output_mesh.vertices[0].cpu().numpy()
            gt_mesh = trimesh.Trimesh(gt_vertices, faces)
            gt_obj_filename = os.path.join(model_output_folder, f"sample_{i}_gt_smplx.obj")
            gt_mesh.export(gt_obj_filename)
            print(f"[{run_name}] Ground truth mesh for sample {i} saved to {gt_obj_filename}")

print("All 10 sample meshes (predicted and ground truth) have been generated and exported for Blender.")