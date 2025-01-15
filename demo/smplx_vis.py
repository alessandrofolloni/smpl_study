import os
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from smplx import create as smplx_create
import trimesh
import pyrender

from utils_smplx import (
    axis_angle_to_rotation_matrices,
    load_smplx_json,
    save_mesh
)

# ----------------------------
# Configuration Parameters
# ----------------------------

config = {
    'smplx_model_file': "/public.hpc/alessandro.folloni2/smpl_study/smplx_usage/files/body_models/SMPLX_NEUTRAL.npz",  # Path to your SMPL-X .npz file
    'json_file_path': "/public.hpc/alessandro.folloni2/smpl_study/runs/Transformer_bs32_lr0.0002_epochs70_dmodel_384_nhead_6_layers_4_dropout_0.2/demo_sample_predicted_smplx.json",  # Path to your SMPL-X JSON file
    'output_mesh_file': "/public.hpc/alessandro.folloni2/smpl_study/runs/Transformer_bs32_lr0.0002_epochs70_dmodel_384_nhead_6_layers_4_dropout_0.2/demo_sample_predicted_smplx.obj"  # Path to save the mesh (optional)
}

# ----------------------------
# Verify File Paths
# ----------------------------

# Verify that the JSON file exists
if not os.path.isfile(config['json_file_path']):
    raise FileNotFoundError(f"JSON file not found at {config['json_file_path']}. Please check the path.")

# Verify that the SMPL-X model file exists
if not os.path.isfile(config['smplx_model_file']):
    raise FileNotFoundError(f"SMPL-X model file not found at {config['smplx_model_file']}. Please check the path.")

# ----------------------------
# Device Configuration
# ----------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------
# Load SMPL-X Parameters from JSON
# ----------------------------

smplx_data = load_smplx_json(config['json_file_path'])
print("SMPL-X parameters loaded from JSON.")

# ----------------------------
# Extract and Process Parameters
# ----------------------------

# Extract parameters
transl = np.array(smplx_data["transl"])  # Shape: (1, 3)
global_orient = np.array(smplx_data["global_orient"])  # Shape: (1, 3, 3)
body_pose = np.array(smplx_data["body_pose"])  # Shape: (21, 3, 3)
betas = np.array(smplx_data["betas"])  # Shape: (1, 10)
left_hand_pose = np.array(smplx_data["left_hand_pose"])  # Shape: (15, 3, 3)
right_hand_pose = np.array(smplx_data["right_hand_pose"])  # Shape: (15, 3, 3)
jaw_pose = np.array(smplx_data["jaw_pose"])  # Shape: (1, 3, 3)
leye_pose = np.array(smplx_data["leye_pose"])  # Shape: (1, 3, 3)
reye_pose = np.array(smplx_data["reye_pose"])  # Shape: (1, 3, 3)
expression = np.array(smplx_data["expression"])  # Shape: (1, 10)

# Convert rotation matrices to axis-angle vectors
def rotation_matrices_to_axis_angle(rotation_matrices):
    """
    Convert rotation matrices to axis-angle vectors.
    :param rotation_matrices: numpy array of shape (N, 3, 3)
    :return: numpy array of shape (N, 3)
    """
    r = R.from_matrix(rotation_matrices)
    return r.as_rotvec()

global_orient_aa = rotation_matrices_to_axis_angle(global_orient)  # Shape: (1, 3)
body_pose_aa = rotation_matrices_to_axis_angle(body_pose)  # Shape: (21, 3)
left_hand_pose_aa = rotation_matrices_to_axis_angle(left_hand_pose)  # Shape: (15, 3)
right_hand_pose_aa = rotation_matrices_to_axis_angle(right_hand_pose)  # Shape: (15, 3)
jaw_pose_aa = rotation_matrices_to_axis_angle(jaw_pose)  # Shape: (1, 3)
leye_pose_aa = rotation_matrices_to_axis_angle(leye_pose)  # Shape: (1, 3)
reye_pose_aa = rotation_matrices_to_axis_angle(reye_pose)  # Shape: (1, 3)

print("Rotation matrices converted to axis-angle vectors.")

# ----------------------------
# Initialize SMPL-X Model
# ----------------------------

smplx_model = smplx_create(
    config['smplx_model_file'],
    model_type='smplx',
    gender='neutral',
    ext='npz',
    use_pca=False
).to(device)

print("SMPL-X model initialized.")

# ----------------------------
# Convert Parameters to Torch Tensors
# ----------------------------

def flatten_pose(pose_aa):
    """
    Flatten axis-angle vectors into a single tensor.
    :param pose_aa: numpy array of shape (N, 3)
    :return: torch tensor of shape (1, N*3)
    """
    return torch.tensor(pose_aa.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

transl_torch = torch.tensor(transl, dtype=torch.float32).to(device)  # Shape: (1, 3)
global_orient_aa_torch = torch.tensor(global_orient_aa, dtype=torch.float32).to(device)  # Shape: (1, 3)
body_pose_aa_torch = flatten_pose(body_pose_aa)  # Shape: (1, 63)
left_hand_pose_aa_torch = flatten_pose(left_hand_pose_aa)  # Shape: (1, 45)
right_hand_pose_aa_torch = flatten_pose(right_hand_pose_aa)  # Shape: (1, 45)
jaw_pose_aa_torch = torch.tensor(jaw_pose_aa, dtype=torch.float32).to(device)  # Shape: (1, 3)
leye_pose_aa_torch = torch.tensor(leye_pose_aa, dtype=torch.float32).to(device)  # Shape: (1, 3)
reye_pose_aa_torch = torch.tensor(reye_pose_aa, dtype=torch.float32).to(device)  # Shape: (1, 3)
betas_torch = torch.tensor(betas, dtype=torch.float32).to(device)  # Shape: (1, 10)
expression_torch = torch.tensor(expression, dtype=torch.float32).to(device)  # Shape: (1, 10)

print("All parameters converted to torch tensors.")

# ----------------------------
# Generate SMPL-X Mesh
# ----------------------------

with torch.no_grad():
    output = smplx_model(
        betas=betas_torch,
        expression=expression_torch,
        global_orient=global_orient_aa_torch,
        body_pose=body_pose_aa_torch,
        left_hand_pose=left_hand_pose_aa_torch,
        right_hand_pose=right_hand_pose_aa_torch,
        jaw_pose=jaw_pose_aa_torch,
        leye_pose=leye_pose_aa_torch,
        reye_pose= reye_pose_aa_torch,
        transl=transl_torch,
        return_verts=True
    )

print("SMPL-X model forward pass completed.")

vertices = output.vertices[0].cpu().numpy()
faces = smplx_model.faces

# ----------------------------
# Create and Save Mesh
# ----------------------------

# Create a mesh using trimesh
mesh = trimesh.Trimesh(vertices, faces)
print("Mesh created using trimesh.")

# Optional: Save the mesh to a file (e.g., OBJ format)
save_mesh(mesh, config['output_mesh_file'])
print(f"Mesh saved to {config['output_mesh_file']}")

# ----------------------------
# Visualize the Mesh
# ----------------------------

# Create a pyrender scene and add the mesh
scene = pyrender.Scene()
mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_pyrender)

print("Mesh added to pyrender scene.")

# Visualize the mesh using pyrender
try:
    pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(800, 600))
except Exception as e:
    print(f"An error occurred during visualization: {e}")

print("Visualization completed.")