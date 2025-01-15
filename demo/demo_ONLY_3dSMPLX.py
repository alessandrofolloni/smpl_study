import os
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from smplx import create as smplx_create
import trimesh
import pyrender

from utils_smplx import (
    TransformerModel,
    load_model,
    run_inference,
    axis_angle_to_rotation_matrices,
    save_smplx_json
)

# ----------------------------
# Configuration Parameters
# ----------------------------

# Update these parameters based on your specific run and paths
config = {
    'model_name': 'Transformer',
    'transformer_d_model': 384,
    'transformer_nhead': 6,
    'transformer_num_layers': 4,
    'dropout': 0.2,
    'num_joints_3d': 25,
    'run_name': 'Transformer_bs32_lr0.0002_epochs70_dmodel_384_nhead_6_layers_4_dropout_0.2',
    'model_file': 'model_Transformer.pth',  # Adjust if different
    'smplx_model_folder': "/public.hpc/alessandro.folloni2/smpl_study/smplx_usage/files/body_models/",
    'smplx_model_file': "/public.hpc/alessandro.folloni2/smpl_study/smplx_usage/files/body_models/SMPLX_NEUTRAL.npz"  # Add this line
}

# ----------------------------
# Paths Setup
# ----------------------------

# Path to the Transformer model directory
output_folder = os.path.join('/public.hpc/alessandro.folloni2/smpl_study/runs', config['run_name'])
model_type = config['model_name']
model_path = os.path.join(output_folder, config['model_file'])

# Verify that the Transformer model file exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path and filename.")

# Path to the SMPL-X model file
smplx_model_file = config['smplx_model_file']

# Verify that the SMPL-X model file exists
if not os.path.isfile(smplx_model_file):
    raise FileNotFoundError(f"SMPL-X model file not found at {smplx_model_file}. Please check the path.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------
# Initialize and Load the Model
# ----------------------------

# Initialize your Transformer model instance with updated parameters
model = TransformerModel(
    input_size=3,  # Assuming each joint has 3 coordinates; adjust if necessary
    output_size=188,  # Number of output parameters for SMPL-X
    d_model=config['transformer_d_model'],
    nhead=config['transformer_nhead'],
    num_layers=config['transformer_num_layers'],
    dropout=config['dropout']
)

# Load the model weights
model = load_model(model, model_path, device)
model.eval()  # Set model to evaluation mode
print("Model loaded and set to evaluation mode.")

# ----------------------------
# Prepare Sample Joints3D Input
# ----------------------------

# Replace this with your actual joints3D data
# Example: Loading from a file or defining manually
# Here, we'll define a sample joints3D array for demonstration purposes

# Example: Define a sample joints3D input (25 joints, 3 coordinates each)
# Replace this with your actual data as needed
sample_joints3d = np.array([
    [0.0, 0.0, 0.0],  # Joint 1
    [0.1, -0.1, 0.2],  # Joint 2
    [0.2, -0.2, 0.4],  # Joint 3
] + [[0.3, -0.3, 0.6]] * (25 - 3))  # Total shape: (25, 3)

# Ensure the input has the correct shape
assert sample_joints3d.shape == (config['num_joints_3d'], 3), \
    f"Expected joints3D shape ({config['num_joints_3d']}, 3), got {sample_joints3d.shape}"

print("Sample joints3D input prepared.")

# ----------------------------
# Run Inference
# ----------------------------

# Convert sample_joints3d to torch tensor
joints3d_input = torch.tensor(sample_joints3d, dtype=torch.float32).to(device)

# If your model expects a batch dimension, add it
joints3d_input = joints3d_input.unsqueeze(0)  # Shape: (1, 25, 3)

# Run inference to get predicted SMPL-X parameters
with torch.no_grad():
    predicted_params = model(joints3d_input)  # Adjust based on your model's forward method

# If 'run_inference' is a separate function in 'utils_smplx', use it instead:
# predicted_params = run_inference(model, sample_joints3d, device)

print("Inference completed. Predicted parameters obtained.")

# ----------------------------
# Process Predicted Parameters
# ----------------------------

# Extract parameters using the defined slices
transl_indices = slice(0, 3)
global_orient_indices = slice(3, 6)
body_pose_indices = slice(6, 69)        # 21 * 3
betas_indices = slice(69, 79)           # 10
left_hand_pose_indices = slice(79, 124) # 15 * 3
right_hand_pose_indices = slice(124, 169) # 15 * 3
jaw_pose_indices = slice(169, 172)      # 3
leye_pose_indices = slice(172, 175)     # 3
reye_pose_indices = slice(175, 178)     # 3
expression_indices = slice(178, 188)    # 10

# Assuming predicted_params is a tensor of shape (1, 188)
predicted_params = predicted_params.squeeze(0).cpu().numpy()

# Extract each parameter set
transl = predicted_params[transl_indices]
global_orient = predicted_params[global_orient_indices].reshape(1, 3)
body_pose = predicted_params[body_pose_indices].reshape(21, 3)
betas = predicted_params[betas_indices]
left_hand_pose = predicted_params[left_hand_pose_indices].reshape(15, 3)
right_hand_pose = predicted_params[right_hand_pose_indices].reshape(15, 3)
jaw_pose = predicted_params[jaw_pose_indices].reshape(1, 3)
leye_pose = predicted_params[leye_pose_indices].reshape(1, 3)
reye_pose = predicted_params[reye_pose_indices].reshape(1, 3)
expression = predicted_params[expression_indices]

# Convert axis-angle to rotation matrices
global_orient_mat = axis_angle_to_rotation_matrices(global_orient)  # Shape: (1, 3, 3)
body_pose_mat = axis_angle_to_rotation_matrices(body_pose)        # Shape: (21, 3, 3)
left_hand_pose_mat = axis_angle_to_rotation_matrices(left_hand_pose)  # Shape: (15, 3, 3)
right_hand_pose_mat = axis_angle_to_rotation_matrices(right_hand_pose)  # Shape: (15, 3, 3)
jaw_pose_mat = axis_angle_to_rotation_matrices(jaw_pose)          # Shape: (1, 3, 3)
leye_pose_mat = axis_angle_to_rotation_matrices(leye_pose)        # Shape: (1, 3, 3)
reye_pose_mat = axis_angle_to_rotation_matrices(reye_pose)        # Shape: (1, 3, 3)

print("Predicted parameters processed and converted to rotation matrices.")

# ----------------------------
# Create SMPL-X JSON Output
# ----------------------------

smplx_output = {
    "transl": [transl.tolist()],
    "global_orient": global_orient_mat.tolist(),
    "body_pose": body_pose_mat.tolist(),
    "betas": [betas.tolist()],
    "left_hand_pose": left_hand_pose_mat.tolist(),
    "right_hand_pose": right_hand_pose_mat.tolist(),
    "jaw_pose": jaw_pose_mat.tolist(),
    "leye_pose": leye_pose_mat.tolist(),
    "reye_pose": reye_pose_mat.tolist(),
    "expression": [expression.tolist()]
}

# Save the generated SMPL-X parameters to JSON
output_json_path = os.path.join(output_folder, "demo_sample_predicted_smplx.json")
save_smplx_json(smplx_output, output_json_path)

print(f"SMPL-X JSON saved to {output_json_path}")

# ----------------------------
# Visualize the SMPL-X Model
# ----------------------------

# Initialize the SMPL-X model
smplx_model = smplx_create(
    config['smplx_model_file'],  # Use the full path to the .npz file
    model_type='smplx',
    gender='neutral',
    ext='npz',
    use_pca=False
).to(device)

print("SMPL-X model initialized.")

def flatten_pose(pose_mat):
    """
    Convert rotation matrices to axis-angle vectors and flatten them.
    :param pose_mat: numpy array of shape (N, 3, 3)
    :return: torch tensor of shape (1, N*3)
    """
    r = R.from_matrix(pose_mat)
    aa = r.as_rotvec()  # Shape: (N, 3)
    return torch.tensor(aa.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

# Convert all parameters to torch tensors
betas_torch = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(device)
expression_torch = torch.tensor(expression, dtype=torch.float32).unsqueeze(0).to(device)
transl_torch = torch.tensor(transl, dtype=torch.float32).unsqueeze(0).to(device)
global_orient_aa = torch.tensor(R.from_matrix(global_orient_mat).as_rotvec(), dtype=torch.float32).unsqueeze(0).to(device)
body_pose_aa = flatten_pose(body_pose_mat)            # Shape: (1, 63)
left_hand_pose_aa = flatten_pose(left_hand_pose_mat)  # Shape: (1, 45)
right_hand_pose_aa = flatten_pose(right_hand_pose_mat)  # Shape: (1, 45)
jaw_pose_aa = torch.tensor(R.from_matrix(jaw_pose_mat).as_rotvec(), dtype=torch.float32).unsqueeze(0).to(device)
leye_pose_aa = torch.tensor(R.from_matrix(leye_pose_mat).as_rotvec(), dtype=torch.float32).unsqueeze(0).to(device)
reye_pose_aa = torch.tensor(R.from_matrix(reye_pose_mat).as_rotvec(), dtype=torch.float32).unsqueeze(0).to(device)

print("All parameters converted to torch tensors.")

# Run the SMPL-X model to get the vertices
with torch.no_grad():
    output = smplx_model(
        betas=betas_torch,
        expression=expression_torch,
        global_orient=global_orient_aa,
        body_pose=body_pose_aa,
        left_hand_pose=left_hand_pose_aa,
        right_hand_pose=right_hand_pose_aa,
        jaw_pose=jaw_pose_aa,
        leye_pose=leye_pose_aa,
        reye_pose=reye_pose_aa,
        transl=transl_torch,
        return_verts=True
    )

print("SMPL-X model forward pass completed.")

vertices = output.vertices[0].cpu().numpy()
faces = smplx_model.faces

# Create a mesh using trimesh
mesh = trimesh.Trimesh(vertices, faces)
print("Mesh created using trimesh.")

# Create a pyrender scene and add the mesh
scene = pyrender.Scene()
mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_pyrender)

print("Mesh added to pyrender scene.")

# Visualize the mesh using pyrender
try:
    pyrender.Viewer(scene, use_raymond_lighting=True)
except Exception as e:
    print(f"An error occurred during visualization: {e}")

print("Visualization completed.")