import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import smplx
import trimesh
import os

# Path to your JSON file
json_file_path = ('/Users/alessandrofolloni/PycharmProjects/smpl_study/'
                  'datasets/FIT3D/train/s03/smplx/band_pull_apart.json')

# Load the JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Total number of frames
num_frames = len(data['transl'])

# Path to your SMPL-X model files (directory, not file)
model_folder = '/Users/alessandrofolloni/PycharmProjects/smpl_study/files/body_models/SMPLX_NEUTRAL.npz'

# Load the SMPL-X model
smplx_model = smplx.create(
    model_folder,
    model_type='smplx',
    gender='neutral',
    ext='npz',
    use_pca=False
)

# Convert parameters to tensors and add batch dimension
device = torch.device('cpu')

# Directory to save the output meshes
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Loop over all frames
for frame_idx in range(num_frames):
    print(f'Processing frame {frame_idx + 1}/{num_frames}')

    # Extract data for the current frame
    transl_np = np.array(data['transl'][frame_idx])  # Shape: (3,)
    betas_np = np.array(data['betas'][frame_idx])  # Shape: (10,)
    expression_np = np.array(data['expression'][frame_idx])  # Shape: (10,)

    # Rotation matrices for various poses
    global_orient_mat = np.array(data['global_orient'][frame_idx])  # Shape: (3, 3)
    body_pose_mat = np.array(data['body_pose'][frame_idx])  # Shape: (21, 3, 3)

    # Uncomment the following lines if you wish to include hand and face poses
    left_hand_pose_mat = np.array(data['left_hand_pose'][frame_idx])  # Shape: (15, 3, 3)
    right_hand_pose_mat = np.array(data['right_hand_pose'][frame_idx])  # Shape: (15, 3, 3)
    jaw_pose_mat = np.array(data['jaw_pose'][frame_idx])  # Shape: (3, 3)
    leye_pose_mat = np.array(data['leye_pose'][frame_idx])  # Shape: (3, 3)
    reye_pose_mat = np.array(data['reye_pose'][frame_idx])  # Shape: (3, 3)

    def rotation_matrices_to_axis_angles(rotation_matrices):
        """
        Convert rotation matrices to axis-angle representations.
        rotation_matrices: numpy array of shape (..., 3, 3)
        Returns: axis-angle vectors of shape (..., 3)
        """
        rot_obj = R.from_matrix(rotation_matrices)
        axis_angles = rot_obj.as_rotvec()  # Shape: (..., 3)
        return axis_angles


    # Convert rotation matrices to axis-angle
    global_orient_axis_angle = rotation_matrices_to_axis_angles(global_orient_mat)  # Shape: (3,)
    body_pose_axis_angle = rotation_matrices_to_axis_angles(body_pose_mat)  # Shape: (21, 3)

    left_hand_pose_axis_angle = rotation_matrices_to_axis_angles(left_hand_pose_mat)  # Shape: (15, 3)
    right_hand_pose_axis_angle = rotation_matrices_to_axis_angles(right_hand_pose_mat)  # Shape: (15, 3)
    jaw_pose_axis_angle = rotation_matrices_to_axis_angles(jaw_pose_mat)  # Shape: (3,)
    leye_pose_axis_angle = rotation_matrices_to_axis_angles(leye_pose_mat)  # Shape: (3,)
    reye_pose_axis_angle = rotation_matrices_to_axis_angles(reye_pose_mat)  # Shape: (3,)

    # Convert parameters to tensors
    betas = torch.tensor(betas_np, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 10)
    expression = torch.tensor(expression_np, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 10)
    transl = torch.tensor(transl_np, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 3)

    global_orient = torch.tensor(global_orient_axis_angle, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 3)
    body_pose = torch.tensor(body_pose_axis_angle.reshape(-1), dtype=torch.float32).unsqueeze(0).to(
        device)  # Shape: (1, 63)

    left_hand_pose = torch.tensor(left_hand_pose_axis_angle.reshape(-1), dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 45)
    right_hand_pose = torch.tensor(right_hand_pose_axis_angle.reshape(-1), dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 45)
    jaw_pose = torch.tensor(jaw_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 3)
    leye_pose = torch.tensor(leye_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 3)
    reye_pose = torch.tensor(reye_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 3)

    # Generate the mesh
    with torch.no_grad():
        output = smplx_model(
            betas=betas,
            expression=expression,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            transl=transl,
            return_verts=True
        )

    vertices = output.vertices.detach().cpu().numpy().squeeze()  # Shape: (number of vertices, 3)
    faces = smplx_model.faces  # Faces of the mesh

    # Save the mesh to an OBJ file
    output_mesh_file = os.path.join(output_dir, f'output_frame{frame_idx:04d}.obj')  # Zero-padded frame index
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(output_mesh_file)
    print(f"Mesh saved to {output_mesh_file}")

    #if frame_idx == 3:
        #break

print('All frames processed and saved.')
