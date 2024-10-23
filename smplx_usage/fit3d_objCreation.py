import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import smplx
import trimesh
import os
from pathlib import Path
import pyrender
import matplotlib.pyplot as plt


class SMPLXProcessor:
    def __init__(self, json_file_path, model_folder, output_dir, device='cpu'):
        """
        Initializes the SMPLXProcessor with the given JSON file, SMPL-X model, and output directory.

        Parameters:
        - json_file_path (str or Path): Path to the JSON file containing pose data.
        - model_folder (str or Path): Directory containing the SMPL-X model files.
        - output_dir (str or Path): Directory where output OBJ files will be saved.
        - device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.json_file_path = Path(json_file_path)
        self.model_folder = Path(model_folder)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)

        self._load_data()
        self._load_model()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self):
        """Loads the JSON data."""
        with open(self.json_file_path, 'r') as f:
            self.data = json.load(f)
        self.num_frames = len(self.data['transl'])
        print(f"Loaded data with {self.num_frames} frames.")

    def _load_model(self):
        """Loads the SMPL-X model."""
        self.smplx_model = smplx.create(
            self.model_folder,
            model_type='smplx',
            gender='neutral',
            ext='npz',
            use_pca=False
        ).to(self.device)
        print("SMPL-X model loaded successfully.")

    def rotation_matrices_to_axis_angles(self, rotation_matrices):
        """
        Convert rotation matrices to axis-angle representations.

        Parameters:
        - rotation_matrices (numpy.ndarray): Rotation matrices of shape (..., 3, 3)

        Returns:
        - axis_angles (numpy.ndarray): Axis-angle vectors of shape (..., 3)
        """
        rot_obj = R.from_matrix(rotation_matrices)
        axis_angles = rot_obj.as_rotvec()
        return axis_angles

    def process_frame(self, frame_idx):
        """
        Processes a specific frame and saves the resulting mesh as an OBJ file.

        Parameters:
        - frame_idx (int): Index of the frame to process (0-based).

        Returns:
        - output_mesh_file (str): Path to the saved OBJ file.
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} is out of bounds (0 to {self.num_frames - 1}).")

        print(f'Processing frame {frame_idx + 1}/{self.num_frames}')

        # Extract data for the current frame
        transl_np = np.array(self.data['transl'][frame_idx])  # Shape: (3,)
        betas_np = np.array(self.data['betas'][frame_idx])  # Shape: (10,)
        expression_np = np.array(self.data['expression'][frame_idx])  # Shape: (10,)

        # Rotation matrices for various poses
        global_orient_mat = np.array(self.data['global_orient'][frame_idx])  # Shape: (3, 3)
        body_pose_mat = np.array(self.data['body_pose'][frame_idx])  # Shape: (21, 3, 3)

        # Uncomment the following lines if you wish to include hand and face poses
        left_hand_pose_mat = np.array(self.data['left_hand_pose'][frame_idx])  # Shape: (15, 3, 3)
        right_hand_pose_mat = np.array(self.data['right_hand_pose'][frame_idx])  # Shape: (15, 3, 3)
        jaw_pose_mat = np.array(self.data['jaw_pose'][frame_idx])  # Shape: (3, 3)
        leye_pose_mat = np.array(self.data['leye_pose'][frame_idx])  # Shape: (3, 3)
        reye_pose_mat = np.array(self.data['reye_pose'][frame_idx])  # Shape: (3, 3)

        # Convert rotation matrices to axis-angle
        global_orient_axis_angle = self.rotation_matrices_to_axis_angles(global_orient_mat)  # Shape: (3,)
        body_pose_axis_angle = self.rotation_matrices_to_axis_angles(body_pose_mat)  # Shape: (21, 3)

        left_hand_pose_axis_angle = self.rotation_matrices_to_axis_angles(left_hand_pose_mat)  # Shape: (15, 3)
        right_hand_pose_axis_angle = self.rotation_matrices_to_axis_angles(right_hand_pose_mat)  # Shape: (15, 3)
        jaw_pose_axis_angle = self.rotation_matrices_to_axis_angles(jaw_pose_mat)  # Shape: (3,)
        leye_pose_axis_angle = self.rotation_matrices_to_axis_angles(leye_pose_mat)  # Shape: (3,)
        reye_pose_axis_angle = self.rotation_matrices_to_axis_angles(reye_pose_mat)  # Shape: (3,)

        # Convert parameters to tensors
        betas = torch.tensor(betas_np, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, 10)
        expression = torch.tensor(expression_np, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, 10)
        transl = torch.tensor(transl_np, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, 3)

        global_orient = torch.tensor(global_orient_axis_angle, dtype=torch.float32).unsqueeze(0).to(
            self.device)  # Shape: (1, 3)
        body_pose = torch.tensor(body_pose_axis_angle.reshape(-1), dtype=torch.float32).unsqueeze(0).to(
            self.device)  # Shape: (1, 63)

        left_hand_pose = torch.tensor(left_hand_pose_axis_angle.reshape(-1), dtype=torch.float32).unsqueeze(0).to(
            self.device)  # Shape: (1, 45)
        right_hand_pose = torch.tensor(right_hand_pose_axis_angle.reshape(-1), dtype=torch.float32).unsqueeze(0).to(
            self.device)  # Shape: (1, 45)
        jaw_pose = torch.tensor(jaw_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, 3)
        leye_pose = torch.tensor(leye_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(
            self.device)  # Shape: (1, 3)
        reye_pose = torch.tensor(reye_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(
            self.device)  # Shape: (1, 3)

        # Generate the mesh
        with torch.no_grad():
            output = self.smplx_model(
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
        faces = self.smplx_model.faces  # Faces of the mesh

        # Save the mesh to an OBJ file
        output_mesh_file = self.output_dir / f'output_frame{frame_idx:04d}.obj'  # Zero-padded frame index
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(output_mesh_file)
        print(f"Mesh saved to {output_mesh_file}")

        return str(output_mesh_file)