import json

import imageio
import numpy as np
import pyrender
import torch
import torch.nn as nn
import trimesh
from scipy.spatial.transform import Rotation as R
from smplx import create as smplx_create


#########################
# Model Implementations #
#########################

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
    def __init__(self, input_channels, output_size, cnn_channels, dropout):
        super(CNNModel, self).__init__()
        layers = []
        in_channels = input_channels
        for out_channels in cnn_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * 25, output_size)  # 25 is the number of joints3d

    def forward(self, x):
        # x expected shape: (batch, seq_length=25, features=3)
        x = self.input_projection(x)          # (batch, 25, d_model)
        x = x.permute(1, 0, 2)                # (25, batch, d_model)
        x = self.transformer_encoder(x)       # (25, batch, d_model)
        x = x.permute(1, 0, 2)                # (batch, 25, d_model)
        x = x.reshape(x.size(0), -1)          # (batch, 25*d_model)
        x = self.fc(x)                        # (batch, 188)
        return x


#########################
# Utility Functions     #
#########################

def axis_angle_to_rotation_matrices(axis_angles):
    """
    Convert axis-angle vectors to rotation matrices.
    axis_angles: numpy array of shape (..., 3)
    returns: rotation matrices of shape (..., 3, 3)
    """
    rot = R.from_rotvec(axis_angles)
    return rot.as_matrix()


def rebuild_smplx_dict(predicted_params_list, exercise_keys, frame_idxs):
    """
    Rebuilds a dictionary with SMPL-X structure from predicted parameter vectors.
    """
    # Define parameter indices
    transl_indices = slice(0, 3)
    global_orient_indices = slice(3, 6)
    body_pose_indices = slice(6, 69)   # 63 values -> 21 joints * 3
    betas_indices = slice(69, 79)      # 10 values
    left_hand_pose_indices = slice(79, 124)   # 45 values -> 15 * 3
    right_hand_pose_indices = slice(124, 169) # 45 values -> 15 * 3
    jaw_pose_indices = slice(169, 172) # 3 values
    leye_pose_indices = slice(172, 175)# 3 values
    reye_pose_indices = slice(175, 178)# 3 values
    expression_indices = slice(178, 188)# 10 values

    predictions_by_exercise = {}
    for params, ex_key, fr_idx in zip(predicted_params_list, exercise_keys, frame_idxs):
        if ex_key not in predictions_by_exercise:
            predictions_by_exercise[ex_key] = {
                "transl": [],
                "global_orient": [],
                "body_pose": [],
                "betas": [],
                "left_hand_pose": [],
                "right_hand_pose": [],
                "jaw_pose": [],
                "leye_pose": [],
                "reye_pose": [],
                "expression": []
            }

        transl = params[transl_indices]
        global_orient = params[global_orient_indices].reshape(1, 3)
        body_pose = params[body_pose_indices].reshape(21, 3)
        betas = params[betas_indices]
        left_hand_pose = params[left_hand_pose_indices].reshape(15, 3)
        right_hand_pose = params[right_hand_pose_indices].reshape(15, 3)
        jaw_pose = params[jaw_pose_indices].reshape(1, 3)
        leye_pose = params[leye_pose_indices].reshape(1, 3)
        reye_pose = params[reye_pose_indices].reshape(1, 3)
        expression = params[expression_indices]

        # Convert axis-angle back to rotation matrices
        global_orient_mat = axis_angle_to_rotation_matrices(global_orient)   # (1,3,3)
        body_pose_mat = axis_angle_to_rotation_matrices(body_pose)           # (21,3,3)
        left_hand_pose_mat = axis_angle_to_rotation_matrices(left_hand_pose) # (15,3,3)
        right_hand_pose_mat = axis_angle_to_rotation_matrices(right_hand_pose)#(15,3,3)
        jaw_pose_mat = axis_angle_to_rotation_matrices(jaw_pose)             # (1,3,3)
        leye_pose_mat = axis_angle_to_rotation_matrices(leye_pose)           # (1,3,3)
        reye_pose_mat = axis_angle_to_rotation_matrices(reye_pose)           # (1,3,3)

        predictions_by_exercise[ex_key]["transl"].append(transl.tolist())
        predictions_by_exercise[ex_key]["global_orient"].append(global_orient_mat.tolist())
        predictions_by_exercise[ex_key]["body_pose"].append(body_pose_mat.tolist())
        predictions_by_exercise[ex_key]["betas"].append(betas.tolist())
        predictions_by_exercise[ex_key]["left_hand_pose"].append(left_hand_pose_mat.tolist())
        predictions_by_exercise[ex_key]["right_hand_pose"].append(right_hand_pose_mat.tolist())
        predictions_by_exercise[ex_key]["jaw_pose"].append(jaw_pose_mat.tolist())
        predictions_by_exercise[ex_key]["leye_pose"].append(leye_pose_mat.tolist())
        predictions_by_exercise[ex_key]["reye_pose"].append(reye_pose_mat.tolist())
        predictions_by_exercise[ex_key]["expression"].append(expression.tolist())

    return predictions_by_exercise


def load_model(model_class, model_path, device):
    """
    Load the trained model weights from a .pth file.

    model_class: the model instance (e.g. an instance of TransformerModel)
    model_path: path to the .pth weights file
    device: torch device
    """
    state_dict = torch.load(model_path, map_location=device)
    model_class.load_state_dict(state_dict)
    model_class.to(device)
    model_class.eval()
    return model_class


def run_inference(model, joints3d_input, device):
    """
    Run inference on a single frame of joints3d data.

    model: the loaded PyTorch model
    joints3d_input: numpy array (25,3) of joint positions
    device: torch device
    """
    inputs = torch.tensor(joints3d_input, dtype=torch.float32).unsqueeze(0).to(device)
    # For Transformer/RNN: shape is (batch_size=1, seq_length=25, features=3)
    with torch.no_grad():
        predicted_params = model(inputs)
    return predicted_params.cpu().numpy().squeeze()


def generate_and_save_mesh(smplx_output, smplx_model_file, mesh_output_path, device):
    """
    Generates the SMPL-X mesh from parameters and saves it as an .obj file.
    """
    # Initialize the SMPL-X model
    smplx_model = smplx_create(
        smplx_model_file,
        model_type='smplx',
        gender='neutral',
        ext='npz',
        use_pca=False
    ).to(device)

    print("SMPL-X model initialized for mesh generation.")

    def flatten_pose(pose_mat):
        """
        Convert rotation matrices to axis-angle vectors and flatten them.
        :param pose_mat: numpy array of shape (N, 3, 3)
        :return: torch tensor of shape (1, N*3)
        """
        r = R.from_matrix(pose_mat)
        aa = r.as_rotvec()  # Shape: (N, 3)
        return torch.tensor(aa.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

    # Extract parameters
    transl = torch.tensor(smplx_output["transl"], dtype=torch.float32).to(device)
    global_orient = torch.tensor(R.from_matrix(np.array(smplx_output["global_orient"])).as_rotvec(),
                                 dtype=torch.float32).unsqueeze(0).to(device)
    body_pose = flatten_pose(np.array(smplx_output["body_pose"]))
    betas = torch.tensor(smplx_output["betas"], dtype=torch.float32).to(device)
    left_hand_pose = flatten_pose(np.array(smplx_output["left_hand_pose"]))
    right_hand_pose = flatten_pose(np.array(smplx_output["right_hand_pose"]))
    jaw_pose = torch.tensor(R.from_matrix(np.array(smplx_output["jaw_pose"])).as_rotvec(),
                            dtype=torch.float32).unsqueeze(0).to(device)
    leye_pose = torch.tensor(R.from_matrix(np.array(smplx_output["leye_pose"])).as_rotvec(),
                             dtype=torch.float32).unsqueeze(0).to(device)
    reye_pose = torch.tensor(R.from_matrix(np.array(smplx_output["reye_pose"])).as_rotvec(),
                             dtype=torch.float32).unsqueeze(0).to(device)
    expression = torch.tensor(smplx_output["expression"], dtype=torch.float32).to(device)

    # Run SMPL-X model to get vertices
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

    print("SMPL-X model forward pass for mesh generation completed.")

    vertices = output.vertices[0].cpu().numpy()
    faces = smplx_model.faces

    # Create a mesh using trimesh
    mesh = trimesh.Trimesh(vertices, faces)
    print("Mesh created using trimesh.")

    # Save the mesh
    mesh.export(mesh_output_path)
    print(f"Mesh saved to {mesh_output_path}")


def save_smplx_json(smplx_data, output_path):
    """
    Save the SMPLX dictionary to a JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(smplx_data, f, indent=4)
    print(f"Saved SMPLX JSON to {output_path}")

def load_smplx_json(json_path):
    """
    Load SMPL-X parameters from a JSON file.
    :param json_path: Path to the JSON file.
    :return: Dictionary with SMPL-X parameters.
    """
    with open(json_path, 'r') as f:
        smplx_data = json.load(f)
    return smplx_data

def save_mesh(mesh, output_path):
    """
    Save the mesh to a file.
    :param mesh: trimesh.Trimesh object.
    :param output_path: Path to save the mesh.
    """
    mesh.export(output_path)


def render_and_save_mesh_image(mesh_path, image_output_path):
    """
    Renders the mesh and saves the image.
    """
    # Load the mesh
    mesh = trimesh.load(mesh_path)

    # Create a pyrender scene
    scene = pyrender.Scene()

    # Add mesh to the scene
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh_pyrender)

    # Add a camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.3],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    # Add a light source
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render the scene to an image
    renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
    color, depth = renderer.render(scene)
    renderer.delete()

    # Save the rendered image
    imageio.imwrite(image_output_path, color)
    print(f"Rendered image saved to {image_output_path}")