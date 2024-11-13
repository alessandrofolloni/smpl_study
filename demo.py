import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from torch import nn
from ultralytics import YOLO


def main():
    # Configuration parameters
    config = {
        'base_video_path': '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train',
        'subject': 's03',  # Subject ID
        'exercise': 'band_pull_apart.mp4',  # Exercise video name
        'camera_ids': ['50591643', '58860488', '60457274', '65906101'],  # Camera IDs
        'num_joints_2d': 17,   # Number of 2D joints from YOLOv8
        'num_joints_3d': 25,   # Number of 3D joints
        'output_dir': 'demo_outputs',  # Directory to save outputs
        'smplx_model_path': '/public.hpc/alessandro.folloni2/smpl_study/'
                            'smplx_usage/files/body_models/SMPLX_NEUTRAL.npz',
        'model_2d3d_path': '/public.hpc/alessandro.folloni2/smpl_study/'
                           'paths/CNN6_bs32_ep200_joints_model.pth',
        'model_3dsmplx_path': '/public.hpc/alessandro.folloni2/smpl_study/'
                              'runs/Transformer_bs16_lr0.0001_epochs50_dmodel_512_nhead_16_layers_6_dropout_0.2/'
                              'joints3d_to_smplx_model_Transformer.pth',
    }

    os.makedirs(config['output_dir'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Construct video paths based on the base path, subject, exercise, and camera IDs
    video_paths = [
        os.path.join(
            config['base_video_path'],
            config['subject'],
            'videos',
            cam_id,
            config['exercise']
        ) for cam_id in config['camera_ids']
    ]

    # Step 1: Extract frames from videos
    frame_idx, frames = extract_frames_from_videos(video_paths)

    # Step 2: Extract 2D keypoints using YOLOv8
    keypoints_list = extract_keypoints(frames, device)

    # Step 3: Plot and save keypoints on images
    plot_keypoints_on_images(frames, keypoints_list, config['output_dir'], config['camera_ids'])

    # Step 4: Prepare input for 2D to 3D model
    joints2d_tensor = prepare_input_2d_to_3d(keypoints_list, config)

    # Step 5: Load 2D to 3D model and predict 3D joints
    predicted_3d_joints = predict_3d_joints(joints2d_tensor, config, device)

    # Step 6: Plot and save 3D joints
    plot_3d_joints(predicted_3d_joints, config['exercise'], frame_idx, output_dir=config['output_dir'])

    # Step 7: Prepare input for 3D to SMPL-X model
    joints3d_tensor = torch.tensor(predicted_3d_joints, dtype=torch.float32).unsqueeze(0).to(device)

    # Step 8: Load 3D to SMPL-X model and predict SMPL-X parameters
    smplx_params = predict_smplx_params(joints3d_tensor, config, device)

    # Step 9: Generate and visualize SMPL-X mesh
    generate_and_visualize_smplx_mesh(smplx_params, config)

def extract_frames_from_videos(video_paths):
    """
    Extracts a random frame from each of the provided video paths.
    Returns the frame index and a list of frames.
    """
    # Get the minimum number of frames across all videos
    frame_counts = [get_video_frame_count(vp) for vp in video_paths]
    min_frame_count = min(frame_counts)

    # Pick a random frame index
    frame_idx = random.randint(0, min_frame_count - 1)
    print(f"Selected frame index: {frame_idx}")

    # Extract frames from each video
    frames = []
    for vp in video_paths:
        frame = extract_frame(vp, frame_idx)
        if frame is not None:
            frames.append(frame)
        else:
            frames.append(None)
    return frame_idx, frames

def get_video_frame_count(video_path):
    """
    Returns the total number of frames in a video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def extract_frame(video_path, frame_idx):
    """
    Extracts a specific frame from a video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Cannot read frame {frame_idx} from video {video_path}")
        return None
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def extract_keypoints(frames, device):
    """
    Uses YOLOv8 to extract 2D keypoints from the frames.
    """
    # Initialize YOLOv8 model for pose estimation
    model_yolo = YOLO('yolov8s-pose.pt')  # Ensure you have the correct model file
    model_yolo.to(device)

    keypoints_list = []
    for idx, frame in enumerate(frames):
        if frame is not None:
            keypoints = extract_keypoints_from_image(model_yolo, frame)
            keypoints_list.append(keypoints)
        else:
            keypoints_list.append(None)
    return keypoints_list

def extract_keypoints_from_image(model, image):
    """
    Extracts keypoints from a single image using the YOLOv8 model.
    """
    # image should be in RGB format
    results = model(image)
    keypoints = []
    for result in results:
        if result.keypoints is not None:
            for kp in result.keypoints:
                kp_array = kp.cpu().numpy()  # Shape: (num_keypoints, 3)
                keypoints.append(kp_array)
    return keypoints

def plot_keypoints_on_images(frames, keypoints_list, output_dir, camera_ids):
    """
    Plots and saves the keypoints overlaid on the images.
    """
    for idx, (frame, keypoints, cam_id) in enumerate(zip(frames, keypoints_list, camera_ids)):
        if frame is not None:
            save_path = os.path.join(output_dir, f'{cam_id}_frame_{idx}_keypoints.png')
            plot_keypoints_on_image(frame, keypoints, save_path=save_path)

def plot_keypoints_on_image(image, keypoints, save_path=None):
    """
    Plots keypoints on a single image and saves it.
    """
    plt.figure()
    plt.imshow(image)
    if keypoints is not None:
        for person_kp in keypoints:
            x = person_kp[:, 0]
            y = person_kp[:, 1]
            plt.scatter(x, y, c='r', s=10)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def prepare_input_2d_to_3d(keypoints_list, config):
    """
    Prepares the input tensor for the 2D to 3D model.
    """
    # Since we don't have mean and std, we'll skip normalization for this demo
    joints2d_list = []
    for keypoints in keypoints_list:
        if keypoints is not None and len(keypoints) > 0:
            # Select the first person
            person_kp = keypoints[0][:, :2]  # Shape: (17, 2)
            joints2d_list.append(person_kp)
        else:
            # If no keypoints, fill with zeros
            joints2d_list.append(np.zeros((config['num_joints_2d'], 2)))

    # Flatten and prepare input tensor for the model
    joints2d_array = np.array(joints2d_list).flatten()
    joints2d_tensor = torch.tensor(joints2d_array, dtype=torch.float32)
    return joints2d_tensor

def predict_3d_joints(joints2d_tensor, config, device):
    """
    Loads the 2D to 3D model and predicts the 3D joints.
    """
    # Define the 2D to 3D model architecture (FCNNModel)
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

    # Instantiate the model and load weights
    input_size = len(config['camera_ids']) * config['num_joints_2d'] * 2  # x and y coordinates
    output_size = config['num_joints_3d'] * 3  # x, y, z coordinates
    hidden_sizes = [1024, 512, 256, 128, 64]
    dropout = 0.3

    model_2d3d = FCNNModel(input_size, output_size, hidden_sizes, dropout)
    model_2d3d.load_state_dict(torch.load(config['model_2d3d_path'], map_location=device))
    model_2d3d.to(device)
    model_2d3d.eval()

    # Make prediction using the 2D to 3D model
    joints2d_tensor = joints2d_tensor.to(device)
    with torch.no_grad():
        predicted_3d_joints = model_2d3d(joints2d_tensor.unsqueeze(0))  # Add batch dimension
        predicted_3d_joints = predicted_3d_joints.cpu().numpy().reshape(config['num_joints_3d'], 3)

    # Since we didn't normalize during training, we can proceed directly
    predicted_3d_joints_denorm = predicted_3d_joints
    return predicted_3d_joints_denorm

def plot_3d_joints(joints, exercise_key, frame_idx, output_dir='3d_plots', save=True):
    """
    Plot 3D joints and save the plot to the specified directory with exercise key and frame number.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    if joints.shape != (25, 3):
        raise ValueError(f"Expected shape (25, 3) for joints, but got {joints.shape}")
    x = joints[:, 0]
    y = joints[:, 1]
    z = joints[:, 2]
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the joints
    ax.scatter(x, y, z, c='r', marker='o')
    # Define the connections between joints based on the updated joint mapping
    connections = [
        # Head and Face
        (10, 9),  # Head to Nose
        (9, 8),  # Nose to Neck
        # Torso
        (8, 7),  # Neck to Stomach
        (7, 0),  # Stomach to Central Hip
        (0, 1),  # Central Hip to Left Hip
        (0, 4),  # Central Hip to Right Hip
        # Right Arm
        (8, 14), (14, 15), (15, 16),  # Neck to Right Shoulder to Right Elbow to Right Wrist
        (16, 23), (16, 24),  # Right Wrist to Right Palm and Fingers
        # Left Arm
        (8, 11), (11, 12), (12, 13),  # Neck to Left Shoulder to Left Elbow to Left Wrist
        (13, 21), (13, 22),  # Left Wrist to Left Palm and Fingers
        # Right Leg
        (4, 5), (5, 6), (6, 19), (6, 20),  # Right Hip to Right Knee to Right Ankle to Right Toe
        # Left Leg
        (1, 2), (2, 3), (3, 17), (3, 18)  # Left Hip to Left Knee to Left Ankle to Left Toe
    ]
    # Plot the lines connecting the joints
    for idx1, idx2 in connections:
        ax.plot(
            [x[idx1], x[idx2]],
            [y[idx1], y[idx2]],
            [z[idx1], z[idx2]],
            c='b'
        )
    # Set equal axis scaling for better visualization
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # Set labels and adjust the view angle
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=60)  # Adjust viewing angle for better understanding
    # Improve layout
    plt.tight_layout()
    if save:
        output_path = os.path.join(output_dir, f'{exercise_key}_frame_{frame_idx}_3djoints.png')
        plt.savefig(output_path, format='png')
        plt.close(fig)
        print(f"Saved 3D joints plot at {output_path}")
    else:
        plt.show()

def predict_smplx_params(joints3d_tensor, config, device):
    """
    Loads the 3D to SMPL-X model and predicts the SMPL-X parameters.
    """
    # Define the Transformer model (used for inference)
    class TransformerModel(nn.Module):
        def __init__(self, input_size, output_size, d_model, nhead, num_layers, dropout):
            super(TransformerModel, self).__init__()
            self.input_projection = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model * config['num_joints_3d'], output_size)
        def forward(self, x):
            # x shape: (batch_size, sequence_length, input_size)
            x = self.input_projection(x)  # Shape: (batch_size, sequence_length, d_model)
            x = x.permute(1, 0, 2)  # Transformer expects: (sequence_length, batch_size, d_model)
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)  # Back to (batch_size, sequence_length, d_model)
            x = x.reshape(x.size(0), -1)  # Flatten
            x = self.fc(x)
            return x

    # Instantiate the model and load weights
    input_size = 3  # x, y, z coordinates per joint
    output_size = 188  # Total SMPL-X parameter size (adjust as per your model)
    d_model = 256
    nhead = 8
    num_layers = 4
    dropout = 0.1

    model_3dsmplx = TransformerModel(input_size, output_size, d_model, nhead, num_layers, dropout)
    model_3dsmplx.load_state_dict(torch.load(config['model_3dsmplx_path'], map_location=device))
    model_3dsmplx.to(device)
    model_3dsmplx.eval()

    # Make prediction using the 3D joints to SMPL-X model
    with torch.no_grad():
        predicted_smplx_params = model_3dsmplx(joints3d_tensor)  # Shape: (1, 188)
        predicted_smplx_params = predicted_smplx_params.cpu().numpy().flatten()

    # Convert predicted parameters to SMPL-X input format
    smplx_params = create_smplx_input_dict(predicted_smplx_params)
    return smplx_params

def create_smplx_input_dict(predicted_params):
    """
    Converts the predicted SMPL-X parameters vector into a dictionary format required by the SMPL-X model.
    """
    idx = 0
    smplx_input = {}
    # Transl
    smplx_input['transl'] = predicted_params[idx:idx+3]
    idx += 3
    # Global orientation
    smplx_input['global_orient'] = predicted_params[idx:idx+3]
    idx += 3
    # Body pose
    smplx_input['body_pose'] = predicted_params[idx:idx+63]  # 21 joints * 3
    idx += 63
    # Betas
    smplx_input['betas'] = predicted_params[idx:idx+10]
    idx += 10
    # Left hand pose
    smplx_input['left_hand_pose'] = predicted_params[idx:idx+45]  # 15 joints * 3
    idx += 45
    # Right hand pose
    smplx_input['right_hand_pose'] = predicted_params[idx:idx+45]
    idx += 45
    # Jaw pose
    smplx_input['jaw_pose'] = predicted_params[idx:idx+3]
    idx += 3
    # Left eye pose
    smplx_input['leye_pose'] = predicted_params[idx:idx+3]
    idx += 3
    # Right eye pose
    smplx_input['reye_pose'] = predicted_params[idx:idx+3]
    idx += 3
    # Expression
    smplx_input['expression'] = predicted_params[idx:idx+10]
    idx +=10
    return smplx_input

def generate_and_visualize_smplx_mesh(smplx_params, config):
    """
    Generates the SMPL-X mesh and visualizes it.
    """
    # Now, generate the SMPL-X mesh
    from smplx import SMPLX

    # Load the SMPL-X model
    smplx_model = SMPLX(
        model_path=os.path.dirname(config['smplx_model_path']),
        model_type='smplx',
        gender='neutral',
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        use_face_contour=True,
        ext='npz'
    )

    # Prepare input tensors for SMPL-X model
    input_dict = {}
    for key in smplx_params:
        param = smplx_params[key]
        input_dict[key] = torch.tensor(param, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Generate the mesh
    output = smplx_model(**input_dict)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = smplx_model.faces

    # Save the mesh as an OBJ file
    mesh = trimesh.Trimesh(vertices, faces)
    mesh_output_path = os.path.join(config['output_dir'], 'predicted_smplx_mesh.obj')
    mesh.export(mesh_output_path)
    print(f"SMPL-X mesh saved at {mesh_output_path}")

    # Visualize the mesh
    mesh_image_path = os.path.join(config['output_dir'], 'predicted_smplx_mesh.png')
    visualize_mesh(mesh_output_path, save_path=mesh_image_path)

def visualize_mesh(mesh_path, save_path=None):
    """
    Visualizes the mesh and saves it as an image.
    """
    mesh = trimesh.load(mesh_path)
    # Render and save an image
    scene = mesh.scene()
    png = scene.save_image(resolution=(800, 800))
    if save_path:
        with open(save_path, 'wb') as f:
            f.write(png)
        print(f"Mesh visualization saved at {save_path}")

if __name__ == '__main__':
    main()