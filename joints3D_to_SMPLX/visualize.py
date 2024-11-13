import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import smplx
import trimesh
import pyrender
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
config = {
    'model_folder': '/public.hpc/alessandro.folloni2/'
                    'smpl_study/files/body_models/',  # Update this path
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': './visualization_results',        # Directory to save images
}

device = torch.device(config['device'])

def reconstruct_smplx_mesh(params_vector):
    # [Same as before]
    idx = 0
    # Translation
    transl = params_vector[idx:idx+3]
    idx += 3

    # Global Orientation
    global_orient_axis_angle = params_vector[idx:idx+3]
    idx += 3

    # Body Pose
    body_pose_axis_angle = params_vector[idx:idx+63]  # 21 joints * 3
    idx += 63

    # Betas
    betas = params_vector[idx:idx+10]
    idx += 10

    # Left Hand Pose
    left_hand_pose_axis_angle = params_vector[idx:idx+45]  # 15 joints * 3
    idx += 45

    # Right Hand Pose
    right_hand_pose_axis_angle = params_vector[idx:idx+45]  # 15 joints * 3
    idx += 45

    # Jaw Pose
    jaw_pose_axis_angle = params_vector[idx:idx+3]
    idx += 3

    # Left Eye Pose
    leye_pose_axis_angle = params_vector[idx:idx+3]
    idx += 3

    # Right Eye Pose
    reye_pose_axis_angle = params_vector[idx:idx+3]
    idx += 3

    # Expression
    expression = params_vector[idx:idx+10]
    idx += 10

    # Prepare parameters for SMPL-X model
    smplx_params = {
        'transl': torch.tensor(transl, dtype=torch.float32).unsqueeze(0).to(device),
        'global_orient': torch.tensor(global_orient_axis_angle, dtype=torch.float32).unsqueeze(0).to(device),
        'body_pose': torch.tensor(body_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(device),
        'betas': torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(device),
        'expression': torch.tensor(expression, dtype=torch.float32).unsqueeze(0).to(device),
        'left_hand_pose': torch.tensor(left_hand_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(device),
        'right_hand_pose': torch.tensor(right_hand_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(device),
        'jaw_pose': torch.tensor(jaw_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(device),
        'leye_pose': torch.tensor(leye_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(device),
        'reye_pose': torch.tensor(reye_pose_axis_angle, dtype=torch.float32).unsqueeze(0).to(device)
    }

    # Load SMPL-X model
    smplx_model = smplx.create(
        model_path=config['model_folder'],
        model_type='smplx',
        gender='neutral',
        ext='npz',
        use_pca=False
    ).to(device)

    with torch.no_grad():
        output = smplx_model(
            betas=smplx_params['betas'],
            expression=smplx_params['expression'],
            global_orient=smplx_params['global_orient'],
            body_pose=smplx_params['body_pose'],
            left_hand_pose=smplx_params['left_hand_pose'],
            right_hand_pose=smplx_params['right_hand_pose'],
            jaw_pose=smplx_params['jaw_pose'],
            leye_pose=smplx_params['leye_pose'],
            reye_pose=smplx_params['reye_pose'],
            transl=smplx_params['transl'],
            return_verts=True
        )

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = smplx_model.faces
    mesh = trimesh.Trimesh(vertices, faces)
    return mesh

def render_mesh(mesh):
    # [Same as before]
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)

    # Set up camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 2.5  # Move camera along z-axis
    scene.add(camera, pose=camera_pose)

    # Add lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    color, _ = r.render(scene)
    r.delete()
    return color

def create_comparison_image(pred_image, gt_image):
    # [Same as before]
    pred_pil = Image.fromarray(pred_image)
    gt_pil = Image.fromarray(gt_image)
    combined_width = pred_pil.width + gt_pil.width
    combined_image = Image.new('RGB', (combined_width, pred_pil.height))
    combined_image.paste(pred_pil, (0, 0))
    combined_image.paste(gt_pil, (pred_pil.width, 0))
    return combined_image

def visualize_results(predicted_params_list, gt_params_list, exercise_keys, frame_idxs):
    os.makedirs(config['output_dir'], exist_ok=True)

    max_samples = 10  # Limit the number of samples to visualize
    for i in range(min(len(predicted_params_list), max_samples)):
        predicted_params = predicted_params_list[i]
        gt_params = gt_params_list[i]
        exercise_key = exercise_keys[i]
        frame_idx = frame_idxs[i]

        # Reconstruct meshes
        pred_mesh = reconstruct_smplx_mesh(predicted_params)
        gt_mesh = reconstruct_smplx_mesh(gt_params)

        # Render meshes to images
        pred_image = render_mesh(pred_mesh)
        gt_image = render_mesh(gt_mesh)

        # Create comparison image
        comparison_image = create_comparison_image(pred_image, gt_image)

        # Save the image
        output_path = os.path.join(config['output_dir'], f'{exercise_key}_frame_{frame_idx}_comparison.png')
        comparison_image.save(output_path)
        print(f"Saved comparison image to {output_path}")

        # Optionally display the image
        # comparison_image.show()

def load_predicted_parameters(predicted_params_path):
    data = np.load(predicted_params_path, allow_pickle=True)
    predicted_params_list = data['predicted_params']
    gt_params_list = data['gt_params']
    exercise_keys = data['exercise_keys']
    frame_idxs = data['frame_idxs']
    return predicted_params_list, gt_params_list, exercise_keys, frame_idxs

if __name__ == '__main__':
    predicted_params_path = 'predicted_parameters.npz'

    # Load the predicted parameters
    predicted_params_list, gt_params_list, exercise_keys, frame_idxs = load_predicted_parameters(predicted_params_path)

    # Visualize the results
    visualize_results(predicted_params_list, gt_params_list, exercise_keys, frame_idxs)