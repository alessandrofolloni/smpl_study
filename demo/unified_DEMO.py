import json
import os
import random
from datetime import datetime

import numpy as np
import pyrender
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from smplx import create as smplx_create

from final_utils import (
    set_seed,
    load_mean_std,
    extract_2d_keypoints,
    load_joints3d,
    plot_pred_and_gt_joints,
    FCNNModel,
    init_yolo_model,
    TransformerModel,
    load_model,
    axis_angle_to_rotation_matrices,
    save_smplx_json,
    save_mesh
)


def main():
    set_seed(42)

    # Paths for the first part (FCNN prediction)
    dataset_root = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train'
    fcnn_model_path = '/public.hpc/alessandro.folloni2/smpl_study/models/FCNN_bs32_ep25_joints_model.pth'
    yolo_weights_path = '/public.hpc/alessandro.folloni2/smpl_study/pretrained_yolos/yolov8s-pose.pt'

    # Paths for the Transformer model (SMPL-X prediction)
    transformer_run_name = 'Transformer_bs32_lr0.0002_epochs70_dmodel_384_nhead_6_layers_4_dropout_0.2'
    transformer_model_path = '/public.hpc/alessandro.folloni2/smpl_study/runs/{}/model_Transformer.pth'.format(transformer_run_name)
    smplx_model_file = "/public.hpc/alessandro.folloni2/smpl_study/smplx_usage/files/body_models/SMPLX_NEUTRAL.npz"

    all_cameras = ['50591643', '58860488', '60457274', '65906101']
    cameras = ['58860488', '60457274', '65906101']

    num_joints_2d = 17
    num_joints_3d = 25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean2d, std2d = load_mean_std(dataset_root)

    pose_model = init_yolo_model(yolo_weights_path, device)
    if pose_model is None:
        raise RuntimeError("Failed to load YOLO model.")

    subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    subjects.sort()

    # Load FCNN model
    input_size = len(cameras) * num_joints_2d * 2
    output_size = num_joints_3d * 3
    hidden_sizes = [512, 256]
    dropout = 0.2
    fcnn = FCNNModel(input_size, output_size, hidden_sizes, dropout).to(device)
    fcnn = load_model(fcnn, fcnn_model_path, device)

    # Load Transformer model
    d_model = 384
    nhead = 6
    transformer_num_layers = 4
    transformer_dropout = 0.2
    transformer_input_size = 3  # (x,y,z) per joint
    transformer_output_size = 188  # SMPL-X params
    transformer_model = TransformerModel(
        input_size=transformer_input_size,
        output_size=transformer_output_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=transformer_num_layers,
        dropout=transformer_dropout
    )
    transformer_model = load_model(transformer_model, transformer_model_path, device)

    max_subject_attempts = 5
    success = False
    for attempt in range(max_subject_attempts):
        subject = random.choice(subjects)
        subject_path = os.path.join(dataset_root, subject)
        videos_root = os.path.join(subject_path, 'videos')

        if not os.path.exists(videos_root):
            continue

        joints3d_folder = os.path.join(subject_path, 'joints3d_25')
        exercises = [f for f in os.listdir(joints3d_folder) if f.endswith('.json')]
        if not exercises:
            continue

        max_exercise_attempts = 5
        for _ in range(max_exercise_attempts):
            exercise_file = random.choice(exercises)
            exercise_name = os.path.splitext(exercise_file)[0]

            cam_videos = {}
            # Find matching videos for all cameras
            for cam_id in all_cameras:
                cam_path = os.path.join(videos_root, cam_id)
                if not os.path.isdir(cam_path):
                    continue
                video_candidates = [f for f in os.listdir(cam_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                matched = [v for v in video_candidates if os.path.splitext(v)[0] == exercise_name]
                if matched:
                    cam_videos[cam_id] = os.path.join(cam_path, matched[0])

            used_cam_videos = {cam_id: v for cam_id, v in cam_videos.items() if cam_id in cameras}
            if len(used_cam_videos) < len(cameras):
                continue

            # Extract 2D keypoints
            joints2d_data = {}
            for cam_id in cameras:
                v_path = used_cam_videos[cam_id]
                all_keypoints = extract_2d_keypoints(v_path, pose_model, device, mean2d, std2d, num_joints_2d)
                joints2d_data[cam_id] = all_keypoints

            lengths = [len(joints2d_data[cam_id]) for cam_id in cameras]
            if any(length == 0 for length in lengths):
                continue

            min_length = min(lengths)
            for cam_id in cameras:
                joints2d_data[cam_id] = joints2d_data[cam_id][:min_length]

            # Load 3D GT
            joints3d_file_path = os.path.join(joints3d_folder, exercise_file)
            joints3d_array = load_joints3d(joints3d_file_path)
            joints3d_array = joints3d_array[:min_length]

            if joints3d_array.shape[0] < 1:
                continue

            frame_idx = random.randint(0, min_length - 1)

            # Prepare FCNN input
            frame_2d = []
            for cam_id in cameras:
                frame_2d.extend(joints2d_data[cam_id][frame_idx])
            frame_2d = np.array(frame_2d).flatten()
            input_tensor = torch.tensor(frame_2d, dtype=torch.float32).unsqueeze(0).to(device)

            print("\n--- Dimension Checks ---")
            print(f"FCNN input shape: {input_tensor.shape} (Should be (1, {input_size}))")

            # Run FCNN inference
            with torch.no_grad():
                pred_3d = fcnn(input_tensor).cpu().numpy().reshape(num_joints_3d, 3)

            gt_3d = joints3d_array[frame_idx]

            print(f"pred_3d shape: {pred_3d.shape}, gt_3d shape: {gt_3d.shape}")

            # Create experiment directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = f"experiment_{subject}_{exercise_name}_{timestamp}"
            os.makedirs(experiment_dir, exist_ok=True)

            # Save videos info
            videos_info_path = os.path.join(experiment_dir, "videos_info.txt")
            with open(videos_info_path, 'w') as f:
                f.write(f"Subject: {subject}\nExercise: {exercise_name}\n\nVideos:\n")
                for cam_id in all_cameras:
                    if cam_id in cam_videos:
                        used_str = " (USED)" if cam_id in cameras else " (NOT USED)"
                        f.write(f"{cam_id}{used_str}: {cam_videos[cam_id]}\n")
                    else:
                        f.write(f"{cam_id}: No video found\n")

            # Save 2D keypoints
            keypoints_dir = os.path.join(experiment_dir, "2d_keypoints")
            os.makedirs(keypoints_dir, exist_ok=True)
            for cam_id in cameras:
                out_file = os.path.join(keypoints_dir, f"{exercise_name}_keypoints_{cam_id}.json")
                with open(out_file, 'w') as f:
                    json.dump(joints2d_data[cam_id], f)

            # Save truncated 3D data
            three_d_dir = os.path.join(experiment_dir, "3d_data")
            os.makedirs(three_d_dir, exist_ok=True)
            truncated_3d_path = os.path.join(three_d_dir, "joints3d.json")
            with open(truncated_3d_path, 'w') as f:
                json.dump(joints3d_array.tolist(), f)

            # Save selected frame data
            selected_frame_dir = os.path.join(experiment_dir, "selected_frame")
            os.makedirs(selected_frame_dir, exist_ok=True)

            # Save frame 2D
            selected_frame_2d_path = os.path.join(selected_frame_dir, "selected_frame_2d.json")
            frame_2d_dict = {}
            start_idx = 0
            joints_per_cam = num_joints_2d
            for cam_id in cameras:
                end_idx = start_idx + joints_per_cam * 2
                cam_2d = frame_2d[start_idx:end_idx].reshape(-1, 2).tolist()
                frame_2d_dict[cam_id] = cam_2d
                start_idx = end_idx
            with open(selected_frame_2d_path, 'w') as f:
                json.dump(frame_2d_dict, f)

            # Save predicted 3D
            selected_frame_pred_3d_path = os.path.join(selected_frame_dir, "selected_frame_predicted_3d.json")
            with open(selected_frame_pred_3d_path, 'w') as f:
                json.dump(pred_3d.tolist(), f)

            # Save ground truth 3D
            selected_frame_gt_3d_path = os.path.join(selected_frame_dir, "selected_frame_groundtruth_3d.json")
            with open(selected_frame_gt_3d_path, 'w') as f:
                json.dump(gt_3d.tolist(), f)

            # Plot comparison
            plot_file = os.path.join(selected_frame_dir, "comparison.jpg")
            plot_pred_and_gt_joints(pred_3d, gt_3d, plot_file)

            # ----------------------------
            # Use Transformer to get SMPL-X params
            # ----------------------------
            # pred_3d shape (25,3), as input to Transformer: (1,25,3)
            transformer_input = pred_3d.astype(np.float32)
            print(f"Transformer input shape before batch dimension: {transformer_input.shape}")
            transformer_input = transformer_input[np.newaxis, ...]
            print(f"Transformer input shape (with batch): {transformer_input.shape}")

            with torch.no_grad():
                transformer_input_tensor = torch.tensor(transformer_input, dtype=torch.float32).to(device)
                smplx_params = transformer_model(transformer_input_tensor).cpu().numpy().squeeze()
            print(f"Transformer output shape: {smplx_params.shape}, expected (188,)")

            # Extract parameters
            transl = smplx_params[0:3]
            global_orient = smplx_params[3:6].reshape(1,3)
            body_pose = smplx_params[6:69].reshape(21,3)
            betas = smplx_params[69:79]
            left_hand_pose = smplx_params[79:124].reshape(15,3)
            right_hand_pose = smplx_params[124:169].reshape(15,3)
            jaw_pose = smplx_params[169:172].reshape(1,3)
            leye_pose = smplx_params[172:175].reshape(1,3)
            reye_pose = smplx_params[175:178].reshape(1,3)
            expression = smplx_params[178:188]

            # Convert to rotation matrices
            global_orient_mat = axis_angle_to_rotation_matrices(global_orient)
            body_pose_mat = axis_angle_to_rotation_matrices(body_pose)
            left_hand_pose_mat = axis_angle_to_rotation_matrices(left_hand_pose)
            right_hand_pose_mat = axis_angle_to_rotation_matrices(right_hand_pose)
            jaw_pose_mat = axis_angle_to_rotation_matrices(jaw_pose)
            leye_pose_mat = axis_angle_to_rotation_matrices(leye_pose)
            reye_pose_mat = axis_angle_to_rotation_matrices(reye_pose)

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

            smplx_json_path = os.path.join(experiment_dir, "smplx_params.json")
            save_smplx_json(smplx_output, smplx_json_path)

            # ----------------------------
            # Generate SMPL-X mesh
            # ----------------------------
            smplx_model = smplx_create(
                smplx_model_file,
                model_type='smplx',
                gender='neutral',
                ext='npz',
                use_pca=False
            ).to(device)

            # Convert axis-angle back from rotation matrices for SMPL-X forward pass
            def flatten_aa_from_mat(mat):
                r = R.from_matrix(mat)
                aa = r.as_rotvec()
                return aa

            global_orient_aa = flatten_aa_from_mat(global_orient_mat)[None,...]
            body_pose_aa = flatten_aa_from_mat(body_pose_mat).flatten()[None,...]
            left_hand_pose_aa = flatten_aa_from_mat(left_hand_pose_mat).flatten()[None,...]
            right_hand_pose_aa = flatten_aa_from_mat(right_hand_pose_mat).flatten()[None,...]
            jaw_pose_aa = flatten_aa_from_mat(jaw_pose_mat)[None,...]
            leye_pose_aa = flatten_aa_from_mat(leye_pose_mat)[None,...]
            reye_pose_aa = flatten_aa_from_mat(reye_pose_mat)[None,...]

            transl_torch = torch.tensor(transl, dtype=torch.float32).unsqueeze(0).to(device)
            global_orient_torch = torch.tensor(global_orient_aa, dtype=torch.float32).to(device)
            body_pose_torch = torch.tensor(body_pose_aa, dtype=torch.float32).to(device)
            left_hand_pose_torch = torch.tensor(left_hand_pose_aa, dtype=torch.float32).to(device)
            right_hand_pose_torch = torch.tensor(right_hand_pose_aa, dtype=torch.float32).to(device)
            jaw_pose_torch = torch.tensor(jaw_pose_aa, dtype=torch.float32).to(device)
            leye_pose_torch = torch.tensor(leye_pose_aa, dtype=torch.float32).to(device)
            reye_pose_torch = torch.tensor(reye_pose_aa, dtype=torch.float32).to(device)
            betas_torch = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(device)
            expression_torch = torch.tensor(expression, dtype=torch.float32).unsqueeze(0).to(device)

            print("Running SMPL-X model forward pass for mesh generation...")
            with torch.no_grad():
                output = smplx_model(
                    betas=betas_torch,
                    expression=expression_torch,
                    global_orient=global_orient_torch,
                    body_pose=body_pose_torch,
                    left_hand_pose=left_hand_pose_torch,
                    right_hand_pose=right_hand_pose_torch,
                    jaw_pose=jaw_pose_torch,
                    leye_pose=leye_pose_torch,
                    reye_pose=reye_pose_torch,
                    transl=transl_torch,
                    return_verts=True
                )

            vertices = output.vertices[0].cpu().numpy()
            faces = smplx_model.faces
            mesh = trimesh.Trimesh(vertices, faces)
            mesh_path = os.path.join(experiment_dir, "smplx_mesh.obj")
            save_mesh(mesh, mesh_path)

            print("Visualization attempt...")
            # Create a pyrender scene and add the mesh
            scene = pyrender.Scene()
            mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
            scene.add(mesh_pyrender)

            print("Done.")
            success = True
            break
        if success:
            break
    else:
        print("Failed to find a suitable subject and exercise after multiple attempts.")


if __name__ == "__main__":
    main()