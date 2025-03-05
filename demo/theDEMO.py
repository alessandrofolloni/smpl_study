import json
import os
import random
import numpy as np
import torch
from smplx import create as smplx_create
from demo_utils import (
    get_subjects,
    select_random_subject,
    get_exercise_files,
    select_random_exercise,
    get_videos_for_exercise,
    save_text_log,
    save_json_data,
    select_n_frames_from_videos,
    process_multiview_frames,
    load_transformer_model_joints3d,
    predict_joints3d,
    load_transformer_model_smplx,
    predict_smplx_params,
    plot_pred_and_gt_joints,
    load_joints3d,
    generate_smplx_mesh
)

def save_ground_truth_obj(joints3d, output_path):
    """Save ground truth 3D joints as an OBJ with vertices and line connectivity."""
    connections = [
        (10, 9), (9, 8), (8, 7), (7, 0), (0, 1), (0, 4),
        (8, 14), (14, 15), (15, 16), (16, 23), (16, 24),
        (8, 11), (11, 12), (12, 13), (13, 21), (13, 22),
        (4, 5), (5, 6), (6, 19), (6, 20),
        (1, 2), (2, 3), (3, 17), (3, 18)
    ]
    with open(output_path, "w") as f:
        for v in joints3d:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for c in connections:
            f.write("l {} {}\n".format(c[0]+1, c[1]+1))

def main():
    dataset_root = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train'
    used_camera_ids = ['58860488', '60457274']
    all_cameras = ['50591643', '58860488', '60457274', '65906101']
    output_dir = 'demo_selection_results'
    os.makedirs(output_dir, exist_ok=True)
    overall_log = ""

    # Subject, exercise, video selection
    subjects = get_subjects(dataset_root)
    subject, subject_path = select_random_subject(subjects, dataset_root)
    overall_log += f"Selected subject: {subject}\n\n"
    exercise_files, _ = get_exercise_files(subject_path, folder_name='joints3d_25')
    exercise_name, exercise_file = select_random_exercise(exercise_files)
    overall_log += f"Selected exercise: {exercise_name}\nExercise file: {exercise_file}\n\n"
    videos_info, video_log = get_videos_for_exercise(subject_path, exercise_name, all_cameras, videos_folder_name='videos')
    overall_log += "Video information:\n" + video_log + "\n"
    save_json_data(output_dir, "selection_data.json", {
        "subject": subject,
        "exercise": exercise_name,
        "exercise_file": exercise_file,
        "videos": videos_info,
        "cameras": all_cameras
    })

    # Select 5 random frames from the videos
    frames_info_all = select_n_frames_from_videos(videos_info, output_dir, exercise_name, n=5)
    save_json_data(output_dir, "frames_info.json", frames_info_all)
    overall_log += f"Selected frames: {list(frames_info_all.keys())}\n\n"

    # Load mean-std for 2D keypoints
    mean_std_path = os.path.join(dataset_root, 'joints2d_mean_std.json')
    with open(mean_std_path, 'r') as f:
        mean_std_data = json.load(f)
    mean2d, std2d = mean_std_data.get('mean2d'), mean_std_data.get('std2d')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path_joints3d = '/public.hpc/alessandro.folloni2/smpl_study/models/Transformer_bs16_ep200_joints_model.pth'
    transformer_model_joints3d = load_transformer_model_joints3d(weights_path_joints3d, device)
    weights_path_smplx = '/public.hpc/alessandro.folloni2/smpl_study/runs/Transformer_bs32_lr0.0002_epochs70_dmodel_384_nhead_6_layers_4_dropout_0.2/model_Transformer.pth'
    transformer_model_smplx = load_transformer_model_smplx(weights_path_smplx, device)

    # Load ground truth joints for the exercise (same file for all frames)
    gt_file = os.path.join(subject_path, "joints3d_25", exercise_file)
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file {gt_file} not found.")
    gt_data, _ = load_joints3d(gt_file)

    # Load SMPL-X model once
    smplx_model = smplx_create(
        "/public.hpc/alessandro.folloni2/smpl_study/smplx_usage/files/body_models/SMPLX_NEUTRAL.npz",
        model_type='smplx',
        gender='neutral',
        ext='npz',
        use_pca=False,
        num_pca_comps=0
    ).to(device)
    print("SMPL-X model loaded.")

    # Process each frame separately
    for frame_idx, frames_info in frames_info_all.items():
        frame_folder = os.path.join(output_dir, f"frame_{frame_idx}")
        os.makedirs(frame_folder, exist_ok=True)

        # Process 2D keypoints
        norm_kps = process_multiview_frames(frames_info, mean2d, std2d, num_joints_2d=17)
        save_json_data(frame_folder, "normalized_keypoints.json", norm_kps)

        # Predict joints3d
        predicted_joints3d = predict_joints3d(norm_kps, transformer_model_joints3d, used_camera_ids, num_joints_2d=17)
        save_json_data(frame_folder, "predicted_joints3d.json", predicted_joints3d.tolist())

        # Predict SMPL-X parameters and generate mesh
        predicted_smplx_params = predict_smplx_params(predicted_joints3d, transformer_model_smplx)
        save_json_data(frame_folder, "predicted_smplx_params.json", predicted_smplx_params.tolist())
        mesh = generate_smplx_mesh(predicted_smplx_params, smplx_model, device)
        pred_mesh_path = os.path.join(frame_folder, f"{exercise_name}_predicted_smplx.obj")
        mesh.export(pred_mesh_path)
        overall_log += f"Frame {frame_idx}: Exported predicted mesh to {pred_mesh_path}\n"

        # Save ground truth joints as OBJ
        frame_key = f"frame_{frame_idx}"
        if frame_key not in gt_data:
            raise ValueError(f"Frame {frame_key} not found in ground truth file {gt_file}.")
        gt_joints3d = np.array(gt_data[frame_key])
        gt_obj_path = os.path.join(frame_folder, f"{exercise_name}_ground_truth_frame_{frame_idx}.obj")
        save_ground_truth_obj(gt_joints3d, gt_obj_path)
        overall_log += f"Frame {frame_idx}: Exported ground truth OBJ to {gt_obj_path}\n"

        # Plot and save comparison
        plot_folder = os.path.join(frame_folder, "plots")
        os.makedirs(plot_folder, exist_ok=True)
        plot_pred_and_gt_joints(predicted_joints3d, gt_joints3d, exercise_name, frame_idx, output_dir=plot_folder, save=True)
        overall_log += f"Frame {frame_idx}: Saved comparison plot in {plot_folder}\n\n"

    save_text_log(output_dir, "demo_all_info.txt", overall_log)
    print("Demo complete. All intermediate results have been saved.")

if __name__ == "__main__":
    main()