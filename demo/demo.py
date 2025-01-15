import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from utils import (
    set_seed,
    load_mean_std,
    extract_2d_keypoints,
    load_joints3d,
    plot_pred_and_gt_joints,
    FCNNModel,
    init_yolo_model
)


def main():
    set_seed(42)

    # Adjust paths
    dataset_root = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train'
    model_path = '/public.hpc/alessandro.folloni2/smpl_study/models/FCNN_bs32_ep25_joints_model.pth'
    yolo_weights_path = '/public.hpc/alessandro.folloni2/smpl_study/pretrained_yolos/yolov8s-pose.pt'

    # Full set of cameras available
    all_cameras = ['50591643', '58860488', '60457274', '65906101']
    # Subset of cameras to actually use
    cameras = ['58860488', '60457274', '65906101']

    num_joints_2d = 17
    num_joints_3d = 25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean2d, std2d = load_mean_std(dataset_root)

    pose_model = init_yolo_model(yolo_weights_path, device)
    if pose_model is None:
        raise RuntimeError("Failed to load YOLO model. pose_model is None.")

    subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    subjects.sort()

    max_subject_attempts = 5
    for attempt in range(max_subject_attempts):
        subject = random.choice(subjects)
        subject_path = os.path.join(dataset_root, subject)
        videos_root = os.path.join(subject_path, 'videos')

        if not os.path.exists(videos_root):
            print(f"No videos directory for subject {subject}, retrying...")
            continue

        joints3d_folder = os.path.join(subject_path, 'joints3d_25')
        exercises = [f for f in os.listdir(joints3d_folder) if f.endswith('.json')]
        if not exercises:
            print(f"No 3D exercises found for subject {subject}, retrying...")
            continue

        max_exercise_attempts = 5
        success = False
        for _ in range(max_exercise_attempts):
            exercise_file = random.choice(exercises)
            exercise_name = os.path.splitext(exercise_file)[0]

            print(f"Selected subject: {subject}")
            print(f"Selected exercise: {exercise_name}")

            cam_videos = {}
            try:
                # Attempt to find videos for all cameras
                # We'll only extract keypoints and predict for the subset 'cameras'
                for cam_id in all_cameras:
                    cam_path = os.path.join(videos_root, cam_id)
                    if not os.path.isdir(cam_path):
                        # Camera directory doesn't exist, skip it but note in videos info later
                        continue
                    video_candidates = [f for f in os.listdir(cam_path) if
                                        f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                    matched = [v for v in video_candidates if os.path.splitext(v)[0] == exercise_name]
                    if matched:
                        # Found a matching video for this camera
                        cam_videos[cam_id] = os.path.join(cam_path, matched[0])

                # Check if at least one of the chosen 'cameras' has a corresponding video
                used_cam_videos = {cam_id: v for cam_id, v in cam_videos.items() if cam_id in cameras}
                if len(used_cam_videos) < len(cameras):
                    print("Not all chosen cameras have a matching video. Trying another exercise...")
                    continue

                # Extract 2D keypoints for the chosen cameras only
                joints2d_data = {}
                for cam_id in cameras:
                    v_path = used_cam_videos[cam_id]
                    print(f"Extracting keypoints from camera {cam_id}, video: {v_path}")
                    all_keypoints = extract_2d_keypoints(v_path, pose_model, device, mean2d, std2d, num_joints_2d)
                    joints2d_data[cam_id] = all_keypoints

                lengths = [len(joints2d_data[cam_id]) for cam_id in cameras]
                if any(length == 0 for length in lengths):
                    print("One of the chosen cameras returned 0 frames. Trying another exercise...")
                    continue

                min_length = min(lengths)
                for cam_id in cameras:
                    joints2d_data[cam_id] = joints2d_data[cam_id][:min_length]

                # Load 3D GT
                joints3d_file_path = os.path.join(joints3d_folder, exercise_file)
                joints3d_array = load_joints3d(joints3d_file_path)
                if joints3d_array.shape[0] == 0:
                    print("No frames in 3D data, trying another exercise...")
                    continue
                joints3d_array = joints3d_array[:min_length]

                if joints3d_array.shape[0] < 1:
                    print("No frames after trimming, trying another exercise...")
                    continue

                frame_idx = random.randint(0, min_length - 1)
                print(f"Selected frame {frame_idx}")

                input_size = len(cameras) * num_joints_2d * 2
                output_size = num_joints_3d * 3
                hidden_sizes = [512, 256]
                dropout = 0.2

                model = FCNNModel(input_size, output_size, hidden_sizes, dropout).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                print("Model loaded.")

                # Prepare frame data
                frame_2d = []
                for cam_id in cameras:
                    frame_2d.extend(joints2d_data[cam_id][frame_idx])
                frame_2d = np.array(frame_2d).flatten()
                input_tensor = torch.tensor(frame_2d, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred_3d = model(input_tensor).cpu().numpy().reshape(num_joints_3d, 3)

                gt_3d = joints3d_array[frame_idx]

                # Now create a unique experiment folder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_dir = f"experiment_{subject}_{exercise_name}_{timestamp}"
                os.makedirs(experiment_dir, exist_ok=True)

                # Save videos info
                videos_info_path = os.path.join(experiment_dir, "videos_info.txt")
                with open(videos_info_path, 'w') as f:
                    f.write(f"Subject: {subject}\nExercise: {exercise_name}\n\nVideos:\n")
                    for cam_id in all_cameras:
                        if cam_id in cam_videos:
                            # Camera video found
                            used_str = " (USED)" if cam_id in cameras else " (NOT USED)"
                            f.write(f"{cam_id}{used_str}: {cam_videos[cam_id]}\n")
                        else:
                            # No video found for this camera
                            f.write(f"{cam_id}: No video found\n")
                print(f"Saved videos info at {videos_info_path}")

                # Save 2D keypoints
                keypoints_dir = os.path.join(experiment_dir, "2d_keypoints")
                os.makedirs(keypoints_dir, exist_ok=True)
                for cam_id in cameras:
                    out_file = os.path.join(keypoints_dir, f"{exercise_name}_keypoints_{cam_id}.json")
                    with open(out_file, 'w') as f:
                        json.dump(joints2d_data[cam_id], f)
                print(f"Saved 2D keypoints in {keypoints_dir}")

                # Save 3D data
                three_d_dir = os.path.join(experiment_dir, "3d_data")
                os.makedirs(three_d_dir, exist_ok=True)
                truncated_3d_path = os.path.join(three_d_dir, "joints3d.json")
                with open(truncated_3d_path, 'w') as f:
                    json.dump(joints3d_array.tolist(), f)
                print(f"Saved truncated 3D data at {truncated_3d_path}")

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
                print(f"Saved selected frame 2D data at {selected_frame_2d_path}")

                # Save predicted 3D
                selected_frame_pred_3d_path = os.path.join(selected_frame_dir, "selected_frame_predicted_3d.json")
                with open(selected_frame_pred_3d_path, 'w') as f:
                    json.dump(pred_3d.tolist(), f)
                print(f"Saved predicted 3D data at {selected_frame_pred_3d_path}")

                # Save ground truth 3D
                selected_frame_gt_3d_path = os.path.join(selected_frame_dir, "selected_frame_groundtruth_3d.json")
                with open(selected_frame_gt_3d_path, 'w') as f:
                    json.dump(gt_3d.tolist(), f)
                print(f"Saved groundtruth 3D data at {selected_frame_gt_3d_path}")

                # Plot comparison
                plot_file = os.path.join(selected_frame_dir, "comparison.png")
                plot_pred_and_gt_joints(pred_3d, gt_3d, plot_file)

                print("Done.")
                success = True
                break
            except Exception as e:
                print(f"Error encountered with {exercise_name}: {e}")
                continue

        if success:
            break
    else:
        print("Failed to find a suitable subject and exercise after multiple attempts.")


if __name__ == "__main__":
    main()