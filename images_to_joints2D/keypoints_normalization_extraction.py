import os
import torch
import json
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from typing import Any, Dict


def get_number_of_frames(data: Any, key: str = None) -> int:
    """
    Determines the number of frames in the data.

    Parameters:
        data (Any): The JSON data to inspect.
        key (str): Optional key to extract data from a dictionary.

    Returns:
        int: The number of frames.
    """
    if key and isinstance(data, dict):
        if key in data:
            data = data[key]
        else:
            raise ValueError(f"Expected key '{key}' not found in data.")
    if isinstance(data, list):
        return len(data)
    else:
        raise ValueError("Data is not a list.")


def check_and_trim_keypoints(subject_folder: str, joints2d_folder_name: str):
    """
    Checks the consistency of frame counts between 2D keypoints files across all cameras and the 3D keypoints file.
    Automatically trims the first n frames to align the frame counts without user input.

    Parameters:
        subject_folder (str): The path to the subject's folder containing 'joints2d' and 'joints3d_25' folders.
        joints2d_folder_name (str): Name of the joints2d folder.
    """
    # Paths to joints2d and joints3d_25 folders
    joints2d_folder = os.path.join(subject_folder, joints2d_folder_name)
    joints3d_folder = os.path.join(subject_folder, 'joints3d_25')

    # Get all camera folders in the joints2d directory
    camera_folders = [
        os.path.join(joints2d_folder, d)
        for d in os.listdir(joints2d_folder)
        if os.path.isdir(os.path.join(joints2d_folder, d))
    ]

    # Collect all JSON files from the first camera folder (assuming all cameras have the same files)
    if not camera_folders:
        print("No camera folders found in joints2d directory.")
        return

    sample_files = [
        f for f in os.listdir(camera_folders[0]) if f.lower().endswith('.json')
    ]

    # Process each sample file
    for sample_file in sample_files:
        print(f"\nChecking sample: {sample_file}")

        # Initialize a dictionary to store frame counts
        frame_counts_2d: Dict[str, int] = {}

        # Check 2D keypoints files across all cameras
        for camera_folder in camera_folders:
            camera_name = os.path.basename(camera_folder)
            file_path = os.path.join(camera_folder, sample_file)
            if not os.path.exists(file_path):
                print(f"  - File '{sample_file}' not found in camera '{camera_name}'.")
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                num_frames = get_number_of_frames(data)
                frame_counts_2d[camera_name] = num_frames
                print(f"  - Camera '{camera_name}': {num_frames} frames")
            except Exception as e:
                print(f"  - Error reading '{file_path}': {e}")

        # Check if we have valid frame counts
        if not frame_counts_2d:
            print("  - No valid 2D keypoints frame counts found.")
            continue

        # Determine the minimum frame count among 2D keypoints files
        min_frames_2d = min(frame_counts_2d.values())

        # Adjust 3D filename (remove '_keypoints' and 'keypoints')
        base_filename = sample_file.replace('_keypoints', '').replace('keypoints', '')
        joints3d_file_name = base_filename
        if not joints3d_file_name.lower().endswith('.json'):
            joints3d_file_name += '.json'
        joints3d_file_path = os.path.join(joints3d_folder, joints3d_file_name)

        # Check corresponding 3D keypoints file
        num_frames_3d = None
        if os.path.exists(joints3d_file_path):
            try:
                with open(joints3d_file_path, 'r', encoding='utf-8') as f:
                    data_3d_dict = json.load(f)
                # Extract the data under 'joints3d_25'
                if 'joints3d_25' in data_3d_dict:
                    data_3d = data_3d_dict['joints3d_25']
                    num_frames_3d = get_number_of_frames(data_3d)
                    print(f"  - 3D keypoints: {num_frames_3d} frames")
                else:
                    print(f"  - Key 'joints3d_25' not found in 3D keypoints file '{joints3d_file_name}'.")
            except Exception as e:
                print(f"  - Error reading 3D keypoints file '{joints3d_file_path}': {e}")
        else:
            print(f"  - 3D keypoints file '{joints3d_file_name}' not found.")

        # Include the 3D frame count in the calculation if available
        frame_counts = list(frame_counts_2d.values())
        if num_frames_3d is not None:
            frame_counts.append(num_frames_3d)

        # Determine the minimum frame count among all files
        min_frame_count = min(frame_counts)

        # Trim 2D keypoints files if necessary
        for camera_folder in camera_folders:
            camera_name = os.path.basename(camera_folder)
            file_path = os.path.join(camera_folder, sample_file)
            if camera_name in frame_counts_2d and frame_counts_2d[camera_name] > min_frame_count:
                frames_to_trim = frame_counts_2d[camera_name] - min_frame_count
                print(f"  - Trimming first {frames_to_trim} frames from 2D keypoints file '{file_path}'.")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    trimmed_data = data[frames_to_trim:]
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(trimmed_data, f)
                    print(f"    Trimmed file '{file_path}' to {len(trimmed_data)} frames.")
                except Exception as e:
                    print(f"    Error trimming '{file_path}': {e}")

        # Trim 3D keypoints file if necessary
        if num_frames_3d is not None and num_frames_3d > min_frame_count:
            frames_to_trim = num_frames_3d - min_frame_count
            print(f"  - Trimming first {frames_to_trim} frames from 3D keypoints file '{joints3d_file_path}'.")
            try:
                data_3d_trimmed = data_3d[frames_to_trim:]
                data_3d_dict['joints3d_25'] = data_3d_trimmed
                with open(joints3d_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_3d_dict, f)
                print(f"    Trimmed 3D keypoints file '{joints3d_file_path}' to {len(data_3d_trimmed)} frames.")
            except Exception as e:
                print(f"    Error trimming '{joints3d_file_path}': {e}")

        print(f"  - All files have been trimmed to {min_frame_count} frames.")


def process_dataset_videos(dataset_root, mean2d, std2d, num_joints_2d=17):
    """
    Processes all videos in the dataset to extract keypoints using YOLOv8,
    ensures the keypoints are in the shape num_frames x 17 x 2,
    normalizes the keypoints using mean and std, and saves them inside each subject's directory under 'joints2d_normalized' as JSON files.
    Also includes the trimming procedure to align frame counts.

    Args:
        dataset_root (str): Root directory of the dataset containing subject folders.
        mean2d (list): [mean_x, mean_y] for normalization.
        std2d (list): [std_x, std_y] for normalization.
        num_joints_2d (int): Number of joints per frame.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the YOLOv8 pose model on the device
    model = YOLO('pretrained_yolos/yolov8s-pose.pt')
    model.to(device)

    mean_x, mean_y = mean2d
    std_x, std_y = std2d
    std_x = std_x if std_x != 0 else 1
    std_y = std_y if std_y != 0 else 1

    # Loop over all subjects
    subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    subjects.sort()

    for subject in subjects:
        subject_path = os.path.join(dataset_root, subject)
        videos_root = os.path.join(subject_path, 'videos')

        # Check if the videos directory exists
        if not os.path.exists(videos_root):
            print(f"No videos directory for subject {subject}")
            continue

        # Loop over cameras or video directories
        for cam in os.listdir(videos_root):
            cam_videos_path = os.path.join(videos_root, cam)
            if not os.path.isdir(cam_videos_path):
                continue

            # Loop over videos
            video_files = [f for f in os.listdir(cam_videos_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            video_files.sort()

            for video_file in video_files:
                video_path = os.path.join(cam_videos_path, video_file)
                print(f"Processing video: {video_path}")

                # Construct output directory inside the subject's 'joints2d_normalized' folder
                output_dir = os.path.join(subject_path, 'joints2d_normalized', cam)
                os.makedirs(output_dir, exist_ok=True)

                # Output name is based on the video file name without extension
                output_name = os.path.splitext(video_file)[0]

                # Initialize a list to hold keypoints for all frames
                all_keypoints = []

                # Run the model on the video with stream=True
                results = model.predict(
                    source=video_path,
                    stream=True,       # Process frames one at a time
                    device=device,     # Specify the device
                    conf=0.5,          # Confidence threshold
                    imgsz=640,         # Image size
                    verbose=False
                )

                # Loop over results_obj for each frame
                for result in tqdm(results, desc=f"Extracting keypoints from {video_file}"):
                    frame_keypoints = []

                    if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints) > 0:
                        # Assuming we are interested in the first detected person
                        kp = result.keypoints[0]
                        kp_array = kp.xy.cpu().numpy()
                        # print(f"kp_array initial shape: {kp_array.shape}")  # Debug print

                        # Ensure kp_array is 2D
                        kp_array = kp_array.reshape(-1, 2)
                        # print(f"kp_array reshaped to: {kp_array.shape}")  # Debug print

                        # Ensure keypoints are in shape (num_joints_2d, 2)
                        if kp_array.shape[0] != num_joints_2d:
                            print(f"Warning: Expected {num_joints_2d} keypoints, but got {kp_array.shape[0]}. Adjusting...")
                            if kp_array.shape[0] > num_joints_2d:
                                kp_array = kp_array[:num_joints_2d, :]
                            else:
                                # Pad with zeros if fewer keypoints
                                padding = np.zeros((num_joints_2d - kp_array.shape[0], 2))
                                kp_array = np.vstack((kp_array, padding))

                        # Normalize keypoints
                        normalized_x = (kp_array[:, 0] - mean_x) / std_x
                        normalized_y = (kp_array[:, 1] - mean_y) / std_y
                        normalized_keypoints = np.stack((normalized_x, normalized_y), axis=1)

                        # Ensure the data is in list format
                        kp_list = normalized_keypoints.tolist()
                    else:
                        # No keypoints detected, fill with zeros
                        kp_list = [[0.0, 0.0] for _ in range(num_joints_2d)]

                    # Append the keypoints for this frame
                    all_keypoints.append(kp_list)

                # Save the keypoints list to a JSON file
                keypoints_file = os.path.join(output_dir, f"{output_name}_keypoints.json")
                with open(keypoints_file, 'w') as f:
                    json.dump(all_keypoints, f)

                print(f"Keypoints saved to: {keypoints_file}")
                print(f"Finished processing video: {video_path}\n")

        # After processing all videos for the subject, perform trimming
        print(f"\nPerforming trimming for subject {subject}...")
        check_and_trim_keypoints(subject_path, joints2d_folder_name='joints2d_normalized')
        print(f"Trimming completed for subject {subject}.\n")


if __name__ == '__main__':
    # Specify your dataset root directory
    dataset_root = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train'
    num_joints_2d = 17

    # Load mean2d and std2d from the existing JSON file
    mean_std_path = os.path.join(dataset_root, 'joints2d_mean_std.json')
    if not os.path.exists(mean_std_path):
        print(f"Mean and std file not found at {mean_std_path}. Please ensure that 'joints2d_mean_std.json' exists.")
        exit()

    # Load mean2d and std2d
    with open(mean_std_path, 'r') as f:
        mean_std_data = json.load(f)
    mean2d = mean_std_data.get('mean2d', None)
    std2d = mean_std_data.get('std2d', None)

    if mean2d is None or std2d is None:
        print("Mean2d or std2d not found in the mean and std file.")
        exit()

    print("Processing dataset videos with normalization and trimming...")
    process_dataset_videos(dataset_root, mean2d, std2d, num_joints_2d=num_joints_2d)
    print("Processing complete.")