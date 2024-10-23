import os
import json
from typing import Any, List, Dict


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


def check_and_trim_keypoints(subject_folder: str):
    """
    Checks the consistency of frame counts between 2D keypoints files across all cameras and the 3D keypoints file.
    Automatically trims the first n frames to align the frame counts without user input.

    Parameters:
        subject_folder (str): The path to the subject's folder containing 'joints2d' and 'joints3d_25' folders.
    """
    # Paths to joints2d and joints3d_25 folders
    joints2d_folder = os.path.join(subject_folder, 'joints2d')
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
                        json.dump(trimmed_data, f, indent=4)
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
                    json.dump(data_3d_dict, f, indent=4)
                print(f"    Trimmed 3D keypoints file '{joints3d_file_path}' to {len(data_3d_trimmed)} frames.")
            except Exception as e:
                print(f"    Error trimming '{joints3d_file_path}': {e}")

        print(f"  - All files have been trimmed to {min_frame_count} frames.")


def main():
    for s in ["s04", "s05", "s07", "s08", "s09", "s10", "s11"]:
        subject_folder = f"/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/{s}"
        check_and_trim_keypoints(subject_folder)


if __name__ == "__main__":
    main()