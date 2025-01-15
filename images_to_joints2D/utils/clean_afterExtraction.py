import os
import json
import numpy as np
from tqdm import tqdm

def is_problematic_file(keypoints_data, num_joints_2d=17, threshold=0.5):
    """
    Determines if a keypoints file is problematic based on the proportion of frames
    with missing or zeroed keypoints.

    Args:
        keypoints_data (list): List of frames, each containing keypoints data.
        num_joints_2d (int): Number of joints expected per frame.
        threshold (float): Proportion of frames with missing keypoints to consider the file problematic.

    Returns:
        bool: True if the file is problematic, False otherwise.
    """
    total_frames = len(keypoints_data)
    if total_frames == 0:
        return True  # Empty file is considered problematic

    problematic_frames = 0

    for frame in keypoints_data:
        frame_array = np.array(frame)
        if frame_array.shape != (num_joints_2d, 2):
            # Frame with incorrect shape is considered problematic
            problematic_frames += 1
            continue
        if np.all(frame_array == 0):
            # Frame with all zero keypoints
            problematic_frames += 1

    proportion_problematic = problematic_frames / total_frames
    if proportion_problematic >= threshold:
        return True
    else:
        return False

def delete_problematic_files(dataset_root, num_joints_2d=17, threshold=0.5):
    """
    Scans the dataset's 'joints2d_normalized' directories and deletes files or folders
    that are considered problematic based on the keypoints data.

    Args:
        dataset_root (str): Root directory of the dataset containing subject folders.
        num_joints_2d (int): Number of joints expected per frame.
        threshold (float): Proportion of problematic frames to consider a file problematic.
    """
    # Loop over all subjects
    subjects = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]
    subjects.sort()

    for subject in tqdm(subjects, desc="Processing subjects"):
        subject_path = os.path.join(dataset_root, subject)
        joints2d_folder = os.path.join(subject_path, 'joints2d_normalized')

        if not os.path.exists(joints2d_folder):
            continue  # Skip subjects without 'joints2d_normalized' folder

        # Loop over camera folders
        camera_folders = [
            os.path.join(joints2d_folder, d)
            for d in os.listdir(joints2d_folder)
            if os.path.isdir(os.path.join(joints2d_folder, d))
        ]

        for camera_folder in camera_folders:
            json_files = [
                f for f in os.listdir(camera_folder)
                if f.lower().endswith('.json')
            ]

            for json_file in json_files:
                json_path = os.path.join(camera_folder, json_file)
                try:
                    with open(json_path, 'r') as f:
                        keypoints_data = json.load(f)

                    if is_problematic_file(
                        keypoints_data,
                        num_joints_2d=num_joints_2d,
                        threshold=threshold
                    ):
                        print(f"Deleting problematic file: {json_path}")
                        os.remove(json_path)
                        # Optionally, delete the entire camera folder if all files are problematic
                        # os.rmdir(camera_folder)
                except Exception as e:
                    print(f"Error processing file {json_path}: {e}")
                    continue

            # Check if the camera folder is empty after deletion
            if not os.listdir(camera_folder):
                print(f"Deleting empty camera folder: {camera_folder}")
                os.rmdir(camera_folder)

        # Check if the 'joints2d_normalized' folder is empty after deletion
        if not os.listdir(joints2d_folder):
            print(f"Deleting empty 'joints2d_normalized' folder for subject {subject}")
            os.rmdir(joints2d_folder)

def main():
    dataset_root = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train'
    num_joints_2d = 17
    threshold = 0.5  # Adjust the threshold as needed

    delete_problematic_files(
        dataset_root,
        num_joints_2d=num_joints_2d,
        threshold=threshold
    )

if __name__ == '__main__':
    main()