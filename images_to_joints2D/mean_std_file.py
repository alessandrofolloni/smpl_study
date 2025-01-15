import os
import json
import numpy as np
from tqdm import tqdm


def compute_global_mean_std(dataset_root, joints2d_folder_name='joints2d_new', num_joints_2d=17):
    """
    Computes the global mean and standard deviation for x and y coordinates across all keypoints.

    Args:
        dataset_root (str): Root directory of the dataset containing subject folders.
        joints2d_folder_name (str): Name of the joints2d directory containing keypoints.
        num_joints_2d (int): Number of 2D joints per frame.

    Returns:
        mean2d (list): [mean_x, mean_y]
        std2d (list): [std_x, std_y]
    """
    all_x = []
    all_y = []
    num_skipped_frames = 0
    error_files = []

    subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    subjects.sort()

    for subject in tqdm(subjects, desc='Processing subjects'):
        subject_path = os.path.join(dataset_root, subject)
        joints2d_folder = os.path.join(subject_path, joints2d_folder_name)
        if not os.path.exists(joints2d_folder):
            print(f"No '{joints2d_folder_name}' directory for subject {subject}")
            continue

        # Get all camera folders in the joints2d directory
        camera_folders = [
            os.path.join(joints2d_folder, d)
            for d in os.listdir(joints2d_folder)
            if os.path.isdir(os.path.join(joints2d_folder, d))
        ]

        for camera_folder in camera_folders:
            # Get all JSON files in the camera folder
            json_files = [f for f in os.listdir(camera_folder) if f.lower().endswith('.json')]
            for json_file in json_files:
                json_path = os.path.join(camera_folder, json_file)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # Determine the structure of the data
                    if isinstance(data, list):
                        # Check if it's a list of frames or a flat list of keypoints
                        if len(data) == 0:
                            print(f"Empty data in file: {json_path}")
                            num_skipped_frames += 1
                            error_files.append(json_path)
                            continue

                        first_element = data[0]
                        if isinstance(first_element, list) and len(first_element) == num_joints_2d and all(
                                isinstance(j, list) and len(j) == 2 for j in first_element):
                            # Structure: [frame1, frame2, ...], each frame is [17, 2]
                            frames = data
                        elif isinstance(first_element, (int, float)):
                            # Structure: flat list of keypoints, must be multiple of 34
                            total_keypoints = len(data)
                            expected_keypoints_per_frame = num_joints_2d * 2
                            if total_keypoints % expected_keypoints_per_frame != 0:
                                print(
                                    f"File {json_path} has incomplete keypoints data (total_keypoints: {total_keypoints})")
                                num_skipped_frames += 1
                                error_files.append(json_path)
                                continue
                            num_frames = total_keypoints // expected_keypoints_per_frame
                            frames = [data[i * expected_keypoints_per_frame:(i + 1) * expected_keypoints_per_frame] for
                                      i in range(num_frames)]
                            # Convert each frame to [17, 2]
                            frames = [frame_keypoints.reshape((num_joints_2d, 2)).tolist() for frame_keypoints in
                                      frames]
                        else:
                            print(f"Unexpected data format in file: {json_path}")
                            num_skipped_frames += 1
                            error_files.append(json_path)
                            continue

                        for frame_idx, frame_keypoints in enumerate(frames):
                            frame_keypoints_array = np.array(frame_keypoints, dtype=np.float64)
                            if frame_keypoints_array.size == 0:
                                num_skipped_frames += 1
                                continue  # Skip empty frames
                            if frame_keypoints_array.shape != (num_joints_2d, 2):
                                print(
                                    f"Skipping frame due to shape mismatch: {frame_keypoints_array.shape} != {(num_joints_2d, 2)} in file: {json_path}")
                                num_skipped_frames += 1
                                error_files.append(json_path)
                                continue
                            # Extract x and y coordinates
                            x_coords = frame_keypoints_array[:, 0]
                            y_coords = frame_keypoints_array[:, 1]
                            all_x.extend(x_coords.tolist())
                            all_y.extend(y_coords.tolist())

                    else:
                        print(f"Unexpected data type (not a list) in file: {json_path}")
                        num_skipped_frames += 1
                        error_files.append(json_path)
                        continue

                except json.JSONDecodeError:
                    print(f"JSON parsing error in file: {json_path}")
                    num_skipped_frames += 1
                    error_files.append(json_path)
                except Exception as e:
                    print(f"Unexpected error in file {json_path}: {e}")
                    num_skipped_frames += 1
                    error_files.append(json_path)

    if num_skipped_frames > 0:
        print(f"Skipped {num_skipped_frames} frames due to errors.")
        # Save error files to a log
        error_log_path = os.path.join(dataset_root, 'error_files.log')
        with open(error_log_path, 'w') as log_file:
            for ef in error_files:
                log_file.write(f"{ef}\n")
        print(f"List of error files saved to {error_log_path}")

    if len(all_x) == 0 or len(all_y) == 0:
        print("No valid keypoints data found.")
        return None, None

    # Compute global mean and std
    mean_x = np.mean(all_x)
    mean_y = np.mean(all_y)
    std_x = np.std(all_x)
    std_y = np.std(all_y)

    mean2d = [mean_x, mean_y]
    std2d = [std_x, std_y]

    return mean2d, std2d


def main():
    # Percorsi dei file (aggiorna questi percorsi secondo la tua configurazione)
    dataset_root = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train'  # Directory radice del dataset
    joints2d_folder_name = 'joints2d_new'

    print("Computing global mean and standard deviation for x and y coordinates...")
    mean2d, std2d = compute_global_mean_std(dataset_root, joints2d_folder_name=joints2d_folder_name, num_joints_2d=17)

    if mean2d is None or std2d is None:
        print("Failed to compute mean2d and std2d due to insufficient data.")
        exit()

    # Save mean2d and std2d to a JSON file
    mean_std_dict = {'mean2d': mean2d, 'std2d': std2d}
    mean_std_path = os.path.join(dataset_root, 'joints2d_mean_std.json')
    with open(mean_std_path, 'w') as f:
        json.dump(mean_std_dict, f)
    print(f"Mean2d and standard deviation values saved to {mean_std_path}.")


if __name__ == '__main__':
    main()