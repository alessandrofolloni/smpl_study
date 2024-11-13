import os
import json
import numpy as np

def load_joints2d(subject_path, exercise, camera_ids):
    """
    Loads and structures 2D joint data from specified cameras.

    Args:
        subject_path (str): Path to the subject's data directory.
        exercise (str): Name of the exercise/video.
        camera_ids (list): List of camera identifiers.

    Returns:
        dict: Structured 2D joint data {frame_key: {camera_id: [[x, y], ...]}}
        int: Number of frames

    Raises:
        ValueError: If any camera data is missing or invalid.
    """
    joints2d_folder = os.path.join(subject_path, 'joints2d')
    camera_joints = {}
    num_frames = None

    for cam_id in camera_ids:
        cam_folder = os.path.join(joints2d_folder, cam_id)
        joints2d_file = os.path.join(cam_folder, f"{exercise}_keypoints.json")

        if not os.path.exists(joints2d_file):
            print(f"Error: {joints2d_file} does not exist. Missing camera data.")
            raise ValueError(f"Missing data for camera {cam_id} in exercise {exercise}")

        with open(joints2d_file, 'r') as f:
            data_2d = json.load(f)

        # Convert data to numpy array
        joints2d = np.array(data_2d)  # Shape: (frames, 17, 2)

        if joints2d.ndim != 3 or joints2d.shape[1:] != (17, 2):
            print(f"Error: Unexpected shape for joints2d in camera {cam_id}: {joints2d.shape}.")
            raise ValueError(f"Invalid data shape for camera {cam_id} in exercise {exercise}")

        if num_frames is None:
            num_frames = joints2d.shape[0]
        elif joints2d.shape[0] != num_frames:
            print(f"Error: Frame count mismatch in camera {cam_id}. Expected {num_frames}, got {joints2d.shape[0]}.")
            raise ValueError(f"Frame count mismatch for camera {cam_id} in exercise {exercise}")

        camera_joints[cam_id] = joints2d

    # Ensure we have data from all cameras
    if len(camera_joints) != len(camera_ids):
        missing_cameras = set(camera_ids) - set(camera_joints.keys())
        print(f"Error: Missing data from cameras: {missing_cameras}")
        raise ValueError(f"Missing data from cameras: {missing_cameras} in exercise {exercise}")

    # Structure the joints2d data per frame and per camera
    joints2d_per_frame = {}
    for frame_idx in range(num_frames):
        frame_key = f"frame_{frame_idx}"
        joints2d_per_frame[frame_key] = {}
        for cam_id in camera_ids:
            joints2d_per_frame[frame_key][cam_id] = camera_joints[cam_id][frame_idx].tolist()

    return joints2d_per_frame, num_frames

def load_joints3d(file_path):
    """
    Loads and structures 3D joint data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing 3D joint data.

    Returns:
        dict: Structured 3D joint data {frame_key: [[x, y, z], ...]}
        int: Number of frames
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    if "joints3d_25" not in data:
        print(f"Error: 'joints3d_25' key not found in {file_path}.")
        return {}, 0

    joints3d_array = np.array(data["joints3d_25"])  # Shape: (frames, 25, 3)

    if joints3d_array.ndim != 3 or joints3d_array.shape[1:] != (25, 3):
        print(f"Error: Unexpected shape for joints3d in {file_path}: {joints3d_array.shape}.")
        return {}, 0

    num_frames = joints3d_array.shape[0]
    joints3d = {}
    for frame_idx in range(num_frames):
        frame_key = f"frame_{frame_idx}"
        joints3d[frame_key] = joints3d_array[frame_idx].tolist()

    return joints3d, num_frames

def compute_frame_difference_3d(current_joints3d, reference_joints3d):
    """
    Computes the total Euclidean distance between current and reference frames over all 3D joints.

    Args:
        current_joints3d (list): Current frame's joints3d data [[x, y, z], ...]
        reference_joints3d (list): Reference frame's joints3d data [[x, y, z], ...]

    Returns:
        float: Total Euclidean distance
    """
    current_joints = np.array(current_joints3d)  # Shape: (25, 3)
    reference_joints = np.array(reference_joints3d)  # Shape: (25, 3)
    diff = np.linalg.norm(current_joints - reference_joints, axis=1)  # Shape: (25,)
    total_diff = np.sum(diff)
    return total_diff

def create_mega_dict(dataset_dir, camera_ids, threshold):
    """
    Creates a mega_dict.json based on aligned frames where all cameras and 3D data are available.
    Includes only frames that differ significantly from the last included frame based on 3D joints.

    Args:
        dataset_dir (str): Path to the main Dataset directory.
        camera_ids (list): List of camera identifiers.
        threshold (float): Threshold for frame difference.

    Returns:
        dict: The constructed mega_dict.
    """
    mega_dict = {}

    # List all subjects in the dataset directory
    subjects = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for subject in subjects:
        subject_path = os.path.join(dataset_dir, subject)
        joints3d_folder = os.path.join(subject_path, 'joints3d_25')

        if not os.path.exists(joints3d_folder):
            print(f"Warning: {joints3d_folder} does not exist. Skipping subject {subject}.")
            continue

        # List all exercises (JSON files) for the subject
        exercises = [f for f in os.listdir(joints3d_folder) if f.endswith('.json')]

        for exercise_file in exercises:
            exercise_name = os.path.splitext(exercise_file)[0]
            exercise_key = f"{subject}_{exercise_name}"
            joints3d_file = os.path.join(joints3d_folder, exercise_file)

            # Load 3D joints
            joints3d, num_frames_3d = load_joints3d(joints3d_file)
            if not joints3d:
                print(f"Skipping {exercise_key} due to invalid 3D joints data.")
                continue
            print(f"Loaded 3D joints for {exercise_key}: {num_frames_3d} frames")

            # Load and structure 2D joints from all specified cameras
            try:
                joints2d, num_frames_2d = load_joints2d(
                    subject_path,
                    exercise_name,
                    camera_ids
                )
            except ValueError as ve:
                print(f"Skipping {exercise_key} due to missing or invalid 2D data: {ve}")
                continue

            if num_frames_2d != num_frames_3d:
                print(f"Frame count mismatch for {exercise_key}. 2D frames: {num_frames_2d}, 3D frames: {num_frames_3d}. Skipping exercise.")
                continue

            print(f"Loaded and structured 2D joints for {exercise_key}: {num_frames_2d} frames")

            # Filter frames based on difference
            filtered_joints2d = {}
            filtered_joints3d = {}
            reference_joints3d = None
            frame_keys = sorted(joints3d.keys(), key=lambda x: int(x.split('_')[1]))

            # Initialize list to store differences for analysis
            all_differences = []

            for frame_key in frame_keys:
                current_joints2d = joints2d[frame_key]  # Dict of camera_id -> joints
                current_joints3d = joints3d[frame_key]  # List of 3D joints

                include_frame = False
                if reference_joints3d is None:
                    # First frame, include it
                    include_frame = True
                else:
                    # Compute difference between current and reference frames
                    diff = compute_frame_difference_3d(current_joints3d, reference_joints3d)
                    all_differences.append(diff)
                    if diff >= threshold:
                        include_frame = True

                if include_frame:
                    filtered_joints2d[frame_key] = current_joints2d
                    filtered_joints3d[frame_key] = current_joints3d
                    reference_joints3d = current_joints3d  # Update reference frame

            # Check if any frames are included
            if not filtered_joints2d:
                print(f"No frames passed the threshold for {exercise_key}. Skipping exercise.")
                continue

            # Initialize dictionary for this subject_video
            mega_dict[exercise_key] = {
                "joints2d": filtered_joints2d,
                "gt": filtered_joints3d
            }

            # Optional: Analyze differences and print statistics
            if all_differences:
                mean_diff = np.mean(all_differences)
                std_diff = np.std(all_differences)
                print(f"Statistics for {exercise_key}: Mean difference = {mean_diff:.4f}, Std Dev = {std_diff:.4f}")

    return mega_dict

def main():
    dataset_directory = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/'  # Adjust to your actual path
    camera_ids = ['50591643', '58860488', '60457274', '65906101']
    threshold = 0.3  # Adjust this threshold based on experimentation

    mega_dictionary = create_mega_dict(
        dataset_dir=dataset_directory,
        camera_ids=camera_ids,
        threshold=threshold
    )

    # output path
    output_json_path = os.path.join(dataset_directory, 'mega_dict_filtered.json')  # Adjust as needed

    try:
        with open(output_json_path, 'w') as json_file:
            json.dump(mega_dictionary, json_file, indent=4)
        print(f"mega_dict_filtered_3d.json has been saved to {output_json_path}")
    except Exception as e:
        print(f"Error saving mega_dict_filtered_3d.json: {e}")

if __name__ == "__main__":
    main()