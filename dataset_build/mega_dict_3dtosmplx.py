import os
import json
import numpy as np

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

def load_smplx(file_path):
    """
    Loads and structures SMPLX data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing SMPLX data.

    Returns:
        dict: Structured SMPLX data {frame_key: smplx_data_for_frame}
        int: Number of frames
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Adjust the key based on the actual structure of your SMPLX files
    # For example, if the data is under 'smplx_params', update the key accordingly
    if "smplx" not in data:
        print(f"Error: 'smplx' key not found in {file_path}. Available keys: {list(data.keys())}")
        return {}, 0

    smplx_array = data["smplx"]  # Adjust if the data is under a different key

    if not isinstance(smplx_array, list):
        print(f"Error: SMPLX data in {file_path} is not a list. Type: {type(smplx_array)}")
        return {}, 0

    num_frames = len(smplx_array)
    smplx_dict = {}
    for frame_idx in range(num_frames):
        frame_key = f"frame_{frame_idx}"
        smplx_dict[frame_key] = smplx_array[frame_idx]

    return smplx_dict, num_frames

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

def create_mega_dict(dataset_dir, threshold):
    """
    Creates a mega_dict_smplx.json based on aligned frames where 3D data and SMPLX data are available.
    Includes only frames that differ significantly from the last included frame based on 3D joints.

    Args:
        dataset_dir (str): Path to the main Dataset directory.
        threshold (float): Threshold for frame difference.

    Returns:
        dict: The constructed mega_dict_smplx.
    """
    mega_dict = {}

    # List all subjects in the dataset directory
    subjects = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for subject in subjects:
        subject_path = os.path.join(dataset_dir, subject)
        joints3d_folder = os.path.join(subject_path, 'joints3d_25')
        smplx_folder = os.path.join(subject_path, 'smplx')

        if not os.path.exists(joints3d_folder):
            print(f"Warning: {joints3d_folder} does not exist. Skipping subject {subject}.")
            continue
        if not os.path.exists(smplx_folder):
            print(f"Warning: {smplx_folder} does not exist. Skipping subject {subject}.")
            continue

        # List all exercises (JSON files) for the subject
        exercises = [f for f in os.listdir(joints3d_folder) if f.endswith('.json')]

        for exercise_file in exercises:
            exercise_name = os.path.splitext(exercise_file)[0]
            exercise_key = f"{subject}_{exercise_name}"
            joints3d_file = os.path.join(joints3d_folder, exercise_file)
            smplx_file = os.path.join(smplx_folder, f"{exercise_name}.json")

            if not os.path.exists(smplx_file):
                print(f"SMPLX file not found for {exercise_name} in subject {subject}. Skipping exercise.")
                continue

            # Load 3D joints
            joints3d, num_frames_3d = load_joints3d(joints3d_file)
            if not joints3d:
                print(f"Skipping {exercise_key} due to invalid 3D joints data.")
                continue
            print(f"Loaded 3D joints for {exercise_key}: {num_frames_3d} frames")

            # Load SMPLX data
            smplx_data, num_frames_smplx = load_smplx(smplx_file)
            if not smplx_data:
                print(f"Skipping {exercise_key} due to invalid SMPLX data.")
                continue
            print(f"Loaded SMPLX data for {exercise_key}: {num_frames_smplx} frames")

            if num_frames_3d != num_frames_smplx:
                print(f"Frame count mismatch for {exercise_key}. 3D frames: {num_frames_3d}, SMPLX frames: {num_frames_smplx}. Skipping exercise.")
                continue

            # Filter frames based on difference
            filtered_joints3d = {}
            filtered_smplx = {}
            reference_joints3d = None
            frame_keys = sorted(joints3d.keys(), key=lambda x: int(x.split('_')[1]))

            # Initialize list to store differences for analysis
            all_differences = []

            for frame_idx, frame_key in enumerate(frame_keys):
                current_joints3d = joints3d[frame_key]  # List of 3D joints
                current_smplx = smplx_data[frame_key]   # SMPLX data for current frame

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
                    filtered_joints3d[frame_key] = current_joints3d
                    filtered_smplx[frame_key] = current_smplx
                    reference_joints3d = current_joints3d  # Update reference frame

            # Check if any frames are included
            if not filtered_joints3d:
                print(f"No frames passed the threshold for {exercise_key}. Skipping exercise.")
                continue

            # Initialize dictionary for this subject_video
            mega_dict[exercise_key] = {
                "joints3d": filtered_joints3d,
                "gt": filtered_smplx
            }

            # Optional: Analyze differences and print statistics
            if all_differences:
                mean_diff = np.mean(all_differences)
                std_diff = np.std(all_differences)
                print(f"Statistics for {exercise_key}: Mean difference = {mean_diff:.4f}, Std Dev = {std_diff:.4f}")

    return mega_dict

def main():
    dataset_directory = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/'
    threshold = 0.3  # Adjust this threshold based on experimentation

    # Create the mega_dict_smplx with frame filtering
    mega_dictionary = create_mega_dict(
        dataset_dir=dataset_directory,
        threshold=threshold
    )

    # Define the output path
    output_json_path = os.path.join(dataset_directory, 'mega_dict_smplx_filtered.json')  # Adjust as needed

    # Save the mega_dict_smplx as JSON
    try:
        with open(output_json_path, 'w') as json_file:
            json.dump(mega_dictionary, json_file, indent=4)
        print(f"mega_dict_smplx_filtered.json has been saved to {output_json_path}")
    except Exception as e:
        print(f"Error saving mega_dict_smplx_filtered.json: {e}")

if __name__ == "__main__":
    main()