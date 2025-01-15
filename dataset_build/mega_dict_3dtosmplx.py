import os
import json
import numpy as np
from pathlib import Path

def get_shape(data):
    """
    Recursively determines the shape of nested lists.

    Parameters:
    - data (list or other): The data to determine the shape of.

    Returns:
    - tuple: The shape of the data.
    """
    if isinstance(data, list):
        if not data:  # Empty list
            return (0,)
        else:
            return (len(data),) + get_shape(data[0])
    else:
        return ()

def validate_smplx_structure(smplx_data, expected_shapes):
    """
    Validates the structure of SMPLX data against expected shapes.

    Parameters:
    - smplx_data (dict): The SMPLX data to validate.
    - expected_shapes (dict): A dictionary mapping SMPLX keys to their expected shapes.

    Returns:
    - list: A list of error messages. Empty if no errors.
    """
    errors = []
    for key, expected_shape in expected_shapes.items():
        if key not in smplx_data:
            errors.append(f"Missing key '{key}'.")
            continue

        data = smplx_data[key]
        actual_frames = len(data)
        if actual_frames == 0:
            errors.append(f"Key '{key}' has no frames.")
            continue

        first_frame = data[0]
        actual_shape = get_shape(first_frame)
        if actual_shape != expected_shape:
            errors.append(f"Key '{key}' first frame shape mismatch. Expected {expected_shape}, Got {actual_shape}")

    return errors

def load_joints3d(file_path):
    """
    Loads and structures 3D joint data from a JSON file.

    Parameters:
    - file_path (str or Path): Path to the JSON file containing 3D joint data.

    Returns:
    - dict: Structured 3D joint data {frame_key: [[x, y, z], ...]}
    - int: Number of frames
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Skipping file '{file_path}' due to JSONDecodeError: {e}")
        return {}, 0
    except Exception as e:
        print(f"Skipping file '{file_path}' due to unexpected error: {e}")
        return {}, 0

    if "joints3d_25" not in data:
        print(f"Error: 'joints3d_25' key not found in {file_path}. Skipping file.")
        return {}, 0

    joints3d_array = np.array(data["joints3d_25"])  # Expected shape: (frames, 25, 3)

    if joints3d_array.ndim != 3 or joints3d_array.shape[1:] != (25, 3):
        print(f"Error: Unexpected shape for joints3d in {file_path}: {joints3d_array.shape}. Skipping file.")
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

    Parameters:
    - current_joints3d (list): Current frame's 3D joints [[x, y, z], ...]
    - reference_joints3d (list): Reference frame's 3D joints [[x, y, z], ...]

    Returns:
    - float: Total Euclidean distance
    """
    current_joints = np.array(current_joints3d)  # Shape: (25, 3)
    reference_joints = np.array(reference_joints3d)  # Shape: (25, 3)
    diff = np.linalg.norm(current_joints - reference_joints, axis=1)  # Shape: (25,)
    total_diff = np.sum(diff)
    return total_diff

def load_smplx(file_path):
    """
    Loads SMPLX data from a JSON file.

    Parameters:
    - file_path (str or Path): Path to the SMPLX JSON file.

    Returns:
    - dict: SMPLX data.
    """
    try:
        with open(file_path, 'r') as f:
            smplx_data = json.load(f)
        return smplx_data
    except json.JSONDecodeError as e:
        print(f"Error loading SMPLX file '{file_path}': {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error loading SMPLX file '{file_path}': {e}")
        return {}

def create_mega_dict(dataset_dir, threshold=0.3):
    """
    Creates a mega_dict.json based on aligned frames where 3D joint data and SMPLX data are available.
    Filters frames based on the specified threshold difference.

    Parameters:
    - dataset_dir (str or Path): Path to the dataset directory.
    - threshold (float): Threshold for frame difference to include the frame.

    Returns:
    - dict: The constructed mega_dict.
    """
    mega_dict = {}

    # Define expected shapes for SMPLX parameters
    expected_shapes = {
        'transl': (3,),
        'global_orient': (1, 3, 3),
        'body_pose': (21, 3, 3),
        'betas': (10,),
        'left_hand_pose': (15, 3, 3),
        'right_hand_pose': (15, 3, 3),
        'jaw_pose': (1, 3, 3),
        'leye_pose': (1, 3, 3),
        'reye_pose': (1, 3, 3),
        'expression': (10,)
    }

    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"Dataset directory '{dataset_dir}' does not exist.")
        return mega_dict

    # List all subjects in the dataset directory
    subjects = [d for d in dataset_path.iterdir() if d.is_dir()]
    subjects.sort()

    for subject in subjects:
        subject_path = subject
        joints3d_folder = subject_path / 'joints3d_25'
        smplx_folder = subject_path / 'smplx'

        # Check if necessary folders exist
        if not joints3d_folder.exists():
            print(f"Warning: '{joints3d_folder}' does not exist. Skipping subject '{subject.name}'.")
            continue
        if not smplx_folder.exists():
            print(f"Warning: '{smplx_folder}' does not exist. Skipping subject '{subject.name}'.")
            continue

        # List all joints3d JSON files for the subject
        exercises = [f for f in joints3d_folder.iterdir() if f.is_file() and f.suffix == '.json']

        for exercise_file in exercises:
            exercise_name = exercise_file.stem
            exercise_key = f"{subject.name}_{exercise_name}"
            joints3d_file_path = exercise_file
            smplx_file_path = smplx_folder / f"{exercise_name}.json"

            # Check if corresponding SMPLX file exists
            if not smplx_file_path.exists():
                print(f"Warning: SMPLX file '{smplx_file_path}' does not exist. Skipping exercise '{exercise_key}'.")
                continue

            # Load joints3d data
            joints3d, num_frames_3d = load_joints3d(joints3d_file_path)
            if not joints3d:
                print(f"Skipping {exercise_key} due to invalid 3D joints data.")
                continue

            # Load SMPLX data
            smplx_data = load_smplx(smplx_file_path)
            if not smplx_data:
                print(f"Skipping {exercise_key} due to invalid SMPLX data.")
                continue

            # Validate SMPLX data structure
            errors = validate_smplx_structure(smplx_data, expected_shapes)
            if errors:
                error_message = "; ".join(errors)
                print(f"Skipping {exercise_key} due to SMPLX data structure errors: {error_message}")
                continue

            # Check frame counts
            smplx_frames = len(smplx_data['transl'])
            if num_frames_3d != smplx_frames:
                print(f"Frame count mismatch for {exercise_key}. 3D frames: {num_frames_3d}, SMPLX frames: {smplx_frames}. Skipping exercise.")
                continue

            print(f"Processing {exercise_key}: {num_frames_3d} frames")

            # Filter frames based on difference
            filtered_frames = {}
            reference_joints3d = None
            frame_keys = sorted(joints3d.keys(), key=lambda x: int(x.split('_')[1]))

            # Initialize list to store differences for analysis
            all_differences = []

            for frame_key in frame_keys:
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
                    frame_index = int(frame_key.split('_')[1])

                    # Extract SMPLX parameters for this frame
                    smplx_frame_data = {}
                    for param_key in expected_shapes.keys():
                        smplx_frame_data[param_key] = smplx_data[param_key][frame_index]

                    # Add data to filtered_frames
                    filtered_frames[frame_key] = {
                        'joints3d': current_joints3d,
                        'gt': smplx_frame_data
                    }

                    # Update reference frame
                    reference_joints3d = current_joints3d

            # Check if any frames are included
            if not filtered_frames:
                print(f"No frames passed the threshold for {exercise_key}. Skipping exercise.")
                continue

            # Add to mega_dict
            mega_dict[exercise_key] = filtered_frames

    return mega_dict

def main():
    """
    Main function to execute the mega_dict creation process.
    """
    dataset_directory = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/'  # Adjust to your actual path
    threshold = 0.3  # Adjust this threshold based on experimentation

    print("Starting mega_dict creation...")
    mega_dictionary = create_mega_dict(
        dataset_dir=dataset_directory,
        threshold=threshold
    )

    # Define output path
    output_json_path = Path(dataset_directory) / 'mega_dict_3dSMPLX_final.json'  # Adjust as needed

    try:
        with open(output_json_path, 'w') as json_file:
            json.dump(mega_dictionary, json_file, indent=4)
        print(f"mega_dict_3dSMPLX.json has been saved to {output_json_path}")
    except Exception as e:
        print(f"Error saving mega_dict_3dSMPLX.json: {e}")

if __name__ == "__main__":
    main()