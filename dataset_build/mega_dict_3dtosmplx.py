import os
import json
import numpy as np

def get_shape(data):
    """
    Recursively determines the shape of nested lists.
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

    joints3d_array = np.array(data["joints3d_25"])  # Shape: (frames, 25, 3)

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
    """
    current_joints = np.array(current_joints3d)  # Shape: (25, 3)
    reference_joints = np.array(reference_joints3d)  # Shape: (25, 3)
    diff = np.linalg.norm(current_joints - reference_joints, axis=1)  # Shape: (25,)
    total_diff = np.sum(diff)
    return total_diff

def create_mega_dict(dataset_dir, threshold):
    """
    Creates a mega_dict.json based on aligned frames where 3D data and SMPLX data are available.
    Excludes files with errors and handles exceptions gracefully.
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

    # List all subjects in the dataset directory
    subjects = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for subject in subjects:
        subject_path = os.path.join(dataset_dir, subject)
        joints3d_folder = os.path.join(subject_path, 'joints3d_25')
        smplx_folder = os.path.join(subject_path, 'smplx')

        # Check if necessary folders exist
        if not os.path.exists(joints3d_folder):
            print(f"Warning: '{joints3d_folder}' does not exist. Skipping subject '{subject}'.")
            continue
        if not os.path.exists(smplx_folder):
            print(f"Warning: '{smplx_folder}' does not exist. Skipping subject '{subject}'.")
            continue

        # List all joints3d JSON files for the subject
        exercises = [f for f in os.listdir(joints3d_folder) if f.endswith('.json')]

        for exercise_file in exercises:
            exercise_name = os.path.splitext(exercise_file)[0]
            exercise_key = f"{subject}_{exercise_name}"
            joints3d_file_path = os.path.join(joints3d_folder, exercise_file)
            smplx_file_path = os.path.join(smplx_folder, f"{exercise_name}.json")

            # Check if corresponding SMPLX file exists
            if not os.path.exists(smplx_file_path):
                print(f"Warning: SMPLX file '{smplx_file_path}' does not exist. Skipping exercise '{exercise_key}'.")
                continue

            # Load joints3d data
            joints3d, num_frames_3d = load_joints3d(joints3d_file_path)
            if not joints3d:
                print(f"Skipping {exercise_key} due to invalid 3D joints data.")
                continue

            # Load SMPLX data
            try:
                with open(smplx_file_path, 'r') as f:
                    smplx_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Skipping {exercise_key} due to JSONDecodeError in SMPLX file: {e}")
                continue
            except Exception as e:
                print(f"Skipping {exercise_key} due to unexpected error when loading SMPLX file: {e}")
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
                    # Collect SMPLX parameters for this frame
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

                    reference_joints3d = current_joints3d  # Update reference frame

            # Check if any frames are included
            if not filtered_frames:
                print(f"No frames passed the threshold for {exercise_key}. Skipping exercise.")
                continue

            # Add to mega_dict
            mega_dict[exercise_key] = filtered_frames

            # Optional: Analyze differences and print statistics
            if all_differences:
                mean_diff = np.mean(all_differences)
                std_diff = np.std(all_differences)
                print(f"Statistics for {exercise_key}: Mean difference = {mean_diff:.4f}, Std Dev = {std_diff:.4f}")

    return mega_dict

def main():
    dataset_directory = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/'  # Adjust to your actual path
    threshold = 0.3  # Adjust this threshold based on experimentation

    mega_dictionary = create_mega_dict(
        dataset_dir=dataset_directory,
        threshold=threshold
    )

    # Output path
    output_json_path = os.path.join(dataset_directory, 'mega_dict_3dSMPLX.json')  # Adjust as needed

    try:
        with open(output_json_path, 'w') as json_file:
            json.dump(mega_dictionary, json_file, indent=4)
        print(f"mega_dict_filtered.json has been saved to {output_json_path}")
    except Exception as e:
        print(f"Error saving mega_dict_filtered.json: {e}")

if __name__ == "__main__":
    main()