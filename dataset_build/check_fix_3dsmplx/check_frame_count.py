import os
import json


def get_shape(data):
    """
    Recursively determines the shape of nested lists.

    Args:
        data (list or other): The data structure to determine the shape of.

    Returns:
        tuple: A tuple representing the shape.
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

    Args:
        smplx_data (dict): The SMPLX JSON data.
        expected_shapes (dict): A dictionary mapping keys to their expected shapes.

    Returns:
        list: A list of error messages. Empty if no errors.
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


def check_frame_counts(dataset_dir):
    """
    Checks that the number of frames in joints3d and smplx JSON files is the same
    and validates the data structures of smplx parameters.

    Args:
        dataset_dir (str): Path to the main Dataset directory.
    """
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

    # Initialize counters and lists for problematic files
    total_exercises = 0
    frame_mismatches = []
    structure_errors = []

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
        joints3d_files = [f for f in os.listdir(joints3d_folder) if f.endswith('.json')]

        for joints3d_file in joints3d_files:
            exercise_name = os.path.splitext(joints3d_file)[0]
            joints3d_file_path = os.path.join(joints3d_folder, joints3d_file)
            smplx_file_path = os.path.join(smplx_folder, f"{exercise_name}.json")

            # Check if corresponding smplx file exists
            if not os.path.exists(smplx_file_path):
                frame_mismatches.append((subject, exercise_name, "SMPLX file missing"))
                continue

            # Load JSON data
            try:
                with open(joints3d_file_path, 'r') as f:
                    joints3d_data = json.load(f)
            except json.JSONDecodeError:
                frame_mismatches.append((subject, exercise_name, "Invalid JSON in joints3d file"))
                continue
            except Exception as e:
                frame_mismatches.append((subject, exercise_name, f"Error reading joints3d file: {e}"))
                continue

            try:
                with open(smplx_file_path, 'r') as f:
                    smplx_data = json.load(f)
            except json.JSONDecodeError:
                frame_mismatches.append((subject, exercise_name, "Invalid JSON in smplx file"))
                continue
            except Exception as e:
                frame_mismatches.append((subject, exercise_name, f"Error reading smplx file: {e}"))
                continue

            # Extract frame counts
            if 'joints3d_25' not in joints3d_data:
                frame_mismatches.append((subject, exercise_name, "Missing 'joints3d_25' key"))
                continue

            joints3d_frames = len(joints3d_data['joints3d_25'])

            if 'transl' not in smplx_data:
                frame_mismatches.append((subject, exercise_name, "Missing 'transl' key in smplx file"))
                continue

            smplx_frames = len(smplx_data['transl'])

            # Check if frame counts match
            if joints3d_frames != smplx_frames:
                frame_mismatches.append((subject, exercise_name,
                                         f"Frame count mismatch (joints3d: {joints3d_frames}, smplx: {smplx_frames})"))
                continue

            # Validate SMPLX data structures
            errors = validate_smplx_structure(smplx_data, expected_shapes)
            if errors:
                error_message = "; ".join(errors)
                structure_errors.append((subject, exercise_name, error_message))
                continue

            # If all checks pass, increment the total_exercises counter
            total_exercises += 1

    # Print problematic files
    if frame_mismatches:
        print("=== Frame Count Mismatches or Missing Files ===")
        for subject, exercise, issue in frame_mismatches:
            print(f"Subject: {subject}, Exercise: {exercise}, Issue: {issue}")
    else:
        print("No frame count mismatches or missing files found.")

    if structure_errors:
        print("\n=== SMPLX Data Structure Errors ===")
        for subject, exercise, error in structure_errors:
            print(f"Subject: {subject}, Exercise: {exercise}, Error: {error}")
    else:
        print("No SMPLX data structure errors found.")

    # Summary Report
    print("\n=== Summary Report ===")
    print(f"Total exercises checked: {total_exercises}")
    print(f"Frame count mismatches or missing files: {len(frame_mismatches)}")
    print(f"SMPLX data structure errors: {len(structure_errors)}")
    print("======================")

if __name__ == "__main__":
    # Manually specify the path to your dataset directory
    dataset_directory = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/'  # Adjust to your actual path
    check_frame_counts(dataset_directory)