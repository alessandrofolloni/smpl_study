import os
import json

def get_shape(data):
    """
    Recursively determines the shape of nested lists.
    """
    if isinstance(data, list):
        if not data:  # Empty list
            return 0,
        else:
            return (len(data),) + get_shape(data[0])
    else:
        return ()

def sanity_check_mega_dict(mega_dict):
    """
    Performs a sanity check on the mega_dict to verify its structure and data integrity.

    Args:
        mega_dict (dict): The mega_dict to check.

    Returns:
        None
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

    total_exercises = len(mega_dict)
    total_frames = 0
    issues_found = False

    print(f"Total exercises in mega_dict: {total_exercises}")

    for exercise_key, frames in mega_dict.items():
        if not isinstance(frames, dict):
            print(f"Error: Exercise '{exercise_key}' does not contain a dictionary of frames.")
            issues_found = True
            continue

        num_frames = len(frames)
        total_frames += num_frames
        print(f"Exercise '{exercise_key}' contains {num_frames} frames.")

        for frame_key, frame_data in frames.items():
            if not isinstance(frame_data, dict):
                print(f"Error: Frame '{frame_key}' in exercise '{exercise_key}' is not a dictionary.")
                issues_found = True
                continue

            # Check for 'joints3d' key
            if 'joints3d' not in frame_data:
                print(f"Error: Missing 'joints3d' in frame '{frame_key}' of exercise '{exercise_key}'.")
                issues_found = True
            else:
                joints3d = frame_data['joints3d']
                joints3d_shape = get_shape(joints3d)
                if joints3d_shape != (25, 3):
                    print(f"Error: 'joints3d' in frame '{frame_key}' of exercise '{exercise_key}' has shape {joints3d_shape}, expected (25, 3).")
                    issues_found = True

            # Check for 'gt' key
            if 'gt' not in frame_data:
                print(f"Error: Missing 'gt' in frame '{frame_key}' of exercise '{exercise_key}'.")
                issues_found = True
            else:
                gt_data = frame_data['gt']
                if not isinstance(gt_data, dict):
                    print(f"Error: 'gt' in frame '{frame_key}' of exercise '{exercise_key}' is not a dictionary.")
                    issues_found = True
                    continue

                # Check each SMPLX parameter
                for param_key, expected_shape in expected_shapes.items():
                    if param_key not in gt_data:
                        print(f"Error: Missing '{param_key}' in 'gt' of frame '{frame_key}' in exercise '{exercise_key}'.")
                        issues_found = True
                    else:
                        param_data = gt_data[param_key]
                        param_shape = get_shape(param_data)
                        if param_shape != expected_shape:
                            print(f"Error: '{param_key}' in 'gt' of frame '{frame_key}' in exercise '{exercise_key}' has shape {param_shape}, expected {expected_shape}.")
                            issues_found = True

    print(f"Total frames in mega_dict: {total_frames}")

    if issues_found:
        print("\nSanity check completed with errors found.")
    else:
        print("\nSanity check completed successfully. No issues found.")

def main():
    # Path to the mega_dict JSON file
    dataset_directory = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/'  # Adjust to your actual path
    mega_dict_path = os.path.join(dataset_directory, 'mega_dict_3dSMPLX.json')  # Adjust as needed

    # Load the mega_dict
    try:
        with open(mega_dict_path, 'r') as json_file:
            mega_dict = json.load(json_file)
        print(f"Loaded mega_dict from '{mega_dict_path}'")
    except Exception as e:
        print(f"Error loading mega_dict from '{mega_dict_path}': {e}")
        return

    # Perform the sanity check
    sanity_check_mega_dict(mega_dict)

if __name__ == "__main__":
    main()