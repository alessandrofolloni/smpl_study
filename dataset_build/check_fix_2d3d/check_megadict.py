import os
import json

def check_mega_dict(mega_dict_path, expected_cameras, expected_joints2d=17, expected_joints3d=25):
    """
    Checks the mega_dict for missing data and reports inconsistencies.

    Args:
        mega_dict_path (str): Path to the mega_dict.json file.
        expected_cameras (list): List of expected camera identifiers.
        expected_joints2d (int): Expected number of joints in 2D data.
        expected_joints3d (int): Expected number of joints in 3D data.

    Returns:
        None
    """
    if not os.path.exists(mega_dict_path):
        print(f"Error: mega_dict.json not found at {mega_dict_path}")
        return

    with open(mega_dict_path, 'r') as f:
        mega_dict = json.load(f)

    if not mega_dict:
        print("Error: mega_dict is empty.")
        return

    total_exercises = len(mega_dict)
    exercises_checked = 0
    errors_found = False

    for exercise_key, exercise_data in mega_dict.items():
        exercises_checked += 1
        print(f"\nChecking exercise: {exercise_key}")

        joints2d = exercise_data.get('joints2d')
        gt = exercise_data.get('gt')

        if not joints2d:
            print(f"  Error: 'joints2d' data is missing.")
            errors_found = True
            continue

        if not gt:
            print(f"  Error: 'gt' data is missing.")
            errors_found = True
            continue

        frame_keys_2d = set(joints2d.keys())
        frame_keys_3d = set(gt.keys())

        # Check for missing frames
        if frame_keys_2d != frame_keys_3d:
            missing_in_2d = frame_keys_3d - frame_keys_2d
            missing_in_3d = frame_keys_2d - frame_keys_3d
            if missing_in_2d:
                print(f"  Error: Frames missing in 'joints2d': {missing_in_2d}")
            if missing_in_3d:
                print(f"  Error: Frames missing in 'gt': {missing_in_3d}")
            errors_found = True
            continue  # Skip further checks for this exercise

        # Check each frame
        for frame_key in frame_keys_2d:
            frame_joints2d = joints2d[frame_key]
            frame_gt = gt[frame_key]

            # Check for missing cameras
            cameras_in_frame = set(frame_joints2d.keys())
            missing_cameras = set(expected_cameras) - cameras_in_frame
            if missing_cameras:
                print(f"  Error in {frame_key}: Missing camera data for cameras: {missing_cameras}")
                errors_found = True
                break  # Stop checking this exercise if cameras are missing

            # Check joints in each camera
            for cam_id in expected_cameras:
                joints2d_list = frame_joints2d.get(cam_id)
                if not joints2d_list or len(joints2d_list) != expected_joints2d:
                    print(f"  Error in {frame_key}, camera {cam_id}: Expected {expected_joints2d} joints, found {len(joints2d_list) if joints2d_list else 'None'}")
                    errors_found = True
                    break  # Stop checking this exercise if joints are missing

            # Check joints in gt
            if not frame_gt or len(frame_gt) != expected_joints3d:
                print(f"  Error in {frame_key}: Expected {expected_joints3d} joints in 'gt', found {len(frame_gt) if frame_gt else 'None'}")
                errors_found = True
                break  # Stop checking this exercise if joints are missing

        if not errors_found:
            print(f"  Exercise {exercise_key} passed all checks.")

    if errors_found:
        print("\nData inconsistencies were found in the mega_dict.")
    else:
        print(f"\nAll {exercises_checked} exercises passed all checks. The mega_dict is okay.")

def main():
    mega_dict_path = '/public.hpc/alessandro.folloni2/smpl_study' \
                     '/datasets/FIT3D/train//mega_dict_filtered.json'  # Update this path accordingly
    expected_cameras = ['50591643', '58860488', '60457274', '65906101']
    expected_joints2d = 17
    expected_joints3d = 25

    check_mega_dict(
        mega_dict_path=mega_dict_path,
        expected_cameras=expected_cameras,
        expected_joints2d=expected_joints2d,
        expected_joints3d=expected_joints3d
    )

if __name__ == "__main__":
    main()