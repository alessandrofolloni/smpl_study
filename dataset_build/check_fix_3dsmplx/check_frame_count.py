import os
import json

def check_frame_counts(dataset_dir):
    """
    Checks that the number of frames in joints3d and smplx JSON files is the same for each exercise.

    Args:
        dataset_dir (str): Path to the main Dataset directory.
    """
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
        joints3d_files = [f for f in os.listdir(joints3d_folder) if f.endswith('.json')]
        smplx_files = [f for f in os.listdir(smplx_folder) if f.endswith('.json')]

        for joints3d_file in joints3d_files:
            exercise_name = os.path.splitext(joints3d_file)[0]
            joints3d_file_path = os.path.join(joints3d_folder, joints3d_file)
            smplx_file_path = os.path.join(smplx_folder, f"{exercise_name}.json")

            if not os.path.exists(smplx_file_path):
                print(f"SMPLX file not found for {exercise_name} in subject {subject}.")
                continue

            with open(joints3d_file_path, 'r') as f:
                joints3d_data = json.load(f)
            with open(smplx_file_path, 'r') as f:
                smplx_data = json.load(f)

            # Adjust the keys based on the actual structure of your JSON files
            joints3d_frames = len(joints3d_data.get('joints3d_25', []))
            smplx_frames = len(smplx_data.get('smplx', []))  # Update 'smplx' key if necessary

            if joints3d_frames != smplx_frames:
                print(f"Frame count mismatch in {exercise_name} for subject {subject}. joints3d frames: {joints3d_frames}, smplx frames: {smplx_frames}.")
            else:
                print(f"Frame counts match for {exercise_name} in subject {subject}: {joints3d_frames} frames.")

if __name__ == "__main__":
    dataset_directory = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/'  # Adjust to your actual path
    check_frame_counts(dataset_directory)