import os
import json

def inspect_smplx_keys(dataset_dir):
    """
    Inspects the keys of the SMPLX JSON files in the dataset.

    Args:
        dataset_dir (str): Path to the main Dataset directory.
    """
    # List all subjects in the dataset directory
    subjects = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for subject in subjects:
        subject_path = os.path.join(dataset_dir, subject)
        smplx_folder = os.path.join(subject_path, 'smplx')

        if not os.path.exists(smplx_folder):
            print(f"Warning: {smplx_folder} does not exist. Skipping subject {subject}.")
            continue

        # List all exercises (JSON files) for the subject
        exercises = [f for f in os.listdir(smplx_folder) if f.endswith('.json')]

        for smplx_file in exercises:
            smplx_file_path = os.path.join(smplx_folder, smplx_file)
            with open(smplx_file_path, 'r') as f:
                data = json.load(f)
            print(f"SMPLX file: {smplx_file_path}")
            print(f"Keys: {list(data.keys())}")
            print("-" * 40)

if __name__ == "__main__":
    dataset_directory = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s10' \
                        '/smplx/warmup_7.json'  # Adjust to your actual path
    inspect_smplx_keys(dataset_directory)