import os
import glob

# Specify the directory containing the files to delete
output_dir = '/Users/alessandrofolloni/PycharmProjects/smpl_study/inspection/results'

# Get the list of all files in the directory
files = glob.glob(os.path.join(output_dir, '*'))

# Loop through the files and delete them
for file_path in files:
    try:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

print("All files in the 'results' folder have been deleted.")