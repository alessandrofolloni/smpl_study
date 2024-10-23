import os
import glob

'''Specify the directory containing the files to delete'''

output_dir = '/Users/alessandrofolloni/PycharmProjects/smpl_study/datasets/post_FIT3D'

files = glob.glob(os.path.join(output_dir, '*'))

for file_path in files:
    try:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

print("All files in the 'results_obj' folder have been deleted.")