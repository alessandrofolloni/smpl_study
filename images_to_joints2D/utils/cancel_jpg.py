import os
import glob

def delete_jpg_files_in_folder(folder_path):
    """
    Deletes all .jpg files in the specified folder.

    Args:
        folder_path (str): Path to the folder.
    """

    # Construct the file pattern
    jpg_files = glob.glob(os.path.join(folder_path, '*.jpg'))

    # Delete each file
    for file_path in jpg_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    print(f"Finished deleting .jpg files in {folder_path}")

if __name__ == '__main__':
    # Specify the folder path
    folder_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train'

    delete_jpg_files_in_folder(folder_path)