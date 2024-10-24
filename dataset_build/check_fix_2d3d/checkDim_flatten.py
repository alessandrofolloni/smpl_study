import os
import json
from typing import Any, List, Set

def get_dimensions(data: Any) -> List[int]:
    """
    Recursively determines the dimensions of the nested list structure.

    Parameters:
        data (Any): The data to inspect.

    Returns:
        List[int]: A list representing the size at each depth level.
    """
    dimensions = []
    current_level = data
    while isinstance(current_level, list):
        dimensions.append(len(current_level))
        if len(current_level) == 0:
            break
        current_level = current_level[0]
    return dimensions

def flatten_data(data: Any, expected_dimensions: List[int]) -> Any:
    """
    Flattens the data from frames x 1 x 1 x 17 x 2 to frames x 17 x 2.

    Parameters:
        data (Any): The data to flatten.
        expected_dimensions (List[int]): The expected dimensions before flattening.

    Returns:
        Any: The flattened data.

    Raises:
        ValueError: If the data cannot be flattened to the expected dimensions.
    """
    # Check if the dimensions are as expected: frames x 1 x 1 x 17 x 2
    if len(expected_dimensions) != 5:
        raise ValueError("Data does not have 5 dimensions as expected.")
    if expected_dimensions[1:3] != [1, 1]:
        raise ValueError("Data does not have singleton dimensions in positions 2 and 3.")

    # Remove the singleton dimensions
    # Since dimensions are frames x 1 x 1 x 17 x 2, we need to remove the two singleton dimensions
    # We can recursively remove singleton lists
    def remove_singleton_dims(data):
        if isinstance(data, list) and len(data) == 1:
            return remove_singleton_dims(data[0])
        elif isinstance(data, list):
            return [remove_singleton_dims(item) for item in data]
        else:
            return data

    flattened_data = remove_singleton_dims(data)
    return flattened_data

def check_dim(json_folder: str):
    # List all JSON files in the folder
    json_files = [f for f in os.listdir(json_folder) if f.lower().endswith('.json')]
    counter = 0
    for file_name in json_files:
        json_file_path = os.path.join(json_folder, file_name)
        #print(f"Processing file: {file_name}")

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            original_dimensions = get_dimensions(data)
            dimensions_str = ' x '.join(map(str, original_dimensions))
            #print(f"Original dimensions: {dimensions_str}")

            if len(original_dimensions) == 3:
                #print("Good")
                continue
            if original_dimensions[1] != 1 or original_dimensions[2] != 1:
                print(f"Error: Unexpected dimensions for {json_folder} at {file_name}")
                counter += 1
        except Exception as e:
            print(f"An error occurred while processing '{json_folder} at {file_name}': {e}\n")

    print("EVIL COUNTER = " + str(counter))

def process_subject_joints2d(subject_folder: str):
    """
    Processes all JSON files in the joints2d folder of a subject.

    Parameters:
        subject_folder (str): The path to the 'joints2d' folder of the subject.
    """
    # Get all camera folders in the subject's joints2d directory
    camera_folders = [os.path.join(subject_folder, d) for d in os.listdir(subject_folder) if os.path.isdir(os.path.join(subject_folder, d))]

    # Initialize a set to keep track of problematic files
    problematic_files: Set[str] = set()

    # First, collect all JSON files across all camera folders
    json_files_per_camera = {}
    for camera_folder in camera_folders:
        json_files = [f for f in os.listdir(camera_folder) if f.lower().endswith('.json')]
        json_files_per_camera[camera_folder] = json_files

    # Process each camera folder
    for camera_folder, json_files in json_files_per_camera.items():
        print(f"\nProcessing camera folder: {camera_folder}")
        for file_name in json_files:
            json_file_path = os.path.join(camera_folder, file_name)
            print(f"Processing file: {file_name}")

            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Get original dimensions
                original_dimensions = get_dimensions(data)
                dimensions_str = ' x '.join(map(str, original_dimensions))
                print(f"Original dimensions: {dimensions_str}")

                # Check for unexpected dimensions (e.g., if there is a 2 instead of a 1)
                # We expect dimensions to be frames x 1 x 1 x 17 x 2
                # So we check that dimensions[1] and dimensions[2] are 1
                if len(original_dimensions) != 5:
                    print(f"Error: Unexpected number of dimensions ({len(original_dimensions)} instead of 5).")
                    # Record the problematic file
                    #problematic_files.add(file_name)
                    continue
                if original_dimensions[1] != 1 or original_dimensions[2] != 1:
                    print(f"Error: Unexpected dimensions at positions 2 and 3 (expected 1 x 1).")
                    # Record the problematic file
                    problematic_files.add(file_name)
                    continue

                # Flatten the data
                flattened_data = flatten_data(data, original_dimensions)

                # Get new dimensions
                new_dimensions = get_dimensions(flattened_data)
                new_dimensions_str = ' x '.join(map(str, new_dimensions))
                print(f"Flattened dimensions: {new_dimensions_str}")

                # Save the flattened data back to the file
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(flattened_data, f, indent=4)

                print(f"Successfully processed and flattened: {file_name}\n")

            except Exception as e:
                print(f"An error occurred while processing '{file_name}': {e}\n")
                # Record the problematic file
                problematic_files.add(file_name)

    # After processing all files, delete problematic files from all camera folders
    if problematic_files:
        print("\nDeleting problematic files from all camera folders:")
        for camera_folder in camera_folders:
            for file_name in problematic_files:
                file_path = os.path.join(camera_folder, file_name)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
                    except Exception as e:
                        print(f"Could not delete file {file_path}: {e}")
    else:
        print("\nNo problematic files to delete.")

def main(json_folder: str):
    """
    Processes all JSON files in the given folder.

    Parameters:
        json_folder (str): The path to the folder containing JSON files.
    """
    # List all JSON files in the folder
    json_files = [f for f in os.listdir(json_folder) if f.lower().endswith('.json')]

    for file_name in json_files:
        json_file_path = os.path.join(json_folder, file_name)
        print(f"Processing file: {file_name}")

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Get original dimensions
            original_dimensions = get_dimensions(data)
            dimensions_str = ' x '.join(map(str, original_dimensions))
            print(f"Original dimensions: {dimensions_str}")

            # Check for unexpected dimensions (e.g., if there is a 2 instead of a 1)
            # We expect dimensions to be frames x 1 x 1 x 17 x 2
            # So we check that dimensions[1] and dimensions[2] are 1
            if len(original_dimensions) != 5:
                print(f"Error: Unexpected number of dimensions ({len(original_dimensions)} instead of 5). Skipping file.\n")
                continue
            if original_dimensions[1] != 1 or original_dimensions[2] != 1:
                print(f"Error: Unexpected dimensions at positions 2 and 3 (expected 1 x 1). Skipping file.\n")
                continue

            # Flatten the data
            flattened_data = flatten_data(data, original_dimensions)

            # Get new dimensions
            new_dimensions = get_dimensions(flattened_data)
            new_dimensions_str = ' x '.join(map(str, new_dimensions))
            print(f"Flattened dimensions: {new_dimensions_str}")

            # Save the flattened data back to the file
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(flattened_data, f, indent=4)

            print(f"Successfully processed and flattened: {file_name}\n")

        except Exception as e:
            print(f"An error occurred while processing '{file_name}': {e}\n")

if __name__ == "__main__":
    #for s in ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]:
    #"65906101", "60457274", "58860488",
    '''for cam in ["50591643"]:
        json_folder = f"/public.hpc/alessandro.folloni2/" \
                      f"smpl_study/datasets/FIT3D/train/s05/joints2d/{cam}"
        main(json_folder)'''

    subject_joints2d_folder = "/public.hpc/alessandro.folloni2/smpl_study/" \
                              "datasets/FIT3D/train/s11/joints2d/"

    # Call the processing function
    process_subject_joints2d(subject_joints2d_folder)