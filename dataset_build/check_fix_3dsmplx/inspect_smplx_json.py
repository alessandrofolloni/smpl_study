import os
import json

def get_shape(data):
    if isinstance(data, list):
        if len(data) == 0:
            return (0,)
        else:
            # Get the shape of the first element
            first_shape = get_shape(data[0])
            # Ensure all elements have the same shape
            for item in data:
                if get_shape(item) != first_shape:
                    return (len(data),)
            return (len(data),) + first_shape
    else:
        return ()

def inspect_smplx_keys(file_path):
    """
    Inspects the keys of a single SMPLX JSON file.

    Args:
        file_path (str): Path to the SMPLX JSON file.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Check if the file has a .json extension
    if not file_path.lower().endswith('.json'):
        print(f"Error: The file '{file_path}' is not a JSON file.")
        return

    # Attempt to open and load the JSON file
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON file '{file_path}'.\n{e}")
        return
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading '{file_path}'.\n{e}")
        return

    # Print the keys and their dimensions
    print(f"SMPLX file: {file_path}")
    print("Keys and Dimensions:")
    for key in data.keys():
        shape = get_shape(data[key])
        shape_str = ' x '.join(map(str, shape)) if shape else 'Scalar'
        print(f" - {key}: {shape_str}")
    print("-" * 40)

if __name__ == "__main__":
    dataset_directory = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s10' \
                        '/smplx/warmup_7.json'
    inspect_smplx_keys(dataset_directory)