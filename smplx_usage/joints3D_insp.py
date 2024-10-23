import os
import json
import numpy as np

# Path to your JSON file
json_file_path = '/Users/alessandrofolloni/PycharmProjects/smpl_study/datasets/FIT3D/train/s03/joints3d_25/band_pull_apart.json'


def inspect_json_file(json_file):
    # Check if the file exists
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        return

    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Print top-level keys
    print("Top-level keys in the JSON file:")
    for key in data.keys():
        print(f" - {key}")
    print()

    # Iterate over each top-level key to inspect the data
    for key, value in data.items():
        print(f"Inspecting key: '{key}'")
        print(f" - Type of value: {type(value)}")

        if isinstance(value, list):
            print(f" - Length: {len(value)}")
            if len(value) > 0:
                # Check the type of the first element
                first_elem = value[0]
                print(f" - Type of first element: {type(first_elem)}")
                if isinstance(first_elem, list):
                    # Assume it's a list of lists (e.g., 2D array)
                    print(f" - First element is a list with length: {len(first_elem)}")
                    # Optionally convert to numpy array to get shape
                    array = np.array(value)
                    print(f" - NumPy array shape: {array.shape}")
                elif isinstance(first_elem, dict):
                    print(f" - First element is a dict with keys: {list(first_elem.keys())}")
                else:
                    print(f" - Elements are of type: {type(first_elem)}")
        elif isinstance(value, dict):
            print(f" - Number of subkeys: {len(value)}")
            print(f" - Subkeys: {list(value.keys())}")
            # Optionally inspect a subkey
            first_subkey = next(iter(value))
            subvalue = value[first_subkey]
            print(f" - Type of value for subkey '{first_subkey}': {type(subvalue)}")
            if isinstance(subvalue, list):
                print(f"   - Length: {len(subvalue)}")
                if len(subvalue) > 0:
                    if isinstance(subvalue[0], list):
                        print(f"   - First element is a list with length: {len(subvalue[0])}")
                        array = np.array(subvalue)
                        print(f"   - NumPy array shape: {array.shape}")
                    else:
                        print(f"   - Elements are of type: {type(subvalue[0])}")
        else:
            print(f" - Value: {value}")
        print()


if __name__ == '__main__':
    inspect_json_file(json_file_path)