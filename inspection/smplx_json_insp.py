import json
import numpy as np
import os

# Path to your JSON file (adjust this to your actual path)
json_file_path = ('/Users/alessandrofolloni/PycharmProjects/smpl_study/datasets/'
                  'FIT3D/train/s03/smplx/band_pull_apart.json')


# Function to load and inspect the JSON file
def inspect_json_file(json_file):
    # Check if the file exists
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        return

    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Print top-level keys
    print("Top-level keys in the JSON file:")
    for key in data.keys():
        print(f" - {key}")
    print()

    # Check the number of frames
    num_frames = None
    for key, value in data.items():
        if isinstance(value, list):
            num_frames = len(value)
            print(f"Parameter '{key}' has {num_frames} frames.")
        else:
            print(f"Parameter '{key}' is not a list (type: {type(value)}).")
    print()

    if num_frames is None:
        print("No parameter contains frame data.")
        return

    # Verify that all parameters have the same number of frames
    inconsistent_params = []
    for key, value in data.items():
        if isinstance(value, list) and len(value) != num_frames:
            inconsistent_params.append(key)
    if inconsistent_params:
        print("The following parameters have inconsistent frame counts:")
        for key in inconsistent_params:
            print(f" - {key}: {len(data[key])} frames")
    else:
        print("All parameters have consistent frame counts.")
    print()

    # Display the shape of each parameter for the first frame
    print("Data structure for each parameter (first frame):")
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 0:
            frame_data = value[0]
            if isinstance(frame_data, list):
                array_data = np.array(frame_data)
                print(f" - {key}: shape {array_data.shape}")
            else:
                print(f" - {key}: type {type(frame_data)}")
        else:
            print(f" - {key}: not a list or empty")
    print()

    # Display sample data from the first few frames
    num_samples = min(3, num_frames)
    print(f"Sample data from the first {num_samples} frames:")
    for key, value in data.items():
        if isinstance(value, list) and len(value) >= num_samples:
            print(f"\nParameter '{key}':")
            for i in range(num_samples):
                frame_data = value[i]
                print(f" Frame {i}: {frame_data}")
        else:
            print(f"\nParameter '{key}': No data to display.")
    print()


# Call the function to inspect your JSON file
inspect_json_file(json_file_path)