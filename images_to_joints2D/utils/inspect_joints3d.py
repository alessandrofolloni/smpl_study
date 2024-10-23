import json

# Load the 3D joints JSON file
with open('/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s03/joints3d_25/band_pull_apart.json', 'r') as f:
    joints3d_data = json.load(f)


def inspect_shape(data, indent=0):
    indentation = " " * indent
    if isinstance(data, list):
        print(f"{indentation}List with {len(data)} elements")
        if len(data) > 0:
            first_element = data[0]
            # Recursively inspect the first element of the list
            inspect_shape(first_element, indent + 4)
    elif isinstance(data, dict):
        print(f"{indentation}Dictionary with {len(data)} keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"{indentation}Key: '{key}' ->", end=" ")
            inspect_shape(value, indent + 4)
    else:
        print(f"{indentation}{type(data).__name__}: {data}")


def get_shape(data):
    """Recursively calculate the shape of the data."""
    if isinstance(data, list):
        if len(data) > 0:
            return f"{len(data)} x {get_shape(data[0])}"
        else:
            return "0"
    elif isinstance(data, dict):
        return f"{len(data)} keys"
    else:
        return str(type(data).__name__)


def detailed_inspect_with_shape(data_dict):
    for key, value in data_dict.items():
        print(f"\nInspecting key: '{key}'")
        shape = get_shape(value)
        print(f"  Shape: {shape}")
        inspect_shape(value)


# Detailed inspection of joints3d_data with shapes
detailed_inspect_with_shape(joints3d_data)