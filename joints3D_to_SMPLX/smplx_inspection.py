import json
import numpy as np

# Load the SMPL-X json file
with open('/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s05/smplx/band_pull_apart.json', 'r') as f:
    data = json.load(f)

# Print keys and shapes
for key in data.keys():
    arr = np.array(data[key])
    print(f"{key}: shape={arr.shape}")