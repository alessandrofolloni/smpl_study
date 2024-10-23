import json

keypoints_json_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/' \
                      'FIT3D/train/s03/joints2d/50591643/band_pull_apart_keypoints.json'

with open(keypoints_json_path, 'r') as f:
    keypoints_data = json.load(f)

print(keypoints_data[0])
print(keypoints_data[0][0])
print(keypoints_data[0][0][0])
print(keypoints_data[0][0][0][0])