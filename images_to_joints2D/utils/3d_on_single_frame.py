import os
import cv2
import json
import numpy as np

def load_camera_parameters(camera_params_path):
    """
    Loads and processes camera parameters from a JSON file.

    Args:
        camera_params_path (str): Path to the JSON file containing camera parameters.

    Returns:
        K (np.ndarray): Intrinsic matrix of shape (3, 3).
        R (np.ndarray): Rotation matrix of shape (3, 3).
        t (np.ndarray): Translation vector of shape (3, 1).
    """
    if not os.path.isfile(camera_params_path):
        raise FileNotFoundError(f"Camera parameters file '{camera_params_path}' does not exist.")
    with open(camera_params_path, 'r') as f:
        camera_params = json.load(f)

    # Extract intrinsic parameters without distortion
    intrinsics = camera_params.get('intrinsics_wo_distortion')
    if not intrinsics:
        raise KeyError("Camera parameters JSON does not contain 'intrinsics_wo_distortion'.")

    # Extract focal lengths and principal point coordinates
    fx = intrinsics['f'][0]
    fy = intrinsics['f'][1]
    cx = intrinsics['c'][0]
    cy = intrinsics['c'][1]

    # Construct the intrinsic matrix K
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    R = np.array(camera_params['extrinsics']['R'], dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix R must be of shape (3, 3).")

    t_list = camera_params['extrinsics']['T']
    if isinstance(t_list[0], list):
        t = np.array(t_list[0], dtype=np.float64).reshape((3, 1))
    else:
        t = np.array(t_list, dtype=np.float64).reshape((3, 1))
    if t.shape != (3, 1):
        raise ValueError("Translation vector t must be of shape (3, 1).")

    return K, R, t

def extract_3d_firstframe(joints3d_path):
    if not os.path.isfile(joints3d_path):
        print(f"Joints3D JSON file '{joints3d_path}' does not exist.")
        return
    with open(joints3d_path, 'r') as f:
        joints3d_data = json.load(f)

    if 'joints3d_25' in joints3d_data:
        joints3d_list = joints3d_data['joints3d_25']
    else:
        print("Error: Key 'joints3d_25' not found in the joints3D JSON file.")
        return
    joints3d = np.array(joints3d_list, dtype=np.float64)
    first_frame = joints3d[0]

    return first_frame

def worldCoordinates_to_cameraCoordinates(keypoints,camera_params):
    R = camera_params[1]
    T = camera_params[2]
    keypoints_cc = (keypoints @ R.T) + T.T
    return keypoints_cc

def camera_to_2d(p_camera, camera_params):
    f_x = camera_params[0][0][0]
    f_y = camera_params[0][1][1]
    c_x = camera_params[0][0][2]
    c_y = camera_params[0][1][2]

    X_camera = p_camera[:, 0]
    Y_camera = p_camera[:, 1]
    Z_camera = p_camera[:, 2]

    u = (f_x * (X_camera / Z_camera)) + c_x
    print(u)
    v = (f_y * (Y_camera / Z_camera)) + c_y
    print(v)

    return u,v

if __name__ == '__main__':
    frame_path = '/public.hpc/alessandro.folloni2/smpl_study/images_to_joints2D/utils/first_frame.png'

    joints3d_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s03/joints3d_25/band_pull_apart.json'
    keypoints = extract_3d_firstframe(joints3d_path)

    camera_params_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s03/camera_parameters' \
                         '/50591643/band_pull_apart.json'
    camera_params = load_camera_parameters(camera_params_path)

    p_camera = worldCoordinates_to_cameraCoordinates(keypoints, camera_params)
    u, v = camera_to_2d(p_camera, camera_params)

    image_width = 900  # Replace with your actual image width
    image_height = 900  # Replace with your actual image height

    u_clipped = np.clip(u, 0, image_width - 1)
    v_clipped = np.clip(v, 0, image_height - 1)

    print("Clipped u values:", u_clipped)
    print("Clipped v values:", v_clipped)

    # Specify the output video file path
    output_video_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/annotated_band_pull_apart.mp4'

