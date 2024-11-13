import os
import cv2
import json
import numpy as np
from tqdm import tqdm

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
    # Check if the camera parameters file exists
    if not os.path.isfile(camera_params_path):
        raise FileNotFoundError(f"Camera parameters file '{camera_params_path}' does not exist.")

    # Load camera parameters from JSON
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

    # Extract rotation matrix R and ensure it's of correct shape
    R = np.array(camera_params['extrinsics']['R'], dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix R must be of shape (3, 3).")

    # Extract translation vector t and reshape it to (3, 1)
    t_list = camera_params['extrinsics']['T']
    if isinstance(t_list[0], list):
        t = np.array(t_list[0], dtype=np.float64).reshape((3, 1))
    else:
        t = np.array(t_list, dtype=np.float64).reshape((3, 1))
    if t.shape != (3, 1):
        raise ValueError("Translation vector t must be of shape (3, 1).")

    return K, R, t

def draw_3d_joints_on_video(video_path, joints3d_json_path, camera_params_path, output_video_path):
    """
    Projects 3D joints onto video frames using camera parameters and draws them.

    Args:
        video_path (str): Path to the input video file.
        joints3d_json_path (str): Path to the JSON file containing 3D joints data.
        camera_params_path (str): Path to the JSON file containing camera parameters.
        output_video_path (str): Path to save the annotated video.
    """

    # Check if the video file exists
    if not os.path.isfile(video_path):
        print(f"Video file '{video_path}' does not exist.")
        return

    # Check if the joints3D JSON file exists
    if not os.path.isfile(joints3d_json_path):
        print(f"Joints3D JSON file '{joints3d_json_path}' does not exist.")
        return

    # Load 3D joints data from JSON
    with open(joints3d_json_path, 'r') as f:
        joints3d_data = json.load(f)

    # Extract the joints data
    if 'joints3d_25' in joints3d_data:
        joints3d_list = joints3d_data['joints3d_25']
    else:
        print("Error: Key 'joints3d_25' not found in the joints3D JSON file.")
        return

    # Convert to NumPy array
    joints3d = np.array(joints3d_list, dtype=np.float64)  # Shape: (num_frames, num_joints, 3)
    num_frames_joints = joints3d.shape[0]
    num_joints = joints3d.shape[1]

    print(f"Loaded joints3d with shape: {joints3d.shape}")

    # Load camera parameters
    try:
        K, R, t = load_camera_parameters(camera_params_path)
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    # Get video properties
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Frame width: {frame_width}")
    print(f"Frame height: {frame_height}")

    # Adjust the principal point to the center of the frame
    cx = frame_width / 2
    cy = frame_height / 2

    # Extract focal lengths from K
    fx = K[0, 0]
    fy = K[1, 1]

    # Reconstruct the intrinsic matrix K with the adjusted principal point
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0, 1]], dtype=np.float64)

    print("Adjusted Intrinsic Matrix K:")
    print(K)

    # Invert the rotation matrix and adjust the translation vector
    R_inv = R.T
    t_inv = -R_inv @ t

    # Build the corrected extrinsic matrix [R|t]
    extrinsic_matrix = np.hstack((R_inv, t_inv))  # Shape: (3, 4)

    print("Inverted Rotation Matrix R_inv:")
    print(R_inv)
    print("Adjusted Translation Vector t_inv:")
    print(t_inv)

    # Ensure units are consistent
    # If t_inv is in millimeters and joints3d are in meters, convert t_inv to meters
    # Uncomment the following line if necessary
    # t_inv = t_inv / 1000.0

    # Determine the number of frames to process
    num_frames_to_process = min(total_frames_video, num_frames_joints)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    pbar = tqdm(total=num_frames_to_process, desc="Processing video")

    while frame_idx < num_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_idx}.")
            break

        # Get 3D joints for the current frame
        joints_world = joints3d[frame_idx]  # Shape: (num_joints, 3)

        # Invert Z-axis to align coordinate systems if necessary
        joints_world[:, 2] = -joints_world[:, 2]  # Invert Z-axis

        # Convert joints to homogeneous coordinates
        ones = np.ones((joints_world.shape[0], 1))
        joints_world_hom = np.hstack((joints_world, ones)).T  # Shape: (4, num_joints)

        # Project joints to camera coordinates
        joints_cam = extrinsic_matrix @ joints_world_hom  # Shape: (3, num_joints)

        # Only use points with positive Z-values (in front of the camera)
        z_values = joints_cam[2, :]
        valid_indices = z_values > 0
        if not np.any(valid_indices):
            print(f"Frame {frame_idx}: No joints in front of the camera.")
            frame_idx += 1
            pbar.update(1)
            continue

        joints_cam_valid = joints_cam[:, valid_indices]

        # Project onto image plane using intrinsic matrix
        x_proj = K @ joints_cam_valid  # Shape: (3, num_valid_joints)

        # Normalize homogeneous coordinates
        x_proj /= x_proj[2, :]  # Divide each coordinate by the third row

        # Get pixel coordinates
        u = x_proj[0, :]
        v = x_proj[1, :]

        # Debugging: Print the first few projected coordinates
        if frame_idx % 50 == 0:  # Print every 50 frames
            print(f"Frame {frame_idx}: Projected coordinates (u, v):")
            for i in range(min(5, u.shape[0])):  # Print up to 5 joints
                print(f"Joint {i}: (u: {u[i]}, v: {v[i]})")

        # Draw the projected joints onto the frame
        for idx, (x_i, y_i) in enumerate(zip(u, v)):
            x, y = int(round(x_i)), int(round(y_i))
            if 0 <= x < frame_width and 0 <= y < frame_height:
                cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
                # Optionally, put the joint index
                cv2.putText(frame, str(idx), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                # Debugging: Joint projected outside frame
                print(f"Frame {frame_idx}, Joint {idx} projected outside frame: (x: {x}, y: {y})")

        # Define skeleton pairs (adjust according to your joint connections)
        skeleton_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Spine
            (0, 5), (5, 6), (6, 7),               # Left arm
            (0, 8), (8, 9), (9, 10),              # Right arm
            (0, 11), (11, 12), (12, 13),          # Left leg
            (0, 14), (14, 15), (15, 16)           # Right leg
            # Add more pairs based on your joint indices and skeleton structure
        ]

        # Adjust skeleton pairs for valid_indices
        valid_indices_array = np.where(valid_indices)[0]
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices_array)}

        # Draw skeleton
        for pair in skeleton_pairs:
            idx_a_orig, idx_b_orig = pair
            if idx_a_orig in valid_indices_array and idx_b_orig in valid_indices_array:
                idx_a = index_mapping[idx_a_orig]
                idx_b = index_mapping[idx_b_orig]
                x1, y1 = int(round(u[idx_a])), int(round(v[idx_a]))
                x2, y2 = int(round(u[idx_b])), int(round(v[idx_b]))

                # Skip if coordinates are invalid
                if (0 <= x1 < frame_width and 0 <= y1 < frame_height and
                    0 <= x2 < frame_width and 0 <= y2 < frame_height):
                    cv2.line(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
                else:
                    # Debugging: Skeleton line projected outside frame
                    print(f"Frame {frame_idx}, Skeleton pair {pair} projected outside frame.")
            else:
                # One or both joints are not valid (behind camera)
                continue

        # Write the frame to the output video
        out.write(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"Finished processing video. Annotated video is saved at '{output_video_path}'.")

if __name__ == '__main__':
    # Specify the path to your video file
    video_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s03/videos/50591643/band_pull_apart.mp4'

    # Path to the 3D joints JSON file
    joints3d_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s03/joints3d_25/band_pull_apart.json'

    # Specify the path to the camera parameters file (.json format)
    camera_params_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s03/camera_parameters/50591643/band_pull_apart.json'

    # Specify the output video file path
    output_video_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/annotated_band_pull_apart.mp4'

    # Call the function to draw 3D joints on video
    draw_3d_joints_on_video(video_path, joints3d_path, camera_params_path, output_video_path)