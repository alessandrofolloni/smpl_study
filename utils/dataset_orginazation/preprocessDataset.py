import os
import cv2
import json


def extract_smplx_frame_data(smplx_data, frame_idx):
    """
    Extracts the SMPL-X parameters for a specific frame from the multi-frame data.

    Args:
        smplx_data (dict): The loaded SMPL-X data containing parameters for all frames.
        frame_idx (int): The index of the frame to extract.

    Returns:
        dict: A dictionary containing the SMPL-X parameters for the specified frame.
    """
    # Initialize a dictionary to hold the frame data
    frame_data = {}

    # Iterate over each parameter in the data
    for key, value in smplx_data.items():
        # Check if the value is a list and has enough elements
        if isinstance(value, list) and len(value) > frame_idx:
            # Take the data for the specified frame
            frame_data[key] = value[frame_idx]
        else:
            # For parameters that are not lists, include them as is
            frame_data[key] = value

    return frame_data


def process_dataset(dataset_root, output_root):
    """
    Processes the dataset by extracting frames and organizing them into the desired structure.

    Args:
        dataset_root (str): Path to the root directory of the dataset.
        output_root (str): Path to the root directory where the processed data will be saved.
    """
    # List all subjects in the dataset
    subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    #subjects.sort()  # Optional: sort for consistency

    for subject in subjects:
        subject_path = os.path.join(dataset_root, subject)
        print(f"Processing subject: {subject}")

        # Paths to camera parameters, SMPL-X data, and videos
        camera_params_root = os.path.join(subject_path, 'camera_parameters')
        smplx_root = os.path.join(subject_path, 'smplx')
        videos_root = os.path.join(subject_path, 'videos')

        # Check if required directories exist
        if not os.path.exists(camera_params_root) or not os.path.exists(smplx_root) or not os.path.exists(videos_root):
            print(f"Missing required directories for subject {subject}")
            continue

        # List exercises based on SMPL-X data
        exercises = [f[:-5] for f in os.listdir(smplx_root) if f.endswith('.json')]
        exercises.sort()

        for exercise in exercises:
            print(f"  Processing exercise: {exercise}")
            # Output folder: /output_root/subject_exercise
            output_folder = os.path.join(output_root, f"{subject}_{exercise}")
            os.makedirs(output_folder, exist_ok=True)

            # Load SMPL-X ground truth data for this exercise
            smplx_gt_path = os.path.join(smplx_root, f"{exercise}.json")
            if not os.path.exists(smplx_gt_path):
                print(f"    SMPL-X ground truth not found for {subject} {exercise}")
                continue

            with open(smplx_gt_path, 'r') as f:
                smplx_data = json.load(f)

            # Initialize video captures for each camera
            caps = {}
            total_frames = None
            cameras = ["50591643", "58860488", "60457274", "65906101"]
            for cam_num in cameras:
                # Video file path for this camera and exercise
                video_path = os.path.join(videos_root, cam_num, f"{exercise}.mp4")
                if not os.path.exists(video_path):
                    print(f"    Video not found: {video_path}")
                    continue
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"    Cannot open video: {video_path}")
                    continue
                caps[cam_num] = cap
                # Ensure all videos have the same number of frames
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames is None:
                    total_frames = frame_count
                else:
                    total_frames = min(total_frames, frame_count)

            if len(caps) < 4:
                print(f"    Not all cameras available for {subject} {exercise}")
                # Release any opened video captures
                for cap in caps.values():
                    cap.release()
                continue

            # Process frames
            for frame_idx in range(total_frames):
                # Read frames from all cameras
                frames = {}
                for cam_num, cap in caps.items():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"    End of video reached for camera {cam_num} at frame {frame_idx}")
                        break
                    frames[cam_num] = frame

                if len(frames) < 4:
                    print(f"    Missing frames at index {frame_idx} for {subject} {exercise}")
                    break

                # Save images from all cameras
                frame_folder = os.path.join(output_folder, f"frame_{frame_idx:05d}")
                os.makedirs(frame_folder, exist_ok=True)
                for cam_num, frame in frames.items():
                    # Construct filename
                    image_filename = f"view_{cam_num}.jpg"
                    image_path = os.path.join(frame_folder, image_filename)
                    # Save the image
                    cv2.imwrite(image_path, frame)

                # Save SMPL-X data for this frame
                # Extract SMPL-X data for this frame using the new function
                smplx_frame_data = extract_smplx_frame_data(smplx_data, frame_idx)
                if not smplx_frame_data:
                    print(f"    SMPL-X data missing for frame {frame_idx} in {subject} {exercise}")
                    continue

                # Save the SMPL-X parameters for this frame
                smplx_filename = os.path.join(frame_folder, 'smplx_params.json')
                with open(smplx_filename, 'w') as f:
                    json.dump(smplx_frame_data, f, indent=4)

                if frame_idx % 100 == 0:
                    print(f"    Processed frame {frame_idx}/{total_frames} for {subject} {exercise}")

            # Release video captures
            for cap in caps.values():
                cap.release()

            print(f"  Finished processing {subject} {exercise}")


if __name__ == "__main__":
    # Path to the dataset root directory
    dataset_root = '/Users/alessandrofolloni/PycharmProjects/smpl_study/datasets/FIT3D/train'

    # Path to the output root directory
    output_root = '/Users/alessandrofolloni/Google Drive/My Drive/FIT3D_post'

    process_dataset(dataset_root, output_root)