import os
import torch
import json
from ultralytics import YOLO
from tqdm import tqdm


def process_dataset_videos(dataset_root):
    """
    Processes all videos in the dataset to extract keypoints using YOLOv8,
    and saves the keypoints inside each subject's directory under 'joints2d' as JSON files.

    Args:
        dataset_root (str): Root directory of the dataset containing subject folders.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the YOLOv8 pose model on the device
    model = YOLO('pretrained_yolos/yolov8s-pose.pt')
    model.to(device)

    # Loop over all subjects
    #subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    subjects = ["s05"]
    subjects.sort()

    for subject in subjects:
        subject_path = os.path.join(dataset_root, subject)
        videos_root = os.path.join(subject_path, 'videos')

        # Check if the videos directory exists
        if not os.path.exists(videos_root):
            print(f"No videos directory for subject {subject}")
            continue

        # Loop over cameras or video directories
        for cam in os.listdir(videos_root):
            cam_videos_path = os.path.join(videos_root, cam)
            if not os.path.isdir(cam_videos_path):
                continue

            # Loop over videos
            video_files = [f for f in os.listdir(cam_videos_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            video_files.sort()

            for video_file in video_files:
                video_path = os.path.join(cam_videos_path, video_file)
                print(f"Processing video: {video_path}")

                # Construct output directory inside the subject's 'joints2d' folder
                output_dir = os.path.join(subject_path, 'joints2d', cam)
                os.makedirs(output_dir, exist_ok=True)

                # Output name is based on the video file name without extension
                output_name = os.path.splitext(video_file)[0]

                # Initialize a list to hold keypoints for all frames
                all_keypoints = []

                # Run the model on the video with stream=True
                results = model.predict(
                    source=video_path,
                    stream=True,       # Process frames one at a time
                    device=device,     # Specify the device
                    conf=0.5,          # Confidence threshold
                    imgsz=640,         # Image size
                    verbose=False
                )

                # Loop over results for each frame
                for result in tqdm(results, desc=f"Extracting keypoints from {video_file}"):
                    frame_keypoints = []

                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        for kp in result.keypoints:
                            kp_array = kp.xy.cpu().numpy()  # Shape: (num_keypoints, 2)
                            kp_list = kp_array.tolist()
                            frame_keypoints.append(kp_list)
                    # Append the keypoints for this frame (empty if no detections)
                    all_keypoints.append(frame_keypoints)

                # Save the keypoints list to a JSON file
                keypoints_file = os.path.join(output_dir, f"{output_name}_keypoints.json")
                with open(keypoints_file, 'w') as f:
                    json.dump(all_keypoints, f)

                print(f"Keypoints saved to: {keypoints_file}")
                print(f"Finished processing video: {video_path}\n")


if __name__ == '__main__':
    # Specify your dataset root directory
    dataset_root = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train'

    process_dataset_videos(dataset_root)