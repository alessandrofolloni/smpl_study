import os
import torch
import json
from ultralytics import YOLO
from tqdm import tqdm


def process_single_video(video_path, output_dir):
    """
    Processes a single video to extract keypoints using YOLOv8,
    and saves the keypoints in the specified output directory as a JSON file.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the keypoints JSON file.
    """

    # Check if the video file exists
    if not os.path.isfile(video_path):
        print(f"Video file {video_path} does not exist.")
        return

    # Determine the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the YOLOv8 pose model on the device
    model = YOLO('yolov8n-pose.pt')
    model.to(device)

    print(f"Processing video: {video_path}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Output name is based on the video file name without extension
    output_name = os.path.splitext(os.path.basename(video_path))[0]

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
    for result in tqdm(results, desc=f"Extracting keypoints from {output_name}"):
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
    # Specify the path to your video file
    video_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s03/videos/50591643/band_pull_apart.mp4'

    # Specify the output directory where you want to save the keypoints JSON file
    output_dir = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D'

    process_single_video(video_path, output_dir)