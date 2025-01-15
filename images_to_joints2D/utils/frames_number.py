import cv2
import os

def get_video_frame_count(video_path):
    """
    Prints the number of frames in the given video file.

    Args:
        video_path (str): Path to the video file.
    """
    # Check if the video file exists
    if not os.path.isfile(video_path):
        print(f"Video file '{video_path}' does not exist.")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file '{video_path}'.")
        return

    # Get the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Number of frames in '{video_path}': {frame_count}")

    # Release the video capture object
    cap.release()

if __name__ == '__main__':
    video_file = "/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s03/videos/50591643/band_pull_apart.mp4"
    get_video_frame_count(video_file)