import os
from pathlib import Path
import cv2
import shutil
import matplotlib.pyplot as plt
from fit3d_objCreation import SMPLXProcessor
from obj_visualize import SMPLX_Visualizer

root = "/Users/alessandrofolloni/PycharmProjects/smpl_study/datasets/FIT3D"
mode = "train"
subject = "s03"
exercise = "band_pull_apart.mp4"
typez = "videos"
cameras = ["50591643", "58860488", "60457274", "65906101"]

frame_idx = 0

first_camera = os.path.join(root, mode, subject, typez, cameras[0], exercise)
second_camera = os.path.join(root, mode, subject, typez, cameras[1], exercise)
third_camera = os.path.join(root, mode, subject, typez, cameras[2], exercise)
fourth_camera = os.path.join(root, mode, subject, typez, cameras[3], exercise)

video_paths = [
    Path(first_camera),
    Path(second_camera),
    Path(third_camera),
    Path(fourth_camera)
]


def extract_first_frame(video_path):
    """
    Extracts the first frame from a video file.

    Parameters:
    - video_path (str or Path): Path to the video file.

    Returns:
    - frame_rgb (numpy.ndarray): The first frame in RGB format, or None if extraction fails.
    """
    # Initialize the video capture object
    cap = cv2.VideoCapture(str(video_path))  # Ensure the path is a string

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None

    # Read the first frame
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Cannot read the first frame of the video {video_path}.")
        cap.release()
        return None

    # Release the video capture object
    cap.release()

    # Convert the frame from BGR (OpenCV's default) to RGB (for correct color display)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame_rgb


# Extract first frames
first_frames = []
for idx, vp in enumerate(video_paths):
    frame = extract_first_frame(vp)
    if frame is not None:
        first_frames.append(frame)
        print(f"Successfully extracted frame from video {idx + 1}: {vp}")
    else:
        first_frames.append(None)  # Placeholder for failed extraction
        print(f"Failed to extract frame from video {idx + 1}: {vp}")


def visualize_frames(frames, titles=None, figsize=(20, 10), save_path=None, display=True):
    """
    Displays multiple frames in a grid layout and optionally saves the figure.

    Parameters:
    - frames (list of numpy.ndarray or None): List of frames to display.
    - titles (list of str, optional): Titles for each subplot.
    - figsize (tuple, optional): Size of the figure.
    - save_path (str or Path, optional): Path to save the figure. If None, the figure is not saved.
    - display (bool, optional): Whether to display the figure. Defaults to True.
    """
    num_frames = len(frames)
    cols = 2  # Number of columns in the grid
    rows = (num_frames + 1) // cols  # Calculate the number of rows needed

    plt.figure(figsize=figsize)

    for idx, frame in enumerate(frames):
        plt.subplot(rows, cols, idx + 1)
        if frame is not None:
            plt.imshow(frame)
            plt.axis('off')
            if titles and idx < len(titles):
                plt.title(titles[idx])
        else:
            plt.text(0.5, 0.5, 'Frame Not Available', horizontalalignment='center',
                     verticalalignment='center', fontsize=12, color='red')
            plt.axis('off')
            if titles and idx < len(titles):
                plt.title(titles[idx])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if display:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory


titles = [
    cameras[0] + ": " + exercise,
    cameras[1] + ": " + exercise,
    cameras[2] + ": " + exercise,
    cameras[3] + ": " + exercise
]

save_path = "/Users/alessandrofolloni/PycharmProjects/smpl_study/smplx_usage/presentation/input.png"

visualize_frames(first_frames, titles=titles, figsize=(20, 10), save_path=save_path, display=False)

""" I saved the ideal input for band_pull_apart.mp4
    now I visualize the ideal output as parameters and as mesh
"""

# Define paths
json_file_path = ('/Users/alessandrofolloni/PycharmProjects/smpl_study/datasets/FIT3D'
                  '/train/s03/smplx/band_pull_apart.json')
model_folder = '/Users/alessandrofolloni/PycharmProjects/smpl_study/files/body_models/SMPLX_NEUTRAL.npz'
output_dir = '/Users/alessandrofolloni/PycharmProjects/smpl_study/smplx_usage/presentation'

file_path = Path(json_file_path)
out = Path(output_dir)

destination_path = out / file_path.name
shutil.copy(file_path, out)
print(f"File copied to {out}")

# Initialize the processor
processor = SMPLXProcessor(
    json_file_path=json_file_path,
    model_folder=model_folder,
    output_dir=output_dir,
    device='cpu'  # Change to 'cuda' if using GPU
)


# Process the specified frame
output_mesh_path = processor.process_frame(frame_idx)
print(f"Processed mesh saved at: {output_mesh_path}")

visualizer = SMPLX_Visualizer(obj_file=output_mesh_path)

output_dir = "/Users/alessandrofolloni/PycharmProjects/smpl_study/smplx_usage/presentation"
# Optionally, save a rendered image without displaying
save_plot_path = os.path.join(output_dir, "mesh_render" + str(frame_idx) + ".png")
visualizer.save_visualization(save_plot_path, resolution=(800, 600))
