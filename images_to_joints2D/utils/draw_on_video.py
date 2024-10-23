import os
import cv2
import json
from tqdm import tqdm

def draw_keypoints_on_video(video_path, keypoints_json_path, output_video_path, skip_keypoint_in=None):
    """
    Draws keypoints on each frame of the video and saves the annotated video.
    Allows skipping the first keypoint in the first frame, last frame, or not at all.

    Args:
        video_path (str): Path to the input video file.
        keypoints_json_path (str): Path to the JSON file containing keypoints.
        output_video_path (str): Path to save the annotated video.
        skip_keypoint_in (str, optional): Specify 'first_frame', 'last_frame', or None to skip the first keypoint accordingly.
    """

    # Check if the video file exists
    if not os.path.isfile(video_path):
        print(f"Video file {video_path} does not exist.")
        return

    # Check if the keypoints JSON file exists
    if not os.path.isfile(keypoints_json_path):
        print(f"Keypoints JSON file {keypoints_json_path} does not exist.")
        return

    # Load keypoints from JSON file
    with open(keypoints_json_path, 'r') as f:
        keypoints_data = json.load(f)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine the number of frames to process
    num_frames_to_process = min(total_frames, len(keypoints_data))
    print(total_frames)
    print(len(keypoints_data))

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    pbar = tqdm(total=num_frames_to_process, desc=f"Processing video")

    while frame_idx < num_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_idx}.")
            break

        # Get keypoints for the current frame
        frame_data = keypoints_data[frame_idx]

        frame_keypoints = []
        for person_data in frame_data:
            person_keypoints = person_data[0]  # Access the keypoints
            frame_keypoints.append(person_keypoints)

        # Determine if we need to skip the first keypoint in this frame
        skip_keypoint = False
        if skip_keypoint_in == 'first_frame' and frame_idx == 0:
            skip_keypoint = True
        elif skip_keypoint_in == 'last_frame' and frame_idx == num_frames_to_process - 1:
            skip_keypoint = True

        if skip_keypoint:
            adjusted_frame_keypoints = []
            for person_keypoints in frame_keypoints:
                adjusted_person_keypoints = person_keypoints[1:]  # Skip the first keypoint
                adjusted_frame_keypoints.append(adjusted_person_keypoints)
            frame_keypoints = adjusted_frame_keypoints

        # Draw keypoints on the frame
        for person_idx, person_keypoints in enumerate(frame_keypoints):

            # Draw keypoints
            for kp_idx, kp in enumerate(person_keypoints):
                try:
                    x, y = int(kp[0]), int(kp[1])
                except (IndexError, TypeError, ValueError) as e:
                    print(f"Error at frame {frame_idx}, person {person_idx}, keypoint {kp_idx}: {e}")
                    continue  # Skip this keypoint

                # Skip if coordinates are invalid
                if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
                    continue
                # Draw the keypoint
                cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
                # Optionally, put the keypoint index number
                cv2.putText(frame, str(kp_idx), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Define skeleton pairs
            skeleton_pairs = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (0, 5), (0, 6), (5, 7), (7, 9),
                (6, 8), (8, 10), (5, 6), (5, 11),
                (6, 12), (11, 12), (11, 13), (13, 15),
                (12, 14), (14, 16)
            ]

            # Adjust skeleton pairs if keypoints were adjusted
            if skip_keypoint:
                adjusted_skeleton_pairs = [(a - 1, b - 1) for a, b in skeleton_pairs if a - 1 >= 0 and b - 1 >= 0]
            else:
                adjusted_skeleton_pairs = skeleton_pairs

            # Draw skeleton
            for pair in adjusted_skeleton_pairs:
                part_a = pair[0]
                part_b = pair[1]
                if part_a < len(person_keypoints) and part_b < len(person_keypoints):
                    try:
                        x1, y1 = int(person_keypoints[part_a][0]), int(person_keypoints[part_a][1])
                        x2, y2 = int(person_keypoints[part_b][0]), int(person_keypoints[part_b][1])
                    except (IndexError, TypeError, ValueError) as e:
                        print(f"Error drawing skeleton at frame {frame_idx}, person {person_idx}, pair {pair}: {e}")
                        continue  # Skip this skeleton line

                    # Skip if coordinates are invalid
                    if (x1 < 0 or y1 < 0 or x1 >= frame.shape[1] or y1 >= frame.shape[0] or
                        x2 < 0 or y2 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]):
                        continue
                    cv2.line(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)

        # Write the frame to the output video
        out.write(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"Finished processing video. Annotated video is saved at {output_video_path}")


if __name__ == '__main__':
    # Specify the path to your video file
    video_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s03/videos/50591643/band_pull_apart.mp4'

    # Specify the path to the keypoints JSON file
    keypoints_json_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/s03/joints2d/50591643/band_pull_apart_keypoints.json'

    # Specify the output video file path
    output_video_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/annotated_band_pull_apart.mp4'

    # Call the function with your desired option
    # Options for skip_keypoint_in: 'first_frame', 'last_frame', or None
    draw_keypoints_on_video(video_path, keypoints_json_path, output_video_path)