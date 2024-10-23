from ultralytics import YOLO
import cv2
import json
import numpy as np

# Carica i parametri della camera dal file JSON
with open('../files/camera_params/band_pull_apart.json') as f:
    camera_params = json.load(f)

intrinsics = camera_params['intrinsics_w_distortion']


def undistort_image(image, intrinsics):
    # Parametri di intrinseci
    fx, fy = intrinsics['f'][0]
    cx, cy = intrinsics['c'][0]
    k1, k2, k3 = intrinsics['k'][0]
    p1, p2 = intrinsics['p'][0]

    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.array([k1, k2, p1, p2, k3])

    # Correggi la distorsione
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    return undistorted_image


model = YOLO('../files/pretrained_yolos/YOLOv8 Pose.pt')


# Funzione per elaborare un batch di frame
def process_batch(frames, frame_indices, keypoints_list):
    results = model(frames)

    for i, result in enumerate(results):
        keypoints = result.keypoints.xy.cpu().numpy().tolist()
        keypoints_list.append({
            "frame": frame_indices[i],
            "keypoints": keypoints
        })


cap = cv2.VideoCapture('../files/video/band_pull_apart.mp4')

keypoints_list = []
batch_size = 4  # Numero di frame da processare in batch
frames = []
frame_indices = []

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    undistorted_frame = undistort_image(frame, intrinsics)
    frames.append(undistorted_frame)
    frame_indices.append(frame_count)

    if len(frames) == batch_size:
        process_batch(frames, frame_indices, keypoints_list)
        frames = []
        frame_indices = []

    frame_count += 1

# Processa eventuali frame rimanenti
if frames:
    process_batch(frames, frame_indices, keypoints_list)

cap.release()

# Salva i keypoints in un file JSON
with open('results_obj/yolo_keypoints.json', 'w') as f:
    json.dump(keypoints_list, f, indent=4)

print("Keypoints saved to yolo_keypoints.json")
with open('results_obj/yolo_keypoints.json', 'w') as f:
    json.dump(keypoints_list, f, indent=4)

print("Keypoints saved to yolo_keypoints.json")