import os
import json
import numpy as np
from tqdm import tqdm

def check_keypoints_dimensions(dataset_root, joints2d_folder_name='joints2d_normalized'):
    """
    Verifica che tutti i file di dati dei keypoint abbiano le dimensioni richieste.

    Args:
        dataset_root (str): Directory radice del dataset contenente le cartelle dei soggetti.
        joints2d_folder_name (str): Nome della directory contenente i keypoint 2D.

    Returns:
        None
    """
    subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    subjects.sort()
    num_errors = 0

    for subject in tqdm(subjects, desc='Verifica delle dimensioni dei keypoint per i soggetti'):
        subject_path = os.path.join(dataset_root, subject)
        joints2d_folder = os.path.join(subject_path, joints2d_folder_name)
        if not os.path.exists(joints2d_folder):
            print(f"Nessuna directory {joints2d_folder_name} per il soggetto {subject}")
            continue
        # Ottieni tutte le cartelle delle telecamere nella directory joints2d
        camera_folders = [
            d for d in os.listdir(joints2d_folder)
            if os.path.isdir(os.path.join(joints2d_folder, d))
        ]
        for camera in camera_folders:
            camera_folder = os.path.join(joints2d_folder, camera)
            # Ottieni tutti i file JSON nella cartella della telecamera
            json_files = [f for f in os.listdir(camera_folder) if f.lower().endswith('.json')]
            for json_file in json_files:
                json_path = os.path.join(camera_folder, json_file)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # data è una lista di frame
                    if not isinstance(data, list):
                        print(f"Il file {json_path} non contiene una lista di frame")
                        num_errors += 1
                        continue
                    num_frames = len(data)
                    for frame_idx, frame_keypoints in enumerate(data):
                        frame_keypoints_array = np.array(frame_keypoints)
                        if frame_keypoints_array.size == 0:
                            # Salta i frame vuoti
                            continue
                        if frame_keypoints_array.shape != (17, 2):
                            print(f"File {json_path}, frame {frame_idx} ha dimensioni {frame_keypoints_array.shape}, atteso (17, 2)")
                            num_errors += 1
                            break  # Interrompe il controllo dei frame in questo file dopo il primo errore
        if num_errors == 0:
            print(f"Soggetto {subject}: Tutti i file hanno le dimensioni corrette.")
        else:
            print(f"Soggetto {subject}: Sono stati trovati {num_errors} file con dimensioni errate.")
            num_errors = 0  # Reset per il prossimo soggetto

if __name__ == '__main__':
    dataset_root = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train'
    joints2d_folder_name = 'joints2d_normalized'

    print('Verifica delle dimensioni dei keypoint...')
    check_keypoints_dimensions(dataset_root, joints2d_folder_name=joints2d_folder_name)