import os
import json
import numpy as np
from tqdm import tqdm

def load_mean_std(mean_std_path):
    """
    Carica mean2d e std2d da un file JSON.

    Args:
        mean_std_path (str): Percorso al file JSON contenente mean2d e std2d.

    Returns:
        mean2d (list): [mean_x, mean_y]
        std2d (list): [std_x, std_y]
    """
    with open(mean_std_path, 'r') as f:
        data = json.load(f)
    return data['mean2d'], data['std2d']

def normalize_keypoints_dataset(dataset_root, mean2d, std2d, joints2d_folder_name='joints2d_new', output_folder_name='joints2d_normalized2', num_joints_2d=17):
    """
    Normalizza tutti i file di keypoints 2D e salva i dati normalizzati.

    Args:
        dataset_root (str): Directory radice del dataset contenente le cartelle dei soggetti.
        mean2d (list): [mean_x, mean_y] per la normalizzazione.
        std2d (list): [std_x, std_y] per la normalizzazione.
        joints2d_folder_name (str): Nome della cartella contenente i keypoints 2D originali.
        output_folder_name (str): Nome della cartella dove salvare i keypoints normalizzati.
        num_joints_2d (int): Numero di giunti 2D per frame.
    """
    mean_x, mean_y = mean2d
    std_x, std_y = std2d
    std_x = std_x if std_x != 0 else 1
    std_y = std_y if std_y != 0 else 1

    subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    subjects.sort()
    num_skipped_frames = 0
    error_files = []

    for subject in tqdm(subjects, desc='Normalizing subjects'):
        subject_path = os.path.join(dataset_root, subject)
        joints2d_folder = os.path.join(subject_path, joints2d_folder_name)
        if not os.path.exists(joints2d_folder):
            print(f"No '{joints2d_folder_name}' directory for subject {subject}")
            continue

        output_joints2d_folder = os.path.join(subject_path, output_folder_name)
        os.makedirs(output_joints2d_folder, exist_ok=True)

        # Ottieni tutte le cartelle delle telecamere nella cartella joints2d_new
        camera_folders = [
            d for d in os.listdir(joints2d_folder)
            if os.path.isdir(os.path.join(joints2d_folder, d))
        ]

        for camera in camera_folders:
            camera_folder = os.path.join(joints2d_folder, camera)
            output_camera_folder = os.path.join(output_joints2d_folder, camera)
            os.makedirs(output_camera_folder, exist_ok=True)

            # Ottieni tutti i file JSON nella cartella della telecamera
            json_files = [f for f in os.listdir(camera_folder) if f.lower().endswith('.json')]

            for json_file in json_files:
                json_path = os.path.join(camera_folder, json_file)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # Determina la struttura dei dati
                    if isinstance(data, list):
                        # Verifica se è una lista di frame o una lista piatta di keypoints
                        if len(data) == 0:
                            print(f"Empty data in file: {json_path}")
                            num_skipped_frames += 1
                            error_files.append(json_path)
                            continue

                        first_element = data[0]
                        if isinstance(first_element, list) and len(first_element) == num_joints_2d and all(isinstance(j, list) and len(j) == 2 for j in first_element):
                            # Struttura: [frame1, frame2, ...], ogni frame è [17, 2]
                            frames = data
                        elif isinstance(first_element, (int, float)):
                            # Struttura: lista piatta di keypoints, deve essere multiplo di 34
                            total_keypoints = len(data)
                            expected_keypoints_per_frame = num_joints_2d * 2
                            if total_keypoints % expected_keypoints_per_frame != 0:
                                print(f"File {json_path} has incomplete keypoints data (total_keypoints: {total_keypoints})")
                                num_skipped_frames += 1
                                error_files.append(json_path)
                                continue
                            num_frames = total_keypoints // expected_keypoints_per_frame
                            frames = [data[i * expected_keypoints_per_frame:(i + 1) * expected_keypoints_per_frame] for i in range(num_frames)]
                            # Converti ogni frame in [17, 2]
                            frames = [np.array(frame_keypoints).reshape((num_joints_2d, 2)).tolist() for frame_keypoints in frames]
                        else:
                            print(f"Unexpected data format in file: {json_path}")
                            num_skipped_frames += 1
                            error_files.append(json_path)
                            continue

                        normalized_data = []
                        for frame_idx, frame_keypoints in enumerate(frames):
                            frame_keypoints_array = np.array(frame_keypoints, dtype=np.float32)
                            if frame_keypoints_array.size == 0:
                                num_skipped_frames += 1
                                # Riempie con zeri
                                normalized_keypoints = np.zeros((num_joints_2d, 2), dtype=np.float32).flatten().tolist()
                                normalized_data.append(normalized_keypoints)
                                continue
                            if frame_keypoints_array.shape != (num_joints_2d, 2):
                                print(f"Skipping frame due to shape mismatch: {frame_keypoints_array.shape} != {(num_joints_2d, 2)} in file: {json_path}")
                                num_skipped_frames += 1
                                error_files.append(json_path)
                                # Riempie con zeri
                                normalized_keypoints = np.zeros((num_joints_2d, 2), dtype=np.float32).flatten().tolist()
                                normalized_data.append(normalized_keypoints)
                                continue
                            # Normalizza x e y separatamente
                            normalized_x = (frame_keypoints_array[:, 0] - mean_x) / std_x
                            normalized_y = (frame_keypoints_array[:, 1] - mean_y) / std_y
                            normalized_keypoints = np.stack((normalized_x, normalized_y), axis=1).flatten().tolist()
                            normalized_data.append(normalized_keypoints)

                    else:
                        print(f"Unexpected data type (not a list) in file: {json_path}")
                        num_skipped_frames += 1
                        error_files.append(json_path)
                        continue

                    # Salva i dati normalizzati
                    output_json_path = os.path.join(output_camera_folder, json_file)
                    with open(output_json_path, 'w') as f:
                        json.dump(normalized_data, f)

                except json.JSONDecodeError:
                    print(f"JSON parsing error in file: {json_path}")
                    num_skipped_frames += 1
                    error_files.append(json_path)
                except Exception as e:
                    print(f"Unexpected error in file {json_path}: {e}")
                    num_skipped_frames += 1
                    error_files.append(json_path)

def main():
    # Percorsi dei file (aggiorna questi percorsi secondo la tua configurazione)
    dataset_root = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train'  # Directory radice del dataset
    mean_std_path = os.path.join(dataset_root, 'joints2d_mean_std.json')
    joints2d_folder_name = 'joints2d_new'
    output_folder_name = 'joints2d_normalized'
    num_joints_2d = 17

    # Verifica se il file mean_std esiste
    if not os.path.exists(mean_std_path):
        print(f"Mean and std file not found at {mean_std_path}. Please run compute_global_mean_std.py first.")
        exit()

    # Carica mean2d e std2d
    with open(mean_std_path, 'r') as f:
        mean_std_data = json.load(f)
    mean2d = mean_std_data.get('mean2d', None)
    std2d = mean_std_data.get('std2d', None)

    if mean2d is None or std2d is None:
        print("Mean2d or std2d not found in the mean and std file.")
        exit()

    print("Normalizing keypoints dataset using global mean and std...")
    normalize_keypoints_dataset(
        dataset_root,
        mean2d,
        std2d,
        joints2d_folder_name=joints2d_folder_name,
        output_folder_name=output_folder_name,
        num_joints_2d=num_joints_2d
    )

    print("Normalization complete. Normalized data saved to the 'joints2d_normalized2' directories.")

    # Salva eventuali file con errori
    error_log_path = os.path.join(dataset_root, 'normalization_error_files.log')
    if os.path.exists(error_log_path):
        print(f"List of error files saved to {error_log_path}")

if __name__ == '__main__':
    main()