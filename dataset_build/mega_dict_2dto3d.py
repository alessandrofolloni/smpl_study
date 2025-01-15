import os
import json
import numpy as np


def load_joints2d(subject_path, exercise, camera_ids):
    """
    Carica e struttura i dati 2D dei keypoint dalle telecamere specificate.

    Args:
        subject_path (str): Percorso alla directory dei dati del soggetto.
        exercise (str): Nome dell'esercizio/video.
        camera_ids (list): Lista degli identificatori delle telecamere.

    Returns:
        dict: Dati strutturati dei keypoint 2D {frame_key: {camera_id: [[x, y], ...]}}
        int: Numero di frame

    Raises:
        ValueError: Se mancano dati per qualche telecamera o se i dati sono invalidi.
    """
    joints2d_folder = os.path.join(subject_path, 'joints2d_normalized')
    camera_joints = {}
    num_frames = None

    for cam_id in camera_ids:
        cam_folder = os.path.join(joints2d_folder, cam_id)
        joints2d_file = os.path.join(cam_folder, f"{exercise}_keypoints.json")

        if not os.path.exists(joints2d_file):
            print(f"Errore: {joints2d_file} non esiste. Dati della telecamera mancanti.")
            raise ValueError(f"Mancano i dati per la telecamera {cam_id} nell'esercizio {exercise}")

        with open(joints2d_file, 'r') as f:
            data_2d = json.load(f)

        # Converti i dati in un array numpy
        joints2d = np.array(data_2d)  # Forma attesa: (num_frames, 17, 2)

        if joints2d.ndim != 3 or joints2d.shape[1:] != (17, 2):
            print(f"Errore: Forma inattesa per joints2d nella telecamera {cam_id}: {joints2d.shape}.")
            raise ValueError(f"Forma dati non valida per la telecamera {cam_id} nell'esercizio {exercise}")

        if num_frames is None:
            num_frames = joints2d.shape[0]
        elif joints2d.shape[0] != num_frames:
            print(
                f"Errore: Mismatch nel numero di frame nella telecamera {cam_id}. Attesi {num_frames}, ottenuti {joints2d.shape[0]}.")
            raise ValueError(f"Mismatch nel numero di frame per la telecamera {cam_id} nell'esercizio {exercise}")

        camera_joints[cam_id] = joints2d

    # Assicuriamoci di avere dati da tutte le telecamere
    if len(camera_joints) != len(camera_ids):
        missing_cameras = set(camera_ids) - set(camera_joints.keys())
        print(f"Errore: Mancano dati dalle telecamere: {missing_cameras}")
        raise ValueError(f"Mancano dati dalle telecamere: {missing_cameras} nell'esercizio {exercise}")

    # Strutturiamo i dati joints2d per frame e per telecamera
    joints2d_per_frame = {}
    for frame_idx in range(num_frames):
        frame_key = f"frame_{frame_idx}"
        joints2d_per_frame[frame_key] = {}
        for cam_id in camera_ids:
            joints2d_per_frame[frame_key][cam_id] = camera_joints[cam_id][frame_idx].tolist()

    return joints2d_per_frame, num_frames


def load_joints3d(file_path):
    """
    Carica e struttura i dati 3D dei keypoint da un file JSON.

    Args:
        file_path (str): Percorso al file JSON contenente i dati 3D dei keypoint.

    Returns:
        dict: Dati strutturati dei keypoint 3D {frame_key: [[x, y, z], ...]}
        int: Numero di frame
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    if "joints3d_25" not in data:
        print(f"Errore: Chiave 'joints3d_25' non trovata in {file_path}.")
        return {}, 0

    joints3d_array = np.array(data["joints3d_25"])  # Forma: (num_frames, 25, 3)

    if joints3d_array.ndim != 3 or joints3d_array.shape[1:] != (25, 3):
        print(f"Errore: Forma inattesa per joints3d in {file_path}: {joints3d_array.shape}.")
        return {}, 0

    num_frames = joints3d_array.shape[0]
    joints3d = {}
    for frame_idx in range(num_frames):
        frame_key = f"frame_{frame_idx}"
        joints3d[frame_key] = joints3d_array[frame_idx].tolist()

    return joints3d, num_frames


def compute_frame_difference_3d(current_joints3d, reference_joints3d):
    """
    Calcola la distanza euclidea totale tra i frame correnti e di riferimento su tutti i keypoint 3D.

    Args:
        current_joints3d (list): Dati joints3d del frame corrente [[x, y, z], ...]
        reference_joints3d (list): Dati joints3d del frame di riferimento [[x, y, z], ...]

    Returns:
        float: Distanza euclidea totale
    """
    current_joints = np.array(current_joints3d)  # Forma: (25, 3)
    reference_joints = np.array(reference_joints3d)  # Forma: (25, 3)
    diff = np.linalg.norm(current_joints - reference_joints, axis=1)  # Forma: (25,)
    total_diff = np.sum(diff)
    return total_diff


def create_mega_dict(dataset_dir, camera_ids, threshold):
    """
    Crea un mega_dict.json basato su frame allineati dove tutti i dati delle telecamere e 3D sono disponibili.
    Include solo i frame che differiscono significativamente dall'ultimo frame incluso basandosi sui joint 3D.

    Args:
        dataset_dir (str): Percorso alla directory principale del dataset.
        camera_ids (list): Lista degli identificatori delle telecamere.
        threshold (float): Soglia per la differenza tra frame.

    Returns:
        dict: Il mega_dict costruito.
    """
    mega_dict = {}

    # Lista di tutti i soggetti nella directory del dataset
    subjects = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    subjects.sort()

    for subject in subjects:
        subject_path = os.path.join(dataset_dir, subject)
        joints3d_folder = os.path.join(subject_path, 'joints3d_25')

        if not os.path.exists(joints3d_folder):
            print(f"Attenzione: {joints3d_folder} non esiste. Salto il soggetto {subject}.")
            continue

        # Lista di tutti gli esercizi (file JSON) per il soggetto
        exercises = [f for f in os.listdir(joints3d_folder) if f.endswith('.json')]

        for exercise_file in exercises:
            exercise_name = os.path.splitext(exercise_file)[0]
            exercise_key = f"{subject}_{exercise_name}"
            joints3d_file = os.path.join(joints3d_folder, exercise_file)

            # Carica i joint 3D
            joints3d, num_frames_3d = load_joints3d(joints3d_file)
            if not joints3d:
                print(f"Salto {exercise_key} a causa di dati joints3d invalidi.")
                continue
            print(f"Caricati joints3d per {exercise_key}: {num_frames_3d} frame")

            # Carica e struttura i joint 2D da tutte le telecamere specificate
            try:
                joints2d, num_frames_2d = load_joints2d(
                    subject_path,
                    exercise_name,
                    camera_ids
                )
            except ValueError as ve:
                print(f"Salto {exercise_key} a causa di dati 2D mancanti o invalidi: {ve}")
                continue

            if num_frames_2d != num_frames_3d:
                print(
                    f"Mismatch nel numero di frame per {exercise_key}. Frame 2D: {num_frames_2d}, Frame 3D: {num_frames_3d}. Salto l'esercizio.")
                continue

            print(f"Caricati e strutturati joints2d per {exercise_key}: {num_frames_2d} frame")

            # Filtra i frame basandosi sulla differenza
            filtered_joints2d = {}
            filtered_joints3d = {}
            reference_joints3d = None
            frame_keys = sorted(joints3d.keys(), key=lambda x: int(x.split('_')[1]))

            # Contatori per i frame
            num_frames_total = len(frame_keys)
            num_frames_included = 0
            num_frames_discarded = 0

            # Lista per memorizzare le differenze per l'analisi
            all_differences = []

            for frame_key in frame_keys:
                current_joints2d = joints2d[frame_key]  # Dict di camera_id -> joints
                current_joints3d = joints3d[frame_key]  # Lista di joints 3D

                include_frame = False
                if reference_joints3d is None:
                    # Primo frame, includilo
                    include_frame = True
                else:
                    # Calcola la differenza tra i frame corrente e di riferimento
                    diff = compute_frame_difference_3d(current_joints3d, reference_joints3d)
                    all_differences.append(diff)
                    if diff >= threshold:
                        include_frame = True

                if include_frame:
                    filtered_joints2d[frame_key] = current_joints2d
                    filtered_joints3d[frame_key] = current_joints3d
                    reference_joints3d = current_joints3d  # Aggiorna il frame di riferimento
                    num_frames_included += 1
                else:
                    num_frames_discarded += 1

            # Controlla se sono stati inclusi frame
            if not filtered_joints2d:
                print(f"Nessun frame ha superato la soglia per {exercise_key}. Salto l'esercizio.")
                continue

            # Inizializza il dizionario per questo soggetto_esercizio
            mega_dict[exercise_key] = {
                "joints2d": filtered_joints2d,
                "gt": filtered_joints3d
            }

            # Stampa le statistiche dei frame
            print(f"Esercizio {exercise_key}:")
            print(f"  - Frame totali: {num_frames_total}")
            print(f"  - Frame inclusi: {num_frames_included}")
            print(f"  - Frame scartati: {num_frames_discarded}")

    return mega_dict


def main():
    dataset_directory = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/'  # Assicurati che questo percorso sia corretto
    camera_ids = ['50591643', '58860488', '60457274', '65906101']
    threshold = 0.3  # Puoi regolare questa soglia in base alle tue esigenze

    mega_dictionary = create_mega_dict(
        dataset_dir=dataset_directory,
        camera_ids=camera_ids,
        threshold=threshold
    )

    # Percorso di output
    output_json_path = os.path.join(dataset_directory,
                                    'mega_dict_2d3d.json')  # Puoi cambiare il nome se necessario

    try:
        with open(output_json_path, 'w') as json_file:
            json.dump(mega_dictionary, json_file, indent=4)
        print(f"mega_dict_2d3d.json è stato salvato in {output_json_path}")
    except Exception as e:
        print(f"Errore durante il salvataggio di mega_dict_filtered2d3d.json: {e}")


if __name__ == "__main__":
    main()