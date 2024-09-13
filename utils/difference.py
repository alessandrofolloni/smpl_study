import os
import trimesh
import numpy as np

# Percorso dei due file .obj
file1_path = "/Users/alessandrofolloni/PycharmProjects/yolo_keypoints/body_poses/smplx_mesh_2.obj"
file2_path = "/Users/alessandrofolloni/PycharmProjects/yolo_keypoints/body_poses/smplx_mesh_1.obj"

# Carica i due file .obj
mesh1 = trimesh.load(file1_path)
mesh2 = trimesh.load(file2_path)

# Assicurati che entrambe le mesh abbiano lo stesso numero di vertici
if len(mesh1.vertices) != len(mesh2.vertices):
    print("Le due mesh hanno un numero diverso di vertici.")
else:
    # Calcola la differenza tra i vertici
    vertex_diff = mesh1.vertices - mesh2.vertices

    # Calcola la distanza euclidea per ogni vertice
    distances = np.linalg.norm(vertex_diff, axis=1)

    # Output dei risultati
    print("Differenza dei vertici (prime 10):", vertex_diff[:10])
    print("Distanza euclidea (prime 10):", distances[:10])
    print("Distanza media:", np.mean(distances))
    print("Distanza massima:", np.max(distances))
    print("Distanza minima:", np.min(distances))