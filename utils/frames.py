import os
import time

import trimesh
import pyrender

directory_path = "/Users/alessandrofolloni/PycharmProjects/yolo_keypoints/body_poses"
files = [f for f in os.listdir(directory_path) if f.endswith('.obj')]

for file in files:
    file_path = os.path.join(directory_path, file)
    mesh = trimesh.load(file_path)

    scene = pyrender.Scene()
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pyrender)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

    time.sleep(0)