import json
import pyrender
import torch
import trimesh
import numpy as np
import smplx

model_path = "body_models/SMPLX_NEUTRAL.npz"

json_path = "files/band_pull_apart.json"
with open(json_path, 'r') as f:
    smplx_params = json.load(f)

smplx_model = smplx.create(model_path, model_type='smplx',
                           gender='neutral',
                           use_face_contour=False,
                           ext='npz')

'''
In 3DFit dataset, after my inspection, I found out that the json has these parameters:
- transl, global_orient, body_pose, betas
- left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression

We want to understand the view of these parameters and how they influence the SMPL model
'''

for i in range(63):
    betas = torch.zeros([1, 10])  # Shape parameters
    body_pose = torch.zeros([1, 63])  # Body pose
    body_pose[0, i] = 1
    global_orient = torch.ones([1, 3])  # Global orientation
    transl = torch.zeros([1, 3])  # Translation

    print(f"VISUALIZING body_pose[0,{i}]")

    output = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl,
                         return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = smplx_model.faces

    mesh = trimesh.Trimesh(vertices, faces)

    # Salva la mesh in un file .obj
    mesh.export(f'body_poses/smplx_mesh_{i}.obj')

    scene = pyrender.Scene()
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pyrender)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)