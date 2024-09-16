import trimesh
import pyrender
import numpy as np

# Path to your OBJ file
obj_file = 'results/output_frame0000.obj'

# Load the mesh using Trimesh
trimesh_mesh = trimesh.load(obj_file)

# Ensure the mesh has vertex normals for proper lighting
if not trimesh_mesh.vertex_normals.any():
    trimesh_mesh.compute_vertex_normals()

# Convert the Trimesh object to a Pyrender mesh
pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)

# Create a Pyrender scene
scene = pyrender.Scene()

# Add the mesh to the scene
scene.add(pyrender_mesh)

# Set up the camera
# You can adjust the camera parameters for better visualization
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
# Position the camera in the scene
# Adjusted camera position to zoom out
camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],    # X-axis position
    [0.0, 1.0, 0.0, -0.5],   # Y-axis position
    [0.0, 0.0, 1.0, 4.0],    # Z-axis position (increase to zoom out)
    [0.0, 0.0, 0.0, 1.0]
])
scene.add(camera, pose=camera_pose)

# Add lights to the scene
# You can adjust the intensity and position of the lights
light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
scene.add(light, pose=camera_pose)

# Render the scene using the Pyrender viewer
pyrender.Viewer(scene, use_raymond_lighting=True)