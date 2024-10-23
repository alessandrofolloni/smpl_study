import trimesh
import pyrender
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os


class SMPLX_Visualizer:
    def __init__(self, obj_file, camera_pose=None, light_intensity=2.0):
        """
        Initializes the SMPLX_Visualizer with the given OBJ file.

        Parameters:
        - obj_file (str or Path): Path to the OBJ file.
        - camera_pose (numpy.ndarray, optional): 4x4 transformation matrix for the camera.
            If None, a default pose is used.
        - light_intensity (float, optional): Intensity of the directional light.
        """
        self.obj_file = Path(obj_file)
        self.camera_pose = camera_pose if camera_pose is not None else self.default_camera_pose()
        self.light_intensity = light_intensity

        self.mesh = self.load_mesh()
        self.scene = self.setup_scene()

    def default_camera_pose(self):
        """
        Returns a default camera pose matrix.
        """
        return np.array([
            [1.0, 0.0, 0.0, 0.0],  # X-axis
            [0.0, 1.0, 0.0, -0.5],  # Y-axis (slightly below)
            [0.0, 0.0, 1.0, 4.0],  # Z-axis position (4 units forward)
            [0.0, 0.0, 0.0, 1.0]
        ])


    def load_mesh(self):
        """
        Loads the mesh from the OBJ file using Trimesh.

        Returns:
        - trimesh.Trimesh: The loaded mesh.
        """
        try:
            mesh = trimesh.load(self.obj_file)
            if not mesh.vertex_normals.any():
                mesh.compute_vertex_normals()
            return mesh
        except Exception as e:
            print(f"Error loading mesh from {self.obj_file}: {e}")
            return None

    def setup_scene(self):
        """
        Sets up the Pyrender scene with mesh, camera, and light.

        Returns:
        - pyrender.Scene: The configured scene.
        """
        scene = pyrender.Scene()

        # Add mesh
        if self.mesh is not None:
            pyrender_mesh = pyrender.Mesh.from_trimesh(self.mesh)
            scene.add(pyrender_mesh)
        else:
            print("No mesh to add to the scene.")
            return scene

        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=self.camera_pose)

        # Add light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.light_intensity)
        scene.add(light, pose=self.camera_pose)

        return scene

    def visualize(self, use_raymond_lighting=True, viewer_flags=None):
        """
        Visualizes the scene using Pyrender's viewer.

        Parameters:
        - use_raymond_lighting (bool, optional): Whether to use Pyrender's default lighting.
        - viewer_flags (dict, optional): Additional flags for the viewer.
        """
        if viewer_flags is None:
            viewer_flags = {}
        if self.scene is not None and len(self.scene.mesh_nodes) > 0:
            pyrender.Viewer(self.scene, use_raymond_lighting=use_raymond_lighting, **viewer_flags)
        else:
            print("Scene is empty. Cannot visualize.")

    def save_visualization(self, save_path, resolution=(800, 600)):
        """
        Saves a rendered image of the scene.

        Parameters:
        - save_path (str or Path): Path to save the rendered image.
        - resolution (tuple, optional): Resolution of the saved image (width, height).
        """
        if self.mesh is None:
            print("No mesh to render and save.")
            return

        r = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
        color, depth = r.render(self.scene)
        r.delete()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(save_path, color)
        print(f"Rendered image saved to {save_path}")

