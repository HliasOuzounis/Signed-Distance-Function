import open3d as o3d
from open3d.visualization import Visualizer, VisualizerWithKeyCallback
import numpy as np

from .callback import Callback
from . import utility


class MeshConstructor(Callback):
    def __init__(self, vis: Visualizer | VisualizerWithKeyCallback, mesh, *args, **kwargs) -> None:
        super().__init__(vis, *args, **kwargs)
        self.mesh = utility.fit_to_unit_sphere(mesh)
        self.mesh.vertex_colors = utility.assign_colors(mesh)

        self.shuffle = True # shuffle the triangles

        self.mesh_triangles = np.asarray(self.mesh.triangles)
        if not self.mesh.has_triangle_normals():
            self.mesh.compute_triangle_normals()
        self.mesh_normals = np.asarray(self.mesh.triangle_normals)
        self.step = self.mesh_triangles.shape[0] // 100

        self.animate_init()

        self.vis.add_geometry(self.mesh)

    def animate_init(self) -> None:
        self.index = 0
        if self.shuffle:
            order = np.random.permutation(np.arange(len(self.mesh_triangles)))
            self.mesh_triangles = self.mesh_triangles[order]
            self.mesh_normals = self.mesh_normals[order]
        # self.mesh.triangles = o3d.utility.Vector3iVector([[0, 0, 0]])
        self.mesh.triangles = o3d.utility.Vector3iVector([self.mesh_triangles[0]])
        self.mesh.triangle_normals = o3d.utility.Vector3dVector([self.mesh_normals[0]])

    def animate(self, _vis):
        self.index += self.step
        self.mesh.triangles = o3d.utility.Vector3iVector(self.mesh_triangles[:self.index + 1])
        self.mesh.triangle_normals = o3d.utility.Vector3dVector(self.mesh_normals[:self.index + 1])

        if self.index > len(self.mesh_triangles):
            self.stop_animate()

        return True
