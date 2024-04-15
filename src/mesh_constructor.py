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
        self.mesh.triangles = o3d.utility.Vector3iVector([self.mesh_triangles[0]])
        self.step = self.mesh_triangles.shape[0] // 200

        self.vis.add_geometry(self.mesh)

    def __call__(self, vis: Visualizer | VisualizerWithKeyCallback, key: int | None = None, action: int | None = None) -> bool:
        if key is not None and key != 1:
            return False

        self.index = 0
        if self.shuffle: 
            np.random.shuffle(self.mesh_triangles)

        self.mesh.triangles = o3d.utility.Vector3iVector([self.mesh_triangles[0]])

        return super().__call__(vis, key, action)

    def animate(self, _vis):
        self.index += self.step
        self.mesh.triangles = o3d.utility.Vector3iVector(self.mesh_triangles[:self.index + 1])

        if self.index >= len(self.mesh_triangles):
            self.stop_animate()

        return True
