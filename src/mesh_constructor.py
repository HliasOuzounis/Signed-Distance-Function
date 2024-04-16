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
        self.step = self.mesh_triangles.shape[0] // 100

        self.animate_init()

        self.vis.add_geometry(self.mesh)

        
    def animate_init(self) -> None:
        self.index = 0
        if self.shuffle: 
            np.random.shuffle(self.mesh_triangles)
        # self.mesh.triangles = o3d.utility.Vector3iVector([[0, 0, 0]])
        self.mesh.triangles = o3d.utility.Vector3iVector([self.mesh_triangles[0]])


    def animate(self, _vis):
        self.index += self.step
        self.mesh.triangles = o3d.utility.Vector3iVector(self.mesh_triangles[:self.index + 1])

        if self.index > len(self.mesh_triangles):
            self.stop_animate()

        return True
