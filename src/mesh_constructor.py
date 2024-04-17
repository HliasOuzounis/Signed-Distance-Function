import numpy as np

from vvrpywork.shapes import Mesh3D

from .callback import Callback
from . import utility


class MeshConstructor(Callback):
    def __init__(self, mesh: Mesh3D) -> None:
        super().__init__()
        self.mesh_id = "mesh"
        self.mesh = utility.fit_to_unit_sphere(mesh)
        self.mesh.color = [0, 0, 0, 1]
        self.mesh.vertex_colors = utility.assign_colors(self.mesh)

        self.shuffle = True # shuffle the triangles

        self.total = len(self.mesh.triangles)
        self.mesh_triangles = self.mesh.triangles
        self.mesh_normals = self.mesh.triangle_normals
        self.step = 1 / 100

    def animate_init(self) -> None:
        self.l = 0
        
        if self.shuffle:
            order = np.random.permutation(np.arange(len(self.mesh_triangles)))
            self.mesh_triangles = self.mesh_triangles[order]
            self.mesh_normals = self.mesh_normals[order]
            
        self.mesh.triangles = [[0, 0, 0]]
        self.mesh.triangle_normals = [[0, 0, 0]]
        
        self.scene.removeShape(self.mesh_id)
        self.scene.addShape(self.mesh, self.mesh_id)
               

    def animate(self) -> bool:
        if self.l > 1:
            self.stop_animate()

        index = int(self.l * self.total)
        self.mesh.triangles = self.mesh_triangles[:index + 1]
        self.mesh.triangle_normals = self.mesh_normals[:index + 1]

        self.l += self.step

        self.scene.updateShape(self.mesh_id)

        return True
