import numpy as np

from vvrpywork.shapes import Mesh3D

from .callback import Callback

class PlaneConstructor(Callback):
    def __init__(self, mesh: Mesh3D) -> None:
        super().__init__()
        self.mesh = mesh

        self.mesh_vertices = self.mesh.vertices
        offset = 0.2
        maxx = np.max(self.mesh_vertices[:, 0]) + offset
        minx = np.min(self.mesh_vertices[:, 0]) - offset
        maxy = np.max(self.mesh_vertices[:, 1]) + offset
        miny = np.min(self.mesh_vertices[:, 1]) - offset
        z    = np.max(self.mesh_vertices[:, 2]) + offset

        self.start1 = np.array([minx, miny, z])
        self.start2 = np.array([maxx, miny, z])

        self.end1 = np.array([minx, maxy, z])
        self.end2 = np.array([maxx, maxy, z])
        
        self.step = 1 / 100

        self.plane = Mesh3D()
        self.plane_name = "plane"

    def animate_init(self) -> None:
        self.l = 0

        self.points = np.array([self.start1, self.start2, self.start1, self.start2])
        self.plane.vertices = self.points
        self.plane.triangles = np.array([[0, 1, 2], [3, 2, 1]])
        self.plane.color = [0.8, 0.8, 0.8]

        self.scene.removeShape(self.plane_name)
        self.scene.addShape(self.plane, self.plane_name)

    def animate(self) -> bool:
        if self.l > 1:
            self.stop_animate()

        self.points[2, :] = self.start1 * (1 - self.l) + self.end1 * self.l
        self.points[3, :] = self.start2 * (1 - self.l) + self.end2 * self.l
        
        self.l += self.step

        self.plane.vertices = self.points
        
        self.scene.updateShape(self.plane_name)

        return True
