import numpy as np
import open3d as o3d

from vvrpywork.shapes import Mesh3D

from .callback import Callback, SequenceHandler, Scene3D
from ..utils import NDPoint3D

class PlaneConstructor(Callback):
    def __init__(self, mesh: Mesh3D) -> None:
        super().__init__()
        self.mesh = mesh

        self.mesh_vertices = self.mesh.vertices
        offset = 0.3
        maxx = np.max(self.mesh_vertices[:, 0]) + offset
        minx = np.min(self.mesh_vertices[:, 0]) - offset
        maxy = np.max(self.mesh_vertices[:, 1]) + offset
        miny = np.min(self.mesh_vertices[:, 1]) - offset
        minz = np.min(self.mesh_vertices[:, 2]) - offset
        maxz = np.max(self.mesh_vertices[:, 2]) + offset

        min_side = min(minx, miny, minz)
        max_side = max(maxx, maxy, maxz)

        self.start1 = np.array([min_side, min_side, -max_side])
        self.start2 = np.array([max_side, min_side, -max_side])

        self.end1 = np.array([min_side, max_side, -max_side])
        self.end2 = np.array([max_side, max_side, -max_side])

        self.step = 1 / 100

        self.plane = Mesh3D()
        self.plane_name = "plane"

    def animate_init(self) -> None:
        self.clear(self.sequence, self.scene)
        
        self.l = 0

        self.points = np.array([self.start1, self.start2, self.start1, self.start2])
        self.plane.vertices = self.points
        self.plane.triangles = np.array([[0, 1, 2], [3, 2, 1]])
        self.plane.color = [0.5, 0.5, 0.5]

        self.scene.addShape(self.plane, self.plane_name)

    def animate(self) -> bool:
        if self.l > self.limit:
            self.stop_animate()

        self.points[2, :] = self.start1 * (1 - self.l) + self.end1 * self.l
        self.points[3, :] = self.start2 * (1 - self.l) + self.end2 * self.l

        self.l += self.step

        self.plane.vertices = self.points

        self.scene.updateShape(self.plane_name)

        return True
    
    def stop_animate(self) -> None:
        self.scene.window.set_on_key(self.handleRotation)
        self.scene.window.set_on_tick_event(None)
    

    def handleRotation(self, key_event) -> None:
        rotate_step = np.pi / 30
        key, action = key_event.key, key_event.type

        if key == o3d.visualization.gui.KeyName.ENTER and action == o3d.visualization.gui.KeyEvent.DOWN:
            super().stop_animate()
            self.scene.window.set_on_key(self.sequence.perform_action)

        # if key == o3d.visualization.gui.KeyName.W:
        #     self.rotate(np.array([rotate_step, 0, 0]))
        # if key == o3d.visualization.gui.KeyName.S:
        #     self.rotate(np.array([-rotate_step, 0, 0]))
        if key == o3d.visualization.gui.KeyName.A:
            self.rotate(np.array([0, rotate_step, 0]))
        if key == o3d.visualization.gui.KeyName.D:
            self.rotate(np.array([0, -rotate_step, 0]))

    def rotate(self, angle: NDPoint3D) -> None:
        R = o3d.geometry.get_rotation_matrix_from_xyz(angle)

        self.plane.vertices = (self.plane.vertices @ R.T)

        self.scene.updateShape(self.plane_name)

    
    def clear(self, _sequence: SequenceHandler, scene: Scene3D) -> bool:
        self.l = 0
        scene.removeShape(self.plane_name)
        return True