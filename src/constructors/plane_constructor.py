import numpy as np
import open3d as o3d

from vvrpywork.shapes import Mesh3D, Cuboid3D, Cuboid3DGeneralized

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
        # z = np.max(self.mesh_vertices[:, 2]) + offset
        z = np.max(np.linalg.norm(self.mesh_vertices, axis=1)) + offset

        self.start1 = np.array([minx, miny, -z])
        self.start2 = np.array([maxx, miny, -z])

        self.end1 = np.array([minx, maxy, -z])
        self.end2 = np.array([maxx, maxy, -z])

        self.step = 1 / 100

        self.plane = Mesh3D()
        self.plane_name = "plane"

    def animate_init(self) -> None:
        self.l = 0

        self.points = np.array([self.start1, self.start2, self.start1, self.start2])
        self.plane.vertices = self.points
        self.plane.triangles = np.array([[0, 1, 2], [3, 2, 1]])
        self.plane.color = [0.5, 0.5, 0.5]


        self.scene.removeShape(self.plane_name)
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
        # return super().stop_animate()
    

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

    def rotate(self, angle: np.array) -> None:
        R = o3d.geometry.get_rotation_matrix_from_xyz(angle)

        self.plane.vertices = (self.plane.vertices @ R.T)

        self.scene.updateShape(self.plane_name)
