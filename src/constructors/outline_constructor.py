import numpy as np

from .callback import Callback

from vvrpywork.shapes import PointSet3D, LineSet3D, Line3D

from ..utils.alpha_shapes import get_outline
from ..utils import utility

class OutlineConstructor(Callback):
    def __init__(self, intesecting_points: PointSet3D) -> None:
        super().__init__()
        self.pointcloud = intesecting_points
        self.outline: dict[tuple[int, int], Line3D] = {}
        self.outline_name = "outline"

    def animate_init(self) -> None:
        plane_normal = np.cross(
            self.pointcloud.points[1] - self.pointcloud.points[0],
            self.pointcloud.points[2] - self.pointcloud.points[0],
        )
        plane_normal /= np.linalg.norm(plane_normal)

        rot_mat = (
            utility.rotation_matrix_from_vectors(plane_normal, np.array([0, 0, 1]))
            if plane_normal[2] != 1
            else np.eye(3)
        )
        inv_rot_mat = np.linalg.inv(rot_mat)

        self.points = self.pointcloud.points
        self.projected_points = np.dot(self.points, inv_rot_mat)[:, :2]

        alpha = 0.05
        self.outline_generator = get_outline(self.points, self.projected_points, alpha)

        for key, line in self.outline.items():
            self.scene.removeShape(key)
        self.outline.clear()

    @utility.show_fps
    def animate(self) -> bool:
        steps = 100
        try:
            for i in range(steps):
                new_line = next(self.outline_generator)
                if new_line[::-1] in self.outline:
                    new_line_str = ",".join(map(str, new_line[::-1]))
                    self.scene.removeShape(new_line_str)
                    self.outline.pop(new_line[::-1])
                else:
                    new_line_str = ",".join(map(str, new_line))
                    self.outline[new_line] = Line3D(self.points[new_line[0]], self.points[new_line[1]], width=2)
                    self.scene.addShape(self.outline[new_line], new_line_str)
        except StopIteration:
            self.stop_animate()
            return True

        return True
