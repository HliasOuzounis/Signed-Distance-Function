import numpy as np

from .callback import Callback

from vvrpywork.shapes import PointSet3D, Line3D

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
        self.outline_generator = get_outline(self.projected_points, alpha)

        for key in self.outline:
            key_str = ",".join(map(str, key))
            self.scene.removeShape(key_str)
        self.outline.clear()

    @utility.show_fps
    def animate(self) -> bool:
        steps = 100
        
        outline_batch = GeneratorBatcher(self.outline_generator, steps)

        for next_line in outline_batch:
            old_line = next_line[::-1]
            if old_line in self.outline:
                old_line_str = ",".join(map(str, old_line))
                self.scene.removeShape(old_line_str)
                self.outline.pop(old_line)
            else:
                next_line_str = ",".join(map(str, next_line))
                self.outline[next_line] = Line3D(self.points[next_line[0]], self.points[next_line[1]], width=2)
                self.scene.addShape(self.outline[next_line], next_line_str)
        
        if outline_batch.value is not None:
            print(f"Area of projection using Outline: {outline_batch.value:.4f} units^2")
            self.stop_animate()
            
        return True

class GeneratorBatcher:
    def __init__(self, generator, batch_size) -> None:
        self.generator = generator
        self.batch_size = batch_size
        self.value = None
        
    def __iter__(self):
        for _ in range(self.batch_size):
            try:
                yield next(self.generator)
            except StopIteration as s:
                self.value = s.value
                break
        