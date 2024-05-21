import numpy as np

from .callback import Callback, Scene3D, SequenceHandler

from vvrpywork.shapes import PointSet3D, Line3D, Scene3D, Mesh3D

from ..utils import alpha_shapes, utility

class OutlineConstructor(Callback):
    def __init__(self, intesecting_points: PointSet3D, plane: Mesh3D) -> None:
        super().__init__()
        self.pointcloud = intesecting_points
        self.plane = plane
        
        self.outline: dict[tuple[int, int], Line3D] = {}
        self.outline_name = "outline"

    def animate_init(self) -> None:
        self.plane_normal = np.cross(
            self.plane.vertices[1] - self.plane.vertices[0],
            self.plane.vertices[2] - self.plane.vertices[0],
        )
        self.plane_normal /= np.linalg.norm(self.plane_normal)

        self.rot_mat = (
            utility.rotation_matrix_from_vectors(self.plane_normal, np.array([0, 0, 1]))
            if not np.isclose(self.plane_normal[2], 1)
            else np.eye(3)
        )
        inv_rot_mat = self.rot_mat.T
        
        self.points = self.pointcloud.points
        self.projected_points = np.dot(self.points, inv_rot_mat)[:, :2]

        alpha = 3e-2
        self.outline_generator = alpha_shapes.get_outline(self.projected_points, alpha)

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
                self.outline[next_line] = Line3D(self.points[next_line[0]], self.points[next_line[1]], width=1.5)
                self.scene.addShape(self.outline[next_line], next_line_str)

        if outline_batch.value is not None:
            print(f"Area of projection using Outline: {outline_batch.value:.4f} units^2")
            self.stop_animate()

        return True

    def clear(self, _sequence: SequenceHandler, _scene: Scene3D) -> bool:
        for key in self.outline:
            key_str = ",".join(map(str, key))
            self.scene.removeShape(key_str)
        self.outline.clear()
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
