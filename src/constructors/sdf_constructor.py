import numpy as np

from vvrpywork.shapes import PointSet3D, Mesh3D

from ..utils import KDTree, SDF
from ..utils import utility

from .callback import Callback, SequenceHandler, Scene3D


class SDFConstructor(Callback):
    def __init__(self, mesh: Mesh3D, kdTree2D: KDTree) -> None:
        super().__init__()
        self.mesh = mesh

        self.total_points = 15**3
        point_per_axis = int(self.total_points ** (1 / 3)) + 1
        self.total_points = point_per_axis**3
        self.step = 1 / 100

        offset = 0.3

        xmin = np.min(self.mesh.vertices[:, 0]) - offset
        xmax = np.max(self.mesh.vertices[:, 0]) + offset
        x = np.linspace(xmin, xmax, point_per_axis)

        ymin = np.min(self.mesh.vertices[:, 1]) - offset
        ymax = np.max(self.mesh.vertices[:, 1]) + offset
        y = np.linspace(ymin, ymax, point_per_axis)

        zmin = np.min(self.mesh.vertices[:, 2]) - offset
        zmax = np.max(self.mesh.vertices[:, 2]) + offset
        z = np.linspace(zmin, zmax, point_per_axis)

        xx, yy, zz = np.meshgrid(x, y, z)
        self.grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        self.grid = (x, y, z)

        self.grid_clouds = {}

        self.sdf = SDF(self.grid)
        self.distances = np.zeros((self.total_points, 1))

        self.kdTree2D = kdTree2D
        self.kdTree3D = KDTree(dimensions=3)

    def animate_init(self) -> None:
        self.clear(self.sequence, self.scene)

        self.l = 0
        self.prev_index = 0

        if not self.kdTree2D.is_built:
            self.kdTree2D.build_tree(
                self.mesh.vertices, self.mesh.triangles, inv_rot_mat=np.eye(3)
            )

        if not self.kdTree3D.is_built:
            self.kdTree3D.build_tree(
                self.mesh.vertices, self.mesh.triangles, inv_rot_mat=np.eye(3)
            )

    @utility.show_fps
    def animate(self) -> bool:
        self.l += self.step

        if self.l > self.limit:
            self.sdf.build(
                self.distances.reshape(
                    (
                        self.grid[0].shape[0],
                        self.grid[1].shape[0],
                        self.grid[2].shape[0],
                    )
                )
            )
            self.stop_animate()

        index = int(self.grid_points.shape[0] * self.l)

        inside = self.kdTree2D.is_inside(
            self.grid_points[self.prev_index: index + 1])

        # self.distances[self.prev_index: index + 1] = self.kdTree3D.min_distance(
        #     self.grid_points[self.prev_index: index + 1]
        # ).reshape(-1, 1)
        _, distances = self.kdTree3D.closest_point(self.grid_points[self.prev_index: index + 1])
        self.distances[self.prev_index: index + 1] = distances.reshape(-1, 1)
        # self.distances[self.prev_index: index + 1] = np.linalg.norm(self.grid_points[self.prev_index: index + 1], axis=1).reshape(-1, 1) - 1

        self.distances[self.prev_index: index + 1][inside] *= -1

        grid_colors = np.zeros((inside.shape[0], 3))
        grid_colors[inside] = (
            np.array([[0, 0, 1]]) * -
            self.distances[self.prev_index: index + 1][inside]
        )
        grid_colors[~inside] = (
            np.array([[1, 0, 0]]) *
            self.distances[self.prev_index: index + 1][~inside]
        )

        grid_cloud = PointSet3D(
            self.grid_points[self.prev_index: index + 1], size=1)
        grid_cloud.colors = grid_colors
        name = f"grid_cloud_{self.prev_index}_{index}"
        self.grid_clouds[name] = grid_cloud

        self.scene.addShape(grid_cloud, name)
        self.prev_index = index

        return True

    def clear(self, _sequence: SequenceHandler, scene: Scene3D) -> bool:
        for key in self.grid_clouds:
            scene.removeShape(key)

        self.grid_clouds = {}
        return True
