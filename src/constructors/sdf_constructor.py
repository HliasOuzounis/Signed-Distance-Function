import numpy as np

from vvrpywork.shapes import PointSet3D, Mesh3D

from ..utils import KDTree, SDF
from ..utils import utility

from .callback import Callback, SequenceHandler, Scene3D


class SDFConstructor(Callback):
    def __init__(self, mesh: Mesh3D, kd_tree: KDTree) -> None:
        super().__init__()
        self.mesh = mesh
        self.kd_tree = kd_tree

        self.total_points = 25_000
        point_per_axis = int(self.total_points ** (1 / 3))
        self.total_points = point_per_axis ** 3
        self.step = 1 / 500

        offset = 0.01

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

        self.grid_cloud_name = "grid"
        
        self.sdf = SDF(self.grid)

    def animate_init(self) -> None:
        self.l = 0
        self.prev_index = 0

        self.grid_cloud = PointSet3D(np.zeros((1, 3)), size=0.5)
        self.grid_colors = np.zeros((self.total_points, 3))
        
        self.distances = np.zeros((self.total_points, 1))

        self.scene.removeShape(self.grid_cloud_name)
        self.scene.addShape(self.grid_cloud, self.grid_cloud_name)

    @utility.show_fps
    def animate(self) -> bool:
        self.l += self.step

        if self.l > self.limit:
            self.sdf.build(self.distances.reshape((
                self.grid[0].shape[0], 
                self.grid[1].shape[0], 
                self.grid[2].shape[0])
            ))
            self.stop_animate()

        index = int(self.grid_points.shape[0] * self.l)

        inside = self.kd_tree.is_inside(self.grid_points[self.prev_index : index + 1])
        
        self.distances[self.prev_index : index + 1] = self.calulate_distanes(self.grid_points[self.prev_index : index + 1])

        self.grid_colors[self.prev_index : index + 1][inside] = np.array([[0, 0, 1]])
        self.grid_colors[self.prev_index : index + 1][~inside] = np.array([[1, 0, 0]])

        self.grid_cloud.points = self.grid_points[: index + 1]
        self.grid_cloud.colors = self.grid_colors[: index + 1]

        self.scene.updateShape(self.grid_cloud_name)

        self.prev_index = index

        return True

    def calulate_distanes(self, points: np.array) -> np.array:
        return np.empty(points.shape[0])
