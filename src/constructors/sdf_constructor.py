import numpy as np

from vvrpywork.shapes import PointSet3D, Mesh3D

from ..utils import KDTree
from ..utils import utility

from .callback import Callback, SequenceHandler, Scene3D


class SDFConstructor(Callback):
    def __init__(self, mesh: Mesh3D, kd_tree: KDTree) -> None:
        super().__init__()
        self.mesh = mesh
        self.kd_tree = kd_tree

        total_points = 10_000
        point_per_axis = int(total_points ** (1 / 3))
        self.step = 1 / 100

        offset = 0.1

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
        self.grid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        self.grid_cloud_name = "grid"

    def animate_init(self) -> None:
        self.l = 0
        self.prev_index = 0

        self.grid_cloud = PointSet3D(np.zeros((1, 3)), size=1)
        self.grid_cloud.colors = np.empty((0, 3))

        self.scene.removeShape(self.grid_cloud_name)
        self.scene.addShape(self.grid_cloud, self.grid_cloud_name)

    # @utility.show_fps
    def animate(self) -> bool:
        self.l += self.step

        if self.l > self.limit:
            self.stop_animate()

        index = int(self.grid.shape[0] * self.l)

        inside = self.kd_tree.is_inside(self.grid[self.prev_index : index + 1])

        colors = np.zeros((inside.shape[0], 3))
        colors[inside] = np.array([[0, 0, 1]])
        colors[~inside] = np.array([[1, 0, 0]])

        self.grid_cloud.points = self.grid[: index + 1]
        self.grid_cloud.colors = np.concatenate([self.grid_cloud.colors[1:, :3], colors], axis=0)
        # self.grid_cloud.colors = np.zeros((self.grid_cloud.points.shape[0], 3))
        print(self.grid_cloud.colors)

        print(self.grid_cloud.points.shape[0], self.grid_cloud.colors.shape[0])
        print(f"{colors.shape[0]=}, {inside.shape[0]=}, {index=}, {self.prev_index=}")
        print(self.grid_cloud.colors[0])

        self.scene.updateShape(self.grid_cloud_name)

        self.prev_index = index

        return True


# pc = PointSet3D(np.array([[0, 0, 0]]), size=1)
# # for i, (ci, ins) in enumerate(zip(c, is_inside)):
# for i, ins in enumerate(is_inside):
#     if not ins:
#         continue
#     pc.add(Point3D(grid[i], color=[1, 0, 0] if not ins else [0, 0, 1]))
# self.scene.addShape(pc)
