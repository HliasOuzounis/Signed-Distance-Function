import numpy as np

from vvrpywork.shapes import Mesh3D, PointSet3D, Cuboid3D, Arrow3D

from .callback import Callback
from ..utils.kd_tree import KDTree, TriangleParams
from ..utils import utility


class PointsConstructor(Callback):
    def __init__(self, mesh: Mesh3D, plane: Mesh3D) -> None:
        super().__init__()

        self.mesh = mesh
        self.plane = plane

        self.point_cloud = PointSet3D()
        self.point_cloud_name = "points"

        self.total_points = 10000
        self.step = 1 / 200

    def animate_init(self) -> None:
        self.l = 0
        self.prev_index = 0

        self.plane_normal = np.cross(
            self.plane.vertices[1] - self.plane.vertices[0],
            self.plane.vertices[2] - self.plane.vertices[0],
        )
        self.plane_normal /= np.linalg.norm(self.plane_normal)

        self.rot_mat = (
            utility.rotation_matrix_from_vectors(self.plane_normal, np.array([0, 0, 1]))
            if self.plane_normal[2] != 1
            else np.eye(3)
        )
        self.inv_rot_mat = np.linalg.inv(self.rot_mat)
        plane_verts = np.dot(self.plane.vertices, self.rot_mat.T)

        # Points are constructed in the plane's coordinate system
        # and then rotated back to world coordinate system
        # self.triangle_params = TriangleParams(self.mesh.triangles, self.mesh.vertices)
        self.kd_tree = KDTree(np.dot(self.mesh.vertices, self.inv_rot_mat), self.mesh.triangles)

        self.intersecting_points = np.empty((0, 3))
        self.projection_pointcloud = PointSet3D()

        self.point_cloud.clear()
        self.point_cloud.createRandom(
            Cuboid3D(
                plane_verts[0] + np.array([0, 0, 0.0001]),
                plane_verts[-1] + np.array([0, 0, 0.0001]),
            ),
            self.total_points,
        )
        self.points = self.point_cloud.points

        self.points_colors = np.zeros((self.total_points, 3))
        self.point_cloud.clear()

        self.scene.removeShape(self.point_cloud_name)
        self.scene.addShape(self.point_cloud, self.point_cloud_name)

    def animate(self) -> bool:
        self.l += self.step

        if self.l > self.limit:
            self.projection_pointcloud = PointSet3D(self.intersecting_points)
            self.estimate_area()
            self.stop_animate()

        index = int(self.l * self.total_points)

        # interecting_points = self.triangle_params.check_points(self.points[self.prev_index : index])
        interecting_points = self.kd_tree.intersects_mesh(
            self.points[self.prev_index : index + 1]
        )  # ~ 4 times faster

        self.points_colors[self.prev_index : index + 1][interecting_points] = [1, 0, 0]
        self.intersecting_points = np.concatenate(
            [
                self.intersecting_points,
                self.points[self.prev_index : index + 1][interecting_points],
            ],
            axis=0,
        )

        self.point_cloud.points = np.dot(self.points[: index + 1], self.rot_mat)
        self.point_cloud.colors = self.points_colors[: index + 1]

        self.prev_index = index
        self.scene.updateShape(self.point_cloud_name)

        return True

    def estimate_area(self) -> None:
        plane1 = self.plane.vertices[0]
        plane2 = self.plane.vertices[-1]
        plane_area = np.linalg.norm(plane1[0] - plane2[0]) * np.linalg.norm(
            plane1[1] - plane2[1]
        )
        print(
            f"Area of projection: {self.intersecting_points.shape[0] / self.prev_index * plane_area:.4f} units^2"
        )
        print(
            f"{self.intersecting_points.shape[0]} points out of {self.prev_index} points"
        )
