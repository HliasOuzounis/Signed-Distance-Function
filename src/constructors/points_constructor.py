import numpy as np

from vvrpywork.shapes import Mesh3D, PointSet3D, Cuboid3D, Point3D

from .callback import Callback
from ..utils.kd_tree import KDTree, TriangleParams
from ..utils import utility


class PointsConstructor(Callback):
    def __init__(self, mesh: Mesh3D, plane: Mesh3D) -> None:
        super().__init__()

        self.mesh = mesh
        self.plane = plane

        self.intersecting = PointSet3D()
        self.intersecting_name = "intersecting points"
        self.non_intersecting = PointSet3D()
        self.non_intersecting_name = "non-intersecting points"

        self.total_points = 15000
        self.step = 1 / 100

    def animate_init(self) -> None:
        self.l = 0
        self.prev_index = 0

        plane_normal = np.cross(
            self.plane.vertices[1] - self.plane.vertices[0],
            self.plane.vertices[2] - self.plane.vertices[0],
        )
        plane_normal /= np.linalg.norm(plane_normal)

        self.rot_mat = (
            utility.rotation_matrix_from_vectors(plane_normal, np.array([0, 0, 1]))
            if plane_normal[2] != 1
            else np.eye(3)
        )
        self.inv_rot_mat = np.linalg.inv(self.rot_mat)
        rotated_plane_verts = np.dot(self.plane.vertices, self.rot_mat.T)

        # Points are constructed in the plane's coordinate system
        # and then rotated back to world coordinate system
        self.triangle_params = TriangleParams(self.mesh.triangles, np.dot(self.mesh.vertices, self.inv_rot_mat))
        self.kd_tree = KDTree(np.dot(self.mesh.vertices, self.inv_rot_mat), self.mesh.triangles)

        self.intersecting.clear()
        self.intersecting_points = np.empty((0, 3))

        self.non_intersecting.clear()
        self.non_intersecting.createRandom(
            Cuboid3D(
                rotated_plane_verts[0]  - np.array([0, 0, 0.0001]),
                rotated_plane_verts[-1] + np.array([0, 0, 0.0001]),
            ),
            self.total_points,
        )

        self.scene.addShape(self.non_intersecting, self.non_intersecting_name)
        self.non_intersecting.remove_duplicated_points()
        self.scene.removeShape(self.non_intersecting_name)
        self.random_points = self.non_intersecting.points

        mesh_v = self.mesh.vertices
        n = int(100_000 ** (1/3))

        xmin = np.min(mesh_v[:, 0])
        xmax = np.max(mesh_v[:, 0])
        x = np.linspace(xmin, xmax, n)

        ymin = np.min(mesh_v[:, 1])
        ymax = np.max(mesh_v[:, 1])
        y = np.linspace(ymin, ymax, n)

        zmin = np.min(mesh_v[:, 2])
        zmax = np.max(mesh_v[:, 2])
        z = np.linspace(zmin, zmax, n)

        xx, yy, zz = np.meshgrid(x, y, z)
        grid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        triangles = set(map(tuple, self.mesh.triangles))

        import time
        t1 = time.time()
        print("Starting", grid.shape[0])
        c, d = self.kd_tree.nearest3(grid)
        print("Time", time.time() - t1)
        
        pc = PointSet3D(np.array([[0, 0, 0]]), size=2)
        for i, ci in enumerate(c):
            from itertools import permutations
            if any((i, j, k) in triangles for i, j, k in permutations(ci, 3)):
                pc.add(Point3D(grid[i], color=[0, 1, 0]))
            else:
                pc.add(Point3D(grid[i], color=[d[i, 0], 0, d[i, 0]]))
        self.scene.addShape(pc)
        

        self.non_intersecting.clear()
        self.non_intersecting_points = np.empty((0, 3))

        self.scene.removeShape(self.intersecting_name)
        self.scene.addShape(self.intersecting, self.intersecting_name)
        self.scene.removeShape(self.non_intersecting_name)
        self.scene.addShape(self.non_intersecting, self.non_intersecting_name)

    @utility.show_fps
    def animate(self) -> bool:
        self.l += self.step

        if self.l > self.limit:
            self.scene.removeShape(self.non_intersecting_name)
            self.estimate_area()
            self.stop_animate()
            return True

        index = int(self.l * self.total_points)

        # interecting_points = self.triangle_params.check_points(self.random_points[self.prev_index : index + 1])
        interecting_points = self.kd_tree.intersects_mesh(self.random_points[self.prev_index : index + 1]) # ~ 2 times faster
        interecting_points_indexes = interecting_points > 0

        self.non_intersecting_points = np.concatenate([
            self.non_intersecting_points[(self.prev_index - index) * 4:],
            np.dot(self.random_points[self.prev_index : index + 1][~interecting_points_indexes], self.rot_mat),
        ], axis=0)

        self.non_intersecting.points = self.non_intersecting_points
        self.non_intersecting.colors = np.zeros((self.non_intersecting.points.shape[0], 3))

        self.intersecting_points = np.concatenate([
            self.intersecting_points,
            np.dot(self.random_points[self.prev_index : index + 1][interecting_points_indexes], self.rot_mat),
        ], axis=0)

        self.intersecting.points = self.intersecting_points            
        self.intersecting.colors = np.zeros((self.intersecting.points.shape[0], 3)) + np.array([1, 0, 0])

        self.scene.updateShape(self.intersecting_name)
        self.scene.updateShape(self.non_intersecting_name)

        self.prev_index = index

        return True

    def estimate_area(self) -> None:
        plane1 = self.plane.vertices[0]
        plane2 = self.plane.vertices[-1]

        diag = plane2 - plane1

        plane_area = np.linalg.norm(diag[[0, 2]]) * diag[1]

        print(f"Area of projection using Monte Carlo: {self.intersecting.points.shape[0] / self.prev_index * plane_area:.4f} units^2")
        print(f"{self.intersecting.points.shape[0]} points out of {self.prev_index} points")
