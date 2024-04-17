import numpy as np

from vvrpywork.shapes import Mesh3D, PointSet3D, Cuboid3D
from vvrpywork.shapes import Triangle2D, Line2D, Point2D

from . import utility, callback


class PointsConstructor(callback.Callback):
    def __init__(self, mesh: Mesh3D, plane: Mesh3D) -> None:
        super().__init__()

        self.mesh = mesh
        self.plane = plane

        self.kd_tree = KDTree(self.mesh.vertices, self.mesh.triangles)

        self.point_cloud = PointSet3D()
        self.point_cloud_name = "points"

        self.total_points = 10000
        self.step = 1 / 1000

    def animate_init(self) -> None:
        self.l = 0
        self.prev_index = 0

        self.point_cloud.clear()
        self.point_cloud.createRandom(
            Cuboid3D(self.plane.vertices[0], self.plane.vertices[-1]),
            self.total_points,
        )
        self.points = self.point_cloud.points
        
        # brute force works
        self.projected_points = self.mesh.vertices[:, :2]
        self.triangles = [Triangle2D(*self.projected_points[triangle]) for triangle in self.mesh.triangles]
        
        self.points_colors = np.zeros((self.total_points, 3))
        self.point_cloud.clear()

        np.random.shuffle(self.points)

        self.scene.removeShape(self.point_cloud_name)
        self.scene.addShape(self.point_cloud, self.point_cloud_name)

    def animate(self) -> bool:
        self.l += self.step

        if self.l > 1:
            self.stop_animate()
        
        index = int(self.l * self.total_points)

        self.point_cloud.points = self.points[: index + 1]
        self.point_cloud.colors = self.points_colors[: index + 1]

        if self.prev_index != index:
            for i, point in enumerate(self.points[self.prev_index:index]):
                intersecting = self.check_intersection(point)
                self.points_colors[self.prev_index + i] = [1, 0, 0] if intersecting else [0, 0, 0]

        self.prev_index = index
        self.scene.updateShape(self.point_cloud_name)

        return True

    def check_intersection(self, point: np.array) -> bool:
        return self.kd_tree.intersects_mesh(Point2D(point[:2]))

        # brute force works
        point = Point2D(point[:2])
        for triangle in self.triangles:
            if triangle.contains(point):
                return True
        return False
            
        


class Node:
    def __init__(self, points, line: Line2D, intersecting_triangles: np.array) -> None:
        self.points = points
        self.line = line
        self.intersecting_triangles = [
            Triangle2D(*points[triangle]) for triangle in intersecting_triangles
        ]

        self.left = None
        self.right = None

    def check_intersection(self, point: Point2D) -> bool:
        for trinagle in self.intersecting_triangles:
            if trinagle.contains(point):
                return True
        
        if self.is_on_right(point):
            if self.right is not None:
                return self.right.check_intersection(point)
        else:
            if self.left is not None:
                return self.left.check_intersection(point)

        return False

    def is_on_right(self, point: Point2D) -> bool:
        return self.line.isOnRight(point)


class KDTree:
    def __init__(self, points, triangles) -> None:
        self.points = points[:, :2]
        self.triangles = triangles
        self.root = self.build_tree(self.points, self.triangles)

    def build_tree(self, points, triangles, depth=0):
        if triangles.shape[0] == 0:
            return None
        if points.shape[0] < 2:
            print("No points left", triangles.shape[0])
            return None

        if depth % 2 == 0:
            median = np.median(points[:, 0])
            is_above_line = points[:, 0] > median
            all_above_line = self.points[:, 0] > median
            line = Line2D((median, 0), (median, 1))
        else:
            median = np.median(points[:, 1])
            is_above_line = points[:, 1] > median
            all_above_line = self.points[:, 1] > median
            line = Line2D((1, median), (0, median))
            
        P1 = points[is_above_line]
        P2 = points[~is_above_line]


        triangle_above_line = np.vectorize(lambda x: all_above_line[x])(triangles)
        
        P1_triangles = triangles[triangle_above_line.all(axis=1)]
        P2_triangles = triangles[~triangle_above_line.any(axis=1)]
        
        intersecting_triangles = triangles[triangle_above_line.any(axis=1) & ~triangle_above_line.all(axis=1)]

        v = Node(self.points, line, intersecting_triangles)
        v.right = self.build_tree(P1, P1_triangles, depth + 1)
        v.left = self.build_tree(P2, P2_triangles, depth + 1)

        return v

    def intersects_mesh(self, point: Point2D) -> bool:
        return self.root.check_intersection(point)
