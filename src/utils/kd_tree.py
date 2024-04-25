import numpy as np

class TriangleParams:
    def __init__(self, triangles, points) -> None:
        # project onto plane
        points = points[:, :2]
        triangles = points[triangles]

        self.v0, v1, v2 = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
        self.edge1 = v1 - self.v0
        self.edge2 = v2 - self.v0

        self.dot00 = np.einsum("ij,ij->i", self.edge1, self.edge1)
        self.dot01 = np.einsum("ij,ij->i", self.edge1, self.edge2)
        self.dot11 = np.einsum("ij,ij->i", self.edge2, self.edge2)

        self.inv_denom = 1 / (self.dot00 * self.dot11 - self.dot01 * self.dot01)

    def check_points(self, points):
        vp = points[:, np.newaxis, :2] - self.v0

        dot20 = np.einsum("ijk,jk->ij", vp, self.edge1)
        dot21 = np.einsum("ijk,jk->ij", vp, self.edge2)

        u = (self.dot11 * dot20 - self.dot01 * dot21) * self.inv_denom
        v = (self.dot00 * dot21 - self.dot01 * dot20) * self.inv_denom

        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        return inside.any(axis=1)


class Node:
    def __init__(self, points, line: np.array, median: float, intersecting_triangles: np.array) -> None:
        self.points = points
        self.line = line
        self.median = median
        self.intersecting_triangles = TriangleParams(intersecting_triangles, points)

        self.left = None
        self.right = None

    def check_intersection(self, points: np.array):
        if points.shape[0] == 0:
            return np.zeros(0, dtype=bool)

        intersects = self.intersecting_triangles.check_points(points)

        on_right = self.is_on_right(points)

        interescts_right = np.zeros_like(intersects, dtype=bool)
        if self.right is not None:
            interescts_right[on_right & ~intersects] = self.right.check_intersection(points[on_right & ~intersects])

        interescts_left = np.zeros_like(intersects, dtype=bool)
        if self.left is not None:
            interescts_left[~on_right & ~intersects] = self.left.check_intersection(points[~on_right & ~intersects])

        return intersects | interescts_right | interescts_left

    def is_on_right(self, points: np.array):
        points = points[:, :2]
        return np.dot(points, self.line) > self.median


class KDTree:
    def __init__(self, points, triangles) -> None:
        self.all_points = points[:, :2]
        self.triangles = triangles
        self.root = self.build_tree(self.all_points, self.triangles)

    def build_tree(self, points, triangles, depth=0):
        if triangles.shape[0] == 0:
            return None

        if depth % 2 == 0:
            median = np.median(points[:, 0])
            is_above_line = points[:, 0] > median
            all_above_line = self.all_points[:, 0] > median
            line = np.array([1, 0]).T
        else:
            median = np.median(points[:, 1])
            is_above_line = points[:, 1] > median
            all_above_line = self.all_points[:, 1] > median
            line = np.array([0, 1]).T

        P1 = points[is_above_line]
        P2 = points[~is_above_line]

        triangle_above_line = np.vectorize(lambda x: all_above_line[x])(triangles)
        all_below_line = ~triangle_above_line.any(axis=1)
        all_above_line = triangle_above_line.all(axis=1)

        P1_triangles = triangles[all_above_line]
        P2_triangles = triangles[all_below_line]

        intersecting_triangles = triangles[~all_below_line & ~all_above_line]

        v = Node(self.all_points, line, median, intersecting_triangles)
        v.right = self.build_tree(P1, P1_triangles, depth + 1)
        v.left = self.build_tree(P2, P2_triangles, depth + 1)

        return v

    def intersects_mesh(self, points: np.array):
        return self.root.check_intersection(points)
