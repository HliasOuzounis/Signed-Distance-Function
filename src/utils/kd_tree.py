import numpy as np

# matrices to shift values in nx3 array. array @ shift
shift1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
shift2 = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]])
shift3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
shift = [shift1, shift2, shift3]


class TriangleParams:
    def __init__(self, triangles, points) -> None:
        # project onto plane
        self.triangles_d = np.max(points[:, 2][triangles], axis=1)
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

        before = points[:, np.newaxis, 2] < self.triangles_d
        inside = (u >= 0) & (v >= 0) & (u + v <= 1) & before
        return np.sum(inside, axis=1)


class Node:
    def __init__(self, value: np.array, line: np.array, idx=0) -> None:
        self.value = value
        self.line = line
        self.idx = idx

        self.left = None
        self.right = None

    def is_on_right(self, points: np.array):
        return np.dot(points, self.line[:3]) > self.line[3]

    def nearest3(self, points, best_nodes, distances):
        if points.shape[0] == 0:
            return best_nodes, distances

        on_right = self.is_on_right(points)

        if self.right is not None:
            new_best, new_distances = self.right.nearest3(points[on_right], best_nodes[on_right], distances[on_right])
            best_nodes[on_right] = new_best
            distances[on_right] = new_distances

        if self.left is not None:
            new_best, new_distances = self.left.nearest3(points[~on_right], best_nodes[~on_right], distances[~on_right])
            best_nodes[~on_right] = new_best
            distances[~on_right] = new_distances

        distance_with_current = np.linalg.norm(points - self.value, axis=1)
        best1 = distances[:, 0] > distance_with_current
        best2 = (distances[:, 1] > distance_with_current) & (~best1)
        best3 = (distances[:, 2] > distance_with_current) & (~(best1 | best2))

        b = [best1, best2, best3]

        for i in range(3):
            besti = b[i]
            best_nodes[besti] @= shift[i]
            best_nodes[besti, i] = self.idx
            distances[besti] @= shift[i]
            distances[besti, i] = distance_with_current[besti]
        

        distance_with_line = np.abs(np.dot(points, self.line[:3]))
        check_opposite = distance_with_line < distances[:, 2]

        if self.left is not None:
            new_best, new_distances = self.left.nearest3(
                points[check_opposite & on_right],
                best_nodes[check_opposite & on_right],
                distances[check_opposite & on_right],
            )
            best_nodes[check_opposite & on_right] = new_best
            distances[check_opposite & on_right] = new_distances

        if self.right is not None:
            new_best, new_distances = self.right.nearest3(
                points[check_opposite & ~on_right],
                best_nodes[check_opposite & ~on_right],
                distances[check_opposite & ~on_right],
            )
            best_nodes[check_opposite & ~on_right] = new_best
            distances[check_opposite & ~on_right] = new_distances

        return best_nodes, distances


class TrianglesNode(Node):
    def __init__(
        self, points, value: np.array, line: np.array, intersecting_triangles: np.array
    ) -> None:
        super().__init__(value, line)
        self.intersecting_triangles = TriangleParams(intersecting_triangles, points)

    def check_intersection(self, points: np.array):
        if points.shape[0] == 0:
            return np.zeros(0)

        intersects = self.intersecting_triangles.check_points(points)

        on_right = self.is_on_right(points)

        interescts_right = np.zeros_like(intersects)
        if self.right is not None:
            interescts_right[on_right] = self.right.check_intersection(points[on_right])

        interescts_left = np.zeros_like(intersects)
        if self.left is not None:
            interescts_left[~on_right] = self.left.check_intersection(points[~on_right])

        return intersects + interescts_right + interescts_left

    def is_on_right(self, points: np.array):
        points = points[:, :2]
        return np.dot(points, self.line[:2]) > self.line[2]


class KDTree:
    def __init__(self, points, triangles) -> None:
        self.all_points = points
        self.triangles = triangles
        self.root = self.build_tree_without_triangles(
            self.all_points, np.arange(self.all_points.shape[0])
        )
        self.troot = self.build_tree(self.all_points, self.triangles)

    def build_tree(self, points, triangles, depth=0) -> TrianglesNode:
        if triangles.shape[0] == 0:
            return None

        dim = depth % 2
        median_idx = np.argpartition(points, points.shape[0] // 2, axis=0)[
            points.shape[0] // 2, dim
        ]
        median = points[median_idx]

        line = np.array([0, 0, median[dim]])
        line[dim] = 1

        is_above_line = points[:, dim] > median[dim]

        P1 = points[is_above_line]
        P2 = points[~is_above_line]

        all_above_line = self.all_points[:, dim] > median[dim]

        triangle_above_line = np.vectorize(lambda x: all_above_line[x])(triangles)
        all_below_line = ~triangle_above_line.any(axis=1)
        all_above_line = triangle_above_line.all(axis=1)

        P1_triangles = triangles[all_above_line]
        P2_triangles = triangles[all_below_line]

        intersecting_triangles = triangles[~all_below_line & ~all_above_line]

        tv = TrianglesNode(self.all_points, median, line, intersecting_triangles)
        tv.right = self.build_tree(P1, P1_triangles, depth + 1)
        tv.left = self.build_tree(P2, P2_triangles, depth + 1)

        return tv

    def build_tree_without_triangles(self, points, indices, depth=0) -> Node:
        if points.shape[0] == 0:
            return None

        dim = depth % points.shape[1]
        median_idx = np.argpartition(points, points.shape[0] // 2, axis=0)[
            points.shape[0] // 2, dim
        ]
        median = points[median_idx]

        line = np.zeros(points.shape[1] + 1)
        line[dim] = 1
        line[-1] = median[dim]

        is_above_line = points[:, dim] > median[dim]
        is_below_line = points[:, dim] < median[dim]

        P1 = points[is_above_line]
        P2 = points[is_below_line]

        I1 = indices[is_above_line]
        I2 = indices[is_below_line]

        v = Node(median, line, indices[median_idx])
        v.right = self.build_tree_without_triangles(P1, I1, depth + 1)
        v.left = self.build_tree_without_triangles(P2, I2, depth + 1)

        return v

    def intersects_mesh(self, points: np.array):
        return self.troot.check_intersection(points)

    def nearest3(self, points: np.array):
        n = points.shape[0]
        return self.root.nearest3(
            points,
            best_nodes=np.zeros((n, 3)),
            distances=np.full((n, 3), 99999, dtype=float),
        )
