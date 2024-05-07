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
        triangles = points[triangles]

        self.v0, self.v1, self.v2 = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
        self.proj_v0 = self.v0[:, :2]
        self.edge1 = self.v1[:, :2] - self.v0[:, :2]
        self.edge2 = self.v2[:, :2] - self.v0[:, :2]


        self.dot00 = np.einsum("ij,ij->i", self.edge1, self.edge1)
        self.dot01 = np.einsum("ij,ij->i", self.edge1, self.edge2)
        self.dot11 = np.einsum("ij,ij->i", self.edge2, self.edge2)

        self.inv_denom = 1 / (self.dot00 * self.dot11 - self.dot01 * self.dot01)

    def check_points(self, points, count_intersections=False):
        vp = points[:, np.newaxis, :2] - self.v0[:, :2]

        dot20 = np.einsum("ijk,jk->ij", vp, self.edge1)
        dot21 = np.einsum("ijk,jk->ij", vp, self.edge2)
        # print(self.edge1.shape, vp.shape, dot20.shape)

        u = (self.dot11 * dot20 - self.dot01 * dot21) * self.inv_denom
        v = (self.dot00 * dot21 - self.dot01 * dot20) * self.inv_denom
        w = 1 - u - v

        valid = np.ones_like(u, dtype=bool)
        if count_intersections:
            # find which intersecting points in 3d are after the start of the ray (going to +z)
            intersection_p = u * self.v0[:, 2] + v * self.v1[:, 2] + w * self.v2[:, 2]
            valid = (intersection_p.T > points[:, 2]).T
            
        inside = (u >= 0) & (v >= 0) & (w >= 0) & valid
        return np.sum(inside, axis=1)




class TrianglesNode():
    def __init__(self, points, value: np.array, line: np.array, intersecting_triangles: np.array) -> None:
        self.value = value
        self.line = line

        self.left = None
        self.right = None

        self.intersecting_triangles = TriangleParams(intersecting_triangles, points)

    def check_intersection(self, points: np.array, count_intersections=False):
        if points.shape[0] == 0:
            return np.zeros(0)

        intersects = self.intersecting_triangles.check_points(points, count_intersections)
        if not count_intersections:
            intersects = intersects > 0

        on_right = self.is_on_right(points)
        
        interescts_right = np.zeros_like(intersects)
        right_condition = on_right if count_intersections else (on_right & ~intersects)
        if self.right is not None:
            interescts_right[right_condition] = self.right.check_intersection(points[right_condition], count_intersections)

        interescts_left = np.zeros_like(intersects)
        left_condition = ~on_right if count_intersections else (~on_right & ~intersects)
        if self.left is not None:
            interescts_left[left_condition] = self.left.check_intersection(points[left_condition], count_intersections)

        return intersects + interescts_right + interescts_left

    def is_on_right(self, points: np.array):
        points = points[:, :2]
        return np.dot(points, self.line[:2]) > self.line[2]


class KDTree:
    def __init__(self) -> None:
        self.all_points = np.empty((0, 3))
        self.troot = None
        self.inv_rot_mat = np.eye(3)
    
    def build_tree(self, points, triangles, inv_rot_mat):
        self.inv_rot_mat = inv_rot_mat
        self.all_points = np.dot(points, inv_rot_mat)
        self.troot = self._build_tree(self.all_points, triangles)

    def _build_tree(self, points, triangles, depth=0) -> TrianglesNode:
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
        tv.right = self._build_tree(P1, P1_triangles, depth + 1)
        tv.left = self._build_tree(P2, P2_triangles, depth + 1)

        return tv
    def intersects_mesh(self, points: np.array):
        points = np.dot(points, self.inv_rot_mat)
        return self.troot.check_intersection(points)

    def is_inside(self, points: np.array):
        points = np.dot(points, self.inv_rot_mat)
        intersections = self.troot.check_intersection(points, count_intersections=True)
        return intersections % 2 == 1