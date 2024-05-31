import numpy as np

from .triangle_params import TriangleParams2D

class TrianglesNode():
    def __init__(self, points, value: np.array, line: np.array, intersecting_triangles: np.array) -> None:
        self.value = value
        self.line = line

        self.left = None
        self.right = None
        
        self.line_name = None
        
        self.empty = intersecting_triangles.shape[0] == 0
        if not self.empty:
            self.intersecting_triangles = TriangleParams2D(intersecting_triangles, points)

    def check_intersection(self, points: np.array, count_intersections=False):
        if points.shape[0] == 0:
            return np.zeros(0)

        if not self.empty:
            intersects = self.intersecting_triangles.check_points(points, count_intersections)
            if not count_intersections:
                intersects = intersects > 0
        else:
            intersects = np.zeros(points.shape[0], dtype=bool)

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
    
    def draw(self, scene, iterations, z, inv_rot_mat, bounds_x=(-0.5, 0.5), bounds_y=(-0.5, 1.0)):
        import vvrpywork.shapes as shapes
        if iterations == 0:
            if self.line[0]:
                start = np.dot(inv_rot_mat, np.array([self.line[2], bounds_y[0], z]))
                end = np.dot(inv_rot_mat, np.array([self.line[2], bounds_y[1], z]))
            else:
                start = np.dot(inv_rot_mat, np.array([bounds_x[0], self.line[2], z]))
                end = np.dot(inv_rot_mat, np.array([bounds_x[1], self.line[2], z]))
                
            line = shapes.Line3D(start, end)
            self.line_name = "".join([str(x) for x in start] + [str(x) for x in end])
            scene.addShape(line, self.line_name)
            
            if not self.empty:
                self.intersecting_triangles.draw(scene, z, inv_rot_mat)
            return

        if self.right is not None:
            if self.line[0]:
                self.right.draw(scene, iterations - 1, z, inv_rot_mat, bounds_x=(self.line[2], bounds_x[1]), bounds_y=bounds_y)
            else:
                self.right.draw(scene, iterations - 1, z, inv_rot_mat, bounds_x=bounds_x, bounds_y=(self.line[2], bounds_y[1]))
        
        if self.left is not None:
            if self.line[0]:
                self.left.draw(scene, iterations - 1, z, inv_rot_mat, bounds_x=(bounds_x[0], self.line[2]), bounds_y=bounds_y)
            else:
                self.left.draw(scene, iterations - 1, z, inv_rot_mat, bounds_x=bounds_x, bounds_y=(bounds_y[0], self.line[2]))

    def clear(self, scene):
        if self.line_name is not None:
            scene.removeShape(self.line_name)
        
        if self.right is not None:
            self.right.clear(scene)
        if self.left is not None:
            self.left.clear(scene)
        if not self.empty:
            self.intersecting_triangles.clear(scene)

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
    
    def draw(self, scene, iterations, z):
        self.troot.draw(scene, iterations, z, self.inv_rot_mat)
        
    def clear(self, scene):
        self.troot.clear(scene)