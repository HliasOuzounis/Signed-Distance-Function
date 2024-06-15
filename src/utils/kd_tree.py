import numpy as np

from .triangle_params import TriangleParams2D

from abc import ABC, abstractmethod

class Node(ABC):
    @abstractmethod
    def check_intersections(self, points: np.array, count_intersections: bool) -> np.array:
        pass
    
    def draw(self, scene, iterations, z, inv_rot_mat, bounds_x, bounds_y) -> None:
        pass
    
    def clear(self, scene) -> None:
        pass
    
class KDNode(Node):
    def __init__(self, value: np.array, line: np.array) -> None:
        self.value = value
        self.line = line

        self.left = None
        self.right = None
        
        self.line_name = None

    def check_intersections(self, points: np.array, count_intersections: bool) -> np.array:
        if points.shape[0] == 0:
            return np.zeros(0)

        on_right = self.is_on_right(points)

        interescts_right = np.zeros_like(points[:, 0])
        if self.right is not None:
            interescts_right[on_right] = self.right.check_intersections(points[on_right], count_intersections)

        interescts_left = np.zeros_like(points[:, 0])
        if self.left is not None:
            interescts_left[~on_right] = self.left.check_intersections(points[~on_right], count_intersections)

        return interescts_right + interescts_left

    def is_on_right(self, points: np.array):
        points = points[:, :2]
        return np.dot(points, self.line[:2]) > self.line[2]
    
    def draw(self, scene, iterations, z, inv_rot_mat, bounds_x=(-0.5, 0.5), bounds_y=(-0.5, 1.0)):
        import vvrpywork.shapes as shapes
        
        def draw_self(scene, z, inv_rot_mat, bounds_x, bounds_y, shapes):
            if self.line[0]:
                start = np.dot(inv_rot_mat, np.array([self.line[2], bounds_y[0], z]))
                end = np.dot(inv_rot_mat, np.array([self.line[2], bounds_y[1], z]))
            else:
                start = np.dot(inv_rot_mat, np.array([bounds_x[0], self.line[2], z]))
                end = np.dot(inv_rot_mat, np.array([bounds_x[1], self.line[2], z]))
                    
            line = shapes.Line3D(start, end)
            self.line_name = "".join([str(x) for x in start] + [str(x) for x in end])
            scene.addShape(line, self.line_name)
            return
        
        def draw_children(scene, iterations, z, inv_rot_mat, bounds_x, bounds_y):
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

        if iterations == 0:
            draw_self(scene, z, inv_rot_mat, bounds_x, bounds_y, shapes)
        else:
            draw_children(scene, iterations, z, inv_rot_mat, bounds_x, bounds_y)

    def clear(self, scene):
        if self.line_name is not None:
            scene.removeShape(self.line_name)
            
        if self.right is not None:
            self.right.clear(scene)
        if self.left is not None:
            self.left.clear(scene)


class Leaf(Node):
    def __init__(self, points, triangles) -> None:
        self.triangle_params = TriangleParams2D(triangles, points)
    
    def check_intersections(self, points: np.array, count_intersections: bool) -> np.array:
        return self.triangle_params.check_points(points, count_intersections)


class KDTree:
    def __init__(self, dimensions) -> None:
        self.all_points = np.empty((0, 3))
        self.troot = None
        self.inv_rot_mat = np.eye(3)
        self.dimensions = dimensions
        self.is_built = False
    
    def build_tree(self, points, triangles, inv_rot_mat):
        self.inv_rot_mat = inv_rot_mat
        self.all_points = np.dot(points, inv_rot_mat)
        self.troot = self._build_tree(self.all_points, triangles)
        self.is_built = True

    def _build_tree(self, points, triangles, depth=0) -> Node:
        if points.shape[0] == 0:
            return Leaf(self.all_points, triangles)

        dim = depth % self.dimensions
        median_idx = np.argpartition(points, points.shape[0] // 2, axis=0)[
            points.shape[0] // 2, dim
        ]
        median = points[median_idx]

        line = np.zeros(self.dimensions + 1)
        line[self.dimensions] = median[dim]
        line[dim] = 1

        is_above_line = points[:, dim] > median[dim]
        is_below_line = points[:, dim] < median[dim]

        P1 = points[is_above_line]
        P2 = points[is_below_line]

        all_above_line = self.all_points[:, dim] > median[dim]

        triangle_above_line = np.vectorize(lambda x: all_above_line[x])(triangles)
        
        all_below_line = ~triangle_above_line.any(axis=1)
        all_above_line = triangle_above_line.all(axis=1)
        interesting = ~all_below_line & ~all_above_line

        P1_triangles = triangles[all_above_line | interesting]
        P2_triangles = triangles[all_below_line | interesting]

        tv = KDNode(median, line)
        tv.right = self._build_tree(P1, P1_triangles, depth + 1)
        tv.left = self._build_tree(P2, P2_triangles, depth + 1)

        return tv

    def intersects_mesh(self, points: np.array):
        points = np.dot(points, self.inv_rot_mat)
        return self.troot.check_intersections(points, count_intersections=False)

    def is_inside(self, points: np.array):
        points = np.dot(points, self.inv_rot_mat)
        intersections = self.troot.check_intersections(points, count_intersections=True)
        return intersections % 2 == 1
    
    def draw(self, scene, iterations, z):
        self.troot.draw(scene, iterations, z, self.inv_rot_mat)
        
    def clear(self, scene):
        self.troot.clear(scene)