import numpy as np

from .triangle_params import TriangleParams2D, TriangleParams3D
from .constants import NDArray1D, NDArrayNx3, Matrix3x3, NDPoint3D
from vvrpywork.scene import Scene3D

from . import utility

from abc import ABC

distances_hash = {}
total_checks = 0

class Node(ABC):
    def check_intersections(
        self, _points: NDArrayNx3, _count_intersections: bool
    ) -> NDArray1D:
        pass

    def draw(
        self,
        _scene: Scene3D,
        _iterations: int,
        _z: float,
        _inv_rot_mat: Matrix3x3,
        _bounds_x: tuple[float, float],
        _bounds_y: tuple[float, float],
    ) -> None:
        pass

    def clear(self, _scene: Scene3D) -> None:
        pass


    def get_closest_points(self, points: NDArrayNx3) -> NDArray1D:
        pass


class KDNode(Node):
    def __init__(self, value: NDPoint3D, line: NDArray1D) -> None:
        self.value = value
        self.line = line

        self.left = None
        self.right = None

        self.line_name = None

    def check_intersections(
        self, points: NDArrayNx3, count_intersections: bool
    ) -> NDArray1D:
        if points.shape[0] == 0:
            return np.zeros(0)

        on_right = self.is_on_right(points)

        interescts = np.zeros_like(points[:, 0])

        if self.right is not None:
            interescts[on_right] = self.right.check_intersections(
                points[on_right], count_intersections
            )

        if self.left is not None:
            interescts[~on_right] = self.left.check_intersections(
                points[~on_right], count_intersections
            )

        return interescts

    def get_closest_points(self, points: NDArrayNx3, closest_points: NDArrayNx3, distances: NDArray1D) -> tuple[NDArrayNx3, NDArray1D]:
        if points.shape[0] == 0:
            return np.zeros((0, 3)), np.zeros(0) 

        on_right = self.is_on_right(points)
        
        if self.right is not None:
            new_closest_points, new_distances = self.right.get_closest_points(points[on_right], closest_points[on_right], distances[on_right])
            improved = new_distances < distances[on_right]
            changed = on_right & improved
            closest_points[changed] = new_closest_points[improved]
            distances[changed] = new_distances[improved]


        if self.left is not None:
            new_closest_points, new_distances = self.left.get_closest_points(points[~on_right], closest_points[~on_right], distances[~on_right])
            improved = new_distances < distances[~on_right]
            changed = ~on_right & improved
            closest_points[changed] = new_closest_points[improved]
            distances[changed] = new_distances[improved]
            
        recheck = distances >= np.abs(self.distance_to_line(points))
        
        if self.right is not None:
            new_closest_points, new_distances = self.right.get_closest_points(points[recheck & on_right], closest_points[recheck & on_right], distances[recheck & on_right])
            improved = new_distances < distances[recheck & on_right]
            changed = recheck & on_right & improved
            closest_points[changed] = new_closest_points[improved]
            distances[changed] = new_distances[improved]

        if self.left is not None:
            new_closest_points, new_distances = self.left.get_closest_points(points[recheck & ~on_right], closest_points[recheck & ~on_right], distances[recheck & ~on_right])
            improved = new_distances < distances[recheck & ~on_right]
            changed = recheck & ~on_right & improved
            closest_points[changed] = new_closest_points[improved]
            distances[changed] = new_distances[improved]
        
        return closest_points, distances
        

    def is_on_right(self, points: NDArrayNx3) -> NDArray1D:
        return self.distance_to_line(points) > 0

    def distance_to_line(self, points: NDArrayNx3) -> NDArray1D:
        points = points[:, : self.line.shape[0] - 1]
        return np.dot(points, self.line[:-1]) - self.line[-1]

    def draw(
        self,
        scene: Scene3D,
        iterations: int,
        z: float,
        inv_rot_mat: Matrix3x3,
        bounds_x: tuple[float, float] = (-0.5, 0.5),
        bounds_y: tuple[float, float] = (-0.5, 1),
    ) -> None:
        import vvrpywork.shapes as shapes

        def draw_self(
            scene: Scene3D,
            z: float,
            inv_rot_mat: Matrix3x3,
            bounds_x: tuple[float, float],
            bounds_y: tuple[float, float],
        ) -> None:
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

        def draw_children(
            scene: Scene3D,
            iterations: int,
            z: float,
            inv_rot_mat: Matrix3x3,
            bounds_x: tuple[float, float],
            bounds_y: tuple[float, float],
        ) -> None:
            if self.left is not None:
                l_bounds_x = (bounds_x[0], self.line[2]) if self.line[0] else bounds_x
                l_bounds_y = bounds_y if self.line[0] else (bounds_y[0], self.line[2])
                self.left.draw(
                    scene,
                    iterations - 1,
                    z,
                    inv_rot_mat,
                    bounds_x=l_bounds_x,
                    bounds_y=l_bounds_y,
                )
            if self.right is not None:
                r_bounds_x = (self.line[2], bounds_x[1]) if self.line[0] else bounds_x
                r_bounds_y = bounds_y if self.line[0] else (self.line[2], bounds_y[1])
                self.right.draw(
                    scene,
                    iterations - 1,
                    z,
                    inv_rot_mat,
                    bounds_x=r_bounds_x,
                    bounds_y=r_bounds_y,
                )

        if iterations == 0:
            draw_self(scene, z, inv_rot_mat, bounds_x, bounds_y)
        else:
            draw_children(scene, iterations, z, inv_rot_mat, bounds_x, bounds_y)

    def clear(self, scene) -> None:
        if self.line_name is not None:
            scene.removeShape(self.line_name)

        if self.right is not None:
            self.right.clear(scene)
        if self.left is not None:
            self.left.clear(scene)


class KDLeaf(Node):
    def __init__(self, points: NDArrayNx3, triangles: NDArrayNx3, is_2D: bool) -> None:
        self.is_empty = triangles.shape[0] == 0
        if not self.is_empty:
            self.triangles = TriangleParams2D(triangles, points) if is_2D else TriangleParams3D(triangles, points)

    def check_intersections(
        self, points: NDArrayNx3, count_intersections: bool
    ) -> NDArray1D:
        if self.is_empty:
            return np.zeros(points.shape[0])
        return self.triangles.find_intersections(points, count_intersections)

    def get_closest_points(self, points: NDArrayNx3, closest_points: NDArrayNx3, distances: NDArray1D) -> tuple[NDArrayNx3, NDArray1D]:
        if self.is_empty:
            return closest_points, distances
        return self.triangles.get_closest_points(points)

class KDTree(Node):
    def __init__(self, dimensions: int) -> None:
        self.all_points = np.empty((0, 3))
        self.troot = None
        self.inv_rot_mat = np.eye(3)
        self.dimensions = dimensions
        self.is_built = False

    def build_tree(self, points, triangles, inv_rot_mat) -> None:
        self.inv_rot_mat = inv_rot_mat
        self.all_points = np.dot(points, inv_rot_mat)
        self.troot = self._build_tree(self.all_points, triangles)
        self.is_built = True

    def _build_tree(
        self, points: NDArrayNx3, triangles: NDArrayNx3, depth: int = 0
    ) -> Node:
        if points.shape[0] == 0:
            return KDLeaf(self.all_points, triangles, self.dimensions == 2)

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

    def intersects_mesh(self, points: NDArrayNx3) -> NDArray1D:
        points = np.dot(points, self.inv_rot_mat)
        return self.troot.check_intersections(points, count_intersections=False)

    def is_inside(self, points: NDArrayNx3) -> NDArray1D:
        points = np.dot(points, self.inv_rot_mat)
        intersections = self.troot.check_intersections(points, count_intersections=True)
        return intersections % 2 == 1

    def draw(self, scene: Scene3D, iterations: int, z: float) -> None:
        self.troot.draw(scene, iterations, z, self.inv_rot_mat)

    def clear(self, scene: Scene3D) -> None:
        self.troot.clear(scene)

    # def get_closest_points(self, points: NDArrayNx3) -> NDArray1D:
    #     global total_checks
    #     points = np.dot(points, self.inv_rot_mat)
    #     # return self.troot.get_closest_points(points)
    #     get_closest_pointss = np.ones(points.shape[0]) * np.inf
    #     a = self.troot.get_closest_points(points, get_closest_pointss)
    #     if points.shape[0] > 0:
    #         print(total_checks / points.shape[0])
    #     total_checks = 0
    #     return a

    def closest_point(self, point: NDArrayNx3) -> NDArrayNx3:
        closest_points = np.zeros((point.shape[0], 3))
        distances = np.ones(point.shape[0]) * np.inf
        return self.troot.get_closest_points(point, closest_points, distances)