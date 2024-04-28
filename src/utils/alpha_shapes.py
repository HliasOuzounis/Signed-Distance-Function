import numpy as np

from vvrpywork.shapes import Line3D
from vvrpywork.shapes import Triangle2D, Point2D


from . import utility
from itertools import combinations
from typing import Generator

from scipy.spatial import Delaunay


def get_outline(
    points: np.array, projected_points: np.array, alpha: float
) -> Generator[None, tuple[int, int], None]:
    """
    Generator that returns indexes of points that define the α-shape outline of the given points.

    Parameters
    - points: np.array The points to compute the α-shape outline from.
    - alpha: float The α value to use for the α-shape.

    Yields
    - Indexes of the verices that belongs to the α-shape outline.
    """
        
    # delauney_triangles = _get_triangulation(projected_points)
    delauney_triangles = Delaunay(projected_points)
    for v1, v2, v3 in delauney_triangles.simplices:
        triangle = Triangle2D(projected_points[v1], projected_points[v2], projected_points[v3])
        if triangle.getCircumCircle().radius < alpha:
            yield (v1, v3)
            yield (v3, v2)
            yield (v2, v1)


def _get_triangulation(points: np.array) -> list[Triangle2D]:
    """
    Returns the Delaunay triangulation of the given points.

    Parameters
    - points: np.array The points to compute the Delaunay triangulation from.

    Returns
    - list[Triangle2D] A list of Delaunay triangles.
    """
    minx = np.min(points[:, 0])
    maxx = np.max(points[:, 0])
    miny = np.min(points[:, 1])
    maxy = np.max(points[:, 1])
    
    p1, p2, p3, p4 = (
        Point2D((minx, miny)),
        Point2D((minx, maxy)),
        Point2D((maxx, miny)),
        Point2D((maxx, maxy)),
    )
    # bounding rectangle for the points
    triangles = {
        f"{p1}_{p2}_{p3}": Triangle2D(p1, p2, p3),
        f"{p2}_{p3}_{p4}": Triangle2D(p2, p3, p4),
    }


    for new_point in points:
        print(new_point)
        _add_point(triangles, Point2D(new_point))

    return triangles


def _add_point(triangles_dict: dict[str, Triangle2D], point: Point2D) -> None:
    for key, triangle in triangles_dict.items():
        if triangle.contains(point):
            triangles_dict.pop(key)

            for p1, p2 in combinations(triangle.getPoints(), 2):
                triangle_name = f"{p1}_{p2}_{point}"
                triangles_dict[triangle_name] = Triangle2D(p1, p2, point)
            
            _fix_delauney(triangles_dict)

            return

    raise RuntimeError("Point is not inside any triangle")


def _fix_delauney(triangles_dict: dict[str, Triangle2D]) -> None:
    """
    Fixes the Delaunay triangulation by removing triangles that are not Delaunay.

    Parameters
    - triangles: list[Triangle2D] The Delaunay triangulation to fix.

    Returns
    - list[Triangle2D] The fixed Delaunay triangulation.
    """
    for key, triangle in reversed(triangles_dict.items()):
        is_delauney, adjecent_key = _is_delauney(triangle, triangles_dict)
        if not is_delauney:
            adjecent_triangle = triangles_dict.pop(adjecent_key)

            common_points = [point for point in triangle.getPoints() if point in adjecent_triangle.getPoints()]
            p1 = [point for point in triangle.getPoints() if point not in common_points][0]
            p2 = [point for point in adjecent_triangle.getPoints() if point not in common_points][0]
            
            triangles_dict.pop(key)
            
            for p3 in  common_points:
                triangles_dict[f"{p1}_{p2}_{p3}"] = Triangle2D(p1, p2, p3)
                
            return _fix_delauney(triangles_dict)


def _is_delauney(triangle: Triangle2D, triangles_dict: dict[str, Triangle2D]) -> tuple[bool, str]:
    for p1, p2 in combinations(triangle.getPoints(), 2):
        for key, tri in triangles_dict.items():
            if tri == triangle:
                continue
            tri_points = tri.getPoints()
            
            if p1 in tri_points and p2 in tri_points:
                p3 = [point for point in tri_points if point not in [p1, p2]][0]
                if _violates_Delauney(triangle, p3):
                    return False, key
                
    return True, ""
        


def _violates_Delauney(t: Triangle2D, p: Point2D) -> bool:
    """Checks if `t` is a Delaunay triangle w.r.t `p`."""

    c = t.getCircumCircle()
    c.radius *= (
        0.99  # Shrink the circle a bit in order to exclude points of its circumference.
    )
    if c.contains(p):
        return True
    return False
