import numpy as np

from vvrpywork.shapes import Triangle2D

from .delaunay import get_triangulation

from typing import Generator

from scipy.spatial import Delaunay


def get_outline(
    projected_points: np.array, alpha: float
) -> Generator[None, tuple[int, int], None]:
    """
    Generator that returns indexes of points that define the α-shape outline of the given points.

    Parameters
    - projected_points: np.array The points to compute the α-shape outline from.
    - alpha: float The α value to use for the α-shape.

    Yields
    - Indexes of the verices that belongs to the α-shape outline.
    """
    # delauney_triangles = get_triangulation(projected_points)
    delauney_triangles = Delaunay(projected_points)
    for vertices in delauney_triangles.simplices:
        v1, v2, v3 = vertices
        triangle = Triangle2D(projected_points[v1], projected_points[v2], projected_points[v3])
        if triangle.getCircumCircle().radius < alpha:
            yield (v1, v3)
            yield (v3, v2)
            yield (v2, v1)
