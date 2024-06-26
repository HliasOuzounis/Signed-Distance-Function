import numpy as np

from vvrpywork.shapes import Mesh3D

from .constants import NDArrayNx3, NDArray1D, NDPoint3D, Matrix3x3


def fit_to_unit_sphere(data: Mesh3D) -> Mesh3D:
    """
    Centers and scales the vertices of a mesh or point cloud to fit within a unit sphere.

    Parameters:
    - data: Mesh3D

    Returns:
    - normalized_data: Open3D TriangleMesh or PointCloud object with centered and scaled vertices/points
    """
    if isinstance(data, Mesh3D):
        vertices = data.vertices
    else:
        raise ValueError("Input must be Mesh3D object.")

    # Scale the vertices/points to fit within a unit sphere
    center_point = np.mean(vertices, axis=0)
    vertices -= center_point
    max_distance = np.max(np.linalg.norm(vertices, axis=1))
    vertices_normalized = vertices / max_distance

    # Update the vertices/points of the input data with the normalized vertices/points
    data.vertices = vertices_normalized

    return data


def assign_colors(data: Mesh3D) -> NDArrayNx3:
    """
    Assigns random colors to the vertices of a mesh or point cloud based on the distance of the vertices from the axis.

    Parameters:
    - data: Mesh3D object

    Returns:
    - colored_data: numpy array of RGB colors assigned to each vertex
    """
    if isinstance(data, Mesh3D):
        vertices = data.vertices
        data.use_material = False
    else:
        raise ValueError("Input must be Mesh3D object.")

    colors = np.zeros_like(vertices)
    colors[:, 0] = np.absolute(vertices[:, 0])
    colors[:, 2] = np.absolute(vertices[:, 2])

    return colors


def points_positions_relative_to_line(
    vertices: NDArrayNx3, line_params: NDArray1D
) -> NDArray1D:
    """
    Determine the positions of points relative to a line defined by its parameters.

    Parameters:
    - vertices: np.ndarray with vertex positions
    - line_params: Parameters of the line [a, b, c] in the equation ax + by + c = 0

    Returns:
    - is_above_line: Boolean array indicating whether each point is above the line
    """

    line_params = line_params.reshape(-1, 1)

    vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    is_above_line = np.dot(vertices, line_params) >= 0

    return is_above_line


def rotation_matrix_from_vectors(vec1: NDPoint3D, vec2: NDPoint3D) -> Matrix3x3:
    """
    Find the rotation matrix that aligns vec1 to vec2

    Parameters:
    - vec1: A 3d "source" vector
    - vec2: A 3d "destination" vector

    Returns
    - mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    if np.allclose(vec1, vec2):
        return np.eye(3)
    
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def rotation_matrix_from_plane_vertices(plane_vertices: np.array) -> np.array:
    """
    Get the rotation matrix that aligns the plane normal with the z-axis.

    Parameters:
    - plane_vertices: np.array with the vertices of the plane

    Returns:
    - rotation_matrix: np.array with the rotation matrix
    """
    plane_normal = np.cross(
        plane_vertices[1] - plane_vertices[0],
        plane_vertices[2] - plane_vertices[0],
    )
    plane_normal /= np.linalg.norm(plane_normal)

    return rotation_matrix_from_vectors(plane_normal, np.array([0, 0, 1]))


def closest_point_on_mesh(triangle: Matrix3x3, point: NDPoint3D) -> float:
    """
    Calculates the distance between a point and a triangle in 3D space.
    Taken from the Embree library

    Parameters:
    - triangle: np.ndarray with the vertices of the triangle
    - point: np.ndarray with the point in 3D space

    Returns:
    - distance: float, the minimum distance between the point and the triangle
    """
    a, b, c = triangle

    ab = b - a
    ac = c - a
    bc = c - b

    ap = point - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:  # 1 (A)
        return a

    bp = point - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    # Project ac and ab on bp. They are equal if bp is perpendicular to bc.
    if d3 >= 0.0 and d4 <= d3:  # 2 (B)
        return b

    cp = point - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d5 >= 0.0 and d5 <= d6:  # 3 (C)
        return c

    # No idea why it works from this point
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:  # 4 (Projection of point on AB)
        v = d1 / (d1 - d3)
        return a + v * ab

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:  # 5 (Projection of point on AC)
        v = d2 / (d2 - d6)
        return a + d2 * ac

    va = d3 * d6 - d5 * d4
    if (
        va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0
    ):  # 6 (Projection of point on BC)
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + v * bc

    # Else, the point is inside the triangle
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + v * ab + w * ac


def show_fps(func):
    from time import time

    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        clear_line()
        print(f"Displaying at {1/(t2-t1):.2f} FPS", end="\r")
        return result

    return wrap_func


def clear_line() -> None:
    import sys

    sys.stdout.write("\033[K")  # Clear to the end of line
    sys.stdout.flush()
