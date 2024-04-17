import open3d as o3d
import numpy as np

from vvrpywork.shapes import Mesh3D

def fit_to_unit_sphere(data):
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


def assign_colors(data):
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