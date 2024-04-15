import open3d as o3d
import numpy as np

def fit_to_unit_sphere(data):
    """
    Centers and scales the vertices of a mesh or point cloud to fit within a unit sphere.

    Parameters:
    - data: Open3D TriangleMesh or PointCloud object

    Returns:
    - normalized_data: Open3D TriangleMesh or PointCloud object with centered and scaled vertices/points
    """
    if isinstance(data, o3d.geometry.TriangleMesh):
        # For a mesh, normalize the vertices
        vertices = np.asarray(data.vertices)
    elif isinstance(data, o3d.geometry.PointCloud):
        # For a point cloud, normalize the points
        vertices = np.asarray(data.points)
    else:
        raise ValueError("Input must be either an Open3D TriangleMesh or PointCloud object.")

    # Scale the vertices/points to fit within a unit sphere
    center_point = np.mean(vertices, axis=0)
    vertices -= center_point
    max_distance = np.max(np.linalg.norm(vertices, axis=1))
    vertices_normalized = vertices / max_distance

    # Update the vertices/points of the input data with the normalized vertices/points
    if isinstance(data, o3d.geometry.TriangleMesh):
        # For a mesh, update the vertices attribute
        data.vertices = o3d.utility.Vector3dVector(vertices_normalized)
    elif isinstance(data, o3d.geometry.PointCloud):
        # For a point cloud, update the points attribute
        data.points = o3d.utility.Vector3dVector(vertices_normalized)

    return data


def assign_colors(data):
    """
    Assigns random colors to the vertices of a mesh or point cloud based on the distance of the vertices from the axis.

    Parameters:
    - data: Open3D TriangleMesh or PointCloud object

    Returns:
    - colored_data: Open3D TriangleMesh or PointCloud object with random vertex colors
    """

    if isinstance(data, o3d.geometry.TriangleMesh):
        # For a mesh, assign colors to the vertices
        vertices = np.asarray(data.vertices)
    elif isinstance(data, o3d.geometry.PointCloud):
        # For a point cloud, assign colors to the points
        vertices = np.asarray(data.points)
    else:
        raise ValueError("Input must be either an Open3D TriangleMesh or PointCloud object.")

    colors = np.zeros_like(vertices)
    colors[:, 0] = np.absolute(vertices[:, 0])
    colors[:, 2] = np.absolute(vertices[:, 2])
    
    colors = o3d.utility.Vector3dVector(colors)

    return colors
