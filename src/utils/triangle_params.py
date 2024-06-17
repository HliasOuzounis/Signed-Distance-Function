import numpy as np


from .constants import NDArrayNx3, NDArray1D, NDPoint3D, Matrix3x3, NDArray
from .utility import rotation_matrix_from_vectors


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


class TriangleParams:
    def __init__(self, v1: NDArrayNx3, v2: NDArrayNx3, v3: NDArrayNx3) -> None:
        self.v0, self.v1, self.v2 = v1, v2, v3
        
        self.edge1 = self.v1[:, :2] - self.v0[:, :2]
        self.edge2 = self.v2[:, :2] - self.v0[:, :2]

        # remove invalid triangles (with 0 area)
        invalid = (
            (np.linalg.norm(self.edge1, axis=1) == 0)
            | (np.linalg.norm(self.edge2, axis=1) == 0)
            | (self.edge1 == self.edge2).all(axis=1)
        )

        self.v0 = self.v0[~invalid]
        self.v1 = self.v1[~invalid]
        self.v2 = self.v2[~invalid]
        self.edge1 = self.edge1[~invalid]
        self.edge2 = self.edge2[~invalid]

        self.dot00 = np.einsum("ij,ij->i", self.edge1, self.edge1)
        self.dot01 = np.einsum("ij,ij->i", self.edge1, self.edge2)
        self.dot11 = np.einsum("ij,ij->i", self.edge2, self.edge2)

        self.inv_denom = 1 / (self.dot00 * self.dot11 - self.dot01 * self.dot01)
        
    def get_barycentric_coordinates(self, points: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        vp = points[:, :, :2] - self.v0[:, :2]
        
        dot20 = np.einsum("ijk,jk->ij", vp, self.edge1)
        dot21 = np.einsum("ijk,jk->ij", vp, self.edge2)

        u = (self.dot11 * dot20 - self.dot01 * dot21) * self.inv_denom
        v = (self.dot00 * dot21 - self.dot01 * dot20) * self.inv_denom
        w = 1 - u - v

        return u, v, w


class TriangleParams2D(TriangleParams):
    def __init__(self, triangles: NDArrayNx3, points: NDArrayNx3) -> None:
        # project onto plane
        triangles = points[triangles]
        
        v0, v1, v2 = (
            triangles[:, 0, :],
            triangles[:, 1, :],
            triangles[:, 2, :],
        )
        
        super().__init__(v0, v1, v2)

    def find_intersections(self, points: NDArrayNx3, count_intersections=False) -> NDArray1D:
        if points.shape[0] == 0:
            return np.zeros(0)

        u, v, w = self.get_barycentric_coordinates(points[:, np.newaxis, :])

        valid = np.ones_like(u, dtype=bool)
        if count_intersections:
            # find which intersecting points in 3d are after the start of the ray (going to +z)
            intersection_p = u * self.v0[:, 2] + \
                v * self.v1[:, 2] + w * self.v2[:, 2]
            valid = (intersection_p.T > points[:, 2]).T

        inside = (u >= 0) & (v >= 0) & (w >= 0) & valid
        return np.sum(inside, axis=1)


class TriangleParams3D(TriangleParams):
    def __init__(self, triangles: NDArrayNx3, points: NDArrayNx3) -> None:
        triangles = points[triangles]

        v0, v1, v2 = (
            triangles[:, 0, :],
            triangles[:, 1, :],
            triangles[:, 2, :],
        )

        edge1 = v1 - v0
        edge2 = v2 - v0

        self.normal = np.cross(edge1, edge2)
        self.normal /= np.linalg.norm(self.normal, axis=1)[:, None]

        self.rot_matrices = np.array(
            [self._rotation_matrix(n) for n in self.normal])
        
        self.inv_rot_matrices = np.array([m.T for m in self.rot_matrices])

        v0, v1, v2 = (
            np.einsum("ijk,ik->ij", self.rot_matrices, v0),
            np.einsum("ijk,ik->ij", self.rot_matrices, v1),
            np.einsum("ijk,ik->ij", self.rot_matrices, v2),
        )
        super().__init__(v0, v1, v2)

    def get_distances(self, points: NDArrayNx3) -> NDArray1D:
        if points.shape[0] == 0:
            return np.zeros(0)
        # Rotate points to the triangle's coordinate system
        # p' = R.T @ p for each triangle, for each point
        # Get a (n, m, 3) array of points where n = nof points, m = nof triangles
        # Then, project onto xy plane
        points = np.einsum("mik,nk->nmi", self.rot_matrices, points)
        u, v, w = self.get_barycentric_coordinates(points)
        
        inside = (u >= 0) & (v >= 0) & (w >= 0)
        return np.sum(inside, axis=1)

    def _rotation_matrix(self, n: NDPoint3D) -> Matrix3x3:
        return rotation_matrix_from_vectors(n, [0, 0, 1])


def _test():
    points = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
    triangles = np.array([[0, 1, 2], [0, 1, 3]])

    tp = TriangleParams3D(triangles, points)

    p = np.array([[0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
    print(tp.get_distances(p))

if __name__ == "__main__":
    _test()
