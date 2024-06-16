import numpy as np

from .constants import NDArrayNx3, NDArray1D

class TriangleParams2D:
    def __init__(self, triangles: NDArrayNx3, points: NDArrayNx3) -> None:
        # project onto plane
        triangles = points[triangles]

        self.v0, self.v1, self.v2 = (
            triangles[:, 0, :],
            triangles[:, 1, :],
            triangles[:, 2, :],
        )

        self.edge1 = self.v1[:, :2] - self.v0[:, :2]
        self.edge2 = self.v2[:, :2] - self.v0[:, :2]

        # remove invalid triangles (with 0 area)
        invalid = (
            (np.linalg.norm(self.edge1, axis=1) == 0) | 
            (np.linalg.norm(self.edge2, axis=1) == 0) |
            (self.edge1 == self.edge2).all(axis=1)
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

    def check_points(self, points: NDArrayNx3, count_intersections=False) -> NDArray1D:        
        if points.shape[0] == 0:
            return np.zeros(0)

        vp = points[:, np.newaxis, :2] - self.v0[:, :2]

        dot20 = np.einsum("ijk,jk->ij", vp, self.edge1)
        dot21 = np.einsum("ijk,jk->ij", vp, self.edge2)

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
