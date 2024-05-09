import numpy as np


class TriangleParams2D:
    def __init__(self, triangles, points) -> None:
        # project onto plane
        self.triangles_d = np.max(points[:, 2][triangles], axis=1)
        triangles = points[triangles]

        self.v0, self.v1, self.v2 = (
            triangles[:, 0, :],
            triangles[:, 1, :],
            triangles[:, 2, :],
        )
        self.proj_v0 = self.v0[:, :2]
        self.edge1 = self.v1[:, :2] - self.v0[:, :2]
        self.edge2 = self.v2[:, :2] - self.v0[:, :2]

        self.dot00 = np.einsum("ij,ij->i", self.edge1, self.edge1)
        self.dot01 = np.einsum("ij,ij->i", self.edge1, self.edge2)
        self.dot11 = np.einsum("ij,ij->i", self.edge2, self.edge2)

        self.inv_denom = 1 / (self.dot00 * self.dot11 - self.dot01 * self.dot01)

    def check_points(self, points, count_intersections=False):
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


class TriangleParams3D:
    def __init__(self, triangles, points) -> None:
        triangles = points[triangles]
        self.v0, self.v1, self.v2 = (
            triangles[:, 0, :],
            triangles[:, 1, :],
            triangles[:, 2, :],
        )

        self.edge1 = self.v1 - self.v0
        self.edge2 = self.v2 - self.v0
        self.edge3 = self.v2 - self.v2

        self.normal = np.cross(self.edge1, self.edge2)
        self.normal /= np.linalg.norm(self.normal, axis=1).reshape(-1, 1)

        self.dot00 = np.einsum("ij,ij->i", self.edge1, self.edge1)
        self.dot01 = np.einsum("ij,ij->i", self.edge1, self.edge2)
        self.dot11 = np.einsum("ij,ij->i", self.edge2, self.edge2)

        self.inv_denom = 1 / (self.dot00 * self.dot11 - self.dot01 * self.dot01)

    def check_barycentric(self, projected_points) -> np.array:
        vp = projected_points - self.v0

        dot20 = np.einsum("ijk,jk->ij", vp, self.edge1)
        dot21 = np.einsum("ijk,jk->ij", vp, self.edge2)

        u = (self.dot11 * dot20 - self.dot01 * dot21) * self.inv_denom
        v = (self.dot00 * dot21 - self.dot01 * dot20) * self.inv_denom
        w = 1 - u - v

        inside = (u >= 0) & (v >= 0) & (w >= 0)
        return inside

    def check_lambda(self, proj_proj_points, to_check) -> np.array:
        pass

    def project_to_plane(self, points) -> np.array:
        # π(P) = P - ((P - V0) · N)N
        points = points[:, np.newaxis, :]
        projected_points = (
            points
            - (np.einsum("ijk,jk->ij", points - self.v0, self.normal))[..., np.newaxis]
            * self.normal
        )
        return projected_points

    def project_to_line(self, projected_points) -> np.array:
        pass

    def distance_to_area(self, points, projected_points) -> np.array:
        # points are (n', 3), projected_points are (n', m, 3)
        # return distances (n', m, 1)
        return np.linalg.norm(points[:, np.newaxis, :] - projected_points, axis=-1)

    def distance_to_edges(self, points) -> np.array:
        pass

    def distance_to_vertices(self, points) -> np.array:
        pass

    def find_min_distance(self, points) -> np.array:
        distance = np.ones((points.shape[0], self.v0.shape[0])) * np.inf # (n, m)

        projected_on_plane = self.project_to_plane(points)
        inside = self.check_barycentric(projected_on_plane)
        print(distance.shape, inside.shape, points.shape, projected_on_plane.shape)
        print(np.any(inside, axis=1)[..., np.newaxis].shape, inside[:, np.newaxis].shape)
        distance[inside] = self.distance_to_area(points[np.any(inside, axis=1), :], projected_on_plane[inside, :])

        ## projected_on_line = self.project_to_line(projected_on_plane[~inside])
        ## valid_lambda = np.zeros_like(inside)
        ## valid_lambda[~inside] = self.check_lambda(projected_on_line, ~inside)
        # else check if proj proj onto edge
        # self.distance_to_edge
        # else self.distance_to_vertices
        return np.min(distance, axis=1)


def _test():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    triangles = np.array([[0, 1, 2], [0, 1, 3]])
    triangle_params = TriangleParams3D(triangles, points)
    test = np.array([[0.5, 0.5, 0.5],[0.1, -0.1, 0.1],])#[-0.5, -0.5, -0.5],])
    
    print(triangle_params.find_min_distance(test))


if __name__ == "__main__":
    _test()
