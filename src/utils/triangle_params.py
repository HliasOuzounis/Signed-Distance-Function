import numpy as np


class TriangleParams2D:
    def __init__(self, triangles, points) -> None:
        # project onto plane
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
    
    def draw(self, scene, z, inv_rot_mat):
        import vvrpywork.shapes as shapes
        mesh = shapes.Mesh3D(color=(1, 0, 0, 0.8))
        for v1, v2, v3 in zip(self.v0, self.v1, self.v2):
            v1[2] = z + 0.01
            v2[2] = z + 0.01
            v3[2] = z + 0.01

            v1 = np.dot(inv_rot_mat, v1)
            v2 = np.dot(inv_rot_mat, v2)
            v3 = np.dot(inv_rot_mat, v3)
            
            mesh.vertices = np.concatenate((mesh.vertices, [v1, v2, v3]))
            index = len(mesh.vertices) - 3
            mesh.triangles = np.concatenate((mesh.triangles, [[index, index + 1, index + 2]]))
            
        self.mesh_name = "".join([str(x) for x in mesh.vertices.ravel()])
        scene.addShape(mesh, self.mesh_name)

    def clear(self, scene):
        scene.removeShape(self.mesh_name)

def _test(): ...


if __name__ == "__main__":
    _test()
