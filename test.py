import numpy as np

# Example input shapes
num_points = 2
num_triangles = 3

triangles = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]])  # shape: (2, 3, 3)
points = np.array([[0.5, 0.5, 0.25], [-0.5, -0.5, -1]])  # shape: (2, 3)

v0, v1, v2 = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
proj_v0 = v0[:, :2]
edge1 = v1[:, :2] - v0[:, :2]
edge2 = v2[:, :2] - v0[:, :2]


dot00 = np.einsum("ij,ij->i", edge1, edge1)
dot01 = np.einsum("ij,ij->i", edge1, edge2)
dot11 = np.einsum("ij,ij->i", edge2, edge2)

inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)

vp = points[:, np.newaxis, :2] - v0[:, :2]

dot20 = np.einsum("ijk,jk->ij", vp, edge1)
dot21 = np.einsum("ijk,jk->ij", vp, edge2)
# print(edge1.shape, vp.shape, dot20.shape)

u = (dot11 * dot20 - dot01 * dot21) * inv_denom
v = (dot00 * dot21 - dot01 * dot20) * inv_denom
w = 1 - u - v

p = v0[:, 2] * u + v1[:, 2] * v + v2[:, 2] * w
print(p.shape, points.shape)
valid = (p.T > points[:, 2]).T

# valid = np.ones_like(u, dtype=bool)
inside = (u >= 0) & (v >= 0) & (w >= 0) & valid

print(u[0][0])
print(v[0][0])
print(w[0][0])


print(inside)