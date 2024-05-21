import numpy as np

from vvrpywork.shapes import Mesh3D, PointSet3D, Cuboid3D

from .callback import Callback, Scene3D, SequenceHandler
from ..utils import KDTree, TriangleParams2D, SDF
from ..utils import utility


class PointsConstructor(Callback):
    def __init__(self, mesh: Mesh3D, plane: Mesh3D, *, useKDTree=False, kdTree: KDTree|None=None, useRayMarching=False, sdf: SDF|None=None) -> None:
        super().__init__()

        self.mesh = mesh
        self.plane = plane

        self.useKDTree = useKDTree
        self.useRayMarching = useRayMarching

        self.intersecting = PointSet3D()
        self.intersecting_name = "intersecting points"
        self.non_intersecting = PointSet3D()
        self.non_intersecting_name = "non-intersecting points"

        if self.useKDTree:
            if not isinstance(kdTree, KDTree):
                raise ValueError("KDTree is required for KDTree")
            self.kd_tree = kdTree
            
        if self.useRayMarching:
            if not isinstance(sdf, SDF):
                raise ValueError("SDF function is required for ray marching")
            self.sdf = sdf

        self.total_points = 15000
        self.step = 1 / 100

    def animate_init(self) -> None:
        self.l = 0
        self.prev_index = 0

        plane_normal = np.cross(
            self.plane.vertices[1] - self.plane.vertices[0],
            self.plane.vertices[2] - self.plane.vertices[0],
        )
        plane_normal /= np.linalg.norm(plane_normal)

        self.rot_mat = (
            utility.rotation_matrix_from_vectors(plane_normal, np.array([0, 0, 1]))
            if not np.isclose(plane_normal[2], 1)
            else np.eye(3)
        )
        
        inv_rot_mat = self.rot_mat.T
        rotated_plane_verts = np.dot(self.plane.vertices, inv_rot_mat)

        # Points are constructed in the plane's coordinate system
        # and then rotated back to world coordinate system
        # self.triangle_params = TriangleParams(self.mesh.triangles, np.dot(self.mesh.vertices, inv_rot_mat))
        if self.useKDTree:
            self.kd_tree.build_tree(self.mesh.vertices, self.mesh.triangles, inv_rot_mat)

        # # To visualize the kd tree (keep plane not rotated)
        # self.kd_tree.draw(self.scene, 3, rotated_plane_verts[0][2])
        
        self.intersecting.clear()
        self.intersecting_points = np.empty((0, 3))

        self.non_intersecting.clear()
        self.non_intersecting.createRandom(
            Cuboid3D(
                rotated_plane_verts[0],
                rotated_plane_verts[-1],
            ),
            self.total_points,
        )

        self.scene.addShape(self.non_intersecting, self.non_intersecting_name)
        self.non_intersecting.remove_duplicated_points()
        self.scene.removeShape(self.non_intersecting_name)
        self.random_points = np.dot(self.non_intersecting.points, self.rot_mat)

        self.non_intersecting.clear()
        self.non_intersecting_points = np.empty((0, 3))

        self.scene.removeShape(self.intersecting_name)
        self.scene.addShape(self.intersecting, self.intersecting_name)
        self.scene.removeShape(self.non_intersecting_name)
        self.scene.addShape(self.non_intersecting, self.non_intersecting_name)

    @utility.show_fps
    def animate(self) -> bool:
        # return self.stop_animate()
        self.l += self.step

        if self.l > self.limit:
            self.scene.removeShape(self.non_intersecting_name)
            self.estimate_area()
            self.stop_animate()
            return True

        index = int(self.l * self.total_points)

        # interecting_points = self.triangle_params.check_points(self.random_points[self.prev_index : index + 1])
        if self.useKDTree:
            interecting_points = self.kd_tree.intersects_mesh(  # ~ 2 times faster
                self.random_points[self.prev_index : index + 1]
            )
            interecting_points_indexes = interecting_points > 0

        if self.useRayMarching:
            interecting_points = self.sdf.ray_marching(
                self.random_points[self.prev_index : index + 1], 
                -self.plane_normal
            )

        self.non_intersecting_points = np.concatenate([
            self.non_intersecting_points[(self.prev_index - index) * 4:],
            self.random_points[self.prev_index : index + 1][~interecting_points_indexes]
        ], axis=0)

        self.non_intersecting.points = self.non_intersecting_points
        self.non_intersecting.colors = np.zeros((self.non_intersecting.points.shape[0], 3))

        self.intersecting_points = np.concatenate([
            self.intersecting_points,
            self.random_points[self.prev_index : index + 1][interecting_points_indexes]
        ], axis=0)

        self.intersecting.points = self.intersecting_points            
        self.intersecting.colors = np.zeros((self.intersecting.points.shape[0], 3)) + np.array([1, 0, 0])

        self.scene.updateShape(self.intersecting_name)
        self.scene.updateShape(self.non_intersecting_name)

        self.prev_index = index

        return True

    def estimate_area(self) -> None:
        plane1 = self.plane.vertices[0]
        plane2 = self.plane.vertices[-1]

        diag = plane2 - plane1

        plane_area = np.linalg.norm(diag[[0, 2]]) * diag[1]

        print(f"Area of projection using Monte Carlo: {self.intersecting.points.shape[0] / self.prev_index * plane_area:.4f} units^2")
        print(f"{self.intersecting.points.shape[0]} points out of {self.prev_index} points")

    def clear(self, _sequence: SequenceHandler, scene: Scene3D) -> bool:
        self.l = 0
        scene.removeShape(self.intersecting_name)
        scene.removeShape(self.non_intersecting_name)
        return True        
