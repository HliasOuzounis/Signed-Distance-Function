import numpy as np

from vvrpywork.shapes import Mesh3D, PointSet3D, Cuboid3D

from .callback import Callback, Scene3D, SequenceHandler
from ..utils import KDTree, TriangleParams2D, SDF
from ..utils import utility


class KDTreeConstructor(Callback):
    def __init__(self, mesh: Mesh3D, plane: Mesh3D, draw=True) -> None:
        super().__init__()

        self.mesh = mesh
        self.plane = plane

        self.kd_tree = KDTree()
        self.iter = 0
        self.total_iters = 15
        
        self.draw = draw

    def animate_init(self) -> None:
        self.clear(self.sequence, self.scene)
        
        self.iter = 0

        plane_normal = np.cross(
            self.plane.vertices[1] - self.plane.vertices[0],
            self.plane.vertices[2] - self.plane.vertices[0],
        )
        plane_normal /= np.linalg.norm(plane_normal)

        rot_mat = (
            utility.rotation_matrix_from_vectors(plane_normal, np.array([0, 0, 1]))
            if not np.isclose(plane_normal[2], 1)
            else np.eye(3)
        )

        inv_rot_mat = rot_mat.T
        self.rotated_plane_verts = np.dot(self.plane.vertices, inv_rot_mat)

        print("Building KD Tree")
        self.kd_tree.build_tree(self.mesh.vertices, self.mesh.triangles, inv_rot_mat)

    def animate(self) -> bool:
        if self.iter == self.total_iters or not self.draw:
            self.stop_animate()
            return False

        self.kd_tree.draw(self.scene, self.iter, self.rotated_plane_verts[0][2])
        self.iter += 1
        
        return True
    
    def clear(self, _sequence: SequenceHandler, scene: Scene3D) -> bool:
        if self.iter != 0:
            self.kd_tree.clear(scene)