import open3d as o3d
import numpy as np

from src.sequence_handler import SequenceHandler
from src.constructors import (
    MeshConstructor,
    PlaneConstructor,
    PointsConstructor,
    OutlineConstructor,
    ClearCallback
)

from src.utils import KDTree

from vvrpywork import scene, shapes

mesh = shapes.Mesh3D("models/DuckMesh.ply")
print(f"Mesh: self-intersecting ({mesh._shape.is_self_intersecting()}), edge-manifold ({mesh._shape.is_edge_manifold()}), vertex-manifold ({mesh._shape.is_vertex_manifold()}), watertight ({mesh._shape.is_watertight()})")
print(f"Triangles: {len(mesh.triangles)}")


WIDTH = 1400
HEIGHT = 900
NAME = "Signed Distance Function"


class Window(scene.Scene3D):
    def __init__(self, mesh) -> None:
        super().__init__(WIDTH, HEIGHT, NAME)
        self.mesh = mesh
        self.sequenceHandler = SequenceHandler(self.window, self)
        self.window.set_on_key(self.sequenceHandler.perform_action)
        self.scene_widget.look_at(np.zeros(3), np.array([-1, 0.5, 1]), np.array([0, 1, 0]))
        self.init_window()

        self.mainLoop()

    def init_window(self):
        # don't need axes or shadows
        self.scene_widget.scene.show_axes(False)
        self.scene_widget.scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS,
            np.array([0.577, -0.577, -0.577]),
        )
        # Task 1: Create the mesh
        meshConstructor = MeshConstructor(self.mesh)
        self.sequenceHandler.next_animation = meshConstructor
        
        # Task 2: Create a plane and uniformly select perpendicualr lines. Calculate if the lines intersect the mesh
        kd_tree = KDTree()
        planeConstructor = PlaneConstructor(meshConstructor.mesh)
        pointsConstructor = PointsConstructor(meshConstructor.mesh, planeConstructor.plane, kd_tree)

        meshConstructor.next_animation = planeConstructor
        planeConstructor.next_animation = pointsConstructor
        
        # Task 3: Calculate outline of the projected points & Task 4: Calculate area of projection
        outlineConstructor = OutlineConstructor(pointsConstructor.intersecting)
        pointsConstructor.next_animation = outlineConstructor
        
        # Clear the scene for part B
        clear = ClearCallback(planeConstructor, pointsConstructor, outlineConstructor)
        outlineConstructor.next_animation = clear
        
        # inf loop
        clear.next_animation = planeConstructor


if __name__ == "__main__":
    w = Window(mesh)
