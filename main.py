import open3d as o3d
import numpy as np

from src.sequence_handler import SequenceHandler
from src.constructors import (
    MeshConstructor,
    PlaneConstructor,
    KDTreeConstructor,
    PointsConstructor,
    OutlineConstructor,
    ClearCallback,
    SDFConstructor,
    SphereConstructor,
)

from vvrpywork import scene, shapes

mesh = shapes.Mesh3D("models/sphere.ply")
print(f"Mesh: self-intersecting ({mesh._shape.is_self_intersecting()}), edge-manifold ({mesh._shape.is_edge_manifold()}), vertex-manifold ({mesh._shape.is_vertex_manifold()}), watertight ({mesh._shape.is_watertight()})")
print(f"Triangles: {len(mesh.triangles)}, Vertices: {len(mesh.vertices)}")


WIDTH = 1400
HEIGHT = 900
NAME = "Signed Distance Function"


class Window(scene.Scene3D):
    def __init__(self, mesh) -> None:
        super().__init__(WIDTH, HEIGHT, NAME)
        self.mesh = mesh
        self.sequenceHandler = SequenceHandler(self.window, self)
        self.window.set_on_key(self.sequenceHandler.perform_action)
        # self.scene_widget.look_at(np.array([0, 0, -1]), np.array([-1, 0.6, 0.7]), np.array([0, 1, 0]))
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
        
        from vvrpywork.shapes import LineSet3D
        self.sequenceHandler.next_animation = meshConstructor

        # Task 2: Create a plane and uniformly select perpendicualr lines. Calculate if the lines intersect the mesh
        planeConstructor = PlaneConstructor(meshConstructor.mesh)
        kdTreeConstructor = KDTreeConstructor(meshConstructor.mesh, planeConstructor.plane)
        clearKdTree = ClearCallback(kdTreeConstructor)
        pointsConstructor = PointsConstructor(meshConstructor.mesh, planeConstructor.plane, useKDTree=True, kdTree=kdTreeConstructor.kdTree)

        meshConstructor.next_animation = planeConstructor
        planeConstructor.next_animation = kdTreeConstructor
        kdTreeConstructor.next_animation = clearKdTree
        clearKdTree.next_animation = pointsConstructor

        # Task 3: Calculate outline of the projected points & Task 4: Calculate area of projection
        outlineConstructor = OutlineConstructor(pointsConstructor.intersecting, planeConstructor.plane)
        pointsConstructor.next_animation = outlineConstructor

        # Clear the scene for part B
        clear_partA = ClearCallback(planeConstructor, pointsConstructor, outlineConstructor)
        outlineConstructor.next_animation = clear_partA

        # Task 5: Check if point is inside or outside the mesh
        # Task 6: Calculate min distance to mesh
        # Task 7: Create SDF function
        sdf_constructor = SDFConstructor(meshConstructor.mesh, kdTreeConstructor.kdTree)
        clear_partA.next_animation = sdf_constructor
        
        sphere_constructor = SphereConstructor(sdf_constructor.grid_points, sdf_constructor.distances)
        sdf_constructor.next_animation = sphere_constructor
        clear_sdf_grid = ClearCallback(sdf_constructor, sphere_constructor)
        sphere_constructor.next_animation = clear_sdf_grid
        
        planeConstructorB = PlaneConstructor(meshConstructor.mesh)
        clear_sdf_grid.next_animation = planeConstructorB
        pointsConstructorB = PointsConstructor(meshConstructor.mesh, planeConstructorB.plane, useRayMarching=True, sdf=sdf_constructor.sdf)
        planeConstructorB.next_animation = pointsConstructorB

        # Task 8 & 9: Calculate outline of the projected points & Calculate area of projection 
        outlineConstructorB = OutlineConstructor(pointsConstructorB.intersecting, planeConstructorB.plane)
        pointsConstructorB.next_animation = outlineConstructorB

        # skip to part B
        meshConstructor.next_animation = sdf_constructor

        # Loop part A
        # clear_partA.next_animation = planeConstructor


if __name__ == "__main__":
    w = Window(mesh)
