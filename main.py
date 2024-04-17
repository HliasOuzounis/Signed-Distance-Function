import open3d as o3d

from src.sequence_handler import SequenceHandler
from src.mesh_constructor import MeshConstructor
from src.plane_constructor import PlaneConstructor
from src.points_constructor import PointsConstructor
from src.constants import *

from vvrpywork import scene, shapes

mesh = shapes.Mesh3D("models/DuckMesh.ply")
print(f"Mesh: self-intersecting ({mesh._shape.is_self_intersecting()}), edge-manifold ({mesh._shape.is_edge_manifold()}), vertex-manifold ({mesh._shape.is_vertex_manifold()}), watertight ({mesh._shape.is_watertight()})")
print(f"Triangles: {len(mesh.triangles)}")


WIDTH = 800
HEIGHT = 800
NAME = "Signed Distance Function"
class Window(scene.Scene3D):
    def __init__(self, mesh) -> None:
        super().__init__(WIDTH, HEIGHT, NAME)
        self.mesh = mesh
        self.sequenceHandler = SequenceHandler(self.window, self)
        self.window.set_on_key(self.sequenceHandler.perform_action)
        self.init_window()
        
        self.mainLoop()

    def init_window(self):
        self.scene_widget.scene.show_axes(False)
        # Task 1: Create the mesh
        meshConstructor = MeshConstructor(self.mesh)
        self.sequenceHandler.next_animation = meshConstructor
        return
        # Task 2: Create a plane and uniformly select perpendicualr lines. Calculate if the lines intersect the mesh
        planeConstructor = PlaneConstructor(self.vis, meshConstructor.mesh)
        pointsConstructor = PointsConstructor(self.vis, meshConstructor.mesh, planeConstructor.plane)

        meshConstructor.next_animation = planeConstructor
        planeConstructor.next_animation = pointsConstructor

        # self.window.run()
        # self.vis.destroy_window()


if __name__ == "__main__":
    w = Window(mesh)
