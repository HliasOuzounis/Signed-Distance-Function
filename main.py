import open3d as o3d

from src.mesh_constructor import MeshConstructor
from src.plane_constructor import PlaneConstructor
from src.points_constructor import PointsConstructor
from src.constants import *

mesh = o3d.io.read_triangle_mesh("models/GnomeMesh.ply")
print(f"Mesh: self-intersecting ({mesh.is_self_intersecting()}), edge-manifold ({mesh.is_edge_manifold()}), vertex-manifold ({mesh.is_vertex_manifold()}), watertight ({mesh.is_watertight()})")
print(f"Triangles: {len(mesh.triangles)}")


class Window:
    def __init__(self, mesh) -> bool:
        super().__init__()
        self.mesh = mesh

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.init_window()

    def init_window(self):
        self.vis.create_window()
        self.camera = self.vis.get_view_control()
        
        # Task 1: Create the mesh
        meshConstructor = MeshConstructor(self.vis, self.mesh)
        self.vis.register_key_action_callback(next_key, meshConstructor)
        # Task 2: Create a plane and uniformly select perpendicualr lines. Calculate if the lines intersect the mesh
        planeConstructor = PlaneConstructor(self.vis, meshConstructor.mesh)
        pointsConstructor = PointsConstructor(self.vis, meshConstructor.mesh, planeConstructor.plane)
        
        meshConstructor.next_animation = planeConstructor
        planeConstructor.next_animation = pointsConstructor
        
        self.vis.run()
        self.vis.destroy_window()

Window(mesh)
