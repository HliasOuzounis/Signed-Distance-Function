import open3d as o3d

from src.utility import fit_to_unit_sphere
from src.mesh_constructor import MeshConstructor
from src.constants import *

mesh = o3d.io.read_triangle_mesh("models/DuckMesh.ply")
print(f"Mesh: self-intersecting ({mesh.is_self_intersecting()}), edge-manifold ({mesh.is_edge_manifold()}), vertex-manifold ({mesh.is_vertex_manifold()}), watertight ({mesh.is_watertight()})")
print(f"Triangles: {len(mesh.triangles)}")

class Window:
    def __init__(self, mesh) -> bool:
        super().__init__()
        self.mesh = fit_to_unit_sphere(mesh)
        self.i = 0
        self.sign = 1

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.init_window()

    def init_window(self):
        self.vis.create_window()
        self.camera = self.vis.get_view_control()
        
        meshConstructor = MeshConstructor(self.vis, self.mesh)
        self.vis.register_key_action_callback(next_key, meshConstructor)
        meshConstructor.next_animation = meshConstructor
                
        self.vis.run()
        self.vis.destroy_window()

Window(mesh)
