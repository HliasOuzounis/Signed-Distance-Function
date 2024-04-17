import open3d as o3d

from src.callback import empty_call
from src.mesh_constructor import MeshConstructor
from src.plane_constructor import PlaneConstructor
from src.points_constructor import PointsConstructor
from src.constants import *

from vvrpywork import scene

mesh = o3d.io.read_triangle_mesh("models/GnomeMesh.ply")
print(f"Mesh: self-intersecting ({mesh.is_self_intersecting()}), edge-manifold ({mesh.is_edge_manifold()}), vertex-manifold ({mesh.is_vertex_manifold()}), watertight ({mesh.is_watertight()})")
print(f"Triangles: {len(mesh.triangles)}")


WIDTH = 800
HEIGHT = 800
NAME = "Signed Distance Function"
class Window(scene.Scene3D):
    def __init__(self, mesh) -> None:
        super().__init__(WIDTH, HEIGHT, NAME)
        self.mesh = mesh
        self.sequenceHandler = SequenceHandler(self.window)
        self.window.set_on_key(self.sequenceHandler.perform_action)
        self.init_window()
        
        self.mainLoop()

    def init_window(self):
        self.scene_widget.scene.show_axes(False)
        # Task 1: Create the mesh
        # meshConstructor = MeshConstructor(self.window, self.mesh)
        # self.next_animation = meshConstructor
        return
        # Task 2: Create a plane and uniformly select perpendicualr lines. Calculate if the lines intersect the mesh
        planeConstructor = PlaneConstructor(self.vis, meshConstructor.mesh)
        pointsConstructor = PointsConstructor(self.vis, meshConstructor.mesh, planeConstructor.plane)

        meshConstructor.next_animation = planeConstructor
        planeConstructor.next_animation = pointsConstructor

        # self.window.run()
        # self.vis.destroy_window()

class SequenceHandler:
    def __init__(self, window) -> None:
        self.window = window

        self._next_animation = empty_call
        self._curr_animation = empty_call
        self._prev_animation = empty_call
        self._skip_animation = empty_call

    def perform_action(self, key_event):
        key, action = key_event.key, key_event.type
        if not action == o3d.visualization.gui.KeyEvent.DOWN or key_event.is_repeat:
            return
        
        if key == o3d.visualization.gui.KeyName.ENTER:
            print("enter pressed (next animation)")
            self.next_animation(self, self.window)
            
        if key == o3d.visualization.gui.KeyName.R:
            print("r pressed (restart current animation)")
            self.curr_animation(self, self.window)
            
        if key == o3d.visualization.gui.KeyName.U:
            print("Not implemented yet")
            
        # if key == o3d.visualization.gui.KeyName.S:
        #     print("s pressed (skip current animation)")
        #     self.skip_animation(self, self.window)
            
    @property
    def next_animation(self):
        return self._next_animation
    
    @next_animation.setter
    def next_animation(self, next):
        self._next_animation = next
    
    @property
    def curr_animation(self):
        return self._curr_animation
    
    @curr_animation.setter
    def curr_animation(self, curr):
        self._curr_animation = curr
        
    @property
    def prev_animation(self):
        return self._prev_animation
    
    @prev_animation.setter
    def prev_animation(self, prev):
        self._prev_animation = prev
        
    @property
    def skip_animation(self):
        return self._skip_animation
    
    @skip_animation.setter
    def skip_animation(self, skip):
        self._skip_animation = skip

if __name__ == "__main__":
    w = Window(mesh)
