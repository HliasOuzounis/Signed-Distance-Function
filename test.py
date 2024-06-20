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
from src.constructors.callback import Callback

from src.utils import KDTree

from vvrpywork import scene, shapes

mesh = shapes.Mesh3D("models/CatMesh.ply")
print(
    f"Mesh: self-intersecting ({mesh._shape.is_self_intersecting()}), edge-manifold ({mesh._shape.is_edge_manifold()}), vertex-manifold ({mesh._shape.is_vertex_manifold()}), watertight ({mesh._shape.is_watertight()})"
)
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
        self.scene_widget.look_at(
            np.zeros(3), np.array([-1, 0.5, 1]), np.array([0, 1, 0])
        )
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
        sdf = testSDF(meshConstructor.mesh)
        meshConstructor.next_animation = sdf


class testSDF(Callback):
    def __init__(self, mesh) -> None:
        super().__init__()
        self.mesh = mesh
        self.kdTree3D = KDTree(dimensions=3)

    def animate_init(self) -> None:
        print("Building KDTree")
        if not self.kdTree3D.is_built:
            self.kdTree3D.build_tree(
                self.mesh.vertices, self.mesh.triangles, inv_rot_mat=np.eye(3)
            )
        print("Built KDTree")

        self.point = [1, 1, 0]

    def animate(self):
        self.stop_animate()

    def stop_animate(self) -> None:
        self.scene.window.set_on_key(self.point_move)
        self.scene.window.set_on_tick_event(None)
        # return super().stop_animate()

    def point_move(self, event):
        self.scene.removeShape("point")
        self.scene.removeShape("closest_point")
        self.scene.removeShape("arrow")
        self.scene.removeShape("center")
        
        key, action = event.key, event.type
        move_step = 0.05

        if key == o3d.visualization.gui.KeyName.A:
            self.point[0] -= move_step
        if key == o3d.visualization.gui.KeyName.D:
            self.point[0] += move_step

        if key == o3d.visualization.gui.KeyName.W:
            self.point[1] += move_step
        if key == o3d.visualization.gui.KeyName.S:
            self.point[1] -= move_step

        if key == o3d.visualization.gui.KeyName.Q:
            self.point[2] -= move_step
        if key == o3d.visualization.gui.KeyName.E:
            self.point[2] += move_step

        point = shapes.Point3D(self.point, color=[0, 1, 0])
        closest_point, dist = self.kdTree3D.closest_point(np.array([self.point]))
        # print(closest_point, dist)
        closest_point = shapes.Point3D(closest_point[0])
        arrow = shapes.Arrow3D(point, closest_point, color=[0.5, 1, 0.5]) 

        self.scene.addShape(point, "point")
        self.scene.addShape(closest_point, "closest_point")
        self.scene.addShape(arrow, "arrow")
        
        center = shapes.Arrow3D(point, np.array([0, 0, 0]), color=[1, 0, 0])
        self.scene.addShape(center, "center")
         
        return True
        
if __name__ == "__main__":
    w = Window(mesh)
