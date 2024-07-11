import open3d as o3d
import numpy as np

from vvrpywork.shapes import Point3D, Arrow3D

from .callback import Callback, Scene3D, SequenceHandler
from ..utils import KDTree, TriangleParams3D

class ClosestPointConstructor(Callback):
    def __init__(self, mesh, kdtree: KDTree) -> None:
        super().__init__()
        self.mesh = mesh
        self.kdTree = kdtree

    def animate_init(self) -> None:
        self.params = TriangleParams3D(self.mesh.triangles, self.mesh.vertices)

        if not self.kdTree.is_built:
            self.kdTree.build_tree(
                self.mesh.vertices, self.mesh.triangles, inv_rot_mat=np.eye(3)
            )
        self.point = [1, 1, 0]

    def animate(self):
        self.stop_animate()

    def stop_animate(self) -> None:
        self.scene.window.set_on_key(self.move_point)
        self.scene.window.set_on_tick_event(None)

    def move_point(self, event):
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
        
        if key == o3d.visualization.gui.KeyName.ENTER and action == o3d.visualization.gui.KeyEvent.DOWN:
            super().stop_animate()
            self.scene.window.set_on_key(self.sequence.perform_action)

        return self.show_points()
    
    def show_points(self):
        self.clear(self.sequence, self.scene)
        
        is_inside = self.kdTree.is_inside(np.array([self.point]))
        
        closest_point, dist = self.params.get_closest_points(np.array([self.point]))
        point = Point3D(self.point, color=[0, 0, 1] * dist if is_inside else [1, 0, 0] * dist)
        closest_point = Point3D(closest_point[0], color=[0, 1, 0])
        arrow = Arrow3D(point, closest_point, color=[0.5, 1, 0.5]) 

        self.scene.addShape(point, "point")
        self.scene.addShape(closest_point, "closest_point")
        self.scene.addShape(arrow, "arrow")
        return True

    def clear(self, _sequence: SequenceHandler, scene: Scene3D) -> bool:
        scene.removeShape("point")
        scene.removeShape("closest_point")
        scene.removeShape("arrow")
        return True