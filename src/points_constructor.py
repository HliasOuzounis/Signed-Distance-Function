import numpy as np

from vvrpywork.shapes import Mesh3D, PointSet3D, Cuboid3D

from .callback import Callback


class PointsConstructor(Callback):
    def __init__(self, mesh: Mesh3D, plane: Mesh3D) -> None:
        super().__init__()
        
        self.mesh = mesh
        self.plane = plane
        
        self.point_cloud = PointSet3D()
        self.point_cloud_name = "points"

        self.total_points = 10000
        self.step = 1 / 100
        
    def animate_init(self) -> None:
        self.l = 0
        
        self.point_cloud.clear()
        self.point_cloud.createRandom(
            Cuboid3D(self.plane.vertices[0], self.plane.vertices[-1]),
            self.total_points,
        )
        self.points = self.point_cloud.points
        self.points_colors = np.zeros((self.total_points, 3))
        self.point_cloud.clear()

        np.random.shuffle(self.points)

        self.scene.removeShape(self.point_cloud_name)
        self.scene.addShape(self.point_cloud, self.point_cloud_name)

    
    def animate(self) -> bool:
        if self.l > 1:
            self.stop_animate()

        index = int(self.l * self.total_points)
            
        self.point_cloud.points = self.points[:index + 1]
        self.point_cloud.colors = self.points_colors[:index + 1]
        self.l += self.step
        
        self.scene.updateShape(self.point_cloud_name)
        
        return True
