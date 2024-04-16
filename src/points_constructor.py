import open3d as o3d
from open3d.visualization import Visualizer, VisualizerWithKeyCallback
import numpy as np

from .callback import Callback
from . import utility

class PointsConstructor(Callback):
    def __init__(self, vis: Visualizer | VisualizerWithKeyCallback, mesh, plane) -> None:
        super().__init__(vis)
        
        self.mesh = mesh
        self.plane = plane
        
    def animate_init(self) -> None:
        self.points = self.plane.sample_points_uniformly(10000)
        self.points.paint_uniform_color([0, 0, 0])
        self.points_array = np.random.permutation(np.asarray(self.points.points))
        self.index = 0
        
        self.points.points = o3d.utility.Vector3dVector(self.points_array[:self.index])
        self.vis.add_geometry(self.points, reset_bounding_box=False)

        return super().animate_init()
    
    def animate(self, vis):
        if self.index > self.points_array.shape[0]:
            self.stop_animate()
            
        self.points.points = o3d.utility.Vector3dVector(self.points_array[:self.index + 1])
        self.index += 100
        
        return True