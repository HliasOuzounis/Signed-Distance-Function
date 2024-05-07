import numpy as np

from vvrpywork.shapes import PointSet3D

from .callback import Callback, SequenceHandler, Scene3D

class SDFConstructor(Callback):
    def __init__(self, mesh) -> None:
        super().__init__()
        self.mesh = mesh
        self.nof_points = 10_000
        self.nof_points = int(self.nof_points ** (1/3))

        offset = 0.1

        xmin = np.min(self.mesh.vertices[:, 0]) - offset
        xmax = np.max(self.mesh.vertices[:, 0]) + offset
        x = np.linspace(xmin, xmax, self.nof_points)
        
        ymin = np.min(self.mesh.vertices[:, 1]) - offset
        ymax = np.max(self.mesh.vertices[:, 1]) + offset
        y = np.linspace(ymin, ymax, self.nof_points)
        
        zmin = np.min(self.mesh.vertices[:, 2]) - offset
        zmax = np.max(self.mesh.vertices[:, 2]) + offset
        z = np.linspace(zmin, zmax, self.nof_points)
        
        xx, yy, zz = np.meshgrid(x, y, z)
        self.grid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        
        self.grid_cloud_name = "grid"
        
    def animate_init(self) -> None:
        self.grid_cloud = PointSet3D(np.zeros((1, 3)), size=1)

        self.scene.removeShape(self.grid_cloud_name)
        self.scene.addShape(self.grid_cloud_name)
        
        
# pc = PointSet3D(np.array([[0, 0, 0]]), size=1)
# # for i, (ci, ins) in enumerate(zip(c, is_inside)):
# for i, ins in enumerate(is_inside):
#     if not ins:
#         continue
#     pc.add(Point3D(grid[i], color=[1, 0, 0] if not ins else [0, 0, 1]))
# self.scene.addShape(pc)
