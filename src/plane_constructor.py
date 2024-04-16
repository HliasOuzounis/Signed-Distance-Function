import open3d as o3d
from open3d.visualization import Visualizer, VisualizerWithKeyCallback
import numpy as np

from .callback import Callback

class PlaneConstructor(Callback):
    def __init__(self, vis: Visualizer | VisualizerWithKeyCallback, mesh) -> None:
        super().__init__(vis)
        self.mesh = mesh

        self.camera = self.vis.get_view_control()

        self.mesh_vertices = np.asarray(self.mesh.vertices)
        offset = 0.3
        maxx = np.max(self.mesh_vertices[:, 0]) + offset
        minx = np.min(self.mesh_vertices[:, 0]) - offset
        maxy = np.max(self.mesh_vertices[:, 1]) + offset
        miny = np.min(self.mesh_vertices[:, 1]) - offset
        z    = np.min(self.mesh_vertices[:, 2]) - offset

        self.start1 = np.array([minx, miny, z])
        self.start2 = np.array([maxx, miny, z])

        self.end1 = np.array([minx, maxy, z])
        self.end2 = np.array([maxx, maxy, z])

        self.plane = o3d.geometry.TriangleMesh()

    def animate_init(self):
        self.vis.remove_geometry(self.plane, reset_bounding_box=False)
        
        self.points = np.array([self.start1, self.start2, self.start1, self.start2])
        self.plane.vertices = o3d.utility.Vector3dVector(self.points)
        self.plane.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [3, 2, 1]])
        
        self.plane.paint_uniform_color([0.8, 0.8, 0.8])
        
        self.vis.add_geometry(self.plane, reset_bounding_box=False)

        self.l = 0

        return True

    def animate(self, vis):
        if self.l > 1:
            self.stop_animate()

        self.points[2, :] = self.start1 * (1 - self.l) + self.end1 * self.l
        self.points[3, :] = self.start2 * (1 - self.l) + self.end2 * self.l
        self.l += 0.01

        self.plane.vertices = o3d.utility.Vector3dVector(self.points)

        return True
