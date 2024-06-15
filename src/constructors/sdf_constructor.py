import numpy as np

from vvrpywork.shapes import PointSet3D, Mesh3D

from ..utils import KDTree, SDF
from ..utils import utility

from .callback import Callback, SequenceHandler, Scene3D


class SDFConstructor(Callback):
    def __init__(self, mesh: Mesh3D, kdTree2D: KDTree) -> None:
        super().__init__()
        self.mesh = mesh

        self.total_points = 30 ** 3
        point_per_axis = int(self.total_points ** (1 / 3)) + 1
        self.total_points = point_per_axis ** 3
        self.step = 1 / 100

        offset = 0.01

        xmin = np.min(self.mesh.vertices[:, 0]) - offset
        xmax = np.max(self.mesh.vertices[:, 0]) + offset
        x = np.linspace(xmin, xmax, point_per_axis)

        ymin = np.min(self.mesh.vertices[:, 1]) - offset
        ymax = np.max(self.mesh.vertices[:, 1]) + offset
        y = np.linspace(ymin, ymax, point_per_axis)

        zmin = np.min(self.mesh.vertices[:, 2]) - offset
        zmax = np.max(self.mesh.vertices[:, 2]) + offset
        z = np.linspace(zmin, zmax, point_per_axis)

        xx, yy, zz = np.meshgrid(x, y, z)
        self.grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        self.grid = (x, y, z)

        self.grid_clouds = {}
        
        self.sdf = SDF(self.grid)
        
        self.kdTree2D = kdTree2D

    def animate_init(self) -> None:
        self.clear(self.sequence, self.scene)
        
        self.l = 0
        self.prev_index = 0
        
        self.distances = np.zeros((self.total_points, 1))
        
        if not self.kdTree2D.is_built:
            self.kdTree2D.build_tree(self.mesh.vertices, self.mesh.triangles, inv_rot_mat=np.eye(3))

    @utility.show_fps
    def animate(self) -> bool:    
        self.l += self.step

        if self.l > self.limit:
            self.sdf.build(self.distances.reshape((
                self.grid[0].shape[0], 
                self.grid[1].shape[0], 
                self.grid[2].shape[0])
            ))
            self.stop_animate()

        index = int(self.grid_points.shape[0] * self.l)

        inside = self.kdTree2D.is_inside(self.grid_points[self.prev_index : index + 1])
        
        # self.distances[self.prev_index : index + 1] = self.calulate_distanes(self.grid_points[self.prev_index : index + 1])
        # self.distances[inside] *= -1

        grid_colors = np.zeros((inside.shape[0], 3))
        grid_colors[inside] = np.array([[0, 0, 1]])
        grid_colors[~inside] = np.array([[1, 0, 0]])

        grid_cloud = PointSet3D(self.grid_points[self.prev_index : index + 1], size=0.5)
        grid_cloud.colors = grid_colors
        name = f"grid_cloud_{self.prev_index}_{index}"
        self.grid_clouds[name] = grid_cloud

        self.scene.addShape(grid_cloud, name)
        self.prev_index = index

        return True

    def calulate_distanes(self, points: np.array) -> np.array:
        mesh_vertices = self.mesh.vertices
        
        points = points[:, np.newaxis, :]
        point_to_vertices_distances = np.linalg.norm(points - mesh_vertices, axis=2)

        distances = np.inf * np.ones((points.shape[0], 1))
        for triangle in self.mesh.triangles:
            distances = utility.distance_to_triangle(mesh_vertices[triangle], points)

        return distances
    
    def clear(self, _sequence: SequenceHandler, scene: Scene3D) -> bool:
        for key in self.grid_clouds:
            scene.removeShape(key)

        self.grid_clouds = {}

        return True