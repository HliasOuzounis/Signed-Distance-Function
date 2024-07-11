import numpy as np

from vvrpywork.shapes import PointSet3D, Mesh3D, Sphere3D

from .callback import Callback, SequenceHandler, Scene3D

from ..utils.constants import NDArrayNx3, NDArray1D

class SphereConstructor(Callback):
    def __init__(self, grid: NDArrayNx3, distances: NDArray1D) -> None:
        super().__init__()
        self.grid = grid
        self.distances = distances

        self.step = 1 / 100

        self.spheres = {}

    def animate_init(self) -> None:
        self.clear(self.sequence, self.scene)
        
        self.l = 0
        self.prev_index = 0
        
        self.inside = (self.distances < 0).reshape(-1)
        self.prev_index = 0

    def animate(self) -> bool:
        self.l += self.step

        if self.l > self.limit:
            self.stop_animate()
            return False
        
        index = int(self.l * self.inside.shape[0])
        distances = self.distances[self.inside][self.prev_index: index + 1]
        points = self.grid[self.inside][self.prev_index: index + 1]
        
        for point, dist in zip(points[:100], distances[:100]):
            sphere = Sphere3D(point, dist[0])
            name = f"sphere_{tuple(point)}"
            # print(name)
            self.spheres[name] = sphere
            self.scene.addShape(sphere, name)
        
        self.prev_index = index + 1
        
        return True

    def clear(self, _sequence: SequenceHandler, scene: Scene3D) -> bool:
        for key in self.spheres:
            scene.removeShape(key)

        self.spheres = {}
        return True
