import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .constants import NDArrayNx3, NDArray1D, NDPoint3D

class SDF:
    def __init__(self, grid):
        self.grid = grid
        
    def build(self, signal):
        self.interpolator = RegularGridInterpolator(
            self.grid,
            signal,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        print()
        print("Interpolator built")

    def ray_marching(self, points: NDArrayNx3, direction: NDPoint3D) -> NDArray1D:
        points = points.copy() # comment out to see where points end up in ray_marching
        intersects = np.zeros(points.shape[0], dtype=bool)
        uncertain = np.ones(points.shape[0], dtype=bool)

        distances = self.interpolator(points[uncertain])

        d = 1e-5
        distance_limit = 1
        
        while np.any(uncertain):
            new_points = points[uncertain] + direction * distances[uncertain].reshape(-1, 1)
            points[uncertain] = new_points
            new_distances = self.interpolator(new_points)

            intersects[uncertain] = new_distances < d

            # increased = new_distances > distances[uncertain]
            outside = new_distances > distance_limit
            distances[uncertain] = new_distances

            uncertain[uncertain] &= ~(outside | intersects[uncertain])
            
        return intersects


def _test():
    def f(x, y, z):
        # return 2 * x**2 + 3 * y - z
        return x + y + z

    x = np.linspace(0, 1, 23)
    y = np.linspace(0, 1, 23)
    z = np.linspace(0, 1, 23)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    signal = f(xx, yy, zz)

    sdf = SDF((x, y, z))
    sdf.build(signal)

    p = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    p = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    for i, j in zip(signal, f(p[:, 0], p[:, 1], p[:, 2]).reshape(23, 23, 23)):
        assert np.allclose(i, j)
    
    print(p[10])


if __name__ == "__main__":
    _test()
