import numpy as np
from scipy.interpolate import RegularGridInterpolator

# from .constants import NDArrayNx3, NDArray1D, NDPoint3D

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
        print(signal.shape)

    # def ray_marching(self, points: NDArrayNx3, direction: NDPoint3D) -> NDArray1D:
    def ray_marching(self, points, direction):
        points = points.copy()
        points += direction * 0.2
        intersects = np.zeros(points.shape[0], dtype=bool)
        uncertain = np.ones(points.shape[0], dtype=bool)

        distances = self.interpolator(points[uncertain])

        d = 1e-5
        
        while np.any(uncertain):
            new_points = points[uncertain] + direction * distances[uncertain].reshape(-1, 1) * 0.9
            points[uncertain] = new_points
            new_distances = self.interpolator(new_points)

            intersects[uncertain] = new_distances < d

            increased = new_distances > distances[uncertain]
            distances[uncertain] = new_distances

            uncertain[uncertain] &= ~(increased | intersects[uncertain])
            
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
    
    for i, j in zip(signal, f(p[:, 0], p[:, 1], p[:, 2]).reshape(23, 23, 23)):
        assert np.allclose(i, j)


if __name__ == "__main__":
    _test()
