import numpy as np
from scipy.interpolate import RegularGridInterpolator


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

    def ray_marching(self, points: np.array, direction: np.array) -> np.array:
        intersects = np.zeros(points.shape[0], dtype=bool)
        uncertain = np.ones(points.shape[0], dtype=bool)

        distances = self.interpolator(points[uncertain])

        while np.any(uncertain):
            new_points = points[uncertain] + direction * distances[uncertain] * 0.9
            new_distances = self.interpolator(new_points)

            intersects[uncertain] = new_distances < 0

            increased = new_distances > distances
            uncertain[uncertain] = (~increased) | (new_distances < 0)

            distances[uncertain] = new_distances[uncertain]

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
