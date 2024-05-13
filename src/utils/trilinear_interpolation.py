import numpy as np
from scipy.interpolate import RegularGridInterpolator


def trilinear_interpolation(signal: np.array, grid: np.array, points) -> np.array:
    interpolator = RegularGridInterpolator(
        grid,
        signal,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    interpolated_values = interpolator(points)
    return interpolated_values


def _test():
    def f(x, y, z):
        return 2 * x**2 + 3 * y - z

    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    z = np.linspace(0, 1, 10)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    signal = f(xx, yy, zz)
    grid =(x, y, z)
    
    points = np.array([[0.55, 0.55, 0.55], [0.1, 0.1, 0.1]])
    interpolated_values = trilinear_interpolation(signal, grid, points)
    print(interpolated_values)
    print(f(points[:, 0], points[:, 1], points[:, 2]))

if __name__ == "__main__":
    _test()
