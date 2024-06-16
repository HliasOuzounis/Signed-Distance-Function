from src.utils.utility import distance_to_triangle
import numpy as np

def _test():
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    point = np.array([1, 0.5, -1])

    print(distance_to_triangle(triangle, point))


if __name__ == "__main__":
    _test()
