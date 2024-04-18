import numpy as np

a = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])

above_5 = a > 5
a[above_5] = np.array([0, 0, 0, 0])
print(a)
