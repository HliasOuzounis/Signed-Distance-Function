import numpy as np

from typing_extensions import Annotated

NDArray1D = Annotated[np.ndarray, "shape=(n,)"]
NDArrayNx2 = Annotated[np.ndarray, "shape=(n, 2)"]
NDArrayNx3 = Annotated[np.ndarray, "shape=(n, 3)"]

Matrix3x3 = Annotated[np.ndarray, "shape=(3, 3)"]

NDPoint3D = Annotated[np.ndarray, "shape=(3,)"]
