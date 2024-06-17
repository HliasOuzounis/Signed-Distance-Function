import numpy as np

from typing_extensions import Annotated

NDArray = np.ndarray

NDArray1D = Annotated[NDArray, "shape=(n,)"]
NDArrayNx2 = Annotated[NDArray, "shape=(n, 2)"]
NDArrayNx3 = Annotated[NDArray, "shape=(n, 3)"]

Matrix3x3 = Annotated[NDArray, "shape=(3, 3)"]

NDPoint3D = Annotated[NDArray, "shape=(3,)"]

