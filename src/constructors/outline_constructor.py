from .callback import Callback

from vvrpywork.shapes import PointSet3D

from ..utils.delaunay import get_delaunay

class OutlineConstructor(Callback):
    def __init__(self, projected_points: PointSet3D) -> None:
        super().__init__()
        self.projected_points = projected_points
        
    def animate_init(self) -> None: ...

    def animate(self) -> bool:
        return
    