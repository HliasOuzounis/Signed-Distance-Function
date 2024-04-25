from .callback import Callback

class OutlineConstructor(Callback):
    def __init__(self) -> None:
        super().__init__()
        
    def animate_init(self) -> None:
        return super().animate_init()
        
    def animate(self) -> bool:
        return super().animate()
    