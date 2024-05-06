from .callback import Callback

class ClearCallback(Callback):
    def __init__(self, *callbacks) -> None:
        super().__init__()
        self.callbacks = callbacks


    def animate(self) -> bool:
        for callback in self.callbacks:
            callback.clear(callback.sequence, callback.scene)
        
        self.stop_animate()
        return True
