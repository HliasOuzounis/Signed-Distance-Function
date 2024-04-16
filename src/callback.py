from abc import ABC, abstractmethod

from open3d.visualization import Visualizer, VisualizerWithKeyCallback

from .constants import *


class Callback(ABC):
    def __init__(self, vis: Visualizer|VisualizerWithKeyCallback, *args, **kwargs) -> None:
        self.vis = vis
        self._next_animation = empty_call


    def __call__(self, vis: Visualizer|VisualizerWithKeyCallback, key: int|None = None, action: int|None = None) -> bool:
        """
        The callback function that will be called when in the callback loop or when the key is pressed.
        Override if more complex functionality is needed.
        
        Params:
            - vis: The visualizer object
            - key: State of the key if it was pressed
            - action: Additional mods (ctrl, shift, etc.) if the key was pressed

        Return: 
            True if UpadteGeometry() needs to be run else False
        """
        # We only want to run the init function once, if the key is pressed don't run it again
        if key is not None and key != 1:
            return False

        self.animate_init()
        self.vis.register_animation_callback(self.animate)
        
        return True
    
    def stop_animate(self):
        """
        Function to stop the animation.
        """
        self.vis.register_animation_callback(None)
        self.vis.register_key_action_callback(next_key, self.next_animation)\
    
    def animate_init(self) -> None:
        """
        Initialize the animation. Will be called when __call__ is called.
        """
        return
        

    @abstractmethod
    def animate(self, vis):
        """
        The animation function that will be put in the callback loop after the key is pressed.
        Needs to have an end condition and call self.stop_animate to stop the animation.
        """
        raise NotImplementedError("Animation not implemented")
    
    @property
    def next_animation(self):
        """
        The next function to be called after the current one is done.
        """
        return self._next_animation

    @next_animation.setter
    def next_animation(self, next):
        if not isinstance(next, Callback):
            raise TypeError("next_animation must be a Callback object")
        self._next_animation = next
        


def empty_call(vis, *args):
    return False
