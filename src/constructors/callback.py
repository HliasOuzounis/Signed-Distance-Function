from abc import ABC, abstractmethod

from vvrpywork.scene import Scene3D

from ..sequence_handler import SequenceHandler, empty_call


class Callback(ABC):
    def __init__(self) -> None:
        self._next_animation = empty_call
        self.l = 0
        self.limit = 1 + 0.00001 # to avoid floating point errors

    def __call__(self, sequence: SequenceHandler, scene: Scene3D) -> None:
        """
        The callback function that will be called when in the callback loop or when the key is pressed.
        Override if more complex functionality is needed.
        
        Params:
            - window: The window that the callback is attached to
            - key: State of the key if it was pressed
            - action: Additional mods (ctrl, shift, etc.) if the key was pressed
        """
        self.sequence = sequence
        self.scene = scene

        self.animate_init()
        self.scene.window.set_on_tick_event(self.animate)
        sequence.next_animation = self.skip
        sequence.curr_animation = self

    def animate_init(self) -> None:
        """
        Initialize the animation. Will be called when __call__ is called.
        """
        return

    def stop_animate(self) -> None:
        """
        Function to stop the animation.
        """

        self.scene.window.set_on_tick_event(None)
        self.sequence.next_animation = self.next_animation

    def skip(self, _sequence: SequenceHandler, _scene: Scene3D) -> None:
        """
        Skip the current animation
        """
        self.l = 1
        return

    # @abstractmethod
    def clear(self, _sequence: SequenceHandler, _scene: Scene3D) -> bool:
        """
        Clear the current animation
        """
        return False

    @abstractmethod
    def animate(self) -> bool:
        """
        The animation function that will be put in the callback loop after the key is pressed.
        Needs to have an end condition and call self.stop_animate to stop the animation.

        Returns:
            - bool: True if a redraw is needed, False otherwise
        """
        raise NotImplementedError("Animation not implemented")

    @property
    def next_animation(self):
        """
        The next function to be called after the current one is done.
        """
        return self._next_animation

    @next_animation.setter
    def next_animation(self, next_anim):
        if not callable(next_anim):
            raise TypeError("next_animation must be callable")
        self._next_animation = next_anim
