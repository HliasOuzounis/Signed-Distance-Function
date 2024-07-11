import open3d as o3d

class SequenceHandler:    
    def __init__(self, window, scene) -> None:
        self.window = window
        self.scene = scene

        self._next_animation = empty_call
        self._curr_animation = empty_call
        self._prev_animation = empty_call

    def perform_action(self, key_event):
        key, action = key_event.key, key_event.type
        if not action == o3d.visualization.gui.KeyEvent.DOWN or key_event.is_repeat:
            return

        if key == o3d.visualization.gui.KeyName.ENTER:
            self.next_animation(self, self.scene)

        if key == o3d.visualization.gui.KeyName.R:
            self.curr_animation(self, self.scene)

        if key == o3d.visualization.gui.KeyName.U:
            raise NotImplementedError("Undo not implemented yet")

    @property
    def next_animation(self):
        return self._next_animation

    @next_animation.setter
    def next_animation(self, next_anim):
        self._next_animation = next_anim

    @property
    def curr_animation(self):
        return self._curr_animation

    @curr_animation.setter
    def curr_animation(self, curr_anim):
        self._curr_animation = curr_anim

    @property
    def prev_animation(self):
        return self._prev_animation

    @prev_animation.setter
    def prev_animation(self, prev_anim):
        self._prev_animation = prev_anim


def empty_call(window, *args):
    return False
