import open3d as o3d

class SequenceHandler:    
    def __init__(self, window, scene) -> None:
        self.window = window
        self.scene = scene

        self._next_animation = empty_call
        self._curr_animation = empty_call
        self._prev_animation = empty_call
        self._skip_animation = empty_call

    def perform_action(self, key_event):
        key, action = key_event.key, key_event.type
        if not action == o3d.visualization.gui.KeyEvent.DOWN or key_event.is_repeat:
            return

        if key == o3d.visualization.gui.KeyName.ENTER:
            print("enter pressed (next animation)")
            self.next_animation(self, self.scene)

        if key == o3d.visualization.gui.KeyName.R:
            print("r pressed (restart current animation)")
            self.curr_animation(self, self.scene)

        if key == o3d.visualization.gui.KeyName.U:
            print("Not implemented yet")

        # if key == o3d.visualization.gui.KeyName.S:
        #     print("s pressed (skip current animation)")
        #     self.skip_animation(self, self.scene)

    @property
    def next_animation(self):
        return self._next_animation

    @next_animation.setter
    def next_animation(self, next):
        self._next_animation = next

    @property
    def curr_animation(self):
        return self._curr_animation

    @curr_animation.setter
    def curr_animation(self, curr):
        self._curr_animation = curr

    @property
    def prev_animation(self):
        return self._prev_animation

    @prev_animation.setter
    def prev_animation(self, prev):
        self._prev_animation = prev

    @property
    def skip_animation(self):
        return self._skip_animation

    @skip_animation.setter
    def skip_animation(self, skip):
        self._skip_animation = skip


def empty_call(window, *args):
    return False
