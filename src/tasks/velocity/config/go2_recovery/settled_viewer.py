"""Recovery-only viewer that exposes the stationary reset state before acting."""
from mjlab.viewer import ViserPlayViewer


class SettledRecoveryViewer(ViserPlayViewer):
    def setup(self):
        super().setup()
        # The viewer starts paused so the randomized fallen pose is stationary.
        # Explicitly request the initial scene update; otherwise the pause-aware
        # render gate can leave a newly connected browser with an empty canvas.
        self._scene.needs_update = True

    def request_reset(self):
        # Stop policy stepping as soon as the UI callback queues the reset.
        self.pause()
        super().request_reset()

    def reset_environment(self):
        self.pause()
        super().reset_environment()
        self.pause()
