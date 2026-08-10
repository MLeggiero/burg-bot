"""Timed blending between named expressions.

Owns the base pose that the animation layers are composited on top of. The one
rule worth stating: a new expression always blends from wherever the face
currently *is*, not from the pose it was last told to hold. Interrupting a
half-finished transition is the normal case, not the exception, and starting
the new blend from the stale target is what produces the visible snap that
makes a face look glitchy.
"""

from typing import Optional

from . import expressions
from .easing import clamp, get as get_curve
from .face_state import AmbientMotion, FaceState, blend


class Animator:
    def __init__(self, initial: str = expressions.DEFAULT):
        spec = expressions.get(initial)
        self._from = spec.state.copy()
        self._to = spec.state.copy()
        self._from_ambient = spec.ambient
        self._to_ambient = spec.ambient
        self._elapsed = 0.0
        self._duration = 0.0
        self._curve = get_curve(spec.curve)

        self.name = initial
        self.source = ""
        self.intensity = 1.0

    @property
    def state(self) -> FaceState:
        """The current base pose, part-way through any running blend."""
        if self._duration <= 0.0:
            return self._to.copy()
        t = clamp(self._elapsed / self._duration)
        return blend(self._from, self._to, self._curve(t))

    @property
    def ambient(self) -> AmbientMotion:
        """The idle-motion character to use right now, blended like the pose.

        Crossfading this alongside the pose is what keeps a transition from
        looking like two different animation styles spliced together -- e.g.
        snapping straight from sleepy's slow heavy breathing into happy's
        bounce the instant the blend starts, before the face has even arrived.
        """
        if self._duration <= 0.0:
            return self._to_ambient
        t = clamp(self._elapsed / self._duration)
        return blend(self._from_ambient, self._to_ambient, self._curve(t))

    @property
    def settled(self) -> bool:
        return self._duration <= 0.0 or self._elapsed >= self._duration

    def set_expression(
        self,
        name: str,
        intensity: float = 1.0,
        blend_time: Optional[float] = None,
        curve: Optional[str] = None,
        source: str = "",
        force: bool = False,
    ) -> bool:
        """Begin a blend toward a named expression.

        Returns True if a new blend actually started. Re-requesting the pose
        already showing is a no-op, so a source republishing at 10 Hz does not
        restart the transition on every message and freeze the face part-way.
        """
        spec = expressions.get(name)
        intensity = clamp(intensity)

        if (
            not force
            and name == self.name
            and abs(intensity - self.intensity) < 0.02
            and source == self.source
        ):
            return False

        neutral_spec = expressions.get(expressions.DEFAULT)
        target = spec.state
        target_ambient = spec.ambient
        if intensity < 0.999:
            # Partial intensity mixes toward neutral, so a source can say
            # "mildly worried" without owning the whole face -- ambient motion
            # mixes the same way, or a 30%-confused face would still fidget
            # like full confusion.
            target = blend(neutral_spec.state, target, intensity)
            target_ambient = blend(neutral_spec.ambient, target_ambient, intensity)

        self._from = self.state
        self._from_ambient = self.ambient
        self._to = target.copy()
        self._to_ambient = target_ambient
        self._duration = spec.blend_time if blend_time is None else max(0.0, blend_time)
        self._curve = get_curve(curve or spec.curve)
        self._elapsed = 0.0

        self.name = name
        self.source = source
        self.intensity = intensity
        return True

    def update(self, dt: float) -> None:
        if self._duration > 0.0 and self._elapsed < self._duration:
            self._elapsed += dt
