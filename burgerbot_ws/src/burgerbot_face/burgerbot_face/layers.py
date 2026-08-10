"""Animation layers composited on top of the blended expression pose.

The base pose alone is a still image. These layers are what make it read as
alive, and they matter more than the keyframes do: a face with two rectangles
for eyes but correct blinking, drift and inertia looks alive, while a
beautifully drawn face that holds perfectly still looks broken.

Each layer is independent and additive, applied in a fixed order by
`Compositor`. Order matters -- blink has to come after the gaze and idle layers
because it overrides lid coverage rather than adding to it.
"""

import math
import random

from .easing import approach, clamp, ease_out_cubic, lerp
from .face_state import AmbientMotion, FaceState

TWO_PI = 2.0 * math.pi


class Layer:
    """Base class. Layers mutate a FaceState in place."""

    def update(self, dt: float) -> None:  # pragma: no cover - trivial
        pass

    def apply(self, state: FaceState) -> None:  # pragma: no cover - trivial
        pass


class BlinkLayer(Layer):
    """Spontaneous blinking.

    Intervals are drawn from an exponential distribution rather than a fixed
    period. A metronomic blink is worse than no blink at all -- the regularity
    is what makes it read as a machine cycling, and humans are startlingly good
    at noticing it.
    """

    def __init__(
        self,
        mean_interval: float = 4.0,
        min_interval: float = 1.2,
        duration: float = 0.13,
        double_blink_chance: float = 0.15,
        rng: random.Random = None,
    ):
        self.mean_interval = mean_interval
        self.min_interval = min_interval
        self.duration = duration
        self.double_blink_chance = double_blink_chance
        #: Manual on/off (used to freeze the layer for reference renders).
        self.enabled = True
        #: Set every frame from the current expression's AmbientMotion.
        #: Sleepy blinks far more often than determined does, and startle
        #: suppresses blinking entirely -- real eyes stop blinking when
        #: frightened, and a face that keeps blinking through a startle pose
        #: undercuts the fright.
        self.interval_scale = 1.0
        self.ambient_enabled = True
        self._rng = rng or random.Random()
        self._t = 0.0
        self._time_to_next = self._draw_interval()
        self._blinking = False
        self._pending_double = False
        self.amount = 0.0

    def set_ambient(self, motion: AmbientMotion) -> None:
        self.interval_scale = max(0.05, motion.blink_interval_scale)
        self.ambient_enabled = motion.blink_enabled

    def _draw_interval(self) -> float:
        base = self.min_interval + self._rng.expovariate(1.0 / self.mean_interval)
        return base * self.interval_scale

    def trigger(self) -> None:
        """Force a blink now. Useful as punctuation on an expression change."""
        if not self._blinking:
            self._blinking = True
            self._t = 0.0

    def update(self, dt: float) -> None:
        if not self.enabled or not self.ambient_enabled:
            # Countdown is simply not advanced while suppressed (e.g. through
            # a startle), rather than reset, so blinking resumes on roughly
            # its normal schedule instead of firing immediately the moment
            # suppression lifts.
            self.amount = 0.0
            return

        if self._blinking:
            self._t += dt
            phase = self._t / self.duration
            if phase >= 1.0:
                self._blinking = False
                self.amount = 0.0
                if self._pending_double:
                    self._pending_double = False
                    self._time_to_next = 0.09
                else:
                    self._time_to_next = self._draw_interval()
            else:
                # Asymmetric: lids snap shut and open more gently, like real
                # eyelids. A symmetric triangle reads as a shutter.
                if phase < 0.42:
                    self.amount = ease_out_cubic(phase / 0.42)
                else:
                    self.amount = 1.0 - ease_out_cubic((phase - 0.42) / 0.58)
        else:
            self._time_to_next -= dt
            if self._time_to_next <= 0.0:
                self._blinking = True
                self._t = 0.0
                self._pending_double = self._rng.random() < self.double_blink_chance

    def apply(self, state: FaceState) -> None:
        if self.amount <= 0.0:
            return
        for eye in (state.left, state.right):
            # Skip eyes that are already near-shut or drawn as flat bars --
            # blinking a 0.1-high rectangle just looks like a glitch.
            if eye.height < 0.15:
                continue
            eye.lid_upper = lerp(eye.lid_upper, 0.55, self.amount)
            eye.lid_lower = lerp(eye.lid_lower, 0.45, self.amount)


class IdleLayer(Layer):
    """The motion a held expression has on its own: breath, sway, tilt, pulse.

    Every oscillator here uses deliberately incommensurate periods, and several
    run through independent phase accumulators rather than a shared clock. A
    single sine is detectable as a loop within a few seconds; periods that
    never quite line up read as organic no matter how small the amplitude.

    All of it is scaled by the current expression's AmbientMotion, which is
    where an expression stops being a shape and starts being a behaviour:
    `sleepy` breathes low and slow, `happy` breathes fast and bounces, `error`
    does not breathe at all.
    """

    def __init__(self, amplitude: float = 1.0):
        self.amplitude = amplitude
        self.ambient = AmbientMotion()
        self._t = 0.0
        #: Independent accumulators, so sway and tilt drift out of phase with
        #: each other instead of both peaking on the same beat.
        self._sway_t = 0.0
        self._tilt_t = 1.7
        self._pulse_t = 3.1

    def set_ambient(self, motion: AmbientMotion) -> None:
        self.ambient = motion

    def update(self, dt: float) -> None:
        self._t += dt
        self._sway_t += dt * self.ambient.sway_rate
        self._tilt_t += dt * self.ambient.tilt_rate
        self._pulse_t += dt

    def apply(self, state: FaceState) -> None:
        a = self.amplitude
        m = self.ambient
        rate = max(0.05, m.breath_rate)
        depth = a * m.breath_depth

        # Breathing: two waves at periods that scale inversely with rate, so a
        # faster breather does not just get louder, it gets quicker too.
        slow = math.sin(TWO_PI * self._t * rate / 4.30)
        fast = math.sin(TWO_PI * self._t * rate / 2.70 + 1.1)

        state.face_offset_y += depth * (0.013 * slow + 0.004 * fast)
        breath = depth * (0.009 * slow + 0.003 * fast)
        state.face_scale_x += breath * 0.5
        state.face_scale_y += breath

        # Sway and tilt are each an expression's own signature -- neutral
        # sways gently, focused holds almost dead still, confused rocks
        # noticeably. Both run at the expression's own rate, not the breath
        # rate, so they do not lock into the same beat as breathing.
        state.face_offset_x += m.sway * math.sin(self._sway_t)
        state.face_tilt += m.tilt * math.sin(self._tilt_t)

        if m.eye_pulse > 1e-4:
            # Eyes pulse a little out of phase with each other -- if they
            # pulsed in lockstep it would read as the whole face breathing
            # again rather than as something happening in the eyes themselves.
            pulse_l = 1.0 + m.eye_pulse * math.sin(self._pulse_t)
            pulse_r = 1.0 + m.eye_pulse * math.sin(self._pulse_t * 1.13 + 0.9)
            state.left.width *= pulse_l
            state.left.height *= pulse_l
            state.right.width *= pulse_r
            state.right.height *= pulse_r


class GazeLayer(Layer):
    """Where the eyes point, including the involuntary parts.

    Owns three things that all write to the same output: the commanded gaze
    target, a slow wander used when nothing is commanded, and microsaccades.
    Real eyes are never still even when fixating, and adding that jitter is
    the cheapest single improvement available here.
    """

    def __init__(
        self,
        follow_rate: float = 9.0,
        eye_shift_x: float = 0.10,
        eye_shift_y: float = 0.07,
        rng: random.Random = None,
    ):
        self.follow_rate = follow_rate
        self.eye_shift_x = eye_shift_x
        self.eye_shift_y = eye_shift_y
        self._rng = rng or random.Random()

        #: Commanded target in eye-normalised units, or None to wander.
        self.target = None
        self.weight = 1.0

        #: How busy the idle gaze is, from the current expression's
        #: AmbientMotion. `determined` barely wanders; `curious` scans
        #: constantly. `gaze_bias_y` is a resting offset applied even while a
        #: commanded target has the gaze, so a sad or sleepy face keeps its
        #: eyes low even when something briefly draws its attention.
        self.activity = 1.0
        self.bias_y = 0.0

        self._x = 0.0
        self._y = 0.0
        self._wander_x = 0.0
        self._wander_y = 0.0
        self._time_to_wander = 0.0
        self._saccade_x = 0.0
        self._saccade_y = 0.0
        self._time_to_saccade = 0.0

    def set_ambient(self, motion: AmbientMotion) -> None:
        self.activity = max(0.0, motion.gaze_activity)
        self.bias_y = motion.gaze_bias_y

    def look_at(self, x: float, y: float, weight: float = 1.0) -> None:
        self.target = (clamp(x, -1.0, 1.0), clamp(y, -1.0, 1.0))
        self.weight = clamp(weight)

    def release(self) -> None:
        self.target = None

    def freeze(self) -> None:
        """Centre the gaze and stop all involuntary motion.

        For reference renders and tests, where the wander and microsaccades
        would otherwise make every frame subtly different.
        """
        self.target = None
        self._x = self._y = 0.0
        self._wander_x = self._wander_y = 0.0
        self._saccade_x = self._saccade_y = 0.0
        self._time_to_wander = float("inf")
        self._time_to_saccade = float("inf")

    def update(self, dt: float) -> None:
        # Activity scales both how often the idle gaze moves and how far it
        # goes. At 0 the interval stretches toward infinity and the amplitude
        # toward zero, which settles into a held stare without needing a
        # separate "locked" code path.
        activity = clamp(self.activity, 0.0, 3.0)
        interval_scale = 1.0 / max(activity, 0.05)

        # Slow wander, used only when nothing is commanding the gaze.
        self._time_to_wander -= dt
        if self._time_to_wander <= 0.0:
            self._time_to_wander = self._rng.uniform(1.4, 4.0) * interval_scale
            self._wander_x = self._rng.uniform(-0.45, 0.45) * activity
            self._wander_y = self._rng.uniform(-0.25, 0.30) * activity + self.bias_y

        # Microsaccades continue even while fixating.
        self._time_to_saccade -= dt
        if self._time_to_saccade <= 0.0:
            self._time_to_saccade = self._rng.uniform(0.35, 1.60) * interval_scale
            self._saccade_x = self._rng.gauss(0.0, 0.035) * activity
            self._saccade_y = self._rng.gauss(0.0, 0.025) * activity

        if self.target is not None:
            # The resting bias still pulls even with a live target, weighted
            # by how much the target does *not* already own the gaze -- a sad
            # face glancing at something briefly still sags back down after.
            tx = lerp(self._wander_x, self.target[0], self.weight)
            ty = lerp(self._wander_y, self.target[1], self.weight)
            rate = self.follow_rate
        else:
            tx, ty = self._wander_x, self._wander_y
            # Unhurried when nothing has its attention.
            rate = self.follow_rate * 0.28

        self._x = approach(self._x, tx, rate, dt)
        self._y = approach(self._y, ty, rate, dt)

    def apply(self, state: FaceState) -> None:
        gx = clamp(self._x + self._saccade_x, -1.0, 1.0)
        gy = clamp(self._y + self._saccade_y, -1.0, 1.0)
        for eye in (state.left, state.right):
            eye.pupil_x += gx
            eye.pupil_y += gy
            # With pupils disabled (the default look) the eye body itself has
            # to carry the gaze, or the face has no way to point at anything.
            eye.center_x += gx * self.eye_shift_x
            eye.center_y += gy * self.eye_shift_y


class MotionLayer(Layer):
    """Inertia and anticipation derived from how the robot is actually moving.

    Two effects, and the interplay between them is the point:

    * Inertia -- the face lags the body. Turning left slides it right, braking
      squashes it. This is what gives the character apparent mass.
    * Anticipation -- the gaze *leads* the turn, going where the robot is about
      to go before the body follows. Straight out of the animation playbook,
      and nearly free here because the planned path is already published.

    They pull in opposite directions during a turn, and that opposition is
    exactly what sells it: eyes ahead, body trailing.
    """

    def __init__(
        self,
        lean_gain: float = 0.055,
        squash_gain: float = 0.030,
        tilt_gain: float = 0.030,
        gaze_lead_gain: float = 0.55,
        smoothing: float = 6.0,
    ):
        self.lean_gain = lean_gain
        self.squash_gain = squash_gain
        self.tilt_gain = tilt_gain
        self.gaze_lead_gain = gaze_lead_gain
        self.smoothing = smoothing

        self.linear = 0.0
        self.angular = 0.0
        #: Signed curvature of the upcoming path, if a planner is running.
        #: Positive turns left. Preferred over `angular` for anticipation
        #: because it describes the turn before the controller commands it.
        self.path_curvature = 0.0

        self._prev_linear = 0.0
        self._accel = 0.0
        self._s_accel = 0.0
        self._s_angular = 0.0
        self._s_lead = 0.0

    def set_velocity(self, linear: float, angular: float) -> None:
        self.linear = linear
        self.angular = angular

    def update(self, dt: float) -> None:
        if dt > 1e-6:
            raw_accel = (self.linear - self._prev_linear) / dt
            # Acceleration from differenced velocity commands is noisy; clamp
            # before smoothing so one bad frame cannot punch the face.
            self._accel = clamp(raw_accel, -4.0, 4.0)
        self._prev_linear = self.linear

        self._s_accel = approach(self._s_accel, self._accel, self.smoothing, dt)
        self._s_angular = approach(self._s_angular, self.angular, self.smoothing, dt)

        lead = self.path_curvature if abs(self.path_curvature) > 1e-4 else self.angular
        self._s_lead = approach(self._s_lead, clamp(lead, -2.0, 2.0), 4.5, dt)

    @property
    def gaze_lead(self) -> float:
        """Horizontal gaze bias, -1..1, pointing into the upcoming turn."""
        return clamp(self._s_lead * self.gaze_lead_gain, -1.0, 1.0)

    def apply(self, state: FaceState) -> None:
        # Lateral inertia: turning left throws the face right.
        state.face_offset_x -= clamp(self._s_angular * self.lean_gain, -0.18, 0.18)
        state.face_tilt -= clamp(self._s_angular * self.tilt_gain, -0.12, 0.12)

        # Squash and stretch. Accelerating stretches the face tall; braking
        # squashes it wide. Volume is roughly preserved, which is what makes
        # it read as a physical material rather than a scale animation.
        stretch = clamp(self._s_accel * self.squash_gain, -0.10, 0.10)
        state.face_scale_y += stretch
        state.face_scale_x -= stretch * 0.6
        state.face_offset_y += stretch * 0.35


class ReactionLayer(Layer):
    """Short-lived additive impulses for things that just happened.

    Separate from expressions because a reaction is an event, not a state: the
    robot can be startled *while* remaining focused, and forcing that through
    the expression channel would mean losing one or the other.
    """

    #: name -> (duration, decay_shape)
    KINDS = {
        "recoil": 0.55,
        "shake": 0.70,
        "bounce": 0.60,
        "squash": 0.35,
    }

    def __init__(self):
        self._active = []  # list of [kind, strength, elapsed, duration]

    def fire(self, kind: str, strength: float = 1.0) -> None:
        duration = self.KINDS.get(kind)
        if duration is None:
            return
        # One instance per kind; re-firing restarts and takes the stronger
        # amplitude rather than stacking into something violent.
        for entry in self._active:
            if entry[0] == kind:
                entry[1] = max(entry[1], clamp(strength))
                entry[2] = 0.0
                return
        self._active.append([kind, clamp(strength), 0.0, duration])

    @property
    def busy(self) -> bool:
        return bool(self._active)

    def update(self, dt: float) -> None:
        for entry in self._active:
            entry[2] += dt
        self._active = [e for e in self._active if e[2] < e[3]]

    def apply(self, state: FaceState) -> None:
        for kind, strength, elapsed, duration in self._active:
            phase = elapsed / duration
            # Every impulse decays to nothing by the end of its window, so a
            # reaction can never leave the face permanently displaced.
            envelope = strength * (1.0 - phase) ** 2

            if kind == "recoil":
                state.face_offset_y -= envelope * 0.10 * math.cos(TWO_PI * phase * 1.5)
                state.face_scale_y -= envelope * 0.10
                state.face_scale_x += envelope * 0.07
            elif kind == "shake":
                state.face_offset_x += envelope * 0.09 * math.sin(TWO_PI * phase * 5.0)
                state.face_tilt += envelope * 0.06 * math.sin(TWO_PI * phase * 5.0)
            elif kind == "bounce":
                state.face_offset_y += envelope * 0.11 * abs(math.sin(TWO_PI * phase * 2.0))
                state.face_scale_y += envelope * 0.06
            elif kind == "squash":
                state.face_scale_y -= envelope * 0.16
                state.face_scale_x += envelope * 0.12
                state.face_offset_y -= envelope * 0.05


class Compositor:
    """Owns the layer stack and produces one finished frame at a time."""

    def __init__(self, rng: random.Random = None):
        rng = rng or random.Random()
        self.blink = BlinkLayer(rng=rng)
        self.idle = IdleLayer()
        self.gaze = GazeLayer(rng=rng)
        self.motion = MotionLayer()
        self.reaction = ReactionLayer()

    def set_ambient(self, motion: AmbientMotion) -> None:
        """Tell the idle-motion layers which expression's behaviour to run.

        Call this with `animator.ambient` once per frame, before `update()`.
        It is what turns 'sleepy' from a shape into slow heavy breathing and
        frequent blinking, and 'startled' into a face that briefly stops
        moving altogether.
        """
        self.blink.set_ambient(motion)
        self.idle.set_ambient(motion)
        self.gaze.set_ambient(motion)

    def update(self, dt: float) -> None:
        for layer in (self.gaze, self.idle, self.motion, self.blink, self.reaction):
            layer.update(dt)

    def compose(self, base: FaceState) -> FaceState:
        """Apply every layer to a copy of the base pose."""
        state = base.copy()

        # Anticipation feeds the gaze layer rather than writing to the face
        # directly, so a commanded gaze target still wins over path-following.
        if self.gaze.target is None:
            lead = self.motion.gaze_lead
            if abs(lead) > 0.01:
                self.gaze._wander_x = lead

        for layer in (self.gaze, self.idle, self.motion, self.blink, self.reaction):
            layer.apply(state)

        _clamp_state(state)
        return state


def _clamp_state(state: FaceState) -> None:
    """Keep composited values inside what the renderer can draw.

    Layers add blindly, so without this a stacked blink, squint and startle
    can push lid coverage past 1.0 and invert the eye geometry.
    """
    for eye in (state.left, state.right):
        eye.lid_upper = clamp(eye.lid_upper, 0.0, 1.0)
        eye.lid_lower = clamp(eye.lid_lower, 0.0, 1.0)
        # Lids meeting in the middle is closed; beyond that is nonsense.
        overlap = eye.lid_upper + eye.lid_lower
        if overlap > 1.0:
            scale = 1.0 / overlap
            eye.lid_upper *= scale
            eye.lid_lower *= scale
        eye.pupil_x = clamp(eye.pupil_x, -1.0, 1.0)
        eye.pupil_y = clamp(eye.pupil_y, -1.0, 1.0)
        eye.width = max(eye.width, 0.0)
        eye.height = max(eye.height, 0.0)
        eye.corner_radius = clamp(eye.corner_radius)

    state.face_scale_x = clamp(state.face_scale_x, 0.4, 1.8)
    state.face_scale_y = clamp(state.face_scale_y, 0.4, 1.8)
    state.color = [clamp(c) for c in state.color]
