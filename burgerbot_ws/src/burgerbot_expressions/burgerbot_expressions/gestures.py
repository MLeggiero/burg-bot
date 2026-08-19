"""The body-gesture library: expressive base motion as pure functions of time.

This is the half of the expression system the Disney work is really about. A
face on a screen is the easy part; what makes a robot read as a character is
that its *body* moves with intent -- it hesitates before setting off, it rocks
when it agrees, it turns to look before it turns to go.

Each gesture is a plain function from (phase, scale) to a velocity pair, with
no notion of safety, obstacles or what the navigation stack is doing. That is
deliberate. Gestures are authored freely as pure character motion; the
feasibility gate in gesture_server.py is solely responsible for deciding
whether the robot is allowed to perform one right now. Keeping intent and
constraint in separate layers means neither is compromised to accommodate the
other, which is exactly the separation Disney gets from wrapping an animator's
motion in a physics-aware policy.

No ROS dependency here, so the shapes can be plotted and tested directly.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

TWO_PI = 2.0 * math.pi

#: (linear m/s, angular rad/s)
Velocity = Tuple[float, float]


@dataclass
class GestureSpec:
    """One authored gesture."""

    name: str
    #: Seconds for a single cycle.
    duration: float
    #: (phase in 0..1, scale in 0..1) -> velocity.
    fn: Callable[[float, float], Velocity]
    #: Human-readable note on what it is for.
    intent: str
    #: True if the gesture translates. Translating gestures need clearance
    #: checked along the direction of travel; pure rotations need only a
    #: swept-radius check, which is a much weaker condition.
    translates: bool = False


def _nod_yes(phase: float, scale: float) -> Velocity:
    # Two forward/back rocks. Small amplitude -- a nod is a gesture, not a
    # manoeuvre, and anything bigger reads as the robot losing its footing.
    return 0.11 * scale * math.sin(TWO_PI * 2.0 * phase), 0.0


def _shake_no(phase: float, scale: float) -> Velocity:
    return 0.0, 1.5 * scale * math.sin(TWO_PI * 2.0 * phase)


def _wiggle(phase: float, scale: float) -> Velocity:
    # Three fast oscillations under a symmetric sin(pi*phase) window, so the
    # wiggle ramps in and out instead of starting and stopping abruptly.
    #
    # The window has to be symmetric. A decaying (1 - phase) envelope reads
    # fine but weights the early half-cycles more than the late ones, so the
    # angular velocity does not integrate to zero and every wiggle leaves the
    # robot yawed a few degrees. Repeat that and wheel odometry drifts for
    # reasons nothing in the navigation stack can explain. A symmetric window
    # against a whole number of cycles integrates to exactly zero.
    envelope = math.sin(math.pi * phase)
    return 0.0, 2.2 * scale * envelope * math.sin(TWO_PI * 3.0 * phase)


def _curious_tilt(phase: float, scale: float) -> Velocity:
    # One slow sweep out and back, as if looking around. A single full sine
    # period returns the robot to its original heading, so the gesture leaves
    # no accumulated yaw error behind for the localizer to absorb.
    return 0.0, 0.9 * scale * math.sin(TWO_PI * phase)


def _celebrate(phase: float, scale: float) -> Velocity:
    # A complete spin. Constant rate with eased ends so it does not start and
    # stop with a jolt.
    ramp = min(1.0, min(phase, 1.0 - phase) / 0.15)
    return 0.0, 2.0 * scale * ramp


def _anticipate(phase: float, scale: float) -> Velocity:
    # A few centimetres of reverse before departing. Textbook animation
    # anticipation: the small opposite move makes the departure read as a
    # decision rather than a motor switching on.
    if phase < 0.55:
        return -0.09 * scale, 0.0
    return 0.0, 0.0


def _recoil(phase: float, scale: float) -> Velocity:
    # Sharp back-off that decays. Pairs with the startled face.
    return -0.22 * scale * (1.0 - phase) ** 2, 0.0


def _segment(phase: float, start: float, end: float) -> float:
    """Local 0..1 phase within one part of a multi-part gesture."""
    return (phase - start) / (end - start)


def _dance(phase: float, scale: float) -> Velocity:
    # Three movements in a row: rock, sweep, shimmy. A single oscillation reads
    # as a twitch; what makes something read as *dancing* is a phrase with
    # parts, and specifically parts of different character -- translation, then
    # a slow rotation, then a fast one.
    #
    # Every segment is a whole number of sine cycles and so individually
    # integrates to zero. That is stricter than making the gesture as a whole
    # come out even, and deliberately: it means the dance can be interrupted at
    # a segment boundary -- which is where the feasibility gate is most likely
    # to stop it, since that is where the direction of travel changes -- and
    # still leave the robot where it started. A gesture that only balances if
    # it runs to completion quietly rotates the robot every time it does not.
    if phase < 0.30:
        return 0.10 * scale * math.sin(TWO_PI * 2.0 * _segment(phase, 0.0, 0.30)), 0.0
    if phase < 0.62:
        return 0.0, 1.6 * scale * math.sin(TWO_PI * _segment(phase, 0.30, 0.62))
    local = _segment(phase, 0.62, 1.0)
    return 0.0, 2.4 * scale * math.sin(math.pi * local) * math.sin(TWO_PI * 3.0 * local)


def _spin_delight(phase: float, scale: float) -> Velocity:
    # Two wide sweeps under the same symmetric window wiggle uses, but slower
    # and further, so it reads as pleased rather than as agitated. The window
    # times a whole number of cycles integrates to exactly zero for any integer
    # count -- the product-to-sum expansion leaves only sine terms evaluated at
    # whole multiples of pi.
    envelope = math.sin(math.pi * phase)
    return 0.0, 2.6 * scale * envelope * math.sin(TWO_PI * 2.0 * phase)


def _bounce(phase: float, scale: float) -> Velocity:
    # Happy forward/back bobbing, the translating counterpart to wiggle. Kept
    # small: this runs within arm's reach of somebody, and the clearance the
    # gesture server demands in front is only 0.35 m.
    envelope = math.sin(math.pi * phase)
    return 0.11 * scale * envelope * math.sin(TWO_PI * 3.0 * phase), 0.0


GESTURES: Dict[str, GestureSpec] = {
    g.name: g
    for g in (
        GestureSpec("nod_yes", 0.9, _nod_yes, "agreement, acknowledgement", True),
        GestureSpec("shake_no", 0.9, _shake_no, "refusal, cannot do that"),
        GestureSpec("wiggle", 1.2, _wiggle, "excitement, greeting"),
        GestureSpec("curious_tilt", 2.0, _curious_tilt, "looking around, searching"),
        GestureSpec("celebrate", 3.2, _celebrate, "goal reached"),
        GestureSpec("anticipate", 0.5, _anticipate, "about to set off", True),
        GestureSpec("recoil", 0.7, _recoil, "surprise, backing away", True),
        # Companion gestures. burgerbot_companion picks between these when it
        # decides to dance for somebody; the feasibility gate still applies, so
        # a dance in a corridor too tight for it simply does not happen.
        GestureSpec("dance", 3.4, _dance, "delight, showing off for someone", True),
        GestureSpec("spin_delight", 1.8, _spin_delight, "pleased to see you"),
        GestureSpec("bounce", 1.3, _bounce, "excited bobbing", True),
    )
}


def get(name: str):
    return GESTURES.get(name)


def names():
    return sorted(GESTURES.keys())


def sample(name: str, elapsed: float, scale: float = 1.0, repeat: int = 1):
    """Velocity at `elapsed` seconds into a gesture.

    Returns (velocity, progress, finished). Progress is 0..1 across all
    repeats, so an action server can report it directly as feedback.
    """
    spec = GESTURES.get(name)
    if spec is None:
        return (0.0, 0.0), 1.0, True

    total = spec.duration * max(1, repeat)
    if elapsed >= total:
        return (0.0, 0.0), 1.0, True

    phase = (elapsed % spec.duration) / spec.duration
    return spec.fn(phase, max(0.0, min(1.0, scale))), elapsed / total, False
