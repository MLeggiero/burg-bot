"""The parametric description of a face, and the maths for blending two of them.

Every visible feature is a number here, which is the whole point: because a
face is just a struct of floats, any two expressions can be mixed continuously.
A sprite-sheet or video-clip face can only cut between fixed states, and that
is what makes most robot faces look like a slideshow instead of a character.

Coordinate system
-----------------
Uniform normalised device coordinates, so circles stay circular on a
non-square panel:

    px_x = width / 2  + x * S
    px_y = height / 2 - y * S      (y is up, unlike raw screen pixels)
    S    = min(width, height) / 2

So y always spans [-1, +1], and x spans [-aspect, +aspect] -- on the 800x480
panel that is x in [-1.667, +1.667]. Sizes are in the same units, meaning a
width of 1.0 is half the screen height regardless of resolution.
"""

from dataclasses import dataclass, field, fields, replace
from typing import List

from .easing import clamp, lerp


@dataclass
class EyeParams:
    """One eye. See module docstring for units."""

    center_x: float = 0.0
    center_y: float = 0.0
    width: float = 0.38
    height: float = 0.62

    #: 0 = sharp rectangle, 1 = a true ellipse. Intermediate values are rounded
    #: rectangles. Rounding reads as friendly; sharp corners read as mechanical
    #: or hostile, which is the whole of what `error` and `focused` rely on.
    corner_radius: float = 1.0

    #: Whole-eye roll in radians, positive = outer edge lifted.
    #:
    #: With no eyebrows on this face, rotation does their job: tilting the
    #: ovals toward each other reads as a scowl, away as worry. It is the most
    #: expressive single parameter here after lid_angle.
    rotation: float = 0.0

    #: Pupil offset in units of the eye's own half-extent, so +-1 is the edge
    #: whatever the eye size. A radius of 0 disables the pupil entirely and the
    #: eye body itself carries the gaze -- the Cozmo/Vector look, and the
    #: default here because it stays readable at a glance across a room.
    pupil_x: float = 0.0
    pupil_y: float = 0.0
    pupil_radius: float = 0.0

    #: Fraction of the eye hidden by each lid, 0..1. Both at 0.5 = shut.
    lid_upper: float = 0.0
    lid_lower: float = 0.0

    #: Upper-lid tilt, radians, positive angles the inner corner downward.
    #: This one parameter does more emotional work than anything else in the
    #: struct: inner-down is angry/determined, inner-up is sad/pleading.
    lid_angle: float = 0.0


@dataclass
class AmbientMotion:
    """The intrinsic, always-running motion that belongs to an expression.

    A pose on its own is a still image, and blending between still images
    produces a face that is only alive during transitions and dead the rest of
    the time -- which is most of the time. So every expression carries its own
    idle behaviour: how it breathes, whether it sways or rocks, how often it
    blinks, how busy its eyes are.

    This is where most of the perceived life comes from. Startle reads as
    startle largely because the face *stops* -- people do not blink or drift
    while frightened -- and sleepy reads as sleepy because everything slows
    down, not because the lids are lower.

    Blended between expressions like any other dataclass, so the character of
    the motion crossfades along with the pose.
    """

    #: Multipliers on the baseline idle breathing period and amplitude.
    breath_rate: float = 1.0
    breath_depth: float = 1.0

    #: Slow horizontal drift, in normalized units.
    sway: float = 0.0
    sway_rate: float = 1.0

    #: Slow roll, in radians.
    tilt: float = 0.0
    tilt_rate: float = 1.0

    #: Out-of-phase size pulsing between the eyes. Small values only; this
    #: reads as liveliness, and large values read as a rendering fault.
    eye_pulse: float = 0.0

    #: Multiplier on the mean interval between blinks. Above 1 is rarer.
    blink_interval_scale: float = 1.0
    #: Blinking off entirely, for frozen or non-organic states.
    blink_enabled: bool = True

    #: Scales both saccade frequency and how far the idle gaze wanders.
    #: 0 is a locked stare, 2 is actively scanning the room.
    gaze_activity: float = 1.0
    #: Resting vertical gaze offset. Negative looks down.
    gaze_bias_y: float = 0.0


def _default_color() -> List[float]:
    # Light sky blue. High contrast on black and legible across a room, while
    # reading as a display rather than a pair of painted eyeballs.
    return [0.43, 0.78, 1.0, 1.0]


@dataclass
class FaceState:
    """A complete face pose.

    Two eyes on black, nothing else -- no mouth, no brows. That is a
    constraint worth keeping: it forces every emotion through eye geometry,
    and eye geometry is what people actually read anyway. It is also why
    EyeParams carries as many levers as it does.
    """

    left: EyeParams = field(default_factory=lambda: EyeParams(center_x=-0.48))
    right: EyeParams = field(default_factory=lambda: EyeParams(center_x=0.48))

    #: Whole-face transform, driven by the squash-stretch and lean layers
    #: rather than authored into keyframes.
    face_offset_x: float = 0.0
    face_offset_y: float = 0.0
    face_tilt: float = 0.0
    face_scale_x: float = 1.0
    face_scale_y: float = 1.0

    #: Feature colour, RGBA 0..1.
    color: List[float] = field(default_factory=_default_color)

    def copy(self) -> "FaceState":
        return replace(
            self,
            left=replace(self.left),
            right=replace(self.right),
            color=list(self.color),
        )


def _blend_value(a, b, t: float):
    """Blend one field, dispatching on type."""
    if isinstance(a, bool):
        # Booleans cannot be interpolated; switch at the midpoint.
        return b if t >= 0.5 else a
    if isinstance(a, (int, float)):
        return lerp(float(a), float(b), t)
    if isinstance(a, (list, tuple)):
        blended = [lerp(float(x), float(y), t) for x, y in zip(a, b)]
        return type(a)(blended) if isinstance(a, tuple) else blended
    if hasattr(a, "__dataclass_fields__"):
        return blend(a, b, t)
    # Strings and anything else: hard switch.
    return b if t >= 0.5 else a


def blend(a, b, t: float):
    """Interpolate between two dataclass instances of the same type.

    Walks the fields generically so adding a parameter to FaceState or
    EyeParams needs no corresponding edit here -- which is what keeps the
    expression library cheap to extend.
    """
    t = clamp(t)
    if t <= 0.0:
        return a.copy() if hasattr(a, "copy") else replace(a)
    if t >= 1.0:
        return b.copy() if hasattr(b, "copy") else replace(b)
    kwargs = {}
    for f in fields(a):
        kwargs[f.name] = _blend_value(getattr(a, f.name), getattr(b, f.name), t)
    return type(a)(**kwargs)


def mirrored(eye: EyeParams) -> EyeParams:
    """Mirror an eye across the vertical centreline.

    Lets an expression be authored once for one eye. Everything directional
    negates: a lid angled inner-down on the left must angle inner-down on the
    right, which is the opposite direction in screen space. Forgetting one of
    these produces a face that is subtly, unplaceably wrong.
    """
    return replace(
        eye,
        center_x=-eye.center_x,
        pupil_x=-eye.pupil_x,
        lid_angle=-eye.lid_angle,
        rotation=-eye.rotation,
    )
