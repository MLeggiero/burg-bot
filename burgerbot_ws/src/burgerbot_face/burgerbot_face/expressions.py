"""The keyframe library: named expressions as FaceState poses.

This is the art direction, and it is meant to be edited. Tuning the robot's
personality happens here and nowhere else -- no renderer or node changes are
needed to make it friendlier, warier or sleepier.

The face is two light-blue ovals on black. Nothing else. Every emotion has to
come out of eye geometry, which sounds limiting and is in fact the reason this
style works: people read eyes far more strongly than mouths, and a face with no
mouth never falls into the uncanny middle ground of a bad one.

The levers, roughly in order of how much work they do:

    rotation    tilting the ovals toward each other reads as a scowl, away as
                worry. This is doing the job eyebrows would.
    lid_angle   an upper lid angled inner-down reads angry or determined,
                inner-up reads sad or pleading
    lid_upper   a lowered upper lid reads focused, suspicious or tired
    lid_lower   a raised lower lid is the cheek-squint of a genuine smile
    size        large is surprised or childlike, small is wary or hostile
    asymmetry   any left/right mismatch reads as confusion or curiosity
    corner      round is soft and alive, rectangular is mechanical or dead

Each expression also carries its own timing, because how fast a face arrives at
a pose is part of the pose's meaning. Startle has to snap in; melancholy has to
seep in. Giving every transition the same duration flattens all of them.
"""

from dataclasses import dataclass, field, replace
from typing import Dict

from .face_state import AmbientMotion, EyeParams, FaceState, mirrored

# Palette. Hue is a blunt but very effective signal -- the eyes going red is
# legible from across a room in a way a shape change is not, so colour is
# reserved for states that genuinely warrant that much attention.
BLUE = [0.43, 0.78, 1.00, 1.0]   # the robot's resting colour
BRIGHT = [0.60, 0.88, 1.00, 1.0]  # lifted, for positive states
DIM = [0.22, 0.44, 0.62, 1.0]    # drained, for tired and sad
RED = [1.00, 0.34, 0.32, 1.0]    # fault only


@dataclass
class ExpressionSpec:
    """A named pose plus the timing that gives it its character."""

    state: FaceState
    #: Seconds to ease in when this becomes the winning expression.
    blend_time: float = 0.25
    #: Named curve from `easing.CURVES`.
    curve: str = "ease_in_out_cubic"
    #: How this expression behaves while it is simply being held. See
    #: AmbientMotion -- this is where most of the perceived life lives.
    ambient: AmbientMotion = field(default_factory=AmbientMotion)


def symmetric(eye: EyeParams, **face_kwargs) -> FaceState:
    """Build a left/right symmetric face from a single authored eye.

    The eye is authored as the *left* eye (negative x) and mirrored.
    """
    return FaceState(left=eye, right=mirrored(eye), **face_kwargs)


#: The resting eye. Tall oval, generously sized -- a cartoon proportion rather
#: than an anatomical one, which is what keeps it readable at a distance.
_EYE = EyeParams(center_x=-0.52, center_y=0.00, width=0.62, height=1.00)


def _build() -> Dict[str, ExpressionSpec]:
    e: Dict[str, ExpressionSpec] = {}

    # --- neutral -------------------------------------------------------
    # Plain tall ovals. Deliberately not narrowed or tilted: this is the pose
    # the robot spends most of its life in, so anything with an attitude in it
    # becomes the robot's whole personality by sheer exposure.
    e["neutral"] = ExpressionSpec(
        symmetric(_EYE),
        blend_time=0.35,
        ambient=AmbientMotion(
            sway=0.012, sway_rate=0.7, tilt=0.010, tilt_rate=0.5,
            eye_pulse=0.006, gaze_activity=1.0,
        ),
    )

    # --- happy ---------------------------------------------------------
    # The squint comes from the *lower* lid, not the upper. That is the
    # difference between a real smile and a polite one, and it survives being
    # rendered as two blue blobs.
    e["happy"] = ExpressionSpec(
        symmetric(
            replace(_EYE, height=1.03, lid_lower=0.40, rotation=-0.06),
            color=BRIGHT,
        ),
        blend_time=0.22,
        curve="ease_out_back",
        ambient=AmbientMotion(
            breath_rate=1.7, breath_depth=1.9,
            tilt=0.028, tilt_rate=1.6,
            eye_pulse=0.018,
            blink_interval_scale=0.75, gaze_activity=1.4,
        ),
    )

    # --- curious -------------------------------------------------------
    # Asymmetry is the entire trick. One eye taller, one slightly squinted, a
    # small head tilt, and the face reads as actively wondering rather than
    # merely switched on.
    e["curious"] = ExpressionSpec(
        FaceState(
            left=replace(_EYE, height=1.15, width=0.65, center_y=0.05),
            right=mirrored(
                replace(_EYE, height=0.78, width=0.59, lid_upper=0.14, rotation=0.10)
            ),
            face_tilt=0.11,
        ),
        blend_time=0.30,
        ambient=AmbientMotion(
            breath_rate=1.2,
            sway=0.030, sway_rate=0.55,
            tilt=0.055, tilt_rate=0.42,
            eye_pulse=0.012,
            gaze_activity=2.0,
        ),
    )

    # --- focused -------------------------------------------------------
    # Narrowed, squared off and steady. Worn while actually following a path,
    # so it has to be legible without being tiring to look at.
    e["focused"] = ExpressionSpec(
        symmetric(
            replace(_EYE, height=0.53, width=0.68, lid_upper=0.12, corner_radius=0.6)
        ),
        blend_time=0.28,
        ambient=AmbientMotion(
            breath_rate=0.8, breath_depth=0.45,
            sway=0.004, tilt=0.004,
            blink_interval_scale=1.8, gaze_activity=0.25,
        ),
    )

    # --- confused ------------------------------------------------------
    # Mismatched everything: one eye large and rolled outward, the other
    # small and squinted, head tilted the other way.
    e["confused"] = ExpressionSpec(
        FaceState(
            left=replace(_EYE, height=1.09, width=0.65, rotation=-0.22),
            right=mirrored(
                replace(_EYE, height=0.62, width=0.56, lid_upper=0.28, rotation=-0.18)
            ),
            face_tilt=-0.09,
        ),
        blend_time=0.30,
        ambient=AmbientMotion(
            breath_rate=1.3,
            sway=0.026, sway_rate=1.3,
            tilt=0.042, tilt_rate=0.85,
            gaze_activity=2.2,
        ),
    )

    # --- startled ------------------------------------------------------
    # Everything wide and round. The elastic curve makes it arrive with a
    # physical jolt; a linear blend into this pose looks like a menu
    # transition, not a fright.
    e["startled"] = ExpressionSpec(
        symmetric(
            replace(_EYE, width=0.93, height=1.21, center_y=0.02),
            color=BRIGHT,
        ),
        blend_time=0.12,
        curve="ease_out_elastic",
        ambient=AmbientMotion(
            breath_rate=2.4, breath_depth=0.20,
            sway=0.0, tilt=0.0,
            blink_enabled=False, gaze_activity=0.1,
        ),
    )

    # --- sad -----------------------------------------------------------
    # Inner corners lifted, lids heavy, eyes dropped and rolled outward. Slow
    # to arrive -- sadness that snaps on reads as a costume change.
    e["sad"] = ExpressionSpec(
        symmetric(
            replace(
                _EYE,
                height=0.84,
                width=0.59,
                center_y=-0.06,
                lid_upper=0.30,
                lid_angle=-0.38,
                rotation=-0.20,
                pupil_y=-0.25,
            ),
            color=DIM,
        ),
        blend_time=0.55,
        ambient=AmbientMotion(
            breath_rate=0.55, breath_depth=1.5,
            sway=0.010, sway_rate=0.35,
            tilt=0.014, tilt_rate=0.3,
            blink_interval_scale=1.4, gaze_activity=0.35, gaze_bias_y=-0.30,
        ),
    )

    # --- sleepy --------------------------------------------------------
    # Mostly shut, sitting low. Distinct from `sad` because the lids are level
    # rather than angled -- tiredness has no opinion, sadness does.
    e["sleepy"] = ExpressionSpec(
        symmetric(
            replace(_EYE, height=0.90, lid_upper=0.62, lid_lower=0.06, center_y=-0.06),
            color=DIM,
        ),
        blend_time=0.70,
        ambient=AmbientMotion(
            breath_rate=0.40, breath_depth=2.4,
            sway=0.014, sway_rate=0.25,
            tilt=0.020, tilt_rate=0.22,
            blink_interval_scale=0.45, gaze_activity=0.2, gaze_bias_y=-0.45,
        ),
    )

    # --- nervous ---------------------------------------------------------
    # Narrowed like `focused`, but where focused holds dead still, nervous
    # trembles: fast sway/tilt, quick blinks, high gaze activity. Same base
    # shape, opposite tempo -- that contrast is what sells "wary" rather than
    # "concentrating" from eye geometry alone.
    e["nervous"] = ExpressionSpec(
        symmetric(
            replace(_EYE, height=0.86, width=0.56, lid_upper=0.16, center_y=0.01)
        ),
        blend_time=0.16,
        curve="ease_out_cubic",
        ambient=AmbientMotion(
            breath_rate=2.2, breath_depth=0.35,
            sway=0.022, sway_rate=2.4,
            tilt=0.020, tilt_rate=2.1,
            eye_pulse=0.022,
            blink_interval_scale=0.5, gaze_activity=2.6,
        ),
    )

    # --- determined ----------------------------------------------------
    # Same narrowing as `focused`, but the lids angle inner-down and the ovals
    # roll toward each other. That sign flip is the whole difference between
    # concentrating and meaning it.
    e["determined"] = ExpressionSpec(
        symmetric(
            replace(
                _EYE,
                height=0.71,
                width=0.65,
                lid_upper=0.22,
                lid_angle=0.34,
                rotation=0.16,
                corner_radius=0.75,
            )
        ),
        blend_time=0.20,
        ambient=AmbientMotion(
            breath_rate=1.25, breath_depth=0.7,
            sway=0.005, tilt=0.006,
            blink_interval_scale=2.2, gaze_activity=0.2,
        ),
    )

    # --- error ---------------------------------------------------------
    # Flat, short, dead-level red bars. Unmistakably "stopped", and visually
    # unlike every other pose here even at a glance.
    e["error"] = ExpressionSpec(
        symmetric(
            replace(_EYE, width=0.71, height=0.17, center_y=0.02, corner_radius=0.25),
            color=RED,
        ),
        blend_time=0.15,
        ambient=AmbientMotion(
            breath_rate=1.0, breath_depth=0.0,
            sway=0.0, tilt=0.0,
            blink_enabled=False, gaze_activity=0.0,
        ),
    )

    return e


EXPRESSIONS: Dict[str, ExpressionSpec] = _build()

#: Fallback whenever an unknown name arrives on the wire.
DEFAULT = "neutral"


def get(name: str) -> ExpressionSpec:
    """Look up an expression, falling back to neutral for unknown names."""
    return EXPRESSIONS.get(name, EXPRESSIONS[DEFAULT])


def names():
    return sorted(EXPRESSIONS.keys())
