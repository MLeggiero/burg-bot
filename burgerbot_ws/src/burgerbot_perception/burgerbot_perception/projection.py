"""Pixel + depth -> 3D point. Pure numpy, no ROS and no camera model object.

Lifted out of object_projector so person_tracker can use exactly the same
sampling rules rather than growing its own subtly different copy. That matters
more than it looks: the two nodes place things on the same map, and if one
rejects a depth reading the other accepts, a chair and the person standing next
to it end up positioned by different rules and their relative geometry is
quietly wrong.

Taking fx/fy/cx/cy as plain floats rather than a PinholeCameraModel keeps this
testable without a CameraInfo message -- back-projection is four numbers and a
multiply, and there is no reason a test of it should need ROS installed.
"""

import math
from typing import Optional, Tuple

import numpy as np


def sample_depth(
    depth: np.ndarray,
    px: float,
    py: float,
    radius: int,
    min_depth: float,
    max_depth: float,
    scale: float = 1.0,
) -> Optional[float]:
    """Median depth in metres over a square patch, or None if nothing is valid.

    A single pixel is too easily a dropout -- depth sensors return zero at any
    given pixel often enough that per-pixel sampling loses a large fraction of
    otherwise good detections. The median (not the mean) of a small patch is
    what makes this robust: at an object's edge, part of the patch lands on the
    background metres behind it, and a mean would place the object somewhere in
    between, in empty space. A median just picks whichever surface most of the
    patch is on.
    """
    h, w = depth.shape[:2]
    cx, cy = int(round(px)), int(round(py))
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return None

    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    patch = depth[y0:y1, x0:x1].astype(np.float32) * scale

    valid = patch[(patch >= min_depth) & (patch <= max_depth)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def depth_scale_for(depth: np.ndarray) -> float:
    """Metres per raw unit. RealSense ships uint16 millimetres; float is metres."""
    return 0.001 if depth.dtype == np.uint16 else 1.0


def backproject(px: float, py: float, z: float,
                fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float]:
    """Pixel + range -> point in the camera optical frame (+Z forward).

    Standard pinhole back-projection, the exact inverse of the projection
    CameraInfo describes -- not an approximation. `z` is depth along the optical
    axis, which is what an aligned depth image stores; it is not range along the
    ray, and treating it as range would push every off-centre detection too far
    away by a factor of 1/cos(angle from the axis).
    """
    return ((px - cx) * z / fx, (py - cy) * z / fy, z)


def yaw_of(dx: float, dy: float) -> float:
    return math.atan2(dy, dx)


def wrap_angle(angle: float) -> float:
    """Fold an angle into [-pi, pi].

    Exactly at half a turn the sign is whichever way floating point rounds
    sin() -- both answers name the same angle, and every consumer here takes
    either abs() or cos() of the result, so it never matters.
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def facing_from_shoulders(
    left: Tuple[float, float, float],
    right: Tuple[float, float, float],
) -> Optional[float]:
    """Map-frame yaw a person is facing, from their two shoulders in 3D.

    The shoulder line is perpendicular to the way somebody faces, which leaves
    two candidate directions -- forwards and backwards. Knowing which shoulder
    is which resolves it: with Z up, a person's facing direction is the cross
    product of (right shoulder -> left shoulder) with the up axis. Facing +X
    puts the left shoulder at +Y, and cross((0,1,0), (0,0,1)) is (1,0,0).

    That is the whole reason this takes anatomically labelled shoulders rather
    than an unordered pair of points: without the labels the answer is only
    ever a line, and a person facing away from the robot is indistinguishable
    from one facing it -- which is precisely the distinction the social layer
    needs.

    Returns None when the shoulders are implausibly close together, which
    happens when the person is edge-on to the camera (the shoulder line
    foreshortens toward a point) or when one keypoint's depth is wrong. Both
    cases produce a yaw dominated by noise, so no answer beats a bad one.
    """
    dx = left[0] - right[0]
    dy = left[1] - right[1]
    if math.hypot(dx, dy) < 0.12:  # metres; adult shoulder span is ~0.4
        return None
    # cross((dx, dy, 0), (0, 0, 1)) = (dy, -dx, 0)
    return math.atan2(-dx, dy)


def engagement_from_facing(person_yaw: float, bearing_person_to_robot: float) -> float:
    """0..1: how squarely a person faces the robot. 1 is head-on, 0 is away.

    A raised cosine rather than a hard "within N degrees" test, because
    engagement is genuinely continuous -- somebody turned 40 degrees away is
    half paying attention, and a threshold would make the robot's behaviour
    flip on a degree of head movement.
    """
    error = wrap_angle(bearing_person_to_robot - person_yaw)
    return float(max(0.0, min(1.0, 0.5 * (1.0 + math.cos(error)))))
