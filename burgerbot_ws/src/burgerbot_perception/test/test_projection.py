"""Depth sampling and back-projection. Pure math, no ROS graph."""

import math

import numpy as np
import pytest

from burgerbot_perception.projection import (
    backproject,
    depth_scale_for,
    engagement_from_facing,
    facing_from_shoulders,
    sample_depth,
    wrap_angle,
)

# A plausible 640x480 colour intrinsic; the exact values don't matter, only
# that projection and back-projection agree on them.
FX, FY, CX, CY = 600.0, 600.0, 320.0, 240.0


# ---- depth sampling ------------------------------------------------------


def test_sample_depth_returns_metres_from_millimetre_input():
    depth = np.full((480, 640), 2500, dtype=np.uint16)  # 2.5 m in mm
    value = sample_depth(depth, 320, 240, radius=3, min_depth=0.2, max_depth=8.0,
                         scale=depth_scale_for(depth))
    assert value == pytest.approx(2.5)


def test_float_depth_is_already_metres():
    depth = np.full((480, 640), 2.5, dtype=np.float32)
    assert depth_scale_for(depth) == 1.0
    value = sample_depth(depth, 320, 240, radius=3, min_depth=0.2, max_depth=8.0,
                         scale=depth_scale_for(depth))
    assert value == pytest.approx(2.5)


def test_dropouts_inside_the_patch_are_ignored_not_averaged_in():
    """Zeros are the sensor saying 'no reading', not 'the object is at 0 m'."""
    depth = np.full((100, 100), 2000, dtype=np.uint16)
    depth[48:52, 48:52] = 0  # a dropout right at the sample point
    value = sample_depth(depth, 50, 50, radius=4, min_depth=0.2, max_depth=8.0, scale=0.001)
    assert value == pytest.approx(2.0)


def test_a_patch_with_no_valid_depth_reports_nothing_rather_than_guessing():
    depth = np.zeros((100, 100), dtype=np.uint16)
    assert sample_depth(depth, 50, 50, radius=3, min_depth=0.2, max_depth=8.0,
                        scale=0.001) is None


def test_the_median_picks_a_surface_rather_than_splitting_the_difference():
    """At an object's edge, part of the patch is on the wall behind it.

    A mean would place the object in the empty space between the two, which is
    the specific failure the median exists to avoid.
    """
    depth = np.full((100, 100), 5000, dtype=np.uint16)  # far wall at 5 m
    depth[45:56, 45:52] = 1500                          # object at 1.5 m
    value = sample_depth(depth, 48, 50, radius=3, min_depth=0.2, max_depth=8.0, scale=0.001)
    assert value == pytest.approx(1.5)


def test_out_of_range_depths_are_rejected():
    depth = np.full((100, 100), 20000, dtype=np.uint16)  # 20 m, beyond max
    assert sample_depth(depth, 50, 50, radius=3, min_depth=0.2, max_depth=8.0,
                        scale=0.001) is None


def test_sampling_outside_the_image_is_refused():
    depth = np.full((100, 100), 2000, dtype=np.uint16)
    assert sample_depth(depth, -5, 50, radius=3, min_depth=0.2, max_depth=8.0,
                        scale=0.001) is None
    assert sample_depth(depth, 50, 999, radius=3, min_depth=0.2, max_depth=8.0,
                        scale=0.001) is None


def test_sampling_at_the_image_edge_clips_the_patch_instead_of_failing():
    depth = np.full((100, 100), 2000, dtype=np.uint16)
    value = sample_depth(depth, 0, 0, radius=5, min_depth=0.2, max_depth=8.0, scale=0.001)
    assert value == pytest.approx(2.0)


# ---- back-projection -----------------------------------------------------


def test_the_principal_point_projects_straight_down_the_optical_axis():
    x, y, z = backproject(CX, CY, 3.0, FX, FY, CX, CY)
    assert (x, y) == pytest.approx((0.0, 0.0))
    assert z == pytest.approx(3.0)


def test_back_projection_inverts_the_pinhole_projection():
    point = (0.4, -0.25, 2.2)
    px = FX * point[0] / point[2] + CX
    py = FY * point[1] / point[2] + CY
    assert backproject(px, py, point[2], FX, FY, CX, CY) == pytest.approx(point)


def test_depth_is_along_the_optical_axis_not_along_the_ray():
    """An aligned depth image stores Z, not range.

    Treating it as range would scale every off-axis point outward. The
    distinguishing check: a point one focal length off-centre has a true range
    of z*sqrt(2), but its Z must come back as exactly z.
    """
    _x, _y, z = backproject(CX + FX, CY, 2.0, FX, FY, CX, CY)
    assert z == pytest.approx(2.0)
    assert math.hypot(_x, z) == pytest.approx(2.0 * math.sqrt(2))


# ---- orientation from shoulders ------------------------------------------


def test_shoulders_give_the_facing_direction_not_just_the_line():
    # Facing +x: with Z up, the person's left shoulder is at +y.
    left, right = (0.0, 0.2, 1.4), (0.0, -0.2, 1.4)
    assert facing_from_shoulders(left, right) == pytest.approx(0.0)


def test_swapping_the_shoulders_reverses_the_facing_direction():
    """The whole reason this takes labelled shoulders rather than two points."""
    left, right = (0.0, 0.2, 1.4), (0.0, -0.2, 1.4)
    forwards = facing_from_shoulders(left, right)
    backwards = facing_from_shoulders(right, left)
    assert abs(wrap_angle(backwards - forwards)) == pytest.approx(math.pi)


def test_a_person_turned_ninety_degrees_reads_as_such():
    # Facing +y: left shoulder now at -x.
    left, right = (-0.2, 0.0, 1.4), (0.2, 0.0, 1.4)
    assert facing_from_shoulders(left, right) == pytest.approx(math.pi / 2)


def test_edge_on_shoulders_report_nothing_rather_than_noise():
    """Foreshortened to nearly a point, the perpendicular is pure noise."""
    left, right = (0.0, 0.02, 1.4), (0.0, -0.02, 1.4)
    assert facing_from_shoulders(left, right) is None


# ---- engagement ----------------------------------------------------------


def test_engagement_peaks_head_on_and_bottoms_out_facing_away():
    assert engagement_from_facing(0.0, 0.0) == pytest.approx(1.0)
    assert engagement_from_facing(0.0, math.pi) == pytest.approx(0.0)


def test_engagement_is_continuous_not_a_threshold():
    """Behaviour must not flip on a degree of head movement."""
    values = [engagement_from_facing(0.0, a) for a in np.linspace(0.0, math.pi, 20)]
    assert values == sorted(values, reverse=True)
    assert 0.4 < engagement_from_facing(0.0, math.pi / 2) < 0.6


def test_engagement_is_symmetric_left_and_right():
    assert engagement_from_facing(0.0, 0.7) == pytest.approx(engagement_from_facing(0.0, -0.7))


def test_engagement_stays_in_range_across_wrapped_angles():
    for a in np.linspace(-10.0, 10.0, 200):
        assert 0.0 <= engagement_from_facing(1.3, float(a)) <= 1.0


def test_wrap_angle_folds_into_a_single_turn():
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert wrap_angle(2 * math.pi + 0.5) == pytest.approx(0.5)
    assert wrap_angle(-2 * math.pi + 0.5) == pytest.approx(0.5)
    for angle in (3 * math.pi, -3 * math.pi, 7.0, -7.0, 100.0):
        assert -math.pi - 1e-9 <= wrap_angle(angle) <= math.pi + 1e-9
        # Same angle, just named once: the difference is a whole number of turns.
        turns = (angle - wrap_angle(angle)) / (2 * math.pi)
        assert turns == pytest.approx(round(turns), abs=1e-9)
