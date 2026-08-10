"""Gesture shape invariants.

These are the properties that keep a gesture from quietly becoming a
manoeuvre: it has to end, it has to stop, and anything that turns has to leave
the robot pointing where it started. A gesture that accumulates yaw slowly
corrupts odometry every time it plays.
"""

import math

import pytest

from burgerbot_expressions import gestures


@pytest.mark.parametrize("name", gestures.names())
def test_gesture_terminates_and_ends_at_rest(name):
    spec = gestures.get(name)
    velocity, progress, finished = gestures.sample(name, spec.duration + 0.01)
    assert finished
    assert progress == pytest.approx(1.0)
    assert velocity == (0.0, 0.0)


@pytest.mark.parametrize("name", gestures.names())
def test_velocities_stay_within_sane_bounds(name):
    """Nothing should command a speed a small indoor robot cannot survive."""
    spec = gestures.get(name)
    steps = 200
    for i in range(steps):
        t = spec.duration * i / steps
        (linear, angular), _, _ = gestures.sample(name, t, scale=1.0)
        assert abs(linear) <= 0.30, f"{name} linear {linear} at t={t}"
        assert abs(angular) <= 2.5, f"{name} angular {angular} at t={t}"


@pytest.mark.parametrize("name", ["nod_yes", "shake_no", "wiggle", "curious_tilt"])
def test_oscillating_gestures_return_to_start(name):
    """Net displacement over one cycle must be ~zero.

    `celebrate` is deliberately excluded: a full spin is the whole point of it.
    `anticipate` and `recoil` are also excluded, being one-way by design.
    """
    spec = gestures.get(name)
    steps = 2000
    dt = spec.duration / steps
    net_linear = 0.0
    net_angular = 0.0
    for i in range(steps):
        (linear, angular), _, _ = gestures.sample(name, i * dt)
        net_linear += linear * dt
        net_angular += angular * dt

    assert abs(net_linear) < 1e-3, f"{name} drifts {net_linear:.4f} m per cycle"
    assert abs(net_angular) < 1e-3, f"{name} drifts {net_angular:.4f} rad per cycle"


def test_celebrate_is_close_to_one_full_turn():
    spec = gestures.get("celebrate")
    steps = 4000
    dt = spec.duration / steps
    total = sum(gestures.sample("celebrate", i * dt)[0][1] * dt for i in range(steps))
    # Eased ends cost some rotation, so allow a generous band -- this is a
    # sanity check that it spins roughly once, not a precision requirement.
    assert 0.7 * 2 * math.pi < abs(total) < 1.3 * 2 * math.pi


def test_scale_is_linear_and_zero_scale_is_still():
    (linear_full, angular_full), _, _ = gestures.sample("wiggle", 0.3, scale=1.0)
    (linear_half, angular_half), _, _ = gestures.sample("wiggle", 0.3, scale=0.5)
    (linear_zero, angular_zero), _, _ = gestures.sample("wiggle", 0.3, scale=0.0)

    assert linear_half == pytest.approx(linear_full * 0.5)
    assert angular_half == pytest.approx(angular_full * 0.5)
    assert (linear_zero, angular_zero) == (0.0, 0.0)


def test_unknown_gesture_is_inert_rather_than_raising():
    velocity, progress, finished = gestures.sample("does_not_exist", 0.0)
    assert velocity == (0.0, 0.0)
    assert finished and progress == 1.0


def test_repeat_extends_duration():
    spec = gestures.get("nod_yes")
    _, _, finished_once = gestures.sample("nod_yes", spec.duration * 1.5, repeat=1)
    _, _, finished_twice = gestures.sample("nod_yes", spec.duration * 1.5, repeat=2)
    assert finished_once
    assert not finished_twice
