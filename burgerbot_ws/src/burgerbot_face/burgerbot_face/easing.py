"""Easing curves for expression blending.

Which curve you pick carries real meaning. Linear interpolation between two
poses always looks mechanical -- it is the single most common reason an
animated robot face reads as a slideshow rather than a character. Everything
here is normalised: each function maps t in [0, 1] to roughly [0, 1].
"""

import math


def linear(t: float) -> float:
    return t


def ease_in_out_cubic(t: float) -> float:
    """Symmetric acceleration then deceleration. The default for mood changes."""
    if t < 0.5:
        return 4.0 * t * t * t
    f = -2.0 * t + 2.0
    return 1.0 - (f * f * f) / 2.0


def ease_out_cubic(t: float) -> float:
    """Fast start, soft landing. Good for reactions that should feel immediate."""
    f = 1.0 - t
    return 1.0 - f * f * f


def ease_in_cubic(t: float) -> float:
    f = t * t * t
    return f


def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    """Overshoots past the target, then settles.

    Gives a pose a bit of physical weight, as if the face carried momentum
    into it. Use sparingly -- on every transition it reads as jitter.
    """
    c3 = overshoot + 1.0
    f = t - 1.0
    return 1.0 + c3 * f * f * f + overshoot * f * f


def ease_out_elastic(t: float) -> float:
    """Springy settle. Reserved for startle and other high-energy reactions."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    c4 = (2.0 * math.pi) / 3.0
    return math.pow(2.0, -10.0 * t) * math.sin((t * 10.0 - 0.75) * c4) + 1.0


def ease_out_quad(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


#: Look-up so expressions and configs can name a curve as a string.
CURVES = {
    "linear": linear,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_out_back": ease_out_back,
    "ease_out_elastic": ease_out_elastic,
    "ease_out_quad": ease_out_quad,
}


def get(name: str):
    """Resolve a curve by name, falling back to the sensible default."""
    return CURVES.get(name, ease_in_out_cubic)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else high if value > high else value


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def approach(current: float, target: float, rate: float, dt: float) -> float:
    """Frame-rate independent exponential smoothing toward a target.

    `rate` is roughly "fraction of the remaining gap closed per second". Using
    the exponential form rather than `current += (target - current) * k` keeps
    the motion identical whether the renderer is managing 60 fps or dropping to
    20 under load, which matters because the Pi does both.
    """
    if rate <= 0.0:
        return target
    alpha = 1.0 - math.exp(-rate * dt)
    return current + (target - current) * alpha
