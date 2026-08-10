"""Arbitration behaviour. Pure logic, no ROS graph needed."""

from burgerbot_expressions.arbiter import (
    PRIORITY_ALERT,
    PRIORITY_AMBIENT,
    PRIORITY_CONCERN,
    PRIORITY_TASK,
    Candidate,
    MoodArbiter,
)


def _c(source, expression, priority, stamp=0.0, expires_at=None):
    return Candidate(
        source=source,
        expression=expression,
        priority=priority,
        stamp=stamp,
        expires_at=expires_at,
    )


def test_nothing_bidding_returns_none():
    assert MoodArbiter().evaluate(0.0) is None


def test_highest_priority_wins():
    a = MoodArbiter(min_hold=0.0)
    a.submit(_c("ambient", "neutral", PRIORITY_AMBIENT))
    a.submit(_c("nav", "focused", PRIORITY_TASK))
    assert a.evaluate(0.0).expression == "focused"


def test_alert_preempts_immediately_despite_hold():
    """A startle must never wait out the anti-churn hold."""
    a = MoodArbiter(min_hold=5.0)
    a.submit(_c("nav", "focused", PRIORITY_TASK, stamp=0.0))
    assert a.evaluate(0.0).expression == "focused"

    a.submit(_c("proximity", "startled", PRIORITY_ALERT, stamp=0.1))
    assert a.evaluate(0.1).expression == "startled"


def test_min_hold_prevents_equal_priority_churn():
    a = MoodArbiter(min_hold=1.0)
    a.submit(_c("nav", "focused", PRIORITY_TASK, stamp=0.0))
    assert a.evaluate(0.0).source == "nav"

    # A second, newer bid at the same priority must not steal the face while
    # the hold is running -- that alternation is what makes a face stutter.
    a.submit(_c("other", "curious", PRIORITY_TASK, stamp=0.5))
    assert a.evaluate(0.5).source == "nav"

    # Once the hold expires the newer bid takes over.
    assert a.evaluate(1.5).source == "other"


def test_expired_candidates_are_dropped():
    a = MoodArbiter(min_hold=0.0)
    a.submit(_c("ambient", "neutral", PRIORITY_AMBIENT))
    a.submit(_c("touch", "happy", PRIORITY_ALERT, stamp=0.0, expires_at=2.0))

    assert a.evaluate(1.0).expression == "happy"
    assert a.evaluate(2.5).expression == "neutral"
    assert "touch" not in a.active_sources(2.5)


def test_source_replaces_rather_than_stacks():
    """A chatty publisher must never accumulate bids."""
    a = MoodArbiter(min_hold=0.0)
    for i in range(50):
        a.submit(_c("proximity", "startled", PRIORITY_ALERT, stamp=float(i)))
    assert a.active_sources(0.0) == ["proximity"]


def test_clear_removes_a_source():
    a = MoodArbiter(min_hold=0.0)
    a.submit(_c("ambient", "neutral", PRIORITY_AMBIENT))
    a.submit(_c("localization", "confused", PRIORITY_CONCERN))
    assert a.evaluate(0.0).expression == "confused"

    a.clear("localization")
    assert a.evaluate(0.0).expression == "neutral"


def test_floor_survives_everything_expiring():
    """The ambient bid has no expiry, so the face always has a defined state."""
    a = MoodArbiter(min_hold=0.0)
    a.submit(_c("ambient", "neutral", PRIORITY_AMBIENT))
    a.submit(_c("nav", "happy", PRIORITY_TASK, expires_at=1.0))
    assert a.evaluate(2.0).expression == "neutral"
