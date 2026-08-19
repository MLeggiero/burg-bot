"""Arbitration behaviour. Pure logic, no ROS graph needed."""

from burgerbot_expressions.arbiter import (
    PRIORITY_ALERT,
    PRIORITY_AMBIENT,
    PRIORITY_CONCERN,
    PRIORITY_TASK,
    Candidate,
    MoodArbiter,
    bid_source,
    candidate_from_bid,
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


# ---- bids from other packages --------------------------------------------


def test_a_bid_is_namespaced_away_from_the_internal_sources():
    """A safety behaviour must not be disable-able by a name collision.

    Submitting replaces that source's previous candidate, so an unprefixed bid
    claiming source="proximity" would take over the startle response with
    nothing in any log to say it had happened.
    """
    arbiter = MoodArbiter(min_hold=0.0)
    arbiter.submit(
        Candidate("proximity", "startled", 1.0, PRIORITY_ALERT, stamp=0.0)
    )
    arbiter.submit(
        candidate_from_bid("proximity", "happy", 1.0, PRIORITY_TASK, 0.0, 0.0, now=0.0)
    )
    assert arbiter.evaluate(0.1).expression == "startled"
    assert arbiter.active_sources(0.1) == ["bid:proximity", "proximity"]


def test_an_unnamed_bid_still_gets_a_stable_key():
    first = candidate_from_bid("", "happy", 1.0, PRIORITY_TASK, 0.0, 0.0, now=0.0)
    second = candidate_from_bid("", "sad", 1.0, PRIORITY_TASK, 0.0, 0.0, now=1.0)
    assert first.source == second.source == "bid:unknown"


def test_a_bid_with_a_duration_expires_on_its_own():
    arbiter = MoodArbiter(min_hold=0.0)
    arbiter.submit(Candidate("ambient", "neutral", 1.0, PRIORITY_AMBIENT, stamp=0.0))
    arbiter.submit(
        candidate_from_bid("companion", "happy", 1.0, PRIORITY_TASK, 0.0,
                           duration=1.0, now=0.0)
    )
    assert arbiter.evaluate(0.5).expression == "happy"
    assert arbiter.evaluate(2.0).expression == "neutral"


def test_a_bid_without_a_duration_holds_until_replaced():
    arbiter = MoodArbiter(min_hold=0.0)
    arbiter.submit(Candidate("ambient", "neutral", 1.0, PRIORITY_AMBIENT, stamp=0.0))
    arbiter.submit(
        candidate_from_bid("companion", "happy", 1.0, PRIORITY_TASK, 0.0,
                           duration=0.0, now=0.0)
    )
    assert arbiter.evaluate(1e6).expression == "happy"


def test_a_bid_does_not_outrank_a_genuine_concern():
    """A flat battery matters more than being pleased to see somebody."""
    arbiter = MoodArbiter(min_hold=0.0)
    arbiter.submit(Candidate("battery", "sleepy", 1.0, PRIORITY_CONCERN, stamp=0.0))
    arbiter.submit(
        candidate_from_bid("companion", "happy", 1.0, 55, 0.0, 0.0, now=0.0)
    )
    assert arbiter.evaluate(0.1).expression == "sleepy"


def test_a_bid_is_still_beaten_by_an_alert():
    arbiter = MoodArbiter(min_hold=0.0)
    arbiter.submit(
        candidate_from_bid("companion", "happy", 1.0, 110, 0.0, 0.0, now=0.0)
    )
    arbiter.submit(Candidate("safety_stop", "startled", 1.0, PRIORITY_ALERT, stamp=0.0))
    assert arbiter.evaluate(0.1).expression == "startled"


def test_clearing_a_bid_uses_the_same_namespaced_key():
    arbiter = MoodArbiter(min_hold=0.0)
    arbiter.submit(
        candidate_from_bid("companion", "happy", 1.0, PRIORITY_TASK, 0.0, 0.0, now=0.0)
    )
    arbiter.clear(bid_source("companion"))
    assert arbiter.evaluate(0.1) is None
