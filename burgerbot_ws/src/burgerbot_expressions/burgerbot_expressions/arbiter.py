"""Priority arbitration between competing expression sources.

Pure logic with no ROS dependency, so it can be reasoned about and unit tested
without a graph running.

The problem this solves: half a dozen independent things all have an opinion
about the robot's face at once -- navigation is running, an obstacle is close,
localization is shaky, the battery is low, someone just poked the screen.
Publishing them all to the renderer directly makes the face flicker between
them at whatever rate they happen to publish, and a flickering face does not
read as conflicted, it reads as broken.

So each source contributes a *candidate* with a priority and an expiry, and
exactly one wins. This is the same shape as twist_mux arbitrating velocity
commands, applied to expression -- which is a good sign it is the right shape.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

# Priority bands, mirroring the constants in ExpressionCommand.msg.
PRIORITY_AMBIENT = 10   # idle personality; the floor, always present
PRIORITY_TASK = 50      # what the robot is currently doing
PRIORITY_CONCERN = 120  # recoverable trouble: lost, stuck, low battery
PRIORITY_ALERT = 200    # immediate physical events that must always show


@dataclass
class Candidate:
    """One source's bid for the face."""

    source: str
    expression: str
    intensity: float = 1.0
    priority: int = PRIORITY_AMBIENT
    #: 0 means "use the expression's own authored blend time".
    blend_time: float = 0.0
    #: Absolute time this stops being valid; None means until replaced.
    expires_at: Optional[float] = None
    #: When it was submitted, used to break priority ties toward the newest.
    stamp: float = 0.0

    def alive(self, now: float) -> bool:
        return self.expires_at is None or now < self.expires_at


#: Prefix applied to the source name of any bid arriving from another package.
#:
#: Sources inside mood_arbiter own bare keys ("proximity", "battery", "nav").
#: Nothing stops a node elsewhere publishing a bid with source="proximity", and
#: since a source's bid *replaces* that source's previous one, an unprefixed
#: collision would silently take over the startle response -- a safety
#: behaviour disabled by a name clash, with nothing in any log to say so.
#: Prefixing also survives into the published winner, which is worth having on
#: its own: "bid:companion" says where a mood came from.
BID_PREFIX = "bid:"


def bid_source(source: str) -> str:
    return BID_PREFIX + (source or "unknown")


def candidate_from_bid(
    source: str,
    expression: str,
    intensity: float,
    priority: int,
    blend_time: float,
    duration: float,
    now: float,
) -> "Candidate":
    """Build a candidate from another package's ExpressionCommand.

    Duration zero means "until replaced or cleared", matching the message's
    documented semantics and how the internal sources behave.
    """
    return Candidate(
        source=bid_source(source),
        expression=expression,
        intensity=float(intensity),
        priority=int(priority),
        blend_time=float(blend_time),
        expires_at=now + duration if duration > 0.0 else None,
        stamp=now,
    )


@dataclass
class MoodArbiter:
    """Holds one live candidate per source and decides which wins."""

    #: Minimum time a winner keeps the face once it has it.
    #:
    #: Without this, two sources at equal priority swap on every update and the
    #: face stutters between them. A higher-priority candidate still preempts
    #: immediately -- the hold only prevents churn among equals, never a
    #: genuine escalation.
    min_hold: float = 0.6

    _candidates: Dict[str, Candidate] = field(default_factory=dict)
    _winner: Optional[Candidate] = None
    _winner_since: float = 0.0

    def submit(self, candidate: Candidate) -> None:
        """Add or replace this source's candidate.

        Replacing rather than stacking means a source publishing at 20 Hz
        cannot crowd out anything: it always holds exactly one bid.
        """
        self._candidates[candidate.source] = candidate

    def clear(self, source: str) -> None:
        self._candidates.pop(source, None)

    def active_sources(self, now: float):
        return sorted(s for s, c in self._candidates.items() if c.alive(now))

    def evaluate(self, now: float) -> Optional[Candidate]:
        """Return the winning candidate, or None if nothing is bidding."""
        for source in [s for s, c in self._candidates.items() if not c.alive(now)]:
            del self._candidates[source]

        if not self._candidates:
            self._winner = None
            return None

        best = max(self._candidates.values(), key=lambda c: (c.priority, c.stamp))

        current = self._winner
        if (
            current is not None
            and current.source in self._candidates
            and self._candidates[current.source] is current
            and best.priority <= current.priority
            and (now - self._winner_since) < self.min_hold
        ):
            return current

        if current is None or best is not current:
            self._winner = best
            self._winner_since = now
        return best
