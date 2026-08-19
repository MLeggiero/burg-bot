"""Turn-taking, timing and rate limits. Pure logic, no ROS and no network.

This is the file that decides what the robot does while it is waiting, which
matters more than it sounds. Even a fast local model takes a second or two, and
a second or two of a robot doing nothing at all reads as a crash rather than as
thought. Everything here is a plain function of a clock, so "what happens when
the model takes nine seconds" is a unit test rather than something you find out
in front of a guest.

Three behaviours are worth naming because they are not obvious:

*Late replies are dropped.* Every request carries the turn it belongs to, and a
result for a turn that is no longer current is thrown away. Without this the
robot answers a question from twenty seconds ago, after the person has already
asked another one -- and it is about six lines to prevent.

*A stalled turn says something.* Past a threshold with nothing back, the robot
emits a holding line from a static list. No model is involved, which is the
point: it works precisely when the model is the thing that is broken.

*Repeated failure stops trying.* After a few consecutive failures the circuit
opens and replies become instantly canned. That turns a dead endpoint from an
eight-second silence on every single turn into a fast, honest "not right now".
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

IDLE = "idle"
THINKING = "thinking"
SPEAKING = "speaking"
DEGRADED = "degraded"

#: What the robot says when a turn is taking a while. No model involved, on
#: purpose -- these have to work when the model is unreachable.
DEFAULT_FILLERS = ("Hang on...", "Let me think.", "One moment.")

#: What it says when a turn has failed, keyed by why. Honest rather than
#: cheerful: a robot that says "sorry, I didn't catch that" when it actually
#: cannot reach its model teaches you to repeat yourself pointlessly.
DEFAULT_FALLBACKS = {
    "timeout": "Sorry, I lost my train of thought. Say that again?",
    "unreachable": "I can't reach my words right now.",
    "circuit": "My head is offline at the moment.",
    "invalid": "I got confused there. Try me again?",
}


@dataclass
class Turn:
    """One exchange in flight."""

    id: int
    text: str
    started: float
    filler_sent: bool = False


@dataclass
class ConversationConfig:
    #: Seconds before the robot visibly starts thinking -- breaking gaze away
    #: and shifting to a focused face. People look away while formulating and
    #: back when they are ready to answer, and with no mouth this is the
    #: strongest "I am working on it" signal available.
    thinking_after: float = 0.8
    #: Seconds before it says something to fill the gap.
    stall_after: float = 3.5
    #: Seconds before it gives up on the turn entirely. An answer nobody is
    #: still waiting for is not an answer.
    give_up_after: float = 12.0
    #: Seconds of silence that end the conversation and drop the history.
    conversation_timeout: float = 90.0

    #: Consecutive failures before the circuit opens, and how long it stays
    #: open.
    failures_to_open: int = 3
    circuit_cooldown: float = 60.0

    #: Turns of history kept. Bounded because history grows into latency.
    max_history_turns: int = 8

    #: Minimum seconds between gestures. Small instruct models are bad at
    #: leaving optional fields out -- they fill in everything -- so a model
    #: that emits a gesture on every turn is the expected case rather than a
    #: surprise, and a robot twitching on every sentence is unwatchable.
    gesture_cooldown: float = 12.0
    #: Ceiling on how often the robot may be told to drive somewhere. Guards
    #: against both an over-eager model and somebody talking it into a loop.
    nav_cooldown: float = 20.0

    fillers: Tuple[str, ...] = DEFAULT_FILLERS


@dataclass
class Conversation:
    """Owns the turn in flight, the history, and everything time-based."""

    config: ConversationConfig = field(default_factory=ConversationConfig)

    _state: str = IDLE
    _turn: Optional[Turn] = None
    _next_id: int = 1
    _history: List[Tuple[str, str]] = field(default_factory=list)
    _last_activity: float = 0.0
    _last_gesture: float = -1e9
    _last_nav: float = -1e9
    _failures: int = 0
    _circuit_opened: float = -1e9

    # ---- starting a turn -------------------------------------------------

    def start_turn(self, text: str, now: float) -> Turn:
        """Begin a turn, superseding any turn already in flight.

        Superseding rather than queueing: people do say "what's that -- oh,
        never mind, can you come here instead", and answering the first half is
        worse than answering late. The old turn's id stops being current, so
        its reply is discarded when it eventually arrives.
        """
        self._turn = Turn(id=self._next_id, text=text, started=now)
        self._next_id += 1
        self._state = THINKING
        self._last_activity = now
        return self._turn

    def is_current(self, turn_id: int) -> bool:
        return self._turn is not None and self._turn.id == turn_id

    def state(self) -> str:
        return self._state

    def turn(self) -> Optional[Turn]:
        return self._turn

    # ---- the clock -------------------------------------------------------

    def thinking_visibly(self, now: float) -> bool:
        """Whether the robot should look like it is working on something."""
        if self._state != THINKING or self._turn is None:
            return False
        return (now - self._turn.started) >= self.config.thinking_after

    def due_filler(self, now: float) -> Optional[str]:
        """A holding line, once per turn, or None.

        Deterministic rather than random: which filler is said matters far less
        than that one is, and a deterministic choice keeps this testable
        without threading an RNG through.
        """
        if self._state != THINKING or self._turn is None:
            return None
        if self._turn.filler_sent:
            return None
        if (now - self._turn.started) < self.config.stall_after:
            return None
        self._turn.filler_sent = True
        fillers = self.config.fillers or DEFAULT_FILLERS
        return fillers[self._turn.id % len(fillers)]

    def expired(self, now: float) -> bool:
        if self._state != THINKING or self._turn is None:
            return False
        return (now - self._turn.started) >= self.config.give_up_after

    def idle_too_long(self, now: float) -> bool:
        if self._state == IDLE or self._last_activity <= 0.0:
            return False
        return (now - self._last_activity) >= self.config.conversation_timeout

    # ---- finishing a turn ------------------------------------------------

    def complete(self, turn_id: int, user_text: str, said: str, now: float) -> bool:
        """Record a successful turn. False if it was superseded meanwhile."""
        if not self.is_current(turn_id):
            return False
        self._state = SPEAKING
        self._last_activity = now
        self._failures = 0
        if user_text:
            self._history.append(("user", user_text))
        if said:
            self._history.append(("assistant", said))
        self._trim_history()
        self._turn = None
        return True

    def fail(self, turn_id: int, now: float) -> bool:
        """Record a failed turn and count it toward opening the circuit."""
        if not self.is_current(turn_id):
            return False
        self._failures += 1
        self._last_activity = now
        self._turn = None
        if self._failures >= self.config.failures_to_open:
            self._circuit_opened = now
            self._state = DEGRADED
        else:
            self._state = IDLE
        return True

    def _trim_history(self) -> None:
        limit = max(0, self.config.max_history_turns) * 2
        if limit and len(self._history) > limit:
            # Drop from the front: losing how the conversation opened is far
            # less damaging than losing what was just said.
            del self._history[:-limit]

    def history(self) -> List[Tuple[str, str]]:
        return list(self._history)

    def end_conversation(self, now: float) -> None:
        """Drop the history and go quiet. Nothing is persisted.

        Transcripts are deliberately not written anywhere: they grow without
        bound, and they are a recording of private conversation in somebody's
        home that nobody consented to. What survives a restart is what the
        companion already stores about people, not what they said.
        """
        self._history.clear()
        self._turn = None
        self._state = IDLE
        self._last_activity = now

    # ---- the circuit -----------------------------------------------------

    def circuit_open(self, now: float) -> bool:
        """Whether to skip the model entirely and answer from the canned list."""
        if self._failures < self.config.failures_to_open:
            return False
        if (now - self._circuit_opened) >= self.config.circuit_cooldown:
            # Half-open: allow exactly one probe through. If it fails, fail()
            # re-opens with a fresh timestamp; if it succeeds, the failure
            # count resets and the circuit closes properly.
            self._failures = self.config.failures_to_open - 1
            return False
        return True

    def failures(self) -> int:
        return self._failures

    # ---- rate limits -----------------------------------------------------

    def allow_gesture(self, now: float) -> bool:
        """Whether a gesture may play, given how recently the last one did."""
        if (now - self._last_gesture) < self.config.gesture_cooldown:
            return False
        self._last_gesture = now
        return True

    def allow_nav(self, now: float) -> bool:
        if (now - self._last_nav) < self.config.nav_cooldown:
            return False
        self._last_nav = now
        return True

    def fallback_line(self, kind: str, lines=None) -> str:
        table = lines or DEFAULT_FALLBACKS
        return table.get(kind, DEFAULT_FALLBACKS["invalid"])


def now() -> float:
    """Wall clock, in one place so tests never reach for time.time()."""
    return time.time()
