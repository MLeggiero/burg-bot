"""The companion's social behaviour: pure logic, no ROS dependency.

Everything the robot does around people is decided here and executed
elsewhere, the same division gestures.py and gesture_server.py already use in
burgerbot_expressions -- intent authored in one place with no knowledge of
obstacles or navigation, feasibility enforced in another. It is worth the
separation twice over here, because social behaviour is the hardest thing in
this workspace to test any other way: you cannot ask a real person to walk away
from a robot three times on cue, and you certainly cannot do it reproducibly.
As a pure function of a list of people and a timestamp, you can.

The three things this exists to do, and how each is actually decided:

*Be happy near people.* Straightforward, and the least interesting of the
three. ENGAGED means somebody the robot chose is within a comfortable distance,
and it bids a happy face for as long as that holds.

*Dance sometimes.* "Sometimes" is doing real work in that sentence. A robot
that dances every time somebody comes near is a vending machine with a
mechanism; one that dances occasionally, more readily for people it has spent
time with, is a character. So a dance needs a long cooldown, a person who is
actually facing the robot, positive affinity, and then a coin toss.

*Be sad when the person it is approaching keeps walking away.* This is the one
with a real risk of being wrong, and being wrong here is expensive: a robot
that reads ordinary passing traffic as rejection will mope constantly and be
tiresome within an hour. So rejection is defined narrowly. It is measured only
during an approach the robot actually committed to, and only from the person's
*own* outward motion -- integrating the component of their velocity along the
line of sight. That deliberately excludes the robot's own movement, so a robot
driving toward somebody stationary never mistakes its own approach for their
departure, and it excludes anyone walking across the robot's view rather than
away from it.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---- States. Mirrors the constants in SocialState.msg. ----
IDLE = "idle"
SEEKING = "seeking"
WATCHING = "watching"
APPROACHING = "approaching"
ENGAGED = "engaged"
DANCING = "dancing"
WITHDRAWN = "withdrawn"
#: Somebody told the robot to go somewhere. Deliberately outranks the robot's
#: own social judgement: being asked to go to the kitchen and then wandering
#: off after whoever walked past is not a robot exercising judgement, it is a
#: robot ignoring you.
COMMANDED = "commanded"

# Expression names, mirroring ExpressionCommand.msg.
EXPR_NONE = ""
EXPR_HAPPY = "happy"
EXPR_CURIOUS = "curious"
EXPR_SAD = "sad"
EXPR_DETERMINED = "determined"

# Priority bands, mirroring burgerbot_expressions/arbiter.py.
PRIORITY_TASK = 50
#: Social moods sit just above ordinary task expression, so being with somebody
#: reads through a routine navigation run -- but below CONCERN (120), because
#: a flat battery or a lost localization estimate is a more important thing for
#: the face to be saying than that the robot is pleased to see you.
PRIORITY_SOCIAL = 55
#: Withdrawal sits higher still, and still under CONCERN.
PRIORITY_WITHDRAWN = 110


@dataclass
class PersonView:
    """One person as the behaviour layer sees them this tick."""

    id: str
    x: float
    y: float
    distance: float
    #: Radians in the robot frame; 0 is dead ahead.
    bearing: float
    #: The person's own velocity along the line of sight, positive = away.
    range_rate: float
    #: 0..1, how squarely they face the robot. 0.5 means unknown.
    engagement: float
    visible: bool
    name: str = ""
    #: Map-frame yaw they are facing, if known.
    facing: Optional[float] = None


@dataclass
class Memory:
    """What the robot remembers about one person between encounters."""

    key: str
    #: -1 (repeatedly rebuffed) to +1 (sticks around). Decays toward 0, so one
    #: bad afternoon does not follow somebody around for a month.
    affinity: float = 0.0
    #: Consecutive approaches this person has walked out of. Reset by a single
    #: successful one -- the claim being made is "you keep leaving", and one
    #: stay is enough to falsify it.
    rejections: int = 0
    encounters: int = 0
    last_seen: float = 0.0
    #: Absolute time until which the robot will not approach them again.
    cooldown_until: float = 0.0


@dataclass
class SocialConfig:
    """Every tunable the behaviour has. Mirrored by config/companion.yaml."""

    # ---- proxemics ----
    #: Where the robot tries to end up: inside conversational range, outside
    #: touching range. Hall's "personal distance" for people who know each
    #: other is roughly 0.5-1.2 m and his "social distance" starts around
    #: 1.2 m; a knee-high robot reads as less intrusive than a person at the
    #: same distance, so the near end of social is about right.
    standoff: float = 1.1
    #: Reaching this counts as having arrived, so the robot stops before Nav2's
    #: own goal tolerance starts nudging it closer.
    engage_distance: float = 1.4
    #: Beyond this, the person has left the conversation.
    disengage_distance: float = 2.4
    #: Never plan a goal closer than this to a person, whatever else is true.
    personal_space: float = 0.75
    #: Widest angle off a person's front the robot will approach from. Coming
    #: at somebody from behind is the single most unsettling thing a mobile
    #: robot can do, and it costs nothing to swing around.
    max_front_offset: float = 1.05  # 60 degrees
    #: Someone further away than this is scenery, not company.
    max_approach_distance: float = 6.0

    # ---- attention ----
    #: How squarely somebody must face the robot before it will approach.
    #: Above the 0.5 that means "unknown", so with a bbox-only detector and a
    #: stationary person the robot watches rather than advances -- which is
    #: the right way to be wrong.
    min_engagement_to_approach: float = 0.55
    #: Hysteresis bonus for the person already targeted, so the robot does not
    #: swap its attention between two people of nearly equal score every tick.
    target_stickiness: float = 0.15
    #: Seconds of nobody around before the robot goes looking.
    seek_after: float = 25.0
    #: How long the robot looks at somebody before deciding to go over. Setting
    #: off the instant a track is confirmed reads as a machine triggering;
    #: a beat of visibly noticing somebody first reads as a decision. This is
    #: the same anticipation idea the gesture library is built on, applied to
    #: a state transition rather than to a velocity profile.
    watch_before_approach: float = 1.0

    # ---- rejection ----
    #: Metres of the person's own outward travel, during one approach, that
    #: count as them walking away. Roughly two paces: enough to exclude
    #: shifting weight or turning on the spot, short enough to notice somebody
    #: leaving before the robot has followed them across the room.
    rejection_travel: float = 1.2
    #: Rejections before the robot gives up on somebody and looks properly sad.
    #: Two is too eager -- people are legitimately busy. Three is a pattern.
    rejections_for_sad: int = 3
    #: How long the robot leaves a withdrawn-from person alone.
    withdraw_cooldown: float = 120.0
    #: How long the sad face holds after withdrawing.
    sad_duration: float = 8.0
    #: An approach that simply never arrives is a navigation problem, not a
    #: social one, and must not be counted as rejection.
    approach_timeout: float = 30.0

    # ---- affinity ----
    engage_reward: float = 0.25
    reject_penalty: float = 0.30
    #: Per second spent engaged.
    engage_rate: float = 0.01
    #: Seconds for affinity to halve. About a fortnight: long enough that a
    #: friendship survives a holiday, short enough that it is not permanent.
    affinity_half_life: float = 1.2e6

    # ---- dancing ----
    dance_cooldown: float = 45.0
    dance_probability: float = 0.35
    dance_min_affinity: float = 0.05
    #: How squarely somebody must be facing the robot for it to dance at them.
    #: Dancing at the back of somebody's head is just spinning.
    dance_min_engagement: float = 0.5
    #: Seconds a dance is allowed to run before the brain stops waiting for it.
    dance_timeout: float = 12.0
    #: How far the target must move before the approach goal is re-sent. Nav2
    #: replans from scratch on every new goal, so streaming one per tick at a
    #: walking person leaves it permanently replanning and never driving.
    replan_distance: float = 0.4
    dance_gestures: Tuple[str, ...] = ("dance", "wiggle", "spin_delight")
    #: One-off gesture when a person the robot knows first shows up.
    greet_gesture: str = "wiggle"
    greet_cooldown: float = 60.0

    #: Height above the floor the eyes aim at. A person's face, not their belt.
    gaze_height: float = 1.5

    #: How long a commanded trip may take before the robot gives up on it and
    #: returns to its own behaviour. Without an expiry a goal Nav2 silently
    #: never completes would leave the companion permanently commanded and
    #: apparently inert -- the failure would look like the whole social layer
    #: had stopped working, rather than like one trip having gone wrong.
    commanded_timeout: float = 60.0


@dataclass
class Decision:
    """What the brain wants done this tick. The node makes it happen, or not."""

    state: str = IDLE
    target_id: str = ""
    target_name: str = ""
    target_distance: float = 0.0
    affinity: float = 0.0
    rejections: int = 0
    people_visible: int = 0
    reason: str = ""

    #: Empty means "bid nothing", which is different from bidding neutral: it
    #: leaves the face entirely to the existing telemetry sources rather than
    #: competing with them for it.
    expression: str = EXPR_NONE
    expression_intensity: float = 1.0
    expression_priority: int = PRIORITY_SOCIAL

    #: Map-frame point to look at, or None to release the eyes to idle drift.
    gaze: Optional[Tuple[float, float, float]] = None
    #: Map-frame (x, y, yaw) to drive to, or None. Only ever set on the tick
    #: the goal changes, so the node is not re-sending the same goal at 10 Hz.
    approach_goal: Optional[Tuple[float, float, float]] = None
    cancel_goal: bool = False
    gesture: Optional[str] = None


def approach_pose(
    person_x: float,
    person_y: float,
    person_facing: Optional[float],
    robot_x: float,
    robot_y: float,
    standoff: float,
    max_front_offset: float,
) -> Tuple[float, float, float]:
    """Where to stand to talk to somebody, and which way to face.

    Two constraints pull against each other. The robot should approach from
    within a person's frontal cone, because arriving from behind or from
    directly beside somebody is startling in a way that arriving in front of
    them is not. But it should also not loop all the way around a room to do
    that when it is already most of the way there.

    So: take the bearing the robot is already coming from, and clamp it into
    the person's frontal cone. When the robot is already in front of them
    nothing moves at all; when it is behind them it swings around to the
    nearest edge of the cone rather than to dead ahead, which is the shorter
    path and looks less like a manoeuvre.

    With no orientation known, the robot's current bearing is used unchanged --
    approaching from where it already is beats guessing at a front that might
    be the back.
    """
    bearing_to_robot = math.atan2(robot_y - person_y, robot_x - person_x)

    if person_facing is None:
        approach_bearing = bearing_to_robot
    else:
        offset = _wrap(bearing_to_robot - person_facing)
        approach_bearing = person_facing + max(-max_front_offset,
                                               min(max_front_offset, offset))

    goal_x = person_x + standoff * math.cos(approach_bearing)
    goal_y = person_y + standoff * math.sin(approach_bearing)
    # Face back toward the person, so the robot arrives looking at them rather
    # than parked at the right spot pointing at a wall.
    return goal_x, goal_y, _wrap(approach_bearing + math.pi)


@dataclass
class SocialBrain:
    """Decides who to attend to and what to do about it."""

    config: SocialConfig = field(default_factory=SocialConfig)
    #: Injected so tests are deterministic. The randomness is real and wanted
    #: -- a companion that dances on a fixed schedule is a metronome -- but it
    #: must be reproducible when something goes wrong.
    rng: random.Random = field(default_factory=random.Random)

    _memories: Dict[str, Memory] = field(default_factory=dict)
    _state: str = IDLE
    _target: str = ""
    _state_since: float = 0.0
    _last_update: Optional[float] = None
    _last_alone: float = 0.0

    #: Metres of the target's own outward travel accumulated this approach.
    _outward_travel: float = 0.0
    _approach_started: float = 0.0
    _goal_sent_for: str = ""
    _last_dance: float = -1e9
    _last_greet: Dict[str, float] = field(default_factory=dict)
    _dance_started: float = 0.0
    _dance_done: bool = True
    _withdrawn_until: float = 0.0
    _robot: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Where the target was standing when the current approach goal was sent.
    _goal_issued_at: Optional[Tuple[float, float]] = None
    #: An externally commanded destination, (x, y, yaw) in the map frame.
    _commanded: Optional[Tuple[float, float, float]] = None
    _commanded_label: str = ""
    _commanded_since: float = 0.0
    _commanded_sent: bool = False

    # ---- memory --------------------------------------------------------

    def memory(self, key: str) -> Memory:
        entry = self._memories.get(key)
        if entry is None:
            entry = Memory(key=key)
            self._memories[key] = entry
        return entry

    def memories(self) -> List[Memory]:
        return list(self._memories.values())

    def _memory_for(self, person: PersonView) -> Memory:
        """Memory for a person, keyed by name when there is one.

        A track id is only stable while the robot can see somebody; a name is
        stable across sessions. When face recognition finally puts a name to a
        track that has been anonymous for a few seconds, whatever was learned
        under the track id is merged into the named record rather than thrown
        away -- without that, the robot forgets a rejection the instant it
        works out who it was talking to.
        """
        if not person.name:
            return self.memory(person.id)

        named = self.memory(person.name)
        anonymous = self._memories.pop(person.id, None)
        if anonymous is not None and anonymous.key != named.key:
            named.affinity = _clamp(named.affinity + anonymous.affinity, -1.0, 1.0)
            named.rejections = max(named.rejections, anonymous.rejections)
            named.encounters += anonymous.encounters
            named.last_seen = max(named.last_seen, anonymous.last_seen)
            named.cooldown_until = max(named.cooldown_until, anonymous.cooldown_until)
        return named

    def to_dict(self) -> dict:
        """Only named people. Track ids mean nothing in the next session.

        Saving anonymous records would grow the file without bound and could
        never match anybody again -- person_4 today is a different human from
        person_4 tomorrow, and restoring a grudge against an id is worse than
        forgetting it.
        """
        return {
            "people": [
                {
                    "name": m.key,
                    "affinity": round(m.affinity, 4),
                    "rejections": m.rejections,
                    "encounters": m.encounters,
                    "last_seen": m.last_seen,
                }
                for m in sorted(self._memories.values(), key=lambda m: m.key)
                if not m.key.startswith("person_") and _worth_keeping(m)
            ]
        }

    def load_dict(self, data: dict) -> int:
        for entry in (data or {}).get("people", []):
            name = entry.get("name")
            if not name:
                continue
            memory = self.memory(name)
            memory.affinity = _clamp(float(entry.get("affinity", 0.0)), -1.0, 1.0)
            memory.rejections = int(entry.get("rejections", 0))
            memory.encounters = int(entry.get("encounters", 0))
            memory.last_seen = float(entry.get("last_seen", 0.0))
            # Cooldowns are absolute times from a previous run's clock and mean
            # nothing now. Starting fresh also means a restart is not a way to
            # bypass one -- the rejection count, which is what actually earns a
            # cooldown, is what persists.
            memory.cooldown_until = 0.0
        return len(self._memories)

    def decay_affinity(self, elapsed: float) -> None:
        """Pull every remembered affinity toward neutral. Call on load."""
        if elapsed <= 0.0 or self.config.affinity_half_life <= 0.0:
            return
        factor = 0.5 ** (elapsed / self.config.affinity_half_life)
        for entry in self._memories.values():
            entry.affinity *= factor

    # ---- commanded trips -------------------------------------------------

    def command(self, x: float, y: float, yaw: float, label: str, t: float) -> None:
        """Send the robot somewhere, overriding its own social judgement.

        Called when something else -- the conversation layer, today -- has
        resolved a request into a pose. It arrives already resolved on purpose:
        working out where the kitchen is is a language problem and belongs
        where the language lives, while owning the Nav2 client is a navigation
        problem and belongs here. What this buys is that there is exactly one
        node sending goals. navigate_to_pose is last-goal-wins, so a second
        client would mean two nodes silently preempting each other, with
        nothing in any log saying which one won -- the same failure the mood
        arbiter exists to prevent for the face.
        """
        self._commanded = (x, y, yaw)
        self._commanded_label = label
        self._commanded_since = t
        self._commanded_sent = False
        self._enter(COMMANDED, t)

    def command_finished(self) -> None:
        """Nav2 reported the commanded goal terminated, however it ended."""
        self._commanded = None

    def commanded(self) -> bool:
        return self._state == COMMANDED

    def _tick_commanded(self, visible: List[PersonView], t: float) -> Decision:
        decision = Decision(state=COMMANDED, people_visible=len(visible))
        decision.expression = EXPR_DETERMINED
        decision.expression_intensity = 0.8
        decision.reason = f"going to {self._commanded_label or 'where I was told'}"

        # Issued once, not every tick. Nav2 replans from scratch on each new
        # goal, and a fixed destination has no reason to be re-sent at all.
        if self._commanded is not None and not self._commanded_sent:
            decision.approach_goal = self._commanded
            self._commanded_sent = True

        # Still look at somebody on the way past. The robot is busy, not rude,
        # and gaze costs nothing.
        if visible:
            nearest = min(visible, key=lambda p: p.distance)
            decision.gaze = (nearest.x, nearest.y, self.config.gaze_height)
            decision.target_id = nearest.id
            decision.target_name = nearest.name
            decision.target_distance = nearest.distance

        arrived = self._commanded is None
        timed_out = (t - self._commanded_since) >= self.config.commanded_timeout
        if arrived or timed_out:
            self._commanded = None
            self._enter(WATCHING if visible else IDLE, t)
            decision.state = self._state
            decision.reason = (
                "got there" if arrived else "gave up on getting there"
            )
            if not visible:
                decision.expression = EXPR_NONE
                decision.gaze = None

        return decision

    # ---- the tick -------------------------------------------------------

    def update(
        self,
        people: List[PersonView],
        t: float,
        robot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Decision:
        """One tick. `robot` is the robot's own (x, y, yaw) in the map frame.

        The robot's pose is only needed to work out which side of a person to
        approach from -- everything else about a person arrives already
        expressed relative to the robot, computed once by person_tracker
        against the pose that matched their timestamp.
        """
        dt = 0.0 if self._last_update is None else max(0.0, t - self._last_update)
        self._last_update = t
        self._robot = robot

        visible = [p for p in people if p.visible]
        for person in people:
            self._memory_for(person).last_seen = t

        # Before target selection, not after. A commanded trip is not one
        # candidate among several -- it is an instruction, and letting the
        # scoring function weigh it against whoever happens to be standing
        # nearby would produce a robot that agrees to go somewhere and then
        # visibly changes its mind.
        if self._state == COMMANDED:
            return self._tick_commanded(visible, t)

        target = self._select_target(visible, t)
        decision = Decision(people_visible=len(visible))

        if target is None:
            self._on_no_target(t, decision)
            return decision

        memory = self._memory_for(target)
        decision.target_id = target.id
        decision.target_name = target.name
        decision.target_distance = target.distance
        decision.affinity = memory.affinity
        decision.rejections = memory.rejections
        # Look at whoever has the robot's attention, in every state that has a
        # target. Gaze is cheap, continuous, and the single strongest signal
        # that a robot is attending to you rather than merely near you.
        decision.gaze = (target.x, target.y, self.config.gaze_height)

        if self._target != target.id:
            self._switch_target(target, t)

        # Transitions and one-shot actions first; the face second, derived from
        # whatever state the tick ended in. Doing both in one pass was the
        # original shape and it was wrong: a handler that set an expression and
        # then transitioned left the decision describing the state the robot
        # had just left, so arriving at somebody published the curious face of
        # walking toward them and the happy face only appeared a tick later.
        handler = {
            IDLE: self._tick_watching,
            SEEKING: self._tick_watching,
            WATCHING: self._tick_watching,
            APPROACHING: self._tick_approaching,
            ENGAGED: self._tick_engaged,
            DANCING: self._tick_dancing,
            WITHDRAWN: self._tick_withdrawn,
        }[self._state]
        handler(target, memory, decision, dt, t)

        decision.state = self._state
        if decision.expression == EXPR_NONE:
            self._express(target, decision)
        decision.rejections = memory.rejections
        decision.affinity = memory.affinity
        return decision

    def _express(self, target: PersonView, decision: Decision) -> None:
        """Face for the state the tick ended in.

        Skipped when a handler already set one, which is how a rejection shows
        disappointment on the tick it happens even though the state it leaves
        behind is an ordinary WATCHING.
        """
        if self._state == APPROACHING:
            # Mildly happy on the way over. A neutral face during the one
            # moment the robot is visibly coming toward you reads as a delivery
            # robot rather than a companion.
            decision.expression = EXPR_HAPPY
            decision.expression_intensity = 0.6
        elif self._state in (ENGAGED, DANCING):
            decision.expression = EXPR_HAPPY
            # Fuller smile for somebody giving the robot their attention.
            decision.expression_intensity = _clamp(
                0.7 + 0.3 * target.engagement, 0.0, 1.0
            )
        elif self._state == WITHDRAWN:
            decision.expression = EXPR_SAD
            decision.expression_intensity = 1.0
            decision.expression_priority = PRIORITY_WITHDRAWN
        else:
            decision.expression = EXPR_CURIOUS
            decision.expression_intensity = 0.8

    # ---- target selection ------------------------------------------------

    def _select_target(self, visible: List[PersonView], t: float) -> Optional[PersonView]:
        """Pick who to attend to, or nobody.

        Scored rather than nearest-wins, because nearest is a poor proxy for
        who wants company: somebody walking past at a metre is nearer than
        somebody standing across the room looking straight at the robot, and
        the second is unambiguously the better choice.
        """
        candidates = [
            p for p in visible if p.distance <= self.config.max_approach_distance
        ]
        if not candidates:
            return None

        def score(person: PersonView) -> float:
            memory = self._memory_for(person)
            proximity = 1.0 - min(1.0, person.distance / self.config.max_approach_distance)
            value = 1.6 * person.engagement + 1.0 * proximity + 0.6 * memory.affinity
            if person.id == self._target:
                value += self.config.target_stickiness
            return value

        return max(candidates, key=score)

    def _switch_target(self, target: PersonView, t: float) -> None:
        self._target = target.id
        self._outward_travel = 0.0
        self._goal_sent_for = ""
        if self._state in (IDLE, SEEKING):
            self._enter(WATCHING, t)

    def _on_no_target(self, t: float, decision: Decision) -> None:
        if self._state == WITHDRAWN and t < self._withdrawn_until:
            # Hold the sad face for its full duration even if they walk out of
            # sight; cutting it short the instant they leave would make the
            # robot look like it stopped caring the moment nobody was watching.
            decision.state = WITHDRAWN
            decision.expression = EXPR_SAD
            decision.expression_priority = PRIORITY_WITHDRAWN
            decision.reason = "they left"
            return

        if self._state != IDLE and self._state != SEEKING:
            self._enter(IDLE, t)
            self._last_alone = t
            decision.cancel_goal = True

        if self._state == IDLE and (t - self._last_alone) >= self.config.seek_after:
            self._enter(SEEKING, t)

        self._target = ""
        decision.state = self._state
        # No expression bid at all when nobody is around: the face belongs to
        # navigation, proximity and battery then, and competing with them for
        # it would be worse than saying nothing.
        decision.expression = EXPR_NONE
        decision.reason = "nobody around"

    # ---- states ----------------------------------------------------------

    def _tick_watching(self, target, memory, decision, dt, t) -> None:
        decision.reason = f"watching {_who(target)}"

        greeting = self._maybe_greet(target, memory, t)
        if greeting is not None:
            decision.gesture = greeting
            decision.reason = f"greeting {_who(target)}"

        if target.distance <= self.config.engage_distance:
            self._enter(ENGAGED, t)
            self._reward_engagement(memory)
            decision.reason = f"{_who(target)} came to me"
            return

        blocked = self._approach_blocked(target, memory, t)
        if blocked:
            decision.reason = f"not approaching {_who(target)}: {blocked}"
            return

        if (t - self._state_since) < self.config.watch_before_approach:
            decision.reason = f"noticing {_who(target)}"
            return

        self._enter(APPROACHING, t)
        self._outward_travel = 0.0
        self._approach_started = t
        decision.approach_goal = self._goal_for(target)
        self._goal_sent_for = target.id
        decision.reason = f"approaching {_who(target)}"

    def _approach_blocked(self, target, memory, t) -> str:
        if t < memory.cooldown_until:
            return f"leaving them alone for another {memory.cooldown_until - t:.0f}s"
        if target.engagement < self.config.min_engagement_to_approach:
            return "they are not facing me"
        if target.distance > self.config.max_approach_distance:
            return "too far away"
        return ""

    def _tick_approaching(self, target, memory, decision, dt, t) -> None:
        decision.reason = f"approaching {_who(target)}"

        # Only the person's own outward motion counts. The robot is closing the
        # gap at the same time, so the actual range rate is usually negative
        # throughout an approach somebody is walking out of.
        if target.range_rate > 0.0:
            self._outward_travel += target.range_rate * dt

        if target.distance <= self.config.engage_distance:
            self._enter(ENGAGED, t)
            self._reward_engagement(memory)
            decision.reason = f"reached {_who(target)}"
            return

        if self._outward_travel >= self.config.rejection_travel:
            self._record_rejection(target, memory, decision, t)
            return

        if (t - self._approach_started) >= self.config.approach_timeout:
            # Never got there, but they never left either. That is a navigation
            # failure, not a snub, and counting it as one would have the robot
            # sulking about its own stuck wheels.
            self._enter(WATCHING, t)
            decision.cancel_goal = True
            self._goal_sent_for = ""
            decision.reason = "approach timed out (not counted as a rejection)"
            return

        # Re-issue the goal only when the person has moved enough to matter.
        # Nav2 replans on every new goal, so streaming one per tick at a
        # walking target keeps it permanently replanning and never driving.
        if self._goal_sent_for != target.id or self._goal_stale(target):
            decision.approach_goal = self._goal_for(target)
            self._goal_sent_for = target.id

    def _tick_engaged(self, target, memory, decision, dt, t) -> None:
        decision.reason = f"with {_who(target)}"
        memory.affinity = _clamp(
            memory.affinity + self.config.engage_rate * dt, -1.0, 1.0
        )

        if target.distance > self.config.disengage_distance:
            self._enter(WATCHING, t)
            decision.reason = f"{_who(target)} moved off"
            return

        gesture = self._maybe_dance(target, memory, t)
        if gesture is not None:
            self._enter(DANCING, t)
            self._dance_started = t
            self._dance_done = False
            decision.gesture = gesture
            decision.cancel_goal = True
            decision.reason = f"dancing for {_who(target)}"

    def _tick_dancing(self, target, memory, decision, dt, t) -> None:
        decision.reason = f"dancing for {_who(target)}"
        # The node reports the gesture finishing; the timeout is the backstop
        # for a gesture that was refused by the feasibility gate and will
        # therefore never report anything at all.
        if self._dance_done or (t - self._dance_started) >= self.config.dance_timeout:
            self._enter(ENGAGED, t)

    def _tick_withdrawn(self, target, memory, decision, dt, t) -> None:
        if t >= self._withdrawn_until:
            # Hand straight over to the watching tick rather than letting this
            # one return, or the decision would spend a tick claiming the robot
            # is still sulking at somebody it has just gone back to watching.
            # One level of re-dispatch, and only in this direction.
            self._enter(WATCHING, t)
            self._tick_watching(target, memory, decision, dt, t)
            return
        decision.reason = f"{_who(target)} kept walking away"

    # ---- events -----------------------------------------------------------

    def _record_rejection(self, target, memory, decision, t) -> None:
        memory.rejections += 1
        memory.affinity = _clamp(
            memory.affinity - self.config.reject_penalty, -1.0, 1.0
        )
        decision.cancel_goal = True
        self._goal_sent_for = ""
        self._outward_travel = 0.0

        if memory.rejections >= self.config.rejections_for_sad:
            memory.cooldown_until = t + self.config.withdraw_cooldown
            self._withdrawn_until = t + self.config.sad_duration
            self._enter(WITHDRAWN, t)
            decision.expression = EXPR_SAD
            decision.expression_priority = PRIORITY_WITHDRAWN
            decision.reason = (
                f"{_who(target)} walked away {memory.rejections} times; "
                f"leaving them alone"
            )
            return

        # Not a pattern yet, just a disappointment. Scale the sadness by how
        # close it is to becoming one, so the first time reads as a flicker
        # and the last as the real thing.
        self._enter(WATCHING, t)
        decision.expression = EXPR_SAD
        decision.expression_intensity = _clamp(
            memory.rejections / max(1, self.config.rejections_for_sad), 0.25, 1.0
        )
        decision.reason = (
            f"{_who(target)} walked away ({memory.rejections} of "
            f"{self.config.rejections_for_sad})"
        )

    def _reward_engagement(self, memory: Memory) -> None:
        # One successful stay clears the record. The claim rejection encodes is
        # "you keep leaving", and staying once falsifies it outright -- carrying
        # the count over would mean a person could never work their way back.
        memory.rejections = 0
        memory.encounters += 1
        memory.affinity = _clamp(
            memory.affinity + self.config.engage_reward, -1.0, 1.0
        )

    def _maybe_dance(self, target, memory, t) -> Optional[str]:
        if (t - self._last_dance) < self.config.dance_cooldown:
            return None
        if memory.affinity < self.config.dance_min_affinity:
            return None
        if target.engagement < self.config.dance_min_engagement:
            return None
        if self.rng.random() >= self.config.dance_probability:
            return None
        self._last_dance = t
        return self.rng.choice(list(self.config.dance_gestures))

    def _maybe_greet(self, target, memory, t) -> Optional[str]:
        """A one-off wiggle for somebody the robot knows, not for a stranger."""
        if not target.name or memory.encounters == 0:
            return None
        last = self._last_greet.get(target.name, -1e9)
        if (t - last) < self.config.greet_cooldown:
            return None
        self._last_greet[target.name] = t
        return self.config.greet_gesture

    def dance_finished(self) -> None:
        self._dance_done = True

    # ---- helpers ----------------------------------------------------------

    def _goal_for(self, target: PersonView) -> Tuple[float, float, float]:
        # personal_space is a floor, not an alternative: a standoff configured
        # tighter than it would walk the robot into somebody.
        standoff = max(self.config.standoff, self.config.personal_space)
        self._goal_issued_at = (target.x, target.y)
        return approach_pose(
            target.x, target.y, target.facing,
            self._robot[0], self._robot[1],
            standoff, self.config.max_front_offset,
        )

    def _goal_stale(self, target: PersonView) -> bool:
        """Has the target moved far enough since the last goal to replan?"""
        if self._goal_issued_at is None:
            return True
        moved = math.hypot(
            target.x - self._goal_issued_at[0], target.y - self._goal_issued_at[1]
        )
        return moved > self.config.replan_distance

    def _enter(self, state: str, t: float) -> None:
        if state != self._state:
            self._state = state
            self._state_since = t

    def state(self) -> str:
        return self._state

    def target(self) -> str:
        return self._target


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _who(person: PersonView) -> str:
    return person.name or person.id


def _worth_keeping(memory: Memory) -> bool:
    """Whether a memory says anything, so blank records are not persisted.

    Rejections count on their own. They usually arrive with a dented affinity,
    but not necessarily -- affinity is clamped and decays, so a long-standing
    friend can be back at neutral while the run of walk-aways that matters for
    whether to approach them is still on the record.
    """
    return (
        memory.encounters > 0
        or memory.rejections > 0
        or abs(memory.affinity) > 1e-6
    )
