"""Turning a place name into somewhere the robot can actually drive.

Pure logic, no ROS, so the resolution rules are testable on synthetic maps.

There is no room segmentation anywhere in this workspace, and nothing here
invents one. The robot has an occupancy grid, a layer of point-shaped detected
objects, and a heatmap of where people tend to be. None of those says that a
region *is* a kitchen, and pretending otherwise produces the worst failure this
feature can have: a robot that drives confidently to the wrong room and has no
way to tell it is wrong.

So resolution happens in three tiers, and the order is the design:

  1. **Taught.** Somebody drove the robot somewhere and named it. Always
     correct, because it is a statement of fact rather than an inference.
  2. **A tracked object.** "go to the chair" resolves against the semantic map.
     Correct as often as the detector is.
  3. **A hotspot.** "go where people are" resolves against the heatmap. Answers
     a social question, not a spatial one.

Nothing infers a room label. When a name matches nothing, this returns a miss
and the robot says it does not know -- which is a better robot than one that
guesses, and which is also exactly the prompt to teach it.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: Words that carry no information in a place name, so "the kitchen" and
#: "kitchen" are the same request.
_FILLER = {"the", "a", "an", "to", "my", "our", "your", "over", "in", "at"}


@dataclass
class Place:
    """Somewhere with a name, in the map frame."""

    name: str
    x: float
    y: float
    #: Which way to face on arrival. For a taught place this is the heading the
    #: robot had when it was named, which is usually the useful direction --
    #: somebody standing it at the kitchen counter was probably facing the
    #: counter.
    yaw: float = 0.0

    def to_dict(self) -> dict:
        return {"name": self.name, "x": round(self.x, 3),
                "y": round(self.y, 3), "yaw": round(self.yaw, 4)}


@dataclass
class Resolution:
    """What a name resolved to, and how confident that is."""

    #: (x, y, yaw), or None when nothing matched.
    goal: Optional[Tuple[float, float, float]] = None
    #: "taught", "object", "hotspot" or "" when nothing matched.
    source: str = ""
    #: The name that actually matched, which may differ from what was asked
    #: for -- worth reporting so a mismatch is visible rather than silent.
    matched: str = ""
    #: Plain language, surfaced in the reply when resolution fails. Written to
    #: be said out loud.
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.goal is not None


def normalise(name: str) -> str:
    """Fold a spoken place name to something comparable.

    Lowercased, punctuation dropped, filler words removed. "The Kitchen!" and
    "kitchen" have to be the same key or teaching a place by voice and asking
    for it by voice will not agree.
    """
    cleaned = re.sub(r"[^a-z0-9\s]", " ", name.lower())
    words = [w for w in cleaned.split() if w and w not in _FILLER]
    return " ".join(words)


@dataclass
class PlaceBook:
    """The places the robot has been taught, and how a name resolves."""

    #: How close a tracked object has to be to the requested label to count.
    #: Objects are points, so this is about matching names, not geometry.
    _places: Dict[str, Place] = field(default_factory=dict)

    # ---- teaching --------------------------------------------------------

    def teach(self, name: str, x: float, y: float, yaw: float = 0.0) -> Optional[Place]:
        """Record a place. Naming an existing one moves it.

        Moving rather than refusing is the useful behaviour: the common reason
        to name somewhere twice is that the first attempt was parked slightly
        wrong.
        """
        key = normalise(name)
        if not key:
            return None
        place = Place(name=key, x=x, y=y, yaw=yaw)
        self._places[key] = place
        return place

    def forget(self, name: str) -> bool:
        return self._places.pop(normalise(name), None) is not None

    def names(self) -> List[str]:
        return sorted(self._places)

    def get(self, name: str) -> Optional[Place]:
        return self._places.get(normalise(name))

    # ---- resolution ------------------------------------------------------

    def resolve(
        self,
        name: str,
        objects: Sequence[Tuple[str, float, float]] = (),
        hotspots: Sequence[Tuple[float, float, float]] = (),
        robot: Tuple[float, float] = (0.0, 0.0),
        standoff: float = 0.9,
    ) -> Resolution:
        """Resolve a spoken name against the three tiers, best first.

        `objects` are (label, x, y) in the map frame from the semantic map;
        `hotspots` are (x, y, heat) from the person heatmap.
        """
        key = normalise(name)
        if not key:
            return Resolution(reason="I did not catch where you meant.")

        taught = self._places.get(key)
        if taught is not None:
            return Resolution(
                goal=(taught.x, taught.y, taught.yaw),
                source="taught", matched=taught.name,
            )

        # Tier 2. Nearest object whose label matches, approached to a standoff
        # rather than driven into -- the stored point is on the object's near
        # surface, so the goal has to be pulled back toward the robot or Nav2
        # is asked to park inside the furniture.
        best_object = _nearest_matching(key, objects, robot)
        if best_object is not None:
            label, ox, oy = best_object
            return Resolution(
                goal=_standoff_pose(ox, oy, robot, standoff),
                source="object", matched=label,
            )

        # Tier 3. Only for genuinely social phrasings. "kitchen" must never
        # fall through to "wherever people happen to stand", which would be a
        # confident answer to a question that was not asked.
        if key in ("people", "everyone", "somebody", "anyone", "company") and hotspots:
            hottest = max(hotspots, key=lambda h: h[2])
            return Resolution(
                goal=_standoff_pose(hottest[0], hottest[1], robot, standoff),
                source="hotspot", matched="where people usually are",
            )

        return Resolution(
            reason=(
                f"I do not know where {key} is. Take me there and tell me, and "
                f"I will remember it."
            )
        )

    # ---- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {"places": [p.to_dict() for _, p in sorted(self._places.items())]}

    def load_dict(self, data: dict) -> int:
        self._places = {}
        for entry in (data or {}).get("places", []):
            name = entry.get("name")
            if not name:
                continue
            try:
                self.teach(name, float(entry["x"]), float(entry["y"]),
                           float(entry.get("yaw", 0.0)))
            except (KeyError, TypeError, ValueError):
                # A place with no usable coordinates is not recoverable, and
                # dropping it beats refusing to load every other place in the
                # file because one line was hand-edited badly.
                continue
        return len(self._places)


def _nearest_matching(
    key: str,
    objects: Sequence[Tuple[str, float, float]],
    robot: Tuple[float, float],
) -> Optional[Tuple[str, float, float]]:
    """The closest object whose label matches the request, or None.

    Matched on whole words in either direction, so "chair" finds "chair" and
    "the dining chair" finds "chair" -- but "air" does not, which a plain
    substring test would allow.
    """
    matches = [
        obj for obj in objects
        if _label_matches(key, normalise(obj[0]))
    ]
    if not matches:
        return None
    return min(matches, key=lambda o: math.hypot(o[1] - robot[0], o[2] - robot[1]))


def _label_matches(request: str, label: str) -> bool:
    if not request or not label:
        return False
    request_words = set(request.split())
    label_words = set(label.split())
    return bool(request_words & label_words)


def _standoff_pose(
    x: float, y: float, robot: Tuple[float, float], standoff: float
) -> Tuple[float, float, float]:
    """A pose `standoff` metres short of a point, facing it.

    Same reasoning as the companion's approach_pose: arrive looking at the
    thing you were sent to, not at a wall. Pulling back along the line from the
    robot is the cheap approximation -- it has no idea what is between the two,
    which is Nav2's problem and Nav2 is good at it.
    """
    dx, dy = x - robot[0], y - robot[1]
    distance = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    if distance <= standoff:
        # Already close enough. Keep the robot where it is and just turn to
        # face the thing, rather than reversing away from it to make room.
        return (robot[0], robot[1], yaw)
    scale = (distance - standoff) / distance
    return (robot[0] + dx * scale, robot[1] + dy * scale, yaw)
