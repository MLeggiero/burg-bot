"""Semantic object clustering: pure math, no ROS dependency.

The raw detection stream is noisy and repetitive -- the robot walks past the
same chair fifty times and produces fifty separate 3D detections, each with
slightly different position and confidence. This turns that stream into a
small number of persistent, de-duplicated objects, matching the pure-logic
pattern used for the expression arbiter, the gesture library, and frontier
detection: plain dataclasses in and out, unit-testable on synthetic input
with no ROS graph running.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrackedObject:
    """One persistent object in the semantic map."""

    id: str
    label: str
    x: float
    y: float
    z: float
    confidence: float
    observation_count: int
    first_seen: float
    last_seen: float

    def fold_in(self, x: float, y: float, z: float, confidence: float, t: float) -> None:
        """Merge a new observation into this object's running estimate.

        Confidence-weighted running average, not a plain mean: a detection
        the model was very sure about should pull the position estimate
        harder than one it was barely confident in. Weighting by the new
        observation's own confidence (rather than, say, always weighting
        newer observations more) keeps a single confident detection from
        being diluted away by a long run of marginal ones.
        """
        total_weight = self.confidence * self.observation_count + confidence
        n = self.observation_count + 1
        if total_weight > 1e-6:
            w_old = self.confidence * self.observation_count / total_weight
            w_new = confidence / total_weight
        else:
            w_old, w_new = (self.observation_count / n, 1.0 / n)

        self.x = self.x * w_old + x * w_new
        self.y = self.y * w_old + y * w_new
        self.z = self.z * w_old + z * w_new
        self.confidence = (self.confidence * self.observation_count + confidence) / n
        self.observation_count = n
        self.last_seen = max(self.last_seen, t)


@dataclass
class ObjectTracker:
    """Owns the current set of tracked objects and folds new detections in."""

    #: Detections of the same label within this radius (metres) are treated
    #: as the same physical object rather than a new one. Too small and one
    #: object gets split into several IDs as detections jitter around its
    #: true position; too large and two real nearby objects of the same
    #: class (two chairs at a table) merge into one.
    #:
    #: Needs to clear the object's own width, not just jitter: the observed
    #: point is on whichever surface faces the robot, so it slides across
    #: the object as the robot moves around it.
    match_radius: float = 0.8
    #: Detections below this confidence are dropped before they ever reach
    #: matching -- a low-confidence false positive shouldn't get to seed a
    #: brand new tracked object, only reinforce or be absorbed by a real one.
    min_confidence: float = 0.4

    _objects: List[TrackedObject] = field(default_factory=list)
    _next_index: dict = field(default_factory=dict)  # label -> next numeric suffix

    def observe(self, label: str, x: float, y: float, z: float,
                confidence: float, t: float) -> Optional[TrackedObject]:
        """Fold in one detection. Returns the object it updated, or None if dropped."""
        if confidence < self.min_confidence:
            return None

        best = None
        best_dist = self.match_radius
        for obj in self._objects:
            if obj.label != label:
                continue
            dist = math.sqrt((obj.x - x) ** 2 + (obj.y - y) ** 2 + (obj.z - z) ** 2)
            if dist <= best_dist:
                best = obj
                best_dist = dist

        if best is not None:
            best.fold_in(x, y, z, confidence, t)
            return best

        index = self._next_index.get(label, 0)
        self._next_index[label] = index + 1
        new_obj = TrackedObject(
            id=f"{label}_{index}",
            label=label,
            x=x, y=y, z=z,
            confidence=confidence,
            observation_count=1,
            first_seen=t,
            last_seen=t,
        )
        self._objects.append(new_obj)
        return new_obj

    def objects(self, min_observations: int = 1) -> List[TrackedObject]:
        """Currently tracked objects, optionally filtered by how many times seen.

        A default of 1 (no filtering) at the storage layer; callers doing
        display or persistence should generally pass something higher --
        min_observations=1 means a single false-positive detection produces
        a permanent phantom object, which is exactly the failure mode
        filtering by observation count exists to prevent.
        """
        return [o for o in self._objects if o.observation_count >= min_observations]

    def prune_stale(self, now: float, max_age: float) -> int:
        """Drop objects not observed within max_age seconds. Returns count dropped."""
        before = len(self._objects)
        self._objects = [o for o in self._objects if now - o.last_seen <= max_age]
        return before - len(self._objects)
