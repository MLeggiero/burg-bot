"""Multi-person tracking in the map frame: pure logic, no ROS dependency.

Object clustering (clustering.py) already folds repeated detections of the same
chair into one tracked object, so it is fair to ask why people need their own
tracker. The answer is that clustering assumes the thing it is tracking does
not move -- it takes a confidence-weighted average of every position it has
ever seen, which is exactly right for furniture and exactly wrong for a person
walking across a room, where it would place them at the centroid of their whole
path.

People need the opposite treatment: position estimated from the *latest*
observation, plus a velocity, plus enough memory to survive them stepping
behind a chair for half a second. That is a constant-velocity alpha-beta
filter, and everything else here exists to feed it -- gating, assignment,
coasting, and the confirmation count that stops a single false positive
becoming a person the robot then goes and looks for.

Same pattern as the rest of this workspace: dataclasses in and out, unit
testable on synthetic tracks with no graph running.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .projection import engagement_from_facing, wrap_angle

#: Engagement value meaning "no information", used when a person's orientation
#: cannot be determined. Deliberately the midpoint: an unknown orientation
#: should neither invite approach nor forbid it.
ENGAGEMENT_UNKNOWN = 0.5


@dataclass
class PersonObservation:
    """One person seen in one frame, already projected into the map frame."""

    x: float
    y: float
    z: float
    confidence: float
    #: Map-frame yaw the person is facing, from pose keypoints. None when the
    #: detector produced no keypoints, or produced them but the shoulders were
    #: too foreshortened to give a trustworthy answer.
    facing: Optional[float] = None


@dataclass
class PersonTrack:
    """One person, persisting across frames."""

    id: str
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    #: Best available estimate of which way they are facing, in the map frame.
    #: From keypoints when there are any, otherwise from which way they are
    #: walking -- people overwhelmingly walk forwards, so heading of travel is
    #: a good orientation estimate and costs nothing.
    facing: Optional[float] = None
    facing_from_keypoints: bool = False
    confidence: float = 0.0
    hits: int = 0
    misses: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    #: Filled in by the identity node, not by tracking. Kept on the track so
    #: that a name, once established, survives the frames where face
    #: recognition fails -- which is most of them, since a face is only
    #: recognisable from the front at close range.
    name: str = ""
    name_confidence: float = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits


@dataclass
class RelativeState:
    """A track expressed relative to the robot, which is how behaviour reads it."""

    distance: float
    #: Radians in the robot frame; 0 is dead ahead, positive is to the left.
    bearing: float
    #: The person's own velocity along the line of sight, positive = moving
    #: away. Deliberately *not* the rate the gap is actually closing, which
    #: would also fold in the robot's own motion: the question the social layer
    #: asks is "is this person leaving?", and a robot driving toward a
    #: stationary person must not read as them walking away.
    range_rate: float
    #: 0..1, how squarely they face the robot. ENGAGEMENT_UNKNOWN when their
    #: orientation could not be established.
    engagement: float


def relative_to_robot(
    track: PersonTrack, robot_x: float, robot_y: float, robot_yaw: float
) -> RelativeState:
    dx = track.x - robot_x
    dy = track.y - robot_y
    distance = math.hypot(dx, dy)

    if distance < 1e-6:
        # Standing on the robot. Bearing is meaningless and range rate is
        # numerically explosive; report something harmless rather than a NaN
        # that propagates into a navigation goal.
        return RelativeState(0.0, 0.0, 0.0, ENGAGEMENT_UNKNOWN)

    bearing = wrap_angle(math.atan2(dy, dx) - robot_yaw)
    range_rate = (track.vx * dx + track.vy * dy) / distance

    if track.facing is None:
        engagement = ENGAGEMENT_UNKNOWN
    else:
        # Bearing from the person to the robot -- the reverse of the vector
        # above, since what matters is whether they are turned toward it.
        engagement = engagement_from_facing(track.facing, math.atan2(-dy, -dx))

    return RelativeState(distance, bearing, range_rate, engagement)


@dataclass
class PersonTracker:
    """Owns the live set of person tracks."""

    #: Metres an observation may sit from a track's predicted position and
    #: still be considered the same person. Has to cover a walking pace times
    #: the detection interval: at 1.5 Hz on the Pi, someone walking at 1.4 m/s
    #: moves nearly a metre between frames. Prediction absorbs most of that,
    #: which is why this is not simply set to the worst case -- too wide and
    #: two people passing each other swap identities.
    match_radius: float = 1.0
    #: Seconds a track survives with no detection before being deleted. Long
    #: enough to cross behind furniture, short enough that a person who has
    #: actually left does not linger as a ghost the robot keeps approaching.
    max_coast: float = 1.5
    #: Longest gap between updates over which constant-velocity extrapolation
    #: is still worth anything. Separate from max_coast, which answers a
    #: different question: max_coast is how long a *person* may go unseen,
    #: this is how long the *pipeline* may stall before its predictions become
    #: fiction. Conflating them extrapolates a walking track metres across the
    #: map when the detector hiccups, and the track then never quite catches
    #: back up to the person because each correction only closes part of the
    #: gap it opened.
    max_predict: float = 0.6
    #: Detections needed before a track is reported to anything downstream.
    #: The false positive this exists to suppress is specific and common: a
    #: coat on a chair reads as a person for a frame or two, and without this
    #: the robot goes over to greet it.
    min_hits: int = 3
    min_confidence: float = 0.35

    #: Alpha-beta filter gains. Alpha weights the position correction, beta the
    #: velocity correction. Low beta on purpose -- velocity is differentiated
    #: from noisy positions, so it amplifies depth jitter, and a person
    #: standing still with a twitchy velocity estimate reads as constantly
    #: about to leave, which is exactly the signal the sadness logic watches.
    position_alpha: float = 0.65
    velocity_beta: float = 0.20
    #: Speed below which heading-of-travel is meaningless and is not used as an
    #: orientation estimate. Set above the drift a stationary person's filtered
    #: velocity shows, or a motionless person appears to be facing whichever
    #: way the noise last pushed them.
    min_speed_for_heading: float = 0.25
    #: Velocity retained per second while a track is coasting unseen. A person
    #: who vanished behind a chair probably kept walking, briefly -- but
    #: extrapolating at full speed for a second and a half puts them through a
    #: wall, so the prediction is allowed to decay toward a stop.
    coast_damping: float = 0.35

    _tracks: Dict[str, PersonTrack] = field(default_factory=dict)
    _next_index: int = 0
    _last_update: Optional[float] = None
    _associations: Dict[int, str] = field(default_factory=dict)

    # ---- input ----------------------------------------------------------

    def update(self, observations: List[PersonObservation], t: float) -> List[PersonTrack]:
        """Fold one frame's detections in. Returns the confirmed tracks."""
        dt = 0.0 if self._last_update is None else max(0.0, t - self._last_update)
        self._last_update = t

        # A long gap means the detector stalled or the node was paused. Both
        # halves of the kinematic state are stale across it, not just the
        # position: a velocity measured before a thirty-second pause says
        # nothing about where somebody is now. So hold every track where it was
        # last actually seen, drop the velocities, and let the next matched
        # observation re-establish both.
        if dt > self.max_predict:
            self._reset_kinematics()
            dt = 0.0

        self._predict(dt)

        # The original index of every observation is carried through the
        # confidence filter, because callers need to map their own detections
        # back onto the tracks they became -- see `associations`.
        usable = [(i, o) for i, o in enumerate(observations) if o.confidence >= self.min_confidence]
        matches, unmatched, unmatched_tracks = self._associate([o for _, o in usable])

        self._associations = {}
        for track_id, index in matches:
            original_index, obs = usable[index]
            self._correct(self._tracks[track_id], obs, dt, t)
            self._associations[original_index] = track_id

        for index in unmatched:
            original_index, obs = usable[index]
            self._associations[original_index] = self._spawn(obs, t)

        for track_id in unmatched_tracks:
            self._tracks[track_id].misses += 1

        self._prune(t)
        return self.tracks()

    def _reset_kinematics(self) -> None:
        for track in self._tracks.values():
            track.vx = 0.0
            track.vy = 0.0
            # A motion-derived heading is only as good as the velocity it came
            # from, so it goes with it. One from keypoints was measured, not
            # inferred, and survives.
            if not track.facing_from_keypoints:
                track.facing = None

    def _predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        for track in self._tracks.values():
            track.x += track.vx * dt
            track.y += track.vy * dt
            if track.misses > 0:
                decay = self.coast_damping ** dt
                track.vx *= decay
                track.vy *= decay

    def _associate(
        self, observations: List[PersonObservation]
    ) -> Tuple[List[Tuple[str, int]], List[int], List[str]]:
        """Global-nearest-neighbour assignment between tracks and observations.

        Every in-gate pair is scored, the whole list is sorted by distance, and
        pairs are taken in order, skipping any whose track or observation is
        already spoken for. This is not optimal the way the Hungarian algorithm
        is, but it fixes the failure that plain per-track greedy matching has:
        iterating tracks in arbitrary order lets the first track claim an
        observation that belonged much more clearly to the second. With the
        handful of people a companion robot ever sees at once, the remaining
        gap to optimal is not worth a dependency on scipy.
        """
        pairs = []
        for track_id, track in self._tracks.items():
            for index, obs in enumerate(observations):
                distance = math.hypot(track.x - obs.x, track.y - obs.y)
                if distance <= self.match_radius:
                    pairs.append((distance, track_id, index))
        pairs.sort(key=lambda p: p[0])

        taken_tracks = set()
        taken_obs = set()
        matches = []
        for _distance, track_id, index in pairs:
            if track_id in taken_tracks or index in taken_obs:
                continue
            taken_tracks.add(track_id)
            taken_obs.add(index)
            matches.append((track_id, index))

        unmatched_obs = [i for i in range(len(observations)) if i not in taken_obs]
        unmatched_tracks = [i for i in self._tracks if i not in taken_tracks]
        return matches, unmatched_obs, unmatched_tracks

    def _correct(self, track: PersonTrack, obs: PersonObservation, dt: float, t: float) -> None:
        residual_x = obs.x - track.x
        residual_y = obs.y - track.y

        track.x += self.position_alpha * residual_x
        track.y += self.position_alpha * residual_y
        track.z = obs.z

        if dt > 1e-3:
            track.vx += self.velocity_beta * residual_x / dt
            track.vy += self.velocity_beta * residual_y / dt

        track.confidence = (
            track.confidence * track.hits + obs.confidence
        ) / (track.hits + 1)
        track.hits += 1
        track.misses = 0
        track.last_seen = t
        self._update_facing(track, obs)

    def _update_facing(self, track: PersonTrack, obs: PersonObservation) -> None:
        """Keypoint orientation when there is any, heading of travel otherwise.

        Keypoints win outright rather than being blended with the motion
        estimate, because the two disagree for a real reason and the
        disagreement is informative: somebody walking sideways while looking at
        the robot is genuinely engaged, and averaging the two would report them
        as half-turned-away.
        """
        if obs.facing is not None:
            track.facing = obs.facing
            track.facing_from_keypoints = True
            return

        if track.speed >= self.min_speed_for_heading:
            track.facing = math.atan2(track.vy, track.vx)
            track.facing_from_keypoints = False
        elif not track.facing_from_keypoints:
            # Standing still with no keypoints: nothing to go on. Drop a stale
            # motion-derived heading rather than keep asserting a direction
            # they may have turned away from several seconds ago.
            track.facing = None

    def _spawn(self, obs: PersonObservation, t: float) -> str:
        track_id = f"person_{self._next_index}"
        self._next_index += 1
        self._tracks[track_id] = PersonTrack(
            id=track_id,
            x=obs.x, y=obs.y, z=obs.z,
            facing=obs.facing,
            facing_from_keypoints=obs.facing is not None,
            confidence=obs.confidence,
            hits=1,
            first_seen=t,
            last_seen=t,
        )
        return track_id

    def _prune(self, t: float) -> None:
        for track_id in [
            i for i, tr in self._tracks.items() if t - tr.last_seen > self.max_coast
        ]:
            del self._tracks[track_id]

    # ---- output ---------------------------------------------------------

    def tracks(self) -> List[PersonTrack]:
        """Confirmed tracks only, including ones currently coasting unseen."""
        return [t for t in self._tracks.values() if t.confirmed(self.min_hits)]

    def all_tracks(self) -> List[PersonTrack]:
        """Including unconfirmed ones. For diagnostics, not for behaviour."""
        return list(self._tracks.values())

    def get(self, track_id: str) -> Optional[PersonTrack]:
        return self._tracks.get(track_id)

    def associations(self) -> Dict[int, str]:
        """Last frame's mapping from observation index to the track it became.

        Exists so a caller can carry per-detection information the tracker has
        no use for -- a face recognised in a bounding box, say -- across to the
        right track. Without this the caller has to re-derive the association by
        finding the nearest track to each detection, which is the same
        computation done a second time, with slightly different rules, and
        therefore occasionally a different answer.

        Includes newly spawned tracks, which are not yet confirmed and so do
        not appear in tracks(). That is deliberate: a name established while a
        track is still provisional is ready the moment it is confirmed.
        """
        return dict(self._associations)

    def visible(self, track_id: str) -> bool:
        track = self._tracks.get(track_id)
        return track is not None and track.misses == 0

    def assign_name(self, track_id: str, name: str, confidence: float) -> bool:
        """Attach a recognised identity to a track.

        Only ever upgrades: a weaker match does not overwrite a stronger one.
        Face recognition on a moving person at 2m produces a scatter of
        low-similarity matches, and without this rule a single poor frame
        renames somebody the robot had already confidently identified.
        """
        track = self._tracks.get(track_id)
        if track is None:
            return False
        if track.name and confidence <= track.name_confidence:
            return False
        track.name = name
        track.name_confidence = confidence
        return True
