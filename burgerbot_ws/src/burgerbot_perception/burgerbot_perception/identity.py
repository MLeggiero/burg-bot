"""Face-embedding gallery and per-track name voting. Pure numpy, no ROS.

Turning "a person" into "Mark" is what lets the companion behaviour remember
anything: affinity, how often somebody actually stops to interact, whether they
have walked away from it three times this week. Without a stable identity all
of that resets the moment a track is lost, which for a robot with a 1.5 Hz
detector is constantly.

Two rules shape this file.

*Enrolment is explicit.* The gallery only ever learns a face when somebody asks
it to, by name. Automatically clustering unknown faces would fill the gallery
with half-seen strangers and, less obviously, would mean the robot builds a
biometric record of everyone who walks past it without anyone choosing that.
Requiring a deliberate "this is Mark" keeps the stored set small, correct, and
something a person opted into.

*Naming a track is a vote, not a lookup.* A single frame's match is unreliable
at any distance, and the failure is asymmetric -- a wrong name is far worse
than no name, because the robot then greets the wrong person and every
per-person memory it has starts accumulating against the wrong record. So
matches accumulate over a window and a name is only claimed once it has won
repeatedly.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


def normalize(embedding) -> np.ndarray:
    """Unit-length copy, so cosine similarity is a plain dot product."""
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


@dataclass
class Identity:
    """One enrolled person and the face embeddings that represent them."""

    name: str
    embeddings: List[np.ndarray] = field(default_factory=list)

    def similarity(self, probe: np.ndarray) -> float:
        """Best match against any stored embedding, -1..1.

        Best rather than mean: the stored embeddings deliberately cover
        different angles and lighting, so averaging their similarities
        penalises a correct match against the one stored view that happens to
        resemble the current frame -- which is the whole point of storing
        several.
        """
        if not self.embeddings:
            return -1.0
        return max(float(np.dot(e, probe)) for e in self.embeddings)


@dataclass
class IdentityGallery:
    """Every enrolled person, and matching against them."""

    #: Cosine similarity below which a probe is called a stranger. Genuinely
    #: model specific -- an ArcFace-style embedding separates identities around
    #: 0.35-0.45, a weaker model far less cleanly. Set it by measuring: enrol
    #: two people, then watch the reported similarity for correct and incorrect
    #: pairings, and put this between the two populations.
    match_threshold: float = 0.42
    #: How far the winner must beat the runner-up. Two people who look alike
    #: both score highly against a given frame, and the gap between them is a
    #: better confidence signal than either absolute score. Without this, the
    #: robot flips between two names frame to frame and looks like it cannot
    #: tell siblings apart -- which, at that point, it cannot.
    margin: float = 0.05
    #: Embeddings kept per person. Enough to cover a few angles; capped so the
    #: gallery does not grow without bound over months of use, and so matching
    #: stays a handful of dot products.
    max_embeddings: int = 12

    _people: Dict[str, Identity] = field(default_factory=dict)

    def enroll(self, name: str, embedding) -> Identity:
        """Add one view of a person's face, creating them if new."""
        vec = normalize(embedding)
        identity = self._people.get(name)
        if identity is None:
            identity = Identity(name=name)
            self._people[name] = identity

        identity.embeddings.append(vec)
        if len(identity.embeddings) > self.max_embeddings:
            # Drop the *most redundant* view rather than the oldest: the point
            # of the stored set is angular coverage, and evicting by age throws
            # away the profile shot that took effort to capture in favour of
            # yet another straight-on one.
            identity.embeddings.pop(_most_redundant_index(identity.embeddings))
        return identity

    def match(self, embedding) -> Tuple[str, float]:
        """Best identity for a probe, or ("", best_similarity) for a stranger."""
        if not self._people:
            return "", -1.0

        probe = normalize(embedding)
        scored = sorted(
            ((identity.similarity(probe), name) for name, identity in self._people.items()),
            reverse=True,
        )
        best_score, best_name = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else -1.0

        if best_score < self.match_threshold or (best_score - runner_up) < self.margin:
            return "", best_score
        return best_name, best_score

    def names(self) -> List[str]:
        return sorted(self._people)

    def forget(self, name: str) -> bool:
        return self._people.pop(name, None) is not None

    # ---- persistence ----------------------------------------------------
    # Plain nested lists so this round-trips through YAML alongside the
    # semantic map's objects.yaml, rather than needing a binary format of its
    # own. A few hundred floats per person is nothing.

    def to_dict(self) -> dict:
        return {
            "people": [
                {"name": name, "embeddings": [e.tolist() for e in identity.embeddings]}
                for name, identity in sorted(self._people.items())
            ]
        }

    def load_dict(self, data: dict) -> int:
        """Replace the gallery from a saved dict. Returns how many people loaded."""
        self._people = {}
        for entry in (data or {}).get("people", []):
            name = entry.get("name")
            if not name:
                continue
            identity = Identity(name=name)
            for raw in entry.get("embeddings", []):
                identity.embeddings.append(normalize(raw))
            if identity.embeddings:
                self._people[name] = identity
        return len(self._people)


def _most_redundant_index(embeddings: List[np.ndarray]) -> int:
    """Index of the embedding closest to all the others."""
    matrix = np.stack(embeddings)
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -1.0)  # ignore self-similarity, always 1
    return int(np.argmax(similarity.max(axis=1)))


@dataclass
class _Vote:
    name: str
    similarity: float
    t: float


@dataclass
class IdentityVoter:
    """Decides a track's name from many frames of matches, not one."""

    #: Seconds of history a vote counts for. Short enough that walking out of
    #: frame and being replaced by somebody else does not inherit the previous
    #: person's name, long enough to survive the many frames where a face is
    #: turned away and matches nothing.
    window: float = 8.0
    #: Winning votes required before a name is claimed at all.
    min_votes: int = 3
    #: Fraction of the votes in the window the winner must hold. Guards the
    #: case min_votes alone misses: three votes for Mark and three for Sam is
    #: not a decision, it is a coin toss with extra steps.
    majority: float = 0.6

    _votes: Dict[str, List[_Vote]] = field(default_factory=dict)

    def vote(self, track_id: str, name: str, similarity: float, t: float) -> None:
        """Record one frame's match. An empty name is a vote for 'stranger'."""
        if not name:
            return
        self._votes.setdefault(track_id, []).append(_Vote(name, similarity, t))

    def best(self, track_id: str, t: float) -> Tuple[str, float]:
        """Winning name and a 0..1 confidence, or ("", 0.0) if undecided."""
        votes = [v for v in self._votes.get(track_id, []) if t - v.t <= self.window]
        self._votes[track_id] = votes
        if len(votes) < self.min_votes:
            return "", 0.0

        tally: Dict[str, List[float]] = {}
        for vote in votes:
            tally.setdefault(vote.name, []).append(vote.similarity)

        name, similarities = max(tally.items(), key=lambda kv: len(kv[1]))
        share = len(similarities) / len(votes)
        if share < self.majority:
            return "", 0.0

        # Confidence folds together how strongly the embeddings matched and how
        # consistently that name won -- either alone is misleading. A single
        # very strong match is not an identification, and ten weak agreeing
        # ones are not either.
        mean_similarity = sum(similarities) / len(similarities)
        return name, float(max(0.0, min(1.0, mean_similarity * share)))

    def forget(self, track_id: str) -> None:
        self._votes.pop(track_id, None)

    def forget_all_except(self, live_track_ids) -> None:
        """Drop vote history for tracks that no longer exist.

        Track ids are never reused while a track is alive, but they are handed
        out afresh as people come and go, and without this the vote table grows
        for as long as the robot is switched on.
        """
        live = set(live_track_ids)
        for track_id in [i for i in self._votes if i not in live]:
            del self._votes[track_id]


def sharpness(gray_patch: np.ndarray) -> float:
    """Variance-of-Laplacian focus measure, for picking a frame worth enrolling.

    Enrolment quality dominates recognition quality: one crisp frontal capture
    beats a dozen motion-blurred ones, and the blurred ones actively hurt by
    dragging the stored set toward a smeared average. This is the cheap
    standard measure -- a 3x3 Laplacian's variance, high for crisp edges, near
    zero for blur -- computed with numpy so it needs no OpenCV import here.
    """
    if gray_patch.size == 0:
        return 0.0
    image = gray_patch.astype(np.float32)
    laplacian = (
        -4.0 * image[1:-1, 1:-1]
        + image[:-2, 1:-1] + image[2:, 1:-1]
        + image[1:-1, :-2] + image[1:-1, 2:]
    )
    if laplacian.size == 0:
        return 0.0
    return float(np.var(laplacian))


def is_frontal(keypoints_xy: Dict[str, Tuple[float, float]], tolerance: float = 0.35) -> bool:
    """Whether a face is square enough to the camera to be worth enrolling.

    Uses the two eyes and the nose: on a frontal face the nose sits near the
    midpoint between the eyes, and it slides toward one of them as the head
    turns. Expressed as a fraction of the inter-eye distance so it is
    resolution and distance independent.
    """
    try:
        left = keypoints_xy["left_eye"]
        right = keypoints_xy["right_eye"]
        nose = keypoints_xy["nose"]
    except KeyError:
        return False

    eye_span = math.hypot(left[0] - right[0], left[1] - right[1])
    if eye_span < 1e-3:
        return False

    midpoint_x = 0.5 * (left[0] + right[0])
    return abs(nose[0] - midpoint_x) / eye_span <= tolerance
