"""Person tracking. Pure logic, no ROS graph."""

import math

import pytest

from burgerbot_perception.person_tracking import (
    ENGAGEMENT_UNKNOWN,
    PersonObservation,
    PersonTracker,
    PersonTrack,
    relative_to_robot,
)


def obs(x, y, confidence=0.9, facing=None, z=0.9):
    return PersonObservation(x=x, y=y, z=z, confidence=confidence, facing=facing)


def feed(tracker, points, start=0.0, dt=0.1, **kwargs):
    """Run a sequence of single-person observations through the tracker."""
    t = start
    for x, y in points:
        tracker.update([obs(x, y, **kwargs)], t)
        t += dt
    return t


# ---- confirmation ------------------------------------------------------


def test_a_single_detection_is_not_yet_a_person():
    tracker = PersonTracker(min_hits=3)
    assert tracker.update([obs(1.0, 0.0)], 0.0) == []
    # It exists internally, it is just not trusted enough to act on.
    assert len(tracker.all_tracks()) == 1


def test_repeated_detections_confirm_a_track():
    tracker = PersonTracker(min_hits=3)
    feed(tracker, [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)])
    tracks = tracker.tracks()
    assert len(tracks) == 1
    assert tracks[0].id == "person_0"
    assert tracks[0].hits == 3


def test_low_confidence_detections_never_confirm_anything():
    tracker = PersonTracker(min_hits=2, min_confidence=0.5)
    feed(tracker, [(1.0, 0.0)] * 5, confidence=0.2)
    assert tracker.tracks() == []
    assert tracker.all_tracks() == []


# ---- association -------------------------------------------------------


def test_two_people_stay_two_tracks():
    tracker = PersonTracker(min_hits=2, match_radius=0.8)
    for i in range(3):
        tracker.update([obs(1.0, 0.0), obs(1.0, 3.0)], i * 0.1)
    assert len(tracker.tracks()) == 2


def test_observations_beyond_the_gate_start_a_new_track():
    tracker = PersonTracker(min_hits=1, match_radius=0.5)
    tracker.update([obs(0.0, 0.0)], 0.0)
    tracker.update([obs(5.0, 0.0)], 0.1)
    assert len(tracker.all_tracks()) == 2


def test_assignment_is_global_not_first_come_first_served():
    """The failure per-track greedy matching has, and this must not have.

    Two tracks, two observations, arranged so that whichever track is
    considered first is closest to the observation that really belongs to the
    other one only if you ignore the better pairing available. Sorting all
    candidate pairs globally picks the assignment with the smallest distances
    overall; iterating tracks in dict order does not.
    """
    tracker = PersonTracker(min_hits=1, match_radius=2.0, position_alpha=1.0)
    tracker.update([obs(0.0, 0.0), obs(1.0, 0.0)], 0.0)
    ids = [t.id for t in tracker.all_tracks()]
    assert ids == ["person_0", "person_1"]

    # person_0 is at 0.0, person_1 at 1.0. Offer observations at 0.9 and 1.05:
    # 1.05 is within gate of both, but pairing it with person_1 (0.05 away)
    # and 0.9 with person_0 is clearly the better global assignment.
    tracker.update([obs(0.9, 0.0), obs(1.05, 0.0)], 0.1)

    by_id = {t.id: t for t in tracker.all_tracks()}
    assert len(by_id) == 2, "a mis-assignment would have spawned a third track"
    assert by_id["person_1"].x == pytest.approx(1.05, abs=1e-6)
    assert by_id["person_0"].x == pytest.approx(0.9, abs=1e-6)


# ---- coasting and deletion ---------------------------------------------


def test_a_track_survives_a_brief_occlusion():
    tracker = PersonTracker(min_hits=2, max_coast=1.0)
    feed(tracker, [(1.0, 0.0), (1.0, 0.0)])
    tracker.update([], 0.5)  # detector saw nobody this frame
    assert len(tracker.tracks()) == 1
    assert not tracker.visible("person_0")


def test_a_track_is_deleted_once_it_has_been_gone_too_long():
    tracker = PersonTracker(min_hits=2, max_coast=1.0)
    feed(tracker, [(1.0, 0.0), (1.0, 0.0)])
    tracker.update([], 5.0)
    assert tracker.tracks() == []


def test_visible_is_false_while_coasting_and_true_again_after_a_hit():
    tracker = PersonTracker(min_hits=2, max_coast=2.0)
    feed(tracker, [(1.0, 0.0), (1.0, 0.0)])
    assert tracker.visible("person_0")
    tracker.update([], 0.3)
    assert not tracker.visible("person_0")
    tracker.update([obs(1.0, 0.0)], 0.4)
    assert tracker.visible("person_0")


def test_coasting_velocity_decays_rather_than_extrapolating_forever():
    """A person who vanished mid-stride should not be predicted through a wall."""
    tracker = PersonTracker(min_hits=2, max_coast=3.0, coast_damping=0.1)
    feed(tracker, [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)], dt=0.5)
    walking_speed = tracker.get("person_0").speed
    assert walking_speed > 0.1

    for i in range(4):
        tracker.update([], 1.5 + i * 0.3)
    assert tracker.get("person_0").speed < walking_speed * 0.5


# ---- velocity ----------------------------------------------------------


def test_velocity_is_estimated_from_consistent_motion():
    tracker = PersonTracker(min_hits=2)
    # Walking +x at 1 m/s, sampled every 0.1s.
    feed(tracker, [(i * 0.1, 0.0) for i in range(30)], dt=0.1)
    track = tracker.get("person_0")
    assert track.vx > 0.5
    assert abs(track.vy) < 0.2


def test_a_stationary_person_does_not_accumulate_velocity():
    tracker = PersonTracker(min_hits=2)
    feed(tracker, [(2.0, 0.0)] * 20, dt=0.1)
    assert tracker.get("person_0").speed < 0.05


# ---- orientation -------------------------------------------------------


def test_keypoint_facing_is_used_directly_when_available():
    tracker = PersonTracker(min_hits=1)
    tracker.update([obs(1.0, 0.0, facing=math.pi)], 0.0)
    track = tracker.get("person_0")
    assert track.facing == pytest.approx(math.pi)
    assert track.facing_from_keypoints


def test_heading_of_travel_is_used_when_there_are_no_keypoints():
    tracker = PersonTracker(min_hits=2, min_speed_for_heading=0.2)
    feed(tracker, [(i * 0.1, 0.0) for i in range(20)], dt=0.1)  # walking +x
    track = tracker.get("person_0")
    assert track.facing == pytest.approx(0.0, abs=0.3)
    assert not track.facing_from_keypoints


def test_a_motionless_person_with_no_keypoints_has_no_claimed_orientation():
    tracker = PersonTracker(min_hits=2, min_speed_for_heading=0.2)
    feed(tracker, [(2.0, 0.0)] * 10, dt=0.1)
    assert tracker.get("person_0").facing is None


def test_keypoint_orientation_is_not_overwritten_by_standing_still():
    tracker = PersonTracker(min_hits=1, min_speed_for_heading=0.2)
    tracker.update([obs(1.0, 0.0, facing=1.0)], 0.0)
    for i in range(5):
        tracker.update([obs(1.0, 0.0)], 0.1 * (i + 1))  # no keypoints now
    assert tracker.get("person_0").facing == pytest.approx(1.0)


# ---- robot-relative view ------------------------------------------------


def test_bearing_is_measured_in_the_robot_frame():
    track = PersonTrack(id="p", x=0.0, y=2.0, z=0.0)
    # Robot at the origin facing +x: someone at +y is 90 degrees to its left.
    state = relative_to_robot(track, 0.0, 0.0, 0.0)
    assert state.distance == pytest.approx(2.0)
    assert state.bearing == pytest.approx(math.pi / 2)

    # Same person, robot now facing +y: they are dead ahead.
    state = relative_to_robot(track, 0.0, 0.0, math.pi / 2)
    assert state.bearing == pytest.approx(0.0, abs=1e-9)


def test_range_rate_is_positive_when_the_person_walks_away():
    track = PersonTrack(id="p", x=2.0, y=0.0, z=0.0, vx=1.0, vy=0.0)
    assert relative_to_robot(track, 0.0, 0.0, 0.0).range_rate == pytest.approx(1.0)


def test_range_rate_is_negative_when_the_person_walks_closer():
    track = PersonTrack(id="p", x=2.0, y=0.0, z=0.0, vx=-1.0, vy=0.0)
    assert relative_to_robot(track, 0.0, 0.0, 0.0).range_rate == pytest.approx(-1.0)


def test_range_rate_ignores_motion_across_the_line_of_sight():
    """Walking past the robot is not walking away from it."""
    track = PersonTrack(id="p", x=2.0, y=0.0, z=0.0, vx=0.0, vy=1.5)
    assert relative_to_robot(track, 0.0, 0.0, 0.0).range_rate == pytest.approx(0.0)


def test_engagement_is_one_facing_the_robot_and_zero_facing_away():
    # Person at +x=2, facing back toward the robot at the origin (yaw pi).
    facing_robot = PersonTrack(id="p", x=2.0, y=0.0, z=0.0, facing=math.pi)
    assert relative_to_robot(facing_robot, 0.0, 0.0, 0.0).engagement == pytest.approx(1.0)

    facing_away = PersonTrack(id="p", x=2.0, y=0.0, z=0.0, facing=0.0)
    assert relative_to_robot(facing_away, 0.0, 0.0, 0.0).engagement == pytest.approx(0.0)


def test_engagement_is_neutral_when_orientation_is_unknown():
    track = PersonTrack(id="p", x=2.0, y=0.0, z=0.0, facing=None)
    assert relative_to_robot(track, 0.0, 0.0, 0.0).engagement == ENGAGEMENT_UNKNOWN


def test_standing_on_the_robot_does_not_produce_nans():
    track = PersonTrack(id="p", x=0.0, y=0.0, z=0.0, vx=1.0, vy=1.0)
    state = relative_to_robot(track, 0.0, 0.0, 0.0)
    assert math.isfinite(state.bearing)
    assert math.isfinite(state.range_rate)
    assert math.isfinite(state.distance)


# ---- naming -------------------------------------------------------------


def test_a_name_can_be_attached_to_a_track():
    tracker = PersonTracker(min_hits=1)
    tracker.update([obs(1.0, 0.0)], 0.0)
    assert tracker.assign_name("person_0", "mark", 0.8)
    assert tracker.get("person_0").name == "mark"


def test_a_weaker_match_does_not_overwrite_an_established_name():
    tracker = PersonTracker(min_hits=1)
    tracker.update([obs(1.0, 0.0)], 0.0)
    tracker.assign_name("person_0", "mark", 0.85)
    assert not tracker.assign_name("person_0", "sam", 0.40)
    assert tracker.get("person_0").name == "mark"


def test_a_stronger_match_does_replace_an_earlier_name():
    tracker = PersonTracker(min_hits=1)
    tracker.update([obs(1.0, 0.0)], 0.0)
    tracker.assign_name("person_0", "sam", 0.45)
    assert tracker.assign_name("person_0", "mark", 0.90)
    assert tracker.get("person_0").name == "mark"


def test_naming_an_unknown_track_is_refused_rather_than_crashing():
    tracker = PersonTracker()
    assert not tracker.assign_name("person_99", "mark", 0.9)


# ---- observation-to-track association ------------------------------------


def test_associations_map_detection_order_onto_track_ids():
    tracker = PersonTracker(min_hits=1, match_radius=0.5)
    tracker.update([obs(0.0, 0.0), obs(5.0, 0.0)], 0.0)
    assert tracker.associations() == {0: "person_0", 1: "person_1"}

    # Same two people, reported in the opposite order this frame.
    tracker.update([obs(5.0, 0.0), obs(0.0, 0.0)], 0.1)
    assert tracker.associations() == {0: "person_1", 1: "person_0"}


def test_associations_survive_the_confidence_filter():
    """Indices must refer to the caller's list, not the filtered one."""
    tracker = PersonTracker(min_hits=1, min_confidence=0.5)
    tracker.update(
        [obs(0.0, 0.0, confidence=0.1), obs(5.0, 0.0, confidence=0.9)], 0.0
    )
    assert tracker.associations() == {1: "person_0"}


def test_associations_are_replaced_each_frame_not_accumulated():
    tracker = PersonTracker(min_hits=1)
    tracker.update([obs(0.0, 0.0)], 0.0)
    tracker.update([], 0.1)
    assert tracker.associations() == {}


# ---- time robustness -----------------------------------------------------


def test_a_long_stall_does_not_extrapolate_tracks_into_nonsense():
    """If the detector stalls, the next frame must not fly the track across the map."""
    tracker = PersonTracker(min_hits=2, max_coast=60.0)
    feed(tracker, [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)], dt=0.5)  # walking
    tracker.update([obs(1.2, 0.0)], 30.0)  # 30 seconds later
    assert tracker.get("person_0").x == pytest.approx(1.2, abs=0.3)


def test_repeated_observations_at_the_same_timestamp_do_not_divide_by_zero():
    tracker = PersonTracker(min_hits=1)
    for _ in range(3):
        tracker.update([obs(1.0, 0.0)], 0.0)
    track = tracker.get("person_0")
    assert math.isfinite(track.vx)
    assert math.isfinite(track.vy)
