"""Semantic object clustering. Pure logic, no ROS graph."""

import pytest

from burgerbot_perception.clustering import ObjectTracker


def test_first_observation_creates_a_new_object():
    t = ObjectTracker()
    obj = t.observe("chair", 1.0, 2.0, 0.5, confidence=0.8, t=0.0)
    assert obj is not None
    assert obj.label == "chair"
    assert obj.id == "chair_0"
    assert obj.observation_count == 1


def test_nearby_same_label_observations_merge_into_one_object():
    t = ObjectTracker(match_radius=0.5)
    t.observe("chair", 1.00, 2.00, 0.5, confidence=0.8, t=0.0)
    t.observe("chair", 1.05, 2.05, 0.5, confidence=0.8, t=1.0)
    t.observe("chair", 0.95, 1.98, 0.5, confidence=0.8, t=2.0)
    objs = t.objects()
    assert len(objs) == 1
    assert objs[0].observation_count == 3


def test_far_apart_observations_become_separate_objects():
    t = ObjectTracker(match_radius=0.5)
    t.observe("chair", 0.0, 0.0, 0.0, confidence=0.8, t=0.0)
    t.observe("chair", 5.0, 5.0, 0.0, confidence=0.8, t=1.0)
    objs = t.objects()
    assert len(objs) == 2
    assert {o.id for o in objs} == {"chair_0", "chair_1"}


def test_different_labels_never_merge_even_at_the_same_point():
    t = ObjectTracker(match_radius=0.5)
    t.observe("chair", 1.0, 1.0, 0.0, confidence=0.9, t=0.0)
    t.observe("table", 1.0, 1.0, 0.0, confidence=0.9, t=0.0)
    objs = t.objects()
    assert len(objs) == 2
    assert {o.label for o in objs} == {"chair", "table"}


def test_low_confidence_detection_is_dropped_not_tracked():
    t = ObjectTracker(min_confidence=0.5)
    obj = t.observe("chair", 1.0, 1.0, 0.0, confidence=0.2, t=0.0)
    assert obj is None
    assert t.objects() == []


def test_low_confidence_detection_does_not_prevent_a_later_real_one():
    t = ObjectTracker(min_confidence=0.5)
    t.observe("chair", 1.0, 1.0, 0.0, confidence=0.2, t=0.0)  # dropped
    obj = t.observe("chair", 1.0, 1.0, 0.0, confidence=0.9, t=1.0)
    assert obj is not None
    assert len(t.objects()) == 1


def test_confidence_weighted_average_favours_the_more_confident_observation():
    # Both observations have to land within match_radius of each other to
    # merge at all -- 1.0 apart, comfortably inside a 2.0 radius.
    t = ObjectTracker(match_radius=2.0)
    # A very confident detection at x=0, a barely-confident one (just above
    # the default min_confidence) at x=1. The merged result should sit much
    # closer to the confident one, not at the midpoint.
    t.observe("chair", x=0.0, y=0.0, z=0.0, confidence=0.95, t=0.0)
    t.observe("chair", x=1.0, y=0.0, z=0.0, confidence=0.41, t=1.0)
    objs = t.objects()
    assert len(objs) == 1  # confirms they actually merged, not two objects
    obj = objs[0]
    assert obj.x < 0.5  # pulled toward the confident observation, not centred
    assert obj.x > 0.0  # but still pulled somewhat by the second observation


def test_last_seen_advances_first_seen_does_not():
    t = ObjectTracker()
    t.observe("chair", 0.0, 0.0, 0.0, confidence=0.9, t=5.0)
    obj = t.observe("chair", 0.0, 0.0, 0.0, confidence=0.9, t=9.0)
    assert obj.first_seen == 5.0
    assert obj.last_seen == 9.0


def test_objects_filtered_by_min_observations():
    t = ObjectTracker(match_radius=0.1)
    t.observe("chair", 0.0, 0.0, 0.0, confidence=0.9, t=0.0)          # seen once
    t.observe("table", 5.0, 5.0, 0.0, confidence=0.9, t=0.0)
    t.observe("table", 5.0, 5.0, 0.0, confidence=0.9, t=1.0)          # seen twice
    assert len(t.objects(min_observations=1)) == 2
    assert len(t.objects(min_observations=2)) == 1
    assert t.objects(min_observations=2)[0].label == "table"


def test_prune_stale_removes_only_old_objects():
    t = ObjectTracker()
    t.observe("chair", 0.0, 0.0, 0.0, confidence=0.9, t=0.0)
    t.observe("table", 5.0, 5.0, 0.0, confidence=0.9, t=100.0)
    dropped = t.prune_stale(now=105.0, max_age=50.0)
    assert dropped == 1
    remaining = t.objects()
    assert len(remaining) == 1
    assert remaining[0].label == "table"


def test_ids_are_stable_and_increment_per_label():
    t = ObjectTracker(match_radius=0.1)
    a = t.observe("chair", 0.0, 0.0, 0.0, confidence=0.9, t=0.0)
    b = t.observe("chair", 10.0, 10.0, 0.0, confidence=0.9, t=0.0)
    c = t.observe("table", 0.0, 0.0, 0.0, confidence=0.9, t=0.0)
    assert a.id == "chair_0"
    assert b.id == "chair_1"
    assert c.id == "table_0"
