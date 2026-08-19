"""Person heatmap. Pure logic, no ROS graph."""

import math

import pytest

from burgerbot_companion.heatmap import PersonHeatmap


def occupy(heatmap, x, y, seconds, step=0.1):
    for _ in range(int(seconds / step)):
        heatmap.observe(x, y, step)


# ---- accumulation --------------------------------------------------------


def test_an_empty_heatmap_suggests_nowhere():
    heatmap = PersonHeatmap()
    assert heatmap.hotspots() == []
    assert heatmap.best_target(0.0, 0.0) is None
    assert heatmap.bounds() is None


def test_time_spent_somewhere_accumulates_there():
    heatmap = PersonHeatmap(resolution=0.5)
    occupy(heatmap, 2.0, 3.0, seconds=10.0)
    assert heatmap.value(2.0, 3.0) == pytest.approx(10.0, abs=0.01)
    assert heatmap.value(-5.0, 0.0) == 0.0


def test_nearby_observations_land_in_the_same_cell():
    heatmap = PersonHeatmap(resolution=1.0)
    heatmap.observe(2.1, 3.1, dt=1.0)
    heatmap.observe(2.4, 3.4, dt=1.0)
    assert heatmap.value(2.2, 3.2) == pytest.approx(2.0)


def test_standing_still_outweighs_walking_through():
    """Weighted by duration, not by detection count.

    Counting detections would just record where the detector ran fastest, and
    would rank a two-second transit above a minute of somebody standing there.
    """
    heatmap = PersonHeatmap(resolution=0.5)
    occupy(heatmap, 0.0, 0.0, seconds=60.0)   # standing in the kitchen
    heatmap.observe(5.0, 0.0, dt=2.0)         # walking down the corridor
    hot = heatmap.hotspots()
    assert hot[0][0] == pytest.approx(0.25)
    assert len(hot) == 1  # the corridor is nowhere near the threshold


def test_negative_or_zero_durations_are_ignored():
    heatmap = PersonHeatmap()
    heatmap.observe(1.0, 1.0, dt=0.0)
    heatmap.observe(1.0, 1.0, dt=-5.0)
    heatmap.observe(1.0, 1.0, dt=1.0, weight=0.0)
    assert heatmap.peak() == 0.0


def test_cells_work_either_side_of_the_origin():
    heatmap = PersonHeatmap(resolution=1.0)
    heatmap.observe(-0.5, -0.5, dt=1.0)
    heatmap.observe(0.5, 0.5, dt=1.0)
    assert len(heatmap.cells()) == 2


# ---- decay ---------------------------------------------------------------


def test_old_sightings_fade():
    heatmap = PersonHeatmap(resolution=0.5, half_life=100.0, prune_below=0.0)
    occupy(heatmap, 1.0, 1.0, seconds=10.0)
    heatmap.decay_to(0.0)
    heatmap.decay_to(100.0)
    assert heatmap.value(1.0, 1.0) == pytest.approx(5.0, abs=0.01)


def test_the_first_decay_call_only_starts_the_clock():
    """Otherwise a heatmap loaded from disk is halved by however long ROS time is."""
    heatmap = PersonHeatmap(half_life=100.0)
    occupy(heatmap, 1.0, 1.0, seconds=10.0)
    heatmap.decay_to(1_000_000.0)
    assert heatmap.value(1.0, 1.0) == pytest.approx(10.0, abs=0.01)


def test_faded_cells_are_eventually_dropped_entirely():
    heatmap = PersonHeatmap(half_life=10.0, prune_below=0.05)
    occupy(heatmap, 1.0, 1.0, seconds=1.0)
    heatmap.decay_to(0.0)
    heatmap.decay_to(200.0)
    assert heatmap.cells() == {}


def test_time_going_backwards_does_not_amplify_anything():
    heatmap = PersonHeatmap(half_life=100.0)
    occupy(heatmap, 1.0, 1.0, seconds=10.0)
    heatmap.decay_to(100.0)
    before = heatmap.value(1.0, 1.0)
    heatmap.decay_to(50.0)
    assert heatmap.value(1.0, 1.0) == pytest.approx(before)


def test_a_zero_half_life_disables_fading():
    heatmap = PersonHeatmap(half_life=0.0)
    occupy(heatmap, 1.0, 1.0, seconds=10.0)
    heatmap.decay_to(0.0)
    heatmap.decay_to(1e6)
    assert heatmap.value(1.0, 1.0) == pytest.approx(10.0, abs=0.01)


# ---- choosing somewhere to wait ------------------------------------------


def test_the_robot_is_sent_to_the_busiest_place():
    heatmap = PersonHeatmap(resolution=0.5)
    occupy(heatmap, 4.0, 0.0, seconds=100.0)
    occupy(heatmap, -4.0, 0.0, seconds=20.0)
    target = heatmap.best_target(0.0, 0.0, min_distance=1.0)
    assert target[0] == pytest.approx(4.25)


def test_it_does_not_cross_the_building_for_a_marginally_warmer_spot():
    """A robot that always chases the global peak spends its day in transit."""
    heatmap = PersonHeatmap(resolution=0.5, half_life=0.0)
    occupy(heatmap, 2.0, 0.0, seconds=90.0)     # close, nearly as busy
    occupy(heatmap, 60.0, 0.0, seconds=100.0)   # far, marginally busier
    target = heatmap.best_target(0.0, 0.0, min_distance=1.0, distance_scale=8.0)
    assert target[0] == pytest.approx(2.25)


def test_the_cell_the_robot_is_already_in_is_not_a_destination():
    heatmap = PersonHeatmap(resolution=0.5)
    occupy(heatmap, 0.1, 0.0, seconds=100.0)
    assert heatmap.best_target(0.0, 0.0, min_distance=1.5) is None


def test_recently_visited_places_can_be_excluded():
    heatmap = PersonHeatmap(resolution=0.5, half_life=0.0)
    occupy(heatmap, 4.0, 0.0, seconds=100.0)
    occupy(heatmap, -4.0, 0.0, seconds=80.0)
    target = heatmap.best_target(
        0.0, 0.0, min_distance=1.0, exclude=[(4.25, 0.25)], exclude_radius=1.0
    )
    assert target[0] < 0.0


def test_hotspots_are_returned_hottest_first():
    heatmap = PersonHeatmap(resolution=0.5, half_life=0.0)
    occupy(heatmap, 0.0, 0.0, seconds=50.0)
    occupy(heatmap, 3.0, 0.0, seconds=100.0)
    occupy(heatmap, 6.0, 0.0, seconds=75.0)
    values = [h[2] for h in heatmap.hotspots(min_fraction=0.0)]
    assert values == sorted(values, reverse=True)


def test_the_hotspot_threshold_is_relative_to_the_peak():
    """The units are arbitrary, so any absolute threshold would need retuning."""
    small = PersonHeatmap(resolution=0.5)
    big = PersonHeatmap(resolution=0.5)
    occupy(small, 0.0, 0.0, seconds=10.0)
    occupy(small, 3.0, 0.0, seconds=1.0)
    occupy(big, 0.0, 0.0, seconds=1000.0)
    occupy(big, 3.0, 0.0, seconds=100.0)
    assert len(small.hotspots(0.35)) == len(big.hotspots(0.35)) == 1


# ---- geometry and persistence --------------------------------------------


def test_bounds_cover_every_occupied_cell():
    heatmap = PersonHeatmap(resolution=1.0)
    heatmap.observe(0.5, 0.5, dt=1.0)
    heatmap.observe(3.5, 2.5, dt=1.0)
    assert heatmap.bounds() == (0.0, 0.0, 4.0, 3.0)


def test_a_heatmap_round_trips_through_a_plain_dict():
    heatmap = PersonHeatmap(resolution=0.5)
    occupy(heatmap, 2.0, 3.0, seconds=10.0)
    occupy(heatmap, -1.0, 0.0, seconds=4.0)

    restored = PersonHeatmap(resolution=0.5)
    assert restored.load_dict(heatmap.to_dict()) == 2
    assert restored.value(2.0, 3.0) == pytest.approx(10.0, abs=0.01)
    assert restored.value(-1.0, 0.0) == pytest.approx(4.0, abs=0.01)


def test_loading_at_a_different_resolution_keeps_the_heat_in_the_right_place():
    """Cell indices are meaningless without the resolution that produced them."""
    coarse = PersonHeatmap(resolution=1.0)
    occupy(coarse, 4.5, 2.5, seconds=10.0)

    fine = PersonHeatmap(resolution=0.25)
    fine.load_dict(coarse.to_dict())
    assert fine.value(4.5, 2.5) == pytest.approx(10.0, abs=0.01)
    assert fine.value(40.0, 40.0) == 0.0


def test_loading_junk_does_not_raise():
    heatmap = PersonHeatmap()
    assert heatmap.load_dict({}) == 0
    assert heatmap.load_dict(None) == 0


def test_a_loaded_heatmap_is_not_immediately_halved_by_the_clock():
    heatmap = PersonHeatmap(resolution=0.5, half_life=100.0)
    occupy(heatmap, 1.0, 1.0, seconds=10.0)
    saved = heatmap.to_dict()

    restored = PersonHeatmap(resolution=0.5, half_life=100.0)
    restored.load_dict(saved)
    restored.decay_to(50_000.0)  # ROS time, not seconds since the save
    assert restored.value(1.0, 1.0) == pytest.approx(10.0, abs=0.01)
