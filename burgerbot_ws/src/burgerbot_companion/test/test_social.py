"""Companion social behaviour. Pure logic, no ROS graph.

The behaviour these cover is the sort you cannot test any other way: nobody can
walk away from a robot three times on cue, and certainly not reproducibly. As a
pure function of a list of people and a timestamp, it is ordinary to test.
"""

import math
import random

import pytest

from burgerbot_companion.social import (
    APPROACHING,
    DANCING,
    ENGAGED,
    EXPR_HAPPY,
    EXPR_NONE,
    EXPR_SAD,
    IDLE,
    SEEKING,
    WATCHING,
    WITHDRAWN,
    PersonView,
    SocialBrain,
    SocialConfig,
    approach_pose,
)


def person(distance=3.0, engagement=1.0, range_rate=0.0, x=None, y=0.0,
           pid="person_0", name="", visible=True, facing=None, bearing=0.0):
    """A person in front of the robot, which sits at the origin facing +x."""
    return PersonView(
        id=pid,
        x=distance if x is None else x,
        y=y,
        distance=distance,
        bearing=bearing,
        range_rate=range_rate,
        engagement=engagement,
        visible=visible,
        name=name,
        facing=facing,
    )


def brain(**overrides):
    return SocialBrain(config=SocialConfig(**overrides), rng=random.Random(0))


def settle(b, who, start=0.0, ticks=3, dt=0.6):
    """Tick a few times so the 'notice them first' dwell elapses.

    The robot deliberately watches somebody for a beat before committing to an
    approach, so any test about what it does *after* deciding has to get past
    that first.
    """
    t = start
    decision = None
    for _ in range(ticks):
        decision = b.update([who] if not isinstance(who, list) else who, t)
        t += dt
    return decision, t


# ---- nobody around -------------------------------------------------------


def test_with_nobody_around_the_face_is_left_alone():
    """Bidding nothing is not the same as bidding neutral.

    In IDLE the face belongs to navigation, proximity and battery; competing
    with them for it would be worse than saying nothing.
    """
    decision = brain().update([], 0.0)
    assert decision.state == IDLE
    assert decision.expression == EXPR_NONE
    assert decision.gaze is None


def test_after_long_enough_alone_the_robot_goes_looking():
    b = brain(seek_after=25.0)
    b.update([], 0.0)
    assert b.update([], 10.0).state == IDLE
    assert b.update([], 30.0).state == SEEKING


def test_someone_appearing_ends_the_search():
    b = brain(seek_after=5.0)
    b.update([], 0.0)
    b.update([], 10.0)
    assert b.state() == SEEKING
    assert b.update([person()], 11.0).state == WATCHING


# ---- noticing and approaching --------------------------------------------


def test_someone_facing_the_robot_is_approached():
    b = brain()
    decision, _ = settle(b, person(distance=3.0, engagement=1.0))
    assert decision.state == APPROACHING


def test_the_robot_notices_somebody_before_setting_off_after_them():
    """Departing the instant a track confirms reads as a trigger, not a choice."""
    b = brain(watch_before_approach=1.0)
    first = b.update([person(distance=3.0, engagement=1.0)], 0.0)
    assert first.state == WATCHING
    assert first.approach_goal is None
    assert b.update([person(distance=3.0, engagement=1.0)], 1.5).state == APPROACHING


def test_the_approach_goal_is_issued_on_the_tick_the_decision_is_made():
    b = brain()
    b.update([person(distance=3.0, engagement=1.0)], 0.0)
    decision = b.update([person(distance=3.0, engagement=1.0)], 1.5)
    assert decision.approach_goal is not None


def test_someone_facing_away_is_watched_not_approached():
    """Distance alone is a bad reason to walk up to somebody."""
    b = brain()
    decision = b.update([person(distance=3.0, engagement=0.1)], 0.0)
    assert decision.state == WATCHING
    assert decision.approach_goal is None
    assert "facing" in decision.reason


def test_unknown_orientation_does_not_trigger_an_approach():
    """0.5 is the tracker's 'no information'. Watching is the right way to be wrong."""
    b = brain()
    assert b.update([person(distance=3.0, engagement=0.5)], 0.0).state == WATCHING


def test_the_robot_looks_at_whoever_it_is_attending_to():
    b = brain(gaze_height=1.5)
    decision = b.update([person(distance=3.0, x=2.0, y=1.0)], 0.0)
    assert decision.gaze == (2.0, 1.0, 1.5)


def test_someone_who_walks_up_to_the_robot_is_engaged_without_an_approach():
    b = brain(engage_distance=1.4)
    decision = b.update([person(distance=1.0)], 0.0)
    assert decision.state == ENGAGED
    assert decision.approach_goal is None


def test_arriving_makes_the_robot_happy():
    """On the tick it arrives, not a tick later."""
    b = brain()
    decision = b.update([person(distance=1.0)], 0.0)
    assert decision.state == ENGAGED
    assert decision.expression == EXPR_HAPPY
    assert decision.expression_intensity > 0.7


def test_engagement_intensity_scales_with_attention():
    attentive = brain().update([person(distance=1.0, engagement=1.0)], 0.0)
    distracted = brain().update([person(distance=1.0, engagement=0.2)], 0.0)
    assert attentive.expression_intensity > distracted.expression_intensity


def test_moving_off_ends_the_engagement():
    b = brain(disengage_distance=2.4)
    b.update([person(distance=1.0)], 0.0)
    assert b.state() == ENGAGED
    assert b.update([person(distance=3.0)], 1.0).state == WATCHING


# ---- target selection ----------------------------------------------------


def test_the_person_paying_attention_beats_the_person_who_is_merely_nearer():
    b = brain()
    passer_by = person(distance=1.5, engagement=0.05, pid="passer")
    interested = person(distance=3.5, engagement=1.0, pid="interested")
    assert b.update([passer_by, interested], 0.0).target_id == "interested"


def test_attention_does_not_flip_between_two_similar_people():
    """Without hysteresis the robot swaps target every tick and reads as broken."""
    b = brain(target_stickiness=0.15)
    a = person(distance=3.0, engagement=0.8, pid="a")
    c = person(distance=3.02, engagement=0.8, pid="c")
    first = b.update([a, c], 0.0).target_id
    for i in range(20):
        # Jitter the other one to just barely better, repeatedly.
        other = person(distance=2.98, engagement=0.81,
                       pid="c" if first == "a" else "a")
        mine = a if first == "a" else c
        assert b.update([mine, other], 0.1 * (i + 1)).target_id == first


def test_people_beyond_the_attention_range_are_ignored():
    b = brain(max_approach_distance=6.0)
    assert b.update([person(distance=12.0)], 0.0).state == IDLE


def test_a_coasting_track_is_not_treated_as_present():
    b = brain()
    assert b.update([person(distance=2.0, visible=False)], 0.0).state == IDLE


# ---- walking away: the sad path ------------------------------------------


def walk_away(b, start_t=0.0, distance=3.0, rate=0.6, dt=0.2, pid="person_0"):
    """Stand still until the robot commits to an approach, then walk away.

    Two phases because the robot deliberately watches for a beat before
    setting off, and rejection is only measured during an approach it actually
    committed to -- somebody leaving before that was never being approached.
    Returns as soon as a rejection is recorded, so a caller can count them one
    at a time.
    """
    t = start_t
    before = b.memory(pid).rejections
    decision = None
    for _ in range(10):
        decision = b.update([person(distance=distance, pid=pid)], t)
        t += dt

    travelled = 0.0
    for _ in range(30):
        travelled += rate * dt
        decision = b.update(
            [person(distance=distance + travelled, range_rate=rate, pid=pid)], t
        )
        t += dt
        if decision.rejections > before:
            break
    return decision, t


def test_a_person_walking_away_during_an_approach_is_a_rejection():
    b = brain(rejection_travel=1.2, rejections_for_sad=3)
    decision, _ = walk_away(b)
    assert decision.rejections == 1
    assert decision.expression == EXPR_SAD
    assert decision.state == WATCHING  # disappointed, not withdrawn


def test_the_first_rejection_is_milder_than_the_last():
    b = brain(rejections_for_sad=3)
    first, t = walk_away(b)
    mild = first.expression_intensity
    second, t = walk_away(b, start_t=t)
    assert second.expression_intensity > mild


def test_three_walk_aways_make_the_robot_withdraw():
    b = brain(rejections_for_sad=3)
    t = 0.0
    for _ in range(3):
        decision, t = walk_away(b, start_t=t)
    assert decision.rejections == 3
    assert decision.state == WITHDRAWN
    assert decision.expression == EXPR_SAD
    assert decision.expression_intensity == pytest.approx(1.0)


def test_a_withdrawn_from_person_is_left_alone_afterwards():
    b = brain(rejections_for_sad=3, withdraw_cooldown=120.0, sad_duration=8.0)
    t = 0.0
    for _ in range(3):
        _, t = walk_away(b, start_t=t)

    # After the sadness passes, the same person standing there attentively must
    # not trigger another chase.
    decision = b.update([person(distance=3.0, engagement=1.0)], t + 20.0)
    assert decision.state == WATCHING
    assert decision.approach_goal is None
    assert "alone" in decision.reason

    # Once the cooldown expires, it is willing to try again.
    b.update([person(distance=3.0, engagement=1.0)], t + 200.0)
    assert b.update([person(distance=3.0, engagement=1.0)], t + 200.1).state == APPROACHING


def test_the_sad_face_holds_even_after_they_are_out_of_sight():
    """Cutting it short the moment nobody is watching would read as pretence."""
    b = brain(rejections_for_sad=3, sad_duration=8.0)
    t = 0.0
    for _ in range(3):
        _, t = walk_away(b, start_t=t)
    decision = b.update([], t + 1.0)
    assert decision.state == WITHDRAWN
    assert decision.expression == EXPR_SAD


def test_someone_walking_across_the_view_is_not_walking_away():
    """Range rate is along the line of sight for exactly this reason."""
    b = brain(rejection_travel=1.2)
    t = 0.0
    for i in range(40):
        decision = b.update(
            [person(distance=3.0, range_rate=0.0, y=-2.0 + i * 0.1)], t
        )
        t += 0.1
    assert decision.rejections == 0
    assert decision.state == APPROACHING


def test_the_robots_own_approach_is_never_mistaken_for_them_leaving():
    """The gap closes fast while the person stands perfectly still."""
    b = brain(rejection_travel=1.2)
    t = 0.0
    for distance in [4.0, 3.5, 3.0, 2.5, 2.0, 1.6]:
        decision = b.update([person(distance=distance, range_rate=0.0)], t)
        t += 0.5
    assert decision.rejections == 0


def test_staying_once_clears_the_record():
    b = brain(rejections_for_sad=3)
    decision, t = walk_away(b)
    assert decision.rejections == 1
    # Now they come back and stay.
    decision = b.update([person(distance=1.0)], t + 1.0)
    assert decision.state == ENGAGED
    assert decision.rejections == 0


def test_an_approach_that_simply_never_arrives_is_not_a_rejection():
    """A stuck robot must not sulk about its own wheels."""
    b = brain(approach_timeout=30.0, rejection_travel=1.2)
    settle(b, person(distance=3.0))
    assert b.state() == APPROACHING
    decision = b.update([person(distance=3.0, range_rate=0.0)], 40.0)
    assert decision.rejections == 0
    assert decision.state == WATCHING
    assert decision.cancel_goal


def test_rejection_lowers_affinity_and_engagement_raises_it():
    b = brain()
    rejected, t = walk_away(b)
    assert rejected.affinity < 0.0
    engaged = b.update([person(distance=1.0)], t + 1.0)
    assert engaged.affinity > rejected.affinity


# ---- dancing --------------------------------------------------------------


def test_the_robot_eventually_dances_for_someone_it_likes():
    b = SocialBrain(
        config=SocialConfig(dance_probability=1.0, dance_cooldown=0.0,
                            dance_min_affinity=-1.0),
        rng=random.Random(1),
    )
    b.update([person(distance=1.0, engagement=1.0)], 0.0)  # arrives
    decision = b.update([person(distance=1.0, engagement=1.0)], 0.5)
    assert decision.state == DANCING
    assert decision.gesture in SocialConfig().dance_gestures
    # Navigation must be out of the way before the robot starts moving to a
    # rhythm of its own.
    assert decision.cancel_goal


def test_dancing_is_occasional_not_automatic():
    """A robot that dances every single time is a mechanism, not a character."""
    b = SocialBrain(
        config=SocialConfig(dance_probability=0.0, dance_cooldown=0.0),
        rng=random.Random(2),
    )
    b.update([person(distance=1.0, engagement=1.0)], 0.0)
    decision = b.update([person(distance=1.0, engagement=1.0)], 0.5)
    assert decision.state == ENGAGED
    assert decision.gesture is None


def test_the_robot_does_not_dance_at_the_back_of_someones_head():
    b = SocialBrain(
        config=SocialConfig(dance_probability=1.0, dance_cooldown=0.0,
                            dance_min_engagement=0.5, dance_min_affinity=-1.0,
                            min_engagement_to_approach=0.0),
        rng=random.Random(3),
    )
    b.update([person(distance=1.0, engagement=0.1)], 0.0)
    assert b.update([person(distance=1.0, engagement=0.1)], 0.5).state == ENGAGED


def test_the_robot_does_not_dance_for_someone_it_has_been_rebuffed_by():
    b = SocialBrain(
        config=SocialConfig(dance_probability=1.0, dance_cooldown=0.0,
                            dance_min_affinity=0.05, rejections_for_sad=9),
        rng=random.Random(4),
    )
    _, t = walk_away(b)  # affinity now negative
    b.update([person(distance=1.0, engagement=1.0)], t + 1.0)
    decision = b.update([person(distance=1.0, engagement=1.0)], t + 1.5)
    assert decision.state == ENGAGED
    assert decision.gesture is None


def test_dances_are_spaced_out_by_the_cooldown():
    b = SocialBrain(
        config=SocialConfig(dance_probability=1.0, dance_cooldown=45.0,
                            dance_min_affinity=-1.0),
        rng=random.Random(5),
    )
    b.update([person(distance=1.0)], 0.0)
    assert b.update([person(distance=1.0)], 0.5).state == DANCING
    b.dance_finished()
    b.update([person(distance=1.0)], 1.0)  # back to ENGAGED
    assert b.update([person(distance=1.0)], 2.0).gesture is None


def test_a_dance_ends_when_the_gesture_reports_finishing():
    b = SocialBrain(
        config=SocialConfig(dance_probability=1.0, dance_cooldown=45.0,
                            dance_min_affinity=-1.0),
        rng=random.Random(6),
    )
    b.update([person(distance=1.0)], 0.0)
    b.update([person(distance=1.0)], 0.5)
    assert b.state() == DANCING
    b.dance_finished()
    assert b.update([person(distance=1.0)], 1.0).state == ENGAGED


def test_a_dance_that_never_reports_back_still_times_out():
    """A gesture refused by the feasibility gate reports nothing at all."""
    b = SocialBrain(
        config=SocialConfig(dance_probability=1.0, dance_cooldown=1e9,
                            dance_min_affinity=-1.0, dance_timeout=12.0),
        rng=random.Random(7),
    )
    b.update([person(distance=1.0)], 0.0)
    b.update([person(distance=1.0)], 0.5)
    assert b.state() == DANCING
    assert b.update([person(distance=1.0)], 20.0).state == ENGAGED


# ---- proxemics ------------------------------------------------------------


def test_the_approach_stops_at_a_standoff_rather_than_at_the_person():
    goal_x, goal_y, _yaw = approach_pose(
        person_x=5.0, person_y=0.0, person_facing=math.pi,
        robot_x=0.0, robot_y=0.0, standoff=1.1, max_front_offset=1.05,
    )
    assert math.hypot(goal_x - 5.0, goal_y - 0.0) == pytest.approx(1.1)


def test_the_robot_arrives_facing_the_person():
    goal_x, goal_y, yaw = approach_pose(
        person_x=5.0, person_y=2.0, person_facing=None,
        robot_x=0.0, robot_y=0.0, standoff=1.1, max_front_offset=1.05,
    )
    bearing_to_person = math.atan2(2.0 - goal_y, 5.0 - goal_x)
    assert math.sin(yaw - bearing_to_person) == pytest.approx(0.0, abs=1e-9)


def test_a_robot_already_in_front_of_someone_does_not_swing_around():
    # Person at x=5 facing back toward the robot at the origin.
    goal_x, goal_y, _ = approach_pose(
        person_x=5.0, person_y=0.0, person_facing=math.pi,
        robot_x=0.0, robot_y=0.0, standoff=1.1, max_front_offset=1.05,
    )
    assert goal_x == pytest.approx(3.9)
    assert goal_y == pytest.approx(0.0, abs=1e-9)


def test_the_robot_comes_around_rather_than_approaching_from_behind():
    """Arriving at somebody's back is the most unsettling thing a robot can do."""
    # Person at the origin facing +x; robot behind them at -5.
    goal_x, goal_y, _ = approach_pose(
        person_x=0.0, person_y=0.0, person_facing=0.0,
        robot_x=-5.0, robot_y=0.0, standoff=1.1, max_front_offset=1.05,
    )
    approach_bearing = math.atan2(goal_y, goal_x)
    assert abs(approach_bearing) <= 1.05 + 1e-9, "goal must lie in the frontal cone"
    assert goal_x > 0.0, "goal is in front of them, not behind"


def test_coming_around_takes_the_nearer_edge_of_the_frontal_cone():
    """The shorter way round, not always the same way round."""
    from_the_left = approach_pose(0.0, 0.0, 0.0, -5.0, 1.0, 1.1, 1.05)
    from_the_right = approach_pose(0.0, 0.0, 0.0, -5.0, -1.0, 1.1, 1.05)
    assert from_the_left[1] > 0.0
    assert from_the_right[1] < 0.0


def test_an_unknown_orientation_is_approached_from_where_the_robot_already_is():
    goal_x, goal_y, _ = approach_pose(
        person_x=0.0, person_y=0.0, person_facing=None,
        robot_x=-5.0, robot_y=0.0, standoff=1.1, max_front_offset=1.05,
    )
    assert goal_x == pytest.approx(-1.1)
    assert goal_y == pytest.approx(0.0, abs=1e-9)


def test_personal_space_is_a_floor_the_standoff_cannot_undercut():
    b = brain(standoff=0.2, personal_space=0.75, min_engagement_to_approach=0.0,
              engage_distance=0.1)
    decision, _ = settle(b, person(distance=3.0, x=3.0))
    goal = decision.approach_goal
    assert math.hypot(goal[0] - 3.0, goal[1]) >= 0.75 - 1e-9


def test_the_goal_is_not_re_sent_every_tick_for_a_stationary_person():
    """Nav2 replans from scratch on each new goal; streaming them never drives."""
    b = brain(replan_distance=0.4)
    decision, t = settle(b, person(distance=3.0))
    assert decision.approach_goal is not None
    resent = [
        b.update([person(distance=3.0)], t + 0.1 * i).approach_goal
        for i in range(10)
    ]
    assert all(goal is None for goal in resent)


def test_the_goal_is_re_sent_once_the_person_has_moved_enough():
    b = brain(replan_distance=0.4, rejection_travel=99.0)
    _, t = settle(b, person(distance=3.0, x=3.0))
    assert b.update([person(distance=3.6, x=3.6)], t).approach_goal is not None


# ---- memory ---------------------------------------------------------------


def test_what_was_learned_anonymously_survives_being_given_a_name():
    """Recognition lands seconds after a track starts; the history must carry."""
    b = brain(rejections_for_sad=9)
    _, t = walk_away(b, pid="person_0")
    assert b.memory("person_0").rejections == 1

    # Face recognition catches up and the same track is now known to be mark.
    b.update([person(distance=3.0, pid="person_0", name="mark")], t + 0.1)
    assert b.memory("mark").rejections == 1
    assert b.memory("mark").affinity < 0.0


def test_affinity_decays_toward_neutral_over_time():
    b = brain(affinity_half_life=100.0)
    b.memory("mark").affinity = 0.8
    b.decay_affinity(100.0)
    assert b.memory("mark").affinity == pytest.approx(0.4)


def test_decaying_by_nothing_changes_nothing():
    b = brain()
    b.memory("mark").affinity = 0.8
    b.decay_affinity(0.0)
    assert b.memory("mark").affinity == pytest.approx(0.8)


def test_affinity_stays_within_range_however_much_happens():
    b = brain(rejections_for_sad=999, engage_reward=0.9, reject_penalty=0.9)
    t = 0.0
    for _ in range(6):
        _, t = walk_away(b, start_t=t)
    assert b.memory("person_0").affinity >= -1.0
    for i in range(20):
        b.update([person(distance=1.0)], t + i)
    assert b.memory("person_0").affinity <= 1.0


def test_a_known_person_gets_greeted_once_not_continuously():
    b = brain(greet_cooldown=60.0, min_engagement_to_approach=2.0)
    b.memory("mark").encounters = 3
    known = person(distance=3.0, name="mark", engagement=1.0)
    assert b.update([known], 0.0).gesture == SocialConfig().greet_gesture
    assert all(b.update([known], 0.1 * (i + 1)).gesture is None for i in range(20))


def test_a_stranger_is_not_greeted_by_name():
    b = brain(min_engagement_to_approach=2.0)
    assert b.update([person(distance=3.0)], 0.0).gesture is None


# ---- persistence ----------------------------------------------------------


def test_memories_of_named_people_round_trip():
    b = brain()
    memory = b.memory("mark")
    memory.affinity = 0.62
    memory.rejections = 2
    memory.encounters = 7

    restored = brain()
    restored.load_dict(b.to_dict())
    assert restored.memory("mark").affinity == pytest.approx(0.62)
    assert restored.memory("mark").rejections == 2
    assert restored.memory("mark").encounters == 7


def test_anonymous_track_records_are_not_saved():
    """person_4 today is a different human from person_4 tomorrow."""
    b = brain()
    b.memory("person_4").affinity = -0.5
    b.memory("mark").affinity = 0.5
    saved = b.to_dict()
    assert [entry["name"] for entry in saved["people"]] == ["mark"]


def test_a_restart_does_not_clear_a_rejection_record():
    b = brain()
    b.memory("mark").rejections = 3
    restored = brain()
    restored.load_dict(b.to_dict())
    assert restored.memory("mark").rejections == 3


def test_cooldowns_do_not_survive_a_restart():
    """They are absolute times from a clock that no longer exists."""
    b = brain()
    b.memory("mark").rejections = 3
    b.memory("mark").cooldown_until = 9e9
    restored = brain()
    restored.load_dict(b.to_dict())
    assert restored.memory("mark").cooldown_until == 0.0


def test_loading_junk_does_not_raise():
    b = brain()
    assert b.load_dict({}) == 0
    assert b.load_dict(None) == 0
    b.load_dict({"people": [{"name": ""}, {"name": "mark"}]})
    assert b.memory("mark").affinity == 0.0


# ---- commanded trips ------------------------------------------------------


def test_being_told_to_go_somewhere_overrides_social_judgement():
    """Agreeing to go somewhere and then chasing a passer-by is not judgement."""
    b = brain()
    b.command(5.0, 2.0, 0.0, "kitchen", t=0.0)
    decision = b.update([person(distance=1.5, engagement=1.0)], 0.1)
    assert decision.state == "commanded"
    assert decision.approach_goal == (5.0, 2.0, 0.0)


def test_the_commanded_goal_is_sent_once_not_every_tick():
    b = brain()
    b.command(5.0, 2.0, 0.0, "kitchen", t=0.0)
    assert b.update([], 0.1).approach_goal is not None
    assert all(b.update([], 0.2 + 0.1 * i).approach_goal is None for i in range(10))


def test_the_robot_still_looks_at_people_on_the_way():
    b = brain()
    b.command(5.0, 2.0, 0.0, "kitchen", t=0.0)
    decision = b.update([person(distance=2.0, x=2.0, y=0.5)], 0.1)
    assert decision.gaze == (2.0, 0.5, SocialConfig().gaze_height)
    assert decision.target_id == "person_0"


def test_arriving_hands_the_robot_back_to_its_own_behaviour():
    b = brain()
    b.command(5.0, 2.0, 0.0, "kitchen", t=0.0)
    b.update([], 0.1)
    b.command_finished()
    decision = b.update([person(distance=3.0, engagement=1.0)], 0.2)
    assert decision.state == WATCHING
    assert "got there" in decision.reason


def test_a_commanded_trip_that_never_finishes_expires():
    """A goal Nav2 silently never completes must not leave the robot inert."""
    b = brain(commanded_timeout=60.0)
    b.command(5.0, 2.0, 0.0, "kitchen", t=0.0)
    assert b.update([], 30.0).state == "commanded"
    decision = b.update([], 90.0)
    assert decision.state == IDLE
    assert "gave up" in decision.reason


def test_a_commanded_trip_with_nobody_around_ends_quietly():
    b = brain()
    b.command(5.0, 2.0, 0.0, "kitchen", t=0.0)
    b.update([], 0.1)
    b.command_finished()
    decision = b.update([], 0.2)
    assert decision.state == IDLE
    assert decision.expression == EXPR_NONE
    assert decision.gaze is None


def test_a_new_command_replaces_one_already_in_flight():
    b = brain()
    b.command(5.0, 2.0, 0.0, "kitchen", t=0.0)
    b.update([], 0.1)
    b.command(-3.0, 1.0, 1.57, "desk", t=1.0)
    decision = b.update([], 1.1)
    assert decision.approach_goal == (-3.0, 1.0, 1.57)
    assert "desk" in decision.reason


def test_the_face_while_driving_somewhere_is_determined_not_neutral():
    b = brain()
    b.command(5.0, 2.0, 0.0, "kitchen", t=0.0)
    assert b.update([], 0.1).expression == "determined"
