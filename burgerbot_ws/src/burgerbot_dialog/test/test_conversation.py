"""Turn-taking and timing. Pure logic, no ROS graph and no network.

Every case here is about what the robot does when the model is slow, dead, or
answering a question that has already been superseded. None of it is
reproducible against a real endpoint, which is exactly why it lives here.
"""

from burgerbot_dialog.conversation import (
    DEGRADED,
    IDLE,
    SPEAKING,
    THINKING,
    Conversation,
    ConversationConfig,
)


def convo(**overrides):
    return Conversation(config=ConversationConfig(**overrides))


# ---- the basic turn -------------------------------------------------------


def test_a_new_conversation_is_idle():
    assert convo().state() == IDLE


def test_starting_a_turn_moves_to_thinking():
    c = convo()
    turn = c.start_turn("hello", now=0.0)
    assert c.state() == THINKING
    assert c.is_current(turn.id)


def test_completing_a_turn_records_both_sides_of_it():
    c = convo()
    turn = c.start_turn("hello", now=0.0)
    assert c.complete(turn.id, "hello", "Hi there.", now=1.0)
    assert c.state() == SPEAKING
    assert c.history() == [("user", "hello"), ("assistant", "Hi there.")]


# ---- superseding ----------------------------------------------------------


def test_a_new_utterance_supersedes_the_turn_in_flight():
    """People do say 'what's that -- oh never mind, come here instead'."""
    c = convo()
    first = c.start_turn("what time is it", now=0.0)
    second = c.start_turn("never mind, come here", now=1.0)
    assert not c.is_current(first.id)
    assert c.is_current(second.id)


def test_a_reply_for_a_superseded_turn_is_dropped():
    """Otherwise the robot answers a question from twenty seconds ago."""
    c = convo()
    first = c.start_turn("what time is it", now=0.0)
    c.start_turn("never mind", now=1.0)
    assert not c.complete(first.id, "what time is it", "It is four.", now=2.0)
    assert c.history() == []


def test_a_failure_for_a_superseded_turn_does_not_count_against_the_circuit():
    c = convo(failures_to_open=2)
    first = c.start_turn("a", now=0.0)
    c.start_turn("b", now=1.0)
    assert not c.fail(first.id, now=2.0)
    assert c.failures() == 0


# ---- the latency ladder ---------------------------------------------------


def test_the_robot_only_looks_like_it_is_thinking_after_a_beat():
    c = convo(thinking_after=0.8)
    c.start_turn("hello", now=0.0)
    assert not c.thinking_visibly(now=0.3)
    assert c.thinking_visibly(now=1.0)


def test_nothing_looks_like_thinking_when_no_turn_is_in_flight():
    assert not convo().thinking_visibly(now=100.0)


def test_a_stalled_turn_says_something_rather_than_going_silent():
    c = convo(stall_after=3.5)
    c.start_turn("hello", now=0.0)
    assert c.due_filler(now=1.0) is None
    assert c.due_filler(now=4.0) is not None


def test_the_filler_is_said_once_per_turn_not_repeatedly():
    c = convo(stall_after=1.0)
    c.start_turn("hello", now=0.0)
    assert c.due_filler(now=2.0) is not None
    assert c.due_filler(now=3.0) is None
    assert c.due_filler(now=9.0) is None


def test_each_turn_gets_its_own_filler():
    c = convo(stall_after=1.0)
    c.start_turn("a", now=0.0)
    assert c.due_filler(now=2.0) is not None
    c.start_turn("b", now=3.0)
    assert c.due_filler(now=5.0) is not None


def test_a_turn_eventually_expires():
    c = convo(give_up_after=12.0)
    c.start_turn("hello", now=0.0)
    assert not c.expired(now=5.0)
    assert c.expired(now=13.0)


def test_an_idle_conversation_eventually_ends():
    c = convo(conversation_timeout=90.0)
    turn = c.start_turn("hello", now=0.0)
    c.complete(turn.id, "hello", "Hi.", now=1.0)
    assert not c.idle_too_long(now=30.0)
    assert c.idle_too_long(now=200.0)


def test_ending_a_conversation_drops_the_history():
    """Transcripts are never persisted; they die with the conversation."""
    c = convo()
    turn = c.start_turn("something private", now=0.0)
    c.complete(turn.id, "something private", "I see.", now=1.0)
    c.end_conversation(now=2.0)
    assert c.history() == []
    assert c.state() == IDLE


# ---- the circuit ----------------------------------------------------------


def test_repeated_failures_open_the_circuit():
    """A dead endpoint should cost one silence, not one per turn."""
    c = convo(failures_to_open=3)
    for i in range(3):
        turn = c.start_turn("hello", now=float(i))
        c.fail(turn.id, now=float(i) + 0.5)
    assert c.state() == DEGRADED
    assert c.circuit_open(now=4.0)


def test_a_couple_of_failures_do_not_open_it():
    c = convo(failures_to_open=3)
    for i in range(2):
        turn = c.start_turn("hello", now=float(i))
        c.fail(turn.id, now=float(i) + 0.5)
    assert not c.circuit_open(now=3.0)


def test_the_circuit_half_opens_after_the_cooldown():
    c = convo(failures_to_open=2, circuit_cooldown=60.0)
    for i in range(2):
        turn = c.start_turn("hello", now=float(i))
        c.fail(turn.id, now=float(i) + 0.5)
    assert c.circuit_open(now=10.0)
    assert not c.circuit_open(now=100.0), "one probe should be allowed through"


def test_a_success_closes_the_circuit_properly():
    c = convo(failures_to_open=2, circuit_cooldown=1.0)
    for i in range(2):
        turn = c.start_turn("hello", now=float(i))
        c.fail(turn.id, now=float(i) + 0.5)
    turn = c.start_turn("hello", now=10.0)
    c.complete(turn.id, "hello", "Hi.", now=10.5)
    assert c.failures() == 0
    assert not c.circuit_open(now=11.0)


def test_fallback_lines_are_honest_about_which_failure_happened():
    c = convo()
    assert c.fallback_line("unreachable") != c.fallback_line("timeout")
    assert c.fallback_line("something_unheard_of")


# ---- rate limits ----------------------------------------------------------


def test_gestures_are_spaced_out():
    """Small models fill in every optional field, so this is the common case."""
    c = convo(gesture_cooldown=12.0)
    assert c.allow_gesture(now=0.0)
    assert not c.allow_gesture(now=5.0)
    assert c.allow_gesture(now=20.0)


def test_navigation_commands_are_rate_limited():
    c = convo(nav_cooldown=20.0)
    assert c.allow_nav(now=0.0)
    assert not c.allow_nav(now=10.0)
    assert c.allow_nav(now=30.0)


# ---- history --------------------------------------------------------------


def test_history_is_capped_and_loses_the_beginning_not_the_end():
    c = convo(max_history_turns=2)
    for i in range(5):
        turn = c.start_turn(f"q{i}", now=float(i))
        c.complete(turn.id, f"q{i}", f"a{i}", now=float(i) + 0.5)
    history = c.history()
    assert len(history) == 4
    assert history[-1] == ("assistant", "a4")
    assert ("user", "q0") not in history


def test_a_turn_with_nothing_said_adds_nothing_to_history():
    c = convo()
    turn = c.start_turn("hello", now=0.0)
    c.complete(turn.id, "", "", now=1.0)
    assert c.history() == []
