"""Place-name resolution. Pure logic, no ROS graph."""

import math

import pytest

from burgerbot_dialog.places import PlaceBook, normalise


def book_with(**places):
    book = PlaceBook()
    for name, (x, y) in places.items():
        book.teach(name, x, y)
    return book


# ---- name normalisation ---------------------------------------------------


def test_filler_words_and_case_do_not_change_a_name():
    assert normalise("The Kitchen") == normalise("kitchen") == "kitchen"
    assert normalise("  to the KITCHEN!  ") == "kitchen"


def test_multi_word_names_survive():
    assert normalise("the living room") == "living room"


def test_a_name_of_nothing_but_filler_is_empty():
    assert normalise("the a to") == ""
    assert normalise("!!!") == ""


# ---- taught places --------------------------------------------------------


def test_a_taught_place_resolves_exactly():
    book = book_with(kitchen=(3.4, -1.1))
    result = book.resolve("kitchen")
    assert result.ok
    assert result.source == "taught"
    assert result.goal[0] == pytest.approx(3.4)
    assert result.goal[1] == pytest.approx(-1.1)


def test_a_taught_place_is_found_however_it_is_phrased():
    book = book_with(kitchen=(3.4, -1.1))
    for phrasing in ("the kitchen", "Kitchen", "  KITCHEN ", "to the kitchen"):
        assert book.resolve(phrasing).ok, phrasing


def test_naming_a_place_again_moves_it():
    """The usual reason to name somewhere twice is that the first try was off."""
    book = book_with(kitchen=(3.4, -1.1))
    book.teach("kitchen", 5.0, 2.0)
    assert book.resolve("kitchen").goal[0] == pytest.approx(5.0)
    assert book.names() == ["kitchen"]


def test_a_taught_place_keeps_the_heading_it_was_named_with():
    book = PlaceBook()
    book.teach("desk", 1.0, 2.0, yaw=1.57)
    assert book.resolve("desk").goal[2] == pytest.approx(1.57)


def test_places_can_be_forgotten():
    book = book_with(kitchen=(1.0, 1.0))
    assert book.forget("The Kitchen")
    assert not book.forget("kitchen")
    assert book.names() == []


def test_a_place_with_no_usable_name_is_refused():
    book = PlaceBook()
    assert book.teach("the", 1.0, 1.0) is None
    assert book.names() == []


# ---- unknown places -------------------------------------------------------


def test_an_unknown_place_does_not_resolve_to_the_nearest_one():
    """Guessing is the worst failure available here.

    A robot that drives confidently to the wrong room is far worse than one
    that says it does not know, and the honest answer is also the prompt that
    teaches it.
    """
    book = book_with(kitchen=(3.0, 0.0), bedroom=(-3.0, 0.0))
    result = book.resolve("bathroom")
    assert not result.ok
    assert result.source == ""
    assert "bathroom" in result.reason


def test_the_failure_reason_tells_the_person_how_to_fix_it():
    result = PlaceBook().resolve("kitchen")
    assert "tell me" in result.reason.lower() or "take me" in result.reason.lower()


def test_an_empty_request_fails_cleanly():
    assert not PlaceBook().resolve("").ok
    assert not PlaceBook().resolve("the a").ok


# ---- objects --------------------------------------------------------------


def test_a_tracked_object_resolves_when_no_place_matches():
    book = PlaceBook()
    result = book.resolve("chair", objects=[("chair", 4.0, 0.0)], robot=(0.0, 0.0))
    assert result.ok
    assert result.source == "object"


def test_the_nearest_matching_object_wins():
    book = PlaceBook()
    result = book.resolve(
        "chair",
        objects=[("chair", 8.0, 0.0), ("chair", 2.0, 0.0)],
        robot=(0.0, 0.0),
    )
    assert result.goal[0] < 4.0


def test_a_taught_place_beats_an_object_of_the_same_name():
    """Being told beats inferring, always."""
    book = book_with(chair=(10.0, 10.0))
    result = book.resolve("chair", objects=[("chair", 1.0, 0.0)], robot=(0.0, 0.0))
    assert result.source == "taught"


def test_object_matching_is_on_whole_words_not_substrings():
    book = PlaceBook()
    assert not book.resolve("air", objects=[("chair", 1.0, 0.0)]).ok


def test_a_multi_word_request_matches_a_single_word_label():
    book = PlaceBook()
    assert book.resolve("the dining chair", objects=[("chair", 2.0, 0.0)]).ok


def test_the_goal_stops_short_of_the_object_and_faces_it():
    book = PlaceBook()
    result = book.resolve("chair", objects=[("chair", 4.0, 0.0)],
                          robot=(0.0, 0.0), standoff=0.9)
    x, y, yaw = result.goal
    assert math.hypot(4.0 - x, 0.0 - y) == pytest.approx(0.9)
    assert yaw == pytest.approx(0.0)


def test_an_object_already_within_the_standoff_does_not_make_the_robot_reverse():
    book = PlaceBook()
    result = book.resolve("chair", objects=[("chair", 0.4, 0.0)],
                          robot=(0.0, 0.0), standoff=0.9)
    assert result.goal[0] == pytest.approx(0.0)
    assert result.goal[1] == pytest.approx(0.0)


# ---- hotspots -------------------------------------------------------------


def test_a_social_request_resolves_to_the_busiest_hotspot():
    book = PlaceBook()
    result = book.resolve(
        "people", hotspots=[(2.0, 0.0, 5.0), (8.0, 0.0, 40.0)], robot=(0.0, 0.0)
    )
    assert result.ok
    assert result.source == "hotspot"
    assert result.goal[0] > 4.0


def test_a_room_name_never_falls_through_to_the_heatmap():
    """That would be a confident answer to a question nobody asked."""
    book = PlaceBook()
    result = book.resolve("kitchen", hotspots=[(2.0, 0.0, 40.0)], robot=(0.0, 0.0))
    assert not result.ok


# ---- persistence ----------------------------------------------------------


def test_places_round_trip_through_a_plain_dict():
    book = book_with(kitchen=(3.4, -1.1), desk=(0.5, 2.0))
    restored = PlaceBook()
    assert restored.load_dict(book.to_dict()) == 2
    assert restored.names() == ["desk", "kitchen"]
    assert restored.resolve("kitchen").goal[0] == pytest.approx(3.4)


def test_loading_replaces_rather_than_merges():
    book = book_with(old=(1.0, 1.0))
    book.load_dict({"places": [{"name": "new", "x": 2.0, "y": 2.0}]})
    assert book.names() == ["new"]


def test_a_hand_edited_bad_entry_does_not_lose_the_whole_file():
    book = PlaceBook()
    loaded = book.load_dict({"places": [
        {"name": "good", "x": 1.0, "y": 2.0},
        {"name": "bad", "x": "over there"},
        {"x": 3.0, "y": 4.0},
    ]})
    assert loaded == 1
    assert book.names() == ["good"]


def test_loading_junk_does_not_raise():
    book = PlaceBook()
    assert book.load_dict({}) == 0
    assert book.load_dict(None) == 0
