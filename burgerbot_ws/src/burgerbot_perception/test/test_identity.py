"""Face-embedding gallery and name voting. Pure logic, no ROS graph."""

import numpy as np
import pytest

from burgerbot_perception.identity import (
    IdentityGallery,
    IdentityVoter,
    is_frontal,
    normalize,
    sharpness,
)


def embedding(*values, dim=8):
    """A short embedding padded to a fixed length, for readable tests."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[: len(values)] = values
    return vec


# Two well-separated identities, plus a near-miss of the first.
MARK = embedding(1.0, 0.0, 0.0)
MARK_ANGLED = embedding(0.94, 0.34, 0.0)
SAM = embedding(0.0, 1.0, 0.0)
STRANGER = embedding(0.0, 0.0, 1.0)


# ---- matching ------------------------------------------------------------


def test_an_empty_gallery_matches_nobody():
    assert IdentityGallery().match(MARK) == ("", -1.0)


def test_an_enrolled_face_is_recognised():
    gallery = IdentityGallery()
    gallery.enroll("mark", MARK)
    name, score = gallery.match(MARK)
    assert name == "mark"
    assert score == pytest.approx(1.0)


def test_an_unenrolled_face_is_a_stranger_not_the_nearest_person():
    gallery = IdentityGallery(match_threshold=0.42)
    gallery.enroll("mark", MARK)
    name, score = gallery.match(STRANGER)
    assert name == ""
    assert score < 0.42


def test_a_slightly_different_view_of_the_same_person_still_matches():
    gallery = IdentityGallery()
    gallery.enroll("mark", MARK)
    assert gallery.match(MARK_ANGLED)[0] == "mark"


def test_enrolling_several_views_covers_angles_one_view_misses():
    profile = embedding(0.3, 0.0, 0.95)
    gallery = IdentityGallery(match_threshold=0.6)
    gallery.enroll("mark", MARK)
    assert gallery.match(profile)[0] == ""  # too far from the frontal view alone
    gallery.enroll("mark", profile)
    assert gallery.match(profile)[0] == "mark"


def test_similarity_uses_the_best_stored_view_not_their_average():
    """Averaging would penalise a correct match against one stored angle."""
    gallery = IdentityGallery(match_threshold=0.9)
    gallery.enroll("mark", MARK)
    gallery.enroll("mark", SAM)  # a wildly different stored view
    assert gallery.match(MARK) == ("mark", pytest.approx(1.0))


def test_two_similar_people_produce_no_answer_rather_than_a_coin_toss():
    """The margin rule: a win by a hair is not an identification."""
    twin_a = embedding(1.0, 0.02, 0.0)
    twin_b = embedding(1.0, 0.00, 0.0)
    gallery = IdentityGallery(match_threshold=0.3, margin=0.05)
    gallery.enroll("twin_a", twin_a)
    gallery.enroll("twin_b", twin_b)
    name, score = gallery.match(embedding(1.0, 0.01, 0.0))
    assert name == ""
    assert score > 0.9  # it matched *something* strongly, just not uniquely


def test_a_clear_winner_beats_the_margin_and_is_named():
    gallery = IdentityGallery(match_threshold=0.42, margin=0.05)
    gallery.enroll("mark", MARK)
    gallery.enroll("sam", SAM)
    assert gallery.match(MARK_ANGLED)[0] == "mark"


def test_unnormalised_embeddings_are_handled():
    gallery = IdentityGallery()
    gallery.enroll("mark", MARK * 17.0)
    assert gallery.match(MARK * 0.003)[0] == "mark"


def test_a_zero_embedding_does_not_divide_by_zero():
    assert np.all(np.isfinite(normalize(np.zeros(8))))


# ---- gallery management --------------------------------------------------


def test_stored_views_are_capped():
    gallery = IdentityGallery(max_embeddings=3)
    for i in range(10):
        gallery.enroll("mark", embedding(1.0, i * 0.05, 0.0))
    assert len(gallery._people["mark"].embeddings) == 3


def test_the_capped_set_keeps_angular_coverage_rather_than_the_newest():
    """Eviction drops the most redundant view, not the oldest.

    Enrol one genuinely distinct profile view first, then flood the gallery
    with near-identical frontal ones. The distinct view is the valuable one and
    must survive; evicting by age would throw it away first.
    """
    profile = embedding(0.0, 0.0, 1.0)
    gallery = IdentityGallery(max_embeddings=4)
    gallery.enroll("mark", profile)
    for i in range(12):
        gallery.enroll("mark", embedding(1.0, i * 0.001, 0.0))
    assert gallery.match(profile)[1] > 0.99


def test_names_are_listed_and_can_be_forgotten():
    gallery = IdentityGallery()
    gallery.enroll("mark", MARK)
    gallery.enroll("sam", SAM)
    assert gallery.names() == ["mark", "sam"]
    assert gallery.forget("mark")
    assert gallery.names() == ["sam"]
    assert not gallery.forget("nobody")


def test_a_gallery_round_trips_through_a_plain_dict():
    gallery = IdentityGallery()
    gallery.enroll("mark", MARK)
    gallery.enroll("mark", MARK_ANGLED)
    gallery.enroll("sam", SAM)

    restored = IdentityGallery()
    assert restored.load_dict(gallery.to_dict()) == 2
    assert restored.names() == ["mark", "sam"]
    assert restored.match(MARK_ANGLED)[0] == "mark"


def test_loading_replaces_rather_than_merges():
    gallery = IdentityGallery()
    gallery.enroll("old", STRANGER)
    gallery.load_dict({"people": [{"name": "mark", "embeddings": [MARK.tolist()]}]})
    assert gallery.names() == ["mark"]


def test_loading_junk_does_not_raise():
    gallery = IdentityGallery()
    assert gallery.load_dict({}) == 0
    assert gallery.load_dict(None) == 0
    assert gallery.load_dict({"people": [{"name": "", "embeddings": []},
                                         {"name": "empty", "embeddings": []}]}) == 0


# ---- voting --------------------------------------------------------------


def test_one_frame_is_never_enough_to_claim_a_name():
    voter = IdentityVoter(min_votes=3)
    voter.vote("person_0", "mark", 0.8, t=0.0)
    assert voter.best("person_0", t=0.1) == ("", 0.0)


def test_repeated_agreeing_matches_settle_on_a_name():
    voter = IdentityVoter(min_votes=3)
    for i in range(4):
        voter.vote("person_0", "mark", 0.8, t=i * 0.5)
    name, confidence = voter.best("person_0", t=2.0)
    assert name == "mark"
    assert confidence == pytest.approx(0.8, abs=0.05)


def test_a_split_vote_is_not_a_decision():
    voter = IdentityVoter(min_votes=3, majority=0.6)
    for i in range(3):
        voter.vote("person_0", "mark", 0.8, t=i * 0.2)
        voter.vote("person_0", "sam", 0.8, t=i * 0.2)
    assert voter.best("person_0", t=1.0) == ("", 0.0)


def test_a_single_stray_match_does_not_overturn_a_consistent_one():
    voter = IdentityVoter(min_votes=3, majority=0.6)
    for i in range(8):
        voter.vote("person_0", "mark", 0.8, t=i * 0.2)
    voter.vote("person_0", "sam", 0.9, t=1.7)
    assert voter.best("person_0", t=1.8)[0] == "mark"


def test_votes_expire_so_a_new_person_does_not_inherit_the_old_name():
    voter = IdentityVoter(min_votes=3, window=5.0)
    for i in range(4):
        voter.vote("person_0", "mark", 0.8, t=i * 0.2)
    assert voter.best("person_0", t=1.0)[0] == "mark"
    assert voter.best("person_0", t=60.0) == ("", 0.0)


def test_stranger_votes_are_not_recorded_as_a_name():
    voter = IdentityVoter(min_votes=1)
    voter.vote("person_0", "", 0.1, t=0.0)
    assert voter.best("person_0", t=0.1) == ("", 0.0)


def test_confidence_reflects_both_match_strength_and_agreement():
    strong = IdentityVoter(min_votes=3)
    weak = IdentityVoter(min_votes=3)
    for i in range(5):
        strong.vote("p", "mark", 0.95, t=i * 0.1)
        weak.vote("p", "mark", 0.50, t=i * 0.1)
    assert strong.best("p", t=0.5)[1] > weak.best("p", t=0.5)[1]


def test_vote_history_for_dead_tracks_is_dropped():
    voter = IdentityVoter()
    voter.vote("person_0", "mark", 0.8, t=0.0)
    voter.vote("person_1", "sam", 0.8, t=0.0)
    voter.forget_all_except(["person_1"])
    assert "person_0" not in voter._votes
    assert "person_1" in voter._votes


# ---- enrolment quality ---------------------------------------------------


def test_sharpness_separates_a_crisp_patch_from_a_blurred_one():
    rng = np.random.default_rng(0)
    crisp = rng.integers(0, 255, size=(40, 40)).astype(np.float32)
    blurred = np.repeat(np.repeat(crisp[::8, ::8], 8, axis=0), 8, axis=1)
    assert sharpness(crisp) > sharpness(blurred) * 2


def test_sharpness_of_an_empty_or_tiny_patch_is_zero_not_an_error():
    assert sharpness(np.zeros((0, 0))) == 0.0
    assert sharpness(np.zeros((1, 1))) == 0.0


def test_a_frontal_face_has_the_nose_between_the_eyes():
    keypoints = {"left_eye": (110.0, 100.0), "right_eye": (90.0, 100.0),
                 "nose": (100.0, 110.0)}
    assert is_frontal(keypoints)


def test_a_turned_head_is_rejected_for_enrolment():
    keypoints = {"left_eye": (110.0, 100.0), "right_eye": (90.0, 100.0),
                 "nose": (122.0, 110.0)}
    assert not is_frontal(keypoints)


def test_missing_keypoints_are_not_frontal():
    assert not is_frontal({"nose": (100.0, 110.0)})
    assert not is_frontal({})


def test_eyes_at_the_same_point_are_not_frontal():
    keypoints = {"left_eye": (100.0, 100.0), "right_eye": (100.0, 100.0),
                 "nose": (100.0, 110.0)}
    assert not is_frontal(keypoints)
