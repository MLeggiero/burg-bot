"""YOLO output parsing, against synthetic tensors with a known answer."""

import numpy as np
import pytest

from burgerbot_perception.detection_postprocess import (
    RawDetection,
    dequantize,
    non_max_suppression,
    parse_yolo_output,
)


def _make_output(boxes_and_classes, num_classes, num_boxes=10):
    """Build a [1, 4+num_classes, num_boxes] tensor from a short spec.

    boxes_and_classes: list of (cx, cy, w, h, class_id, score) tuples, placed
    in the first N box slots; the remainder are left as zero-confidence.
    """
    arr = np.zeros((4 + num_classes, num_boxes), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(boxes_and_classes):
        arr[0, i], arr[1, i], arr[2, i], arr[3, i] = cx, cy, w, h
        arr[4 + cls, i] = score
    return arr[np.newaxis, :, :]  # add the leading batch dim


def test_finds_single_confident_detection():
    out = _make_output([(100, 100, 50, 50, 2, 0.9)], num_classes=5)
    dets = parse_yolo_output(out, num_classes=5, score_threshold=0.4)
    assert len(dets) == 1
    assert dets[0].class_id == 2
    assert dets[0].score == pytest.approx(0.9)
    assert dets[0].cx == pytest.approx(100)


def test_below_threshold_detections_are_dropped():
    out = _make_output([(100, 100, 50, 50, 0, 0.1)], num_classes=3)
    dets = parse_yolo_output(out, num_classes=3, score_threshold=0.4)
    assert dets == []


def test_accepts_pre_squeezed_2d_input():
    out = _make_output([(50, 50, 20, 20, 1, 0.8)], num_classes=4)
    squeezed = out[0]  # drop the batch dim
    dets = parse_yolo_output(squeezed, num_classes=4, score_threshold=0.4)
    assert len(dets) == 1


def test_transposed_layout_is_tolerated():
    out = _make_output([(50, 50, 20, 20, 1, 0.8)], num_classes=4)
    transposed = np.transpose(out[0])  # [num_boxes, 4+num_classes]
    dets = parse_yolo_output(transposed, num_classes=4, score_threshold=0.4)
    assert len(dets) == 1
    assert dets[0].class_id == 1


def test_wrong_shape_raises_with_a_helpful_message():
    bad = np.zeros((1, 7, 10), dtype=np.float32)  # doesn't match 4+num_classes for any obvious num_classes
    with pytest.raises(ValueError, match="unexpected YOLO output shape"):
        parse_yolo_output(bad, num_classes=5, score_threshold=0.4)


def test_nms_suppresses_overlapping_same_class_boxes():
    a = RawDetection(cx=100, cy=100, w=50, h=50, class_id=0, score=0.9)
    b = RawDetection(cx=105, cy=100, w=50, h=50, class_id=0, score=0.7)  # heavy overlap with a
    kept = non_max_suppression([a, b], iou_threshold=0.45)
    assert kept == [a]


def test_nms_keeps_overlapping_boxes_of_different_classes():
    a = RawDetection(cx=100, cy=100, w=50, h=50, class_id=0, score=0.9)
    b = RawDetection(cx=100, cy=100, w=50, h=50, class_id=1, score=0.8)
    kept = non_max_suppression([a, b], iou_threshold=0.45)
    assert {d.class_id for d in kept} == {0, 1}


def test_nms_keeps_non_overlapping_same_class_boxes():
    a = RawDetection(cx=10, cy=10, w=5, h=5, class_id=0, score=0.9)
    b = RawDetection(cx=200, cy=200, w=5, h=5, class_id=0, score=0.8)
    kept = non_max_suppression([a, b], iou_threshold=0.45)
    assert len(kept) == 2


def test_end_to_end_multiple_boxes_multiple_classes():
    out = _make_output(
        [
            (50, 50, 20, 20, 0, 0.95),
            (200, 200, 30, 30, 3, 0.6),
            (52, 51, 20, 20, 0, 0.55),  # near-duplicate of the first, should be NMS'd away
        ],
        num_classes=5,
    )
    dets = parse_yolo_output(out, num_classes=5, score_threshold=0.4, iou_threshold=0.45)
    assert len(dets) == 2
    classes = sorted(d.class_id for d in dets)
    assert classes == [0, 3]


def test_dequantize_matches_affine_formula():
    raw = np.array([0, 128, 255], dtype=np.uint8)
    out = dequantize(raw, scale=0.5, zero_point=128)
    expected = np.array([-64.0, 0.0, 63.5], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_bounding_box_corners_derived_correctly():
    d = RawDetection(cx=100, cy=50, w=20, h=10, class_id=0, score=0.9)
    assert d.x1 == pytest.approx(90)
    assert d.x2 == pytest.approx(110)
    assert d.y1 == pytest.approx(45)
    assert d.y2 == pytest.approx(55)
