"""Parses a raw YOLOv8-family TFLite output tensor into boxes/scores/classes.

Kept separate from the ROS node and from any specific TFLite runtime so it
can be unit tested against a synthetic tensor with a known answer, the same
pure-logic pattern used throughout this workspace.

!! VERIFY AGAINST YOUR OWN EXPORTED MODEL !!
YOLOv8's TFLite export layout is well documented and stable across recent
ultralytics releases: a single output tensor of shape
[1, 4 + num_classes, num_boxes] (box params then per-class scores, no
separate objectness column -- YOLOv8 folds that into the class scores
directly), boxes as (cx, cy, w, h) in pixels of the model's input size. This
module was written against that documented layout; scripts/export_detection_model.sh
prints the actual exported tensor shape so a layout drift in some future
ultralytics version is caught immediately rather than silently producing
garbage detections.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RawDetection:
    """One detection in pixel coordinates of the model's input image."""

    cx: float
    cy: float
    w: float
    h: float
    class_id: int
    score: float

    @property
    def x1(self) -> float:
        return self.cx - self.w / 2.0

    @property
    def y1(self) -> float:
        return self.cy - self.h / 2.0

    @property
    def x2(self) -> float:
        return self.cx + self.w / 2.0

    @property
    def y2(self) -> float:
        return self.cy + self.h / 2.0


def _iou(a: RawDetection, b: RawDetection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def non_max_suppression(
    detections: List[RawDetection], iou_threshold: float = 0.45
) -> List[RawDetection]:
    """Greedy NMS, highest score first, per-class.

    Per-class (not global) NMS is deliberate: an overlapping "chair" and
    "person" box are two real, different objects worth keeping both of, and
    only redundant same-class boxes around one physical object should be
    suppressed.
    """
    kept: List[RawDetection] = []
    by_class = {}
    for d in detections:
        by_class.setdefault(d.class_id, []).append(d)

    for dets in by_class.values():
        dets = sorted(dets, key=lambda d: d.score, reverse=True)
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets if _iou(best, d) < iou_threshold]

    return kept


def parse_yolo_output(
    output: np.ndarray,
    num_classes: int,
    score_threshold: float = 0.4,
    iou_threshold: float = 0.45,
) -> List[RawDetection]:
    """Decode a YOLOv8-layout tensor into filtered, NMS'd detections.

    `output` is the raw tensor, either [1, 4+num_classes, num_boxes] or
    already squeezed to [4+num_classes, num_boxes] -- both are accepted so
    the caller doesn't have to know which shape a given export produced.
    """
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[0] != 4 + num_classes and arr.shape[1] == 4 + num_classes:
        arr = arr.T  # tolerate a [num_boxes, 4+num_classes] export too
    if arr.shape[0] != 4 + num_classes:
        raise ValueError(
            f"unexpected YOLO output shape {output.shape} for {num_classes} classes "
            f"-- re-run scripts/export_detection_model.sh and check the printed "
            f"tensor shape against detection_postprocess.py's assumptions"
        )

    boxes = arr[:4, :]  # (4, num_boxes): cx, cy, w, h
    class_scores = arr[4:, :]  # (num_classes, num_boxes)

    best_class = np.argmax(class_scores, axis=0)
    best_score = np.max(class_scores, axis=0)

    keep = best_score >= score_threshold
    candidates = [
        RawDetection(
            cx=float(boxes[0, i]), cy=float(boxes[1, i]),
            w=float(boxes[2, i]), h=float(boxes[3, i]),
            class_id=int(best_class[i]), score=float(best_score[i]),
        )
        for i in np.nonzero(keep)[0]
    ]
    return non_max_suppression(candidates, iou_threshold)


def dequantize(raw: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Undo int8/uint8 quantization: real_value = (raw - zero_point) * scale.

    Standard TFLite affine quantization; `scale`/`zero_point` come from the
    interpreter's own get_output_details(), never guessed or hardcoded --
    they are specific to each exported model and will not match a different
    export.
    """
    return (raw.astype(np.float32) - zero_point) * scale
