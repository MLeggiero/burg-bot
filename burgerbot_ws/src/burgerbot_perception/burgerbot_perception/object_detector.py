"""Runs a quantized TFLite object detector on the color camera stream.

The Pi 4 has no GPU or NPU, so this is the one part of the pipeline that
actually strains the hardware -- everything else (camera capture, TF,
message passing) is cheap by comparison. Two choices keep it realistic:

  * ai-edge-litert (Google's actively maintained successor to the abandoned
    tflite-runtime package -- confirmed to ship real Python 3.12 wheels for
    both x86_64 and aarch64, where the old package's last release did not).
    A minimal inference-only runtime, not a full ML framework.
  * Inference is throttled to a low rate (default ~1.5 Hz) rather than run on
    every incoming frame. Objects worth labelling on a map are, almost by
    definition, not moving fast enough to need 30fps detection -- throttling
    costs nothing real and saves the Pi's only scarce resource here.

Model and label map are parameters, not hardcoded, so a Coral/Hailo-
accelerated model can be swapped in later without touching this file.
"""

from typing import List, Optional

import numpy as np
import rclpy
from ai_edge_litert.interpreter import Interpreter
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from .detection_postprocess import dequantize, parse_yolo_output


class ObjectDetector(Node):
    def __init__(self):
        super().__init__("object_detector")

        self.declare_parameter("model_path", "")
        self.declare_parameter("labels_path", "")
        self.declare_parameter("input_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("inference_rate_hz", 1.5)
        self.declare_parameter("score_threshold", 0.45)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("num_threads", 3)

        model_path = self.get_parameter("model_path").value
        if not model_path:
            raise RuntimeError(
                "object_detector requires 'model_path' -- run "
                "scripts/export_detection_model.sh and point this at the "
                "resulting .tflite file (see burgerbot_perception/config)"
            )

        self._labels = self._load_labels(self.get_parameter("labels_path").value)

        # Default delegates (XNNPACK) left on deliberately. An earlier version
        # disabled them to get past "failed to delegate TRANSPOSE node" on the
        # int8 model, which then hit the reference sigmoid kernel's
        # "output->params.scale == 1./256" constraint on the first real
        # inference -- no delegate setting satisfies both. The shipped model
        # is float32 instead (see scripts/export_detection_model.sh), where
        # XNNPACK handles transposes and sigmoid without complaint and the
        # accelerated path is exactly what a Pi 4 wants.
        self._interpreter = Interpreter(
            model_path=model_path,
            num_threads=int(self.get_parameter("num_threads").value),
        )
        self._interpreter.allocate_tensors()
        self._input_detail = self._interpreter.get_input_details()[0]
        self._output_detail = self._interpreter.get_output_details()[0]
        _, self._input_h, self._input_w, _ = self._input_detail["shape"]
        self._num_classes = self._output_detail["shape"][1] - 4

        self.get_logger().info(
            f"model loaded: input {self._input_w}x{self._input_h}, "
            f"{self._num_classes} classes, output shape {self._output_detail['shape']}"
        )

        self._bridge = CvBridge()
        self._min_period = 1.0 / max(float(self.get_parameter("inference_rate_hz").value), 0.01)
        self._last_run = 0.0

        self.create_subscription(
            Image, self.get_parameter("input_topic").value, self._on_image, 1
        )
        self._pub = self.create_publisher(Detection2DArray, "/perception/detections2d", 10)

    @staticmethod
    def _load_labels(path: str) -> List[str]:
        if not path:
            raise RuntimeError("object_detector requires 'labels_path' (one label per line)")
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    def _on_image(self, msg: Image) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_run < self._min_period:
            return  # Throttled: drop this frame, the pipeline doesn't need it.
        self._last_run = now

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        detections = self._infer(frame)
        if not detections:
            return

        out = Detection2DArray()
        out.header = msg.header
        scale_x = msg.width / self._input_w
        scale_y = msg.height / self._input_h

        for det in detections:
            d2d = Detection2D()
            d2d.header = msg.header

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = (
                self._labels[det.class_id]
                if det.class_id < len(self._labels)
                else f"class_{det.class_id}"
            )
            hyp.hypothesis.score = det.score
            d2d.results.append(hyp)

            bbox = BoundingBox2D()
            # Scale from the model's fixed input resolution back to the
            # actual camera frame's pixel coordinates -- the depth image
            # object_projector samples from is at the camera's native
            # resolution, not the model's.
            bbox.center.position.x = det.cx * scale_x
            bbox.center.position.y = det.cy * scale_y
            bbox.size_x = det.w * scale_x
            bbox.size_y = det.h * scale_y
            d2d.bbox = bbox

            out.detections.append(d2d)

        self._pub.publish(out)

    def _infer(self, frame_rgb: np.ndarray):
        import cv2

        resized = cv2.resize(frame_rgb, (self._input_w, self._input_h))

        if self._input_detail["dtype"] == np.uint8:
            input_data = resized.astype(np.uint8)
        else:
            input_data = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        self._interpreter.set_tensor(self._input_detail["index"], input_data)
        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output_detail["index"])

        quant = self._output_detail.get("quantization", (0.0, 0))
        if quant[0]:  # scale != 0 means this output is quantized
            raw = dequantize(raw, quant[0], quant[1])

        return parse_yolo_output(
            raw,
            self._num_classes,
            score_threshold=float(self.get_parameter("score_threshold").value),
            iou_threshold=float(self.get_parameter("iou_threshold").value),
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
