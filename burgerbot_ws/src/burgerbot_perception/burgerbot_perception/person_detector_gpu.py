"""Pose-model person detection, meant to run on the control PC's GPU.

This is the offloaded half of the split described in
burgerbot_perception/launch/people.launch.py: the Pi publishes a compressed
colour stream, this node runs a real pose model on it somewhere with a GPU, and
small detection messages come back. The Pi keeps the depth image and does its
own 3D work, so nothing large ever travels in the return direction.

Two things this buys that the onboard detector cannot:

  * Rate. Person tracking wants 15-20 Hz. The Pi's TFLite path runs at 1.5 Hz,
    which is ample for labelling a chair and marginal for following somebody
    across a room.
  * Keypoints. A pose model gives shoulders, which give body orientation, which
    is what separates "walking past" from "standing here looking at the robot".
    A bounding box cannot express that difference at all.

Both are optional. Nothing here is required for the companion behaviour to
work; person_detector_lite covers the same topic with what the Pi already has.

A note specific to the hardware this was written for: an RTX 5070 Ti is
Blackwell (compute capability 12.0), which needs CUDA 12.8 or newer and a
matching PyTorch build. An older torch installs and imports perfectly happily
and then fails at the first kernel launch with "no kernel image is available
for execution on the device", which reads like a broken model rather than a
version mismatch. `torch.cuda.get_arch_list()` on a working install includes
`sm_120`; this node checks that at startup and says so plainly rather than
letting it surface mid-inference.
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from burgerbot_msgs.msg import PersonDetection2D, PersonDetection2DArray

#: COCO class index for "person" in every ultralytics detection/pose model.
PERSON_CLASS = 0


class PersonDetectorGPU(Node):
    def __init__(self):
        super().__init__("person_detector_gpu")

        self.declare_parameter("model", "yolo11n-pose.pt")
        self.declare_parameter("device", "cuda:0")
        # fp16 roughly halves inference time on any recent NVIDIA card at no
        # accuracy cost that matters for finding a person-shaped blob.
        self.declare_parameter("half", True)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("score_threshold", 0.35)
        self.declare_parameter("max_rate_hz", 20.0)
        self.declare_parameter("use_compressed", True)
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("output_topic", "/perception/people_detections")

        self._threshold = float(self.get_parameter("score_threshold").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        self._device = self.get_parameter("device").value
        self._half = bool(self.get_parameter("half").value)
        self._min_period = 1.0 / max(float(self.get_parameter("max_rate_hz").value), 0.01)
        self._last_run = 0.0
        self._bridge = CvBridge()

        self._check_device()
        self._model = self._load_model()

        self._pub = self.create_publisher(
            PersonDetection2DArray, self.get_parameter("output_topic").value, 10
        )

        topic = self.get_parameter("image_topic").value
        if bool(self.get_parameter("use_compressed").value):
            # The whole point of the offload: JPEG over the network instead of
            # raw frames. A 640x480 RGB frame is 900 kB and 15 of them a second
            # is 110 Mbit/s, which a Pi on WiFi cannot sustain and which will
            # starve the rest of the ROS graph long before it saturates the
            # link. The same stream compressed is a couple of megabits.
            self.create_subscription(
                CompressedImage, f"{topic}/compressed", self._on_compressed,
                qos_profile_sensor_data,
            )
            self.get_logger().info(f"subscribed to {topic}/compressed")
        else:
            self.create_subscription(
                Image, topic, self._on_raw, qos_profile_sensor_data
            )
            self.get_logger().info(f"subscribed to {topic} (raw)")

    # ---- setup ----------------------------------------------------------

    def _check_device(self) -> None:
        if not str(self._device).startswith("cuda"):
            return
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "person_detector_gpu needs PyTorch. This node is meant to run "
                "on the control PC, not the Pi -- see the offload section of "
                "the README."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device '{self._device}' requested but torch.cuda.is_available() "
                "is False. Set device:=cpu to run anyway (slowly)."
            )

        capability = torch.cuda.get_device_capability()
        arch = f"sm_{capability[0]}{capability[1]}"
        supported = torch.cuda.get_arch_list()
        if arch not in supported:
            # Fail here with an explanation rather than at the first kernel
            # launch with a message nobody can act on.
            raise RuntimeError(
                f"this PyTorch build has no kernels for {arch} "
                f"({torch.cuda.get_device_name()}). It was built for "
                f"{', '.join(supported)}. A 50-series card needs a cu128 or "
                f"newer wheel: pip install --index-url "
                f"https://download.pytorch.org/whl/cu128 torch"
            )
        self.get_logger().info(
            f"{torch.cuda.get_device_name()} ({arch}), torch {torch.__version__}"
        )

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "person_detector_gpu needs ultralytics: pip install ultralytics"
            )

        name = self.get_parameter("model").value
        model = YOLO(name)
        model.to(self._device)
        self.get_logger().info(f"loaded {name} on {self._device}")
        return model

    # ---- inference -------------------------------------------------------

    def _on_compressed(self, msg: CompressedImage) -> None:
        if self._throttled():
            return
        frame = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._detect_and_publish(frame, msg.header)

    def _on_raw(self, msg: Image) -> None:
        if self._throttled():
            return
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._detect_and_publish(frame, msg.header)

    def _throttled(self) -> bool:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_run < self._min_period:
            return True
        self._last_run = now
        return False

    def _detect_and_publish(self, frame: np.ndarray, header) -> None:
        results = self._model.predict(
            frame,
            imgsz=self._imgsz,
            conf=self._threshold,
            classes=[PERSON_CLASS],
            device=self._device,
            half=self._half,
            verbose=False,
        )

        out = PersonDetection2DArray()
        # The source image's stamp, not now(). person_tracker matches this
        # against a depth frame on the robot, and with inference happening on
        # another machine the round trip is easily 100ms -- long enough that
        # restamping would pair every detection with the wrong depth image.
        out.header = header

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            keypoints = result.keypoints
            xywh = boxes.xywh.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            kp_data = (
                keypoints.data.cpu().numpy() if keypoints is not None else None
            )

            for i in range(len(xywh)):
                entry = PersonDetection2D()
                entry.center_x = float(xywh[i][0])
                entry.center_y = float(xywh[i][1])
                entry.size_x = float(xywh[i][2])
                entry.size_y = float(xywh[i][3])
                entry.score = float(scores[i])
                if kp_data is not None and i < len(kp_data):
                    # (17, 3) -> flat x,y,confidence triples in COCO order.
                    entry.keypoints = kp_data[i].reshape(-1).astype(float).tolist()
                out.detections.append(entry)

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorGPU()
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
