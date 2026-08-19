"""Re-labels the Pi's existing detections as people. No extra inference.

The onboard TFLite detector already finds the COCO `person` class on every
frame it runs; this pulls those out and republishes them in the shape the
social stack expects. That is the entire node -- about thirty lines of real
work -- and it exists so the companion behaviour has a Pi-only path that costs
nothing at all beyond what perception was already spending.

What it cannot supply is keypoints, so everything downstream falls back to
estimating orientation from which way somebody is walking. And it inherits the
detector's 1.5 Hz throttle, which is fine for furniture and marginal for
people: at a walking pace of 1.4 m/s a person covers most of a metre between
frames, so tracks are coarse and a fast walker can outrun the association gate
entirely. Both limits go away when person_detector_gpu runs on a machine with
a GPU -- see burgerbot_perception/launch/people.launch.py.
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

from burgerbot_msgs.msg import PersonDetection2D, PersonDetection2DArray


class PersonDetectorLite(Node):
    def __init__(self):
        super().__init__("person_detector_lite")

        self.declare_parameter("detections_topic", "/perception/detections2d")
        self.declare_parameter("output_topic", "/perception/people_detections")
        self.declare_parameter("person_label", "person")
        # Lower than the shared detector's own threshold on purpose. A person
        # is a deformable, partly-occluded, often back-turned target and scores
        # lower than a chair does; the tracker's confirmation count is the real
        # false-positive filter, so this can afford to be permissive.
        self.declare_parameter("score_threshold", 0.30)

        self._label = self.get_parameter("person_label").value
        self._threshold = float(self.get_parameter("score_threshold").value)

        self._pub = self.create_publisher(
            PersonDetection2DArray, self.get_parameter("output_topic").value, 10
        )
        self.create_subscription(
            Detection2DArray,
            self.get_parameter("detections_topic").value,
            self._on_detections,
            10,
        )
        self.get_logger().info(
            f"person_detector_lite up: '{self._label}' from "
            f"{self.get_parameter('detections_topic').value} (no keypoints)"
        )

    def _on_detections(self, msg: Detection2DArray) -> None:
        out = PersonDetection2DArray()
        # Carried through unchanged: this is the stamp of the image inference
        # ran on, and person_tracker needs it to find the matching depth frame.
        # Restamping here with the current time would silently pair every
        # detection with a depth image from after the robot had already moved.
        out.header = msg.header

        for detection in msg.detections:
            person = max(
                (r for r in detection.results if r.hypothesis.class_id == self._label),
                key=lambda r: r.hypothesis.score,
                default=None,
            )
            if person is None or person.hypothesis.score < self._threshold:
                continue

            entry = PersonDetection2D()
            entry.center_x = float(detection.bbox.center.position.x)
            entry.center_y = float(detection.bbox.center.position.y)
            entry.size_x = float(detection.bbox.size_x)
            entry.size_y = float(detection.bbox.size_y)
            entry.score = float(person.hypothesis.score)
            entry.keypoints = []  # bbox-only detector; orientation comes from motion
            out.detections.append(entry)

        # Published even when empty. "Nobody is here" is a fact the tracker
        # needs in order to age its tracks out; staying silent would leave the
        # last person the robot saw coasting forever.
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorLite()
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
