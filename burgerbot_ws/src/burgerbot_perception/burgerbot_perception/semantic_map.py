"""Maintains the persistent, de-duplicated object layer and its RViz view.

Subscribes to the raw Detection3DArray stream from object_projector and folds
every detection into ObjectTracker (clustering.py). Publishes a MarkerArray
so the labels show up overlaid on the map in RViz, and saves/loads a simple
YAML file alongside the SLAM map -- mirroring how nav2_map_server persists
map.yaml/map.pgm, so a saved map and its object labels travel together.

Deliberately not part of the geometric map itself (the OccupancyGrid is a
probability grid, not a place to store "this cell is a chair") -- this is a
parallel layer, the standard pattern for semantic mapping.
"""

import math
import os
from typing import Optional

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.node import Node
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from burgerbot_msgs.msg import SemanticMap, SemanticObject

from .clustering import ObjectTracker, TrackedObject

# A handful of visually distinct colours, cycled by a hash of the label so
# the same class is always the same colour across a run without needing a
# hand-maintained palette for every possible COCO class.
_PALETTE = [
    (0.90, 0.30, 0.30), (0.30, 0.75, 0.90), (0.95, 0.75, 0.20),
    (0.55, 0.85, 0.35), (0.75, 0.40, 0.90), (0.95, 0.55, 0.20),
    (0.30, 0.90, 0.65), (0.90, 0.40, 0.65),
]


def _color_for_label(label: str):
    return _PALETTE[hash(label) % len(_PALETTE)]


class SemanticMapNode(Node):
    def __init__(self):
        super().__init__("semantic_map")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("match_radius", 0.5)
        self.declare_parameter("min_confidence", 0.4)
        self.declare_parameter("min_observations_for_display", 2)
        self.declare_parameter("publish_rate", 2.0)
        self.declare_parameter("map_name", "small_house")
        # Default alongside the SLAM maps this workspace already saves to
        # (see burgerbot_mapping/maps/) -- kept as its own parameter rather
        # than hardcoded so a map saved somewhere else still works.
        self.declare_parameter(
            "maps_directory",
            os.path.join(get_package_share_directory("burgerbot_mapping"), "maps"),
        )
        self.declare_parameter("prune_max_age", 0.0)  # 0 = never prune

        self._map_frame = self.get_parameter("map_frame").value
        self._tracker = ObjectTracker(
            match_radius=float(self.get_parameter("match_radius").value),
            min_confidence=float(self.get_parameter("min_confidence").value),
        )

        self.create_subscription(
            Detection3DArray, "/perception/detections3d", self._on_detections, 10
        )
        self._marker_pub = self.create_publisher(MarkerArray, "/perception/semantic_map", 10)
        self._map_pub = self.create_publisher(SemanticMap, "/perception/semantic_map_msg", 10)

        rate = float(self.get_parameter("publish_rate").value)
        self.create_timer(1.0 / max(rate, 0.1), self._publish)

        self.create_service(Trigger, "~/save_map", self._on_save)
        self.create_service(Trigger, "~/load_map", self._on_load)

        self.get_logger().info("semantic_map up")

    # ---- input --------------------------------------------------------

    def _on_detections(self, msg: Detection3DArray) -> None:
        t = self._stamp_seconds(msg.header.stamp)
        for det in msg.detections:
            if not det.results:
                continue
            best = max(det.results, key=lambda r: r.hypothesis.score)
            p = det.bbox.center.position
            self._tracker.observe(
                best.hypothesis.class_id, p.x, p.y, p.z, best.hypothesis.score, t
            )

        max_age = float(self.get_parameter("prune_max_age").value)
        if max_age > 0.0:
            dropped = self._tracker.prune_stale(t, max_age)
            if dropped:
                self.get_logger().debug(f"pruned {dropped} stale object(s)")

    # ---- output ---------------------------------------------------------

    def _publish(self) -> None:
        min_obs = int(self.get_parameter("min_observations_for_display").value)
        objects = self._tracker.objects(min_observations=min_obs)

        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, obj in enumerate(objects):
            markers.markers.append(self._sphere_marker(obj, i, now))
            markers.markers.append(self._label_marker(obj, i, now))
        self._marker_pub.publish(markers)

        semantic_map = SemanticMap()
        semantic_map.header.stamp = now
        semantic_map.header.frame_id = self._map_frame
        semantic_map.objects = [self._to_msg(o) for o in objects]
        self._map_pub.publish(semantic_map)

    def _sphere_marker(self, obj: TrackedObject, i: int, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = self._map_frame
        m.header.stamp = stamp
        m.ns = "semantic_objects"
        m.id = i * 2
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = obj.x, obj.y, obj.z
        m.pose.orientation.w = 1.0
        # Fixed, modest size -- this is a label pin, not a to-scale model of
        # the object (we have no size estimate for it, only a point).
        m.scale.x = m.scale.y = m.scale.z = 0.15
        r, g, b = _color_for_label(obj.label)
        m.color.r, m.color.g, m.color.b = r, g, b
        # More confident/more-observed objects render more opaque, so a
        # glance at the map distinguishes "seen once, uncertain" from
        # "seen repeatedly, trust this."
        m.color.a = min(1.0, 0.35 + 0.1 * obj.observation_count)
        return m

    def _label_marker(self, obj: TrackedObject, i: int, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = self._map_frame
        m.header.stamp = stamp
        m.ns = "semantic_labels"
        m.id = i * 2 + 1
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y = obj.x, obj.y
        m.pose.position.z = obj.z + 0.25
        m.pose.orientation.w = 1.0
        m.scale.z = 0.18
        m.color.r = m.color.g = m.color.b = m.color.a = 1.0
        m.text = f"{obj.label} ({obj.confidence:.2f})"
        return m

    @staticmethod
    def _to_msg(obj: TrackedObject) -> SemanticObject:
        msg = SemanticObject()
        msg.id = obj.id
        msg.label = obj.label
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = obj.x, obj.y, obj.z
        msg.pose.orientation.w = 1.0
        msg.confidence = float(obj.confidence)
        msg.observation_count = int(obj.observation_count)
        msg.first_seen = SemanticMapNode._seconds_to_stamp(obj.first_seen)
        msg.last_seen = SemanticMapNode._seconds_to_stamp(obj.last_seen)
        return msg

    # ---- persistence ------------------------------------------------------

    def _objects_path(self) -> str:
        directory = os.path.join(
            self.get_parameter("maps_directory").value,
            self.get_parameter("map_name").value,
        )
        return os.path.join(directory, "objects.yaml")

    def _on_save(self, request, response):
        path = self._objects_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            objects = self._tracker.objects(min_observations=1)
            data = {
                "objects": [
                    {
                        "id": o.id, "label": o.label,
                        "x": o.x, "y": o.y, "z": o.z,
                        "confidence": o.confidence,
                        "observation_count": o.observation_count,
                    }
                    for o in objects
                ]
            }
            with open(path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            response.success = True
            response.message = f"saved {len(objects)} object(s) to {path}"
        except OSError as exc:
            response.success = False
            response.message = str(exc)
        self.get_logger().info(response.message)
        return response

    def _on_load(self, request, response):
        path = self._objects_path()
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except OSError as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().warn(response.message)
            return response

        now = self.get_clock().now().nanoseconds * 1e-9
        count = 0
        for entry in data.get("objects", []):
            self._tracker.observe(
                entry["label"], entry["x"], entry["y"], entry.get("z", 0.0),
                confidence=entry.get("confidence", 1.0), t=now,
            )
            count += 1
        response.success = True
        response.message = f"loaded {count} object(s) from {path}"
        self.get_logger().info(response.message)
        return response

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def _seconds_to_stamp(seconds: float) -> TimeMsg:
        msg = TimeMsg()
        msg.sec = int(seconds)
        msg.nanosec = int((seconds - int(seconds)) * 1e9)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapNode()
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
