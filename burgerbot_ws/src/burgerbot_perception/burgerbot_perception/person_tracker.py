"""Turns per-frame person detections into persistent map-frame tracks.

Structurally the same job object_projector + semantic_map do for furniture,
and deliberately a separate pipeline rather than a filter bolted onto that one,
because people break the assumption the object pipeline is built on. Semantic
mapping averages every observation of a thing into one position, which is
correct for a chair and nonsense for somebody walking. Semantic mapping also
wants to remember what it saw an hour ago; a companion needs to know where
somebody is *now*, and a track nobody has seen for two seconds is worse than no
track at all, because the robot will go and stand next to where they were.

What this node owns is the ROS-shaped work -- pairing detections with the depth
frame they belong to, back-projecting into the map, and reading the robot's own
pose. The tracking itself is person_tracking.py, which has no ROS in it.
"""

import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)
from builtin_interfaces.msg import Time as TimeMsg
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray

from burgerbot_msgs.msg import Person, PersonArray, PersonDetection2DArray

from .identity import IdentityVoter
from .person_tracking import (
    PersonObservation,
    PersonTrack,
    PersonTracker,
    relative_to_robot,
)
from .projection import (
    backproject,
    depth_scale_for,
    facing_from_shoulders,
    sample_depth,
)

# Keypoint indices, mirroring the constants in PersonDetection2D.msg.
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
#: Keypoints whose depth is worth sampling to locate the body as a whole.
#: Torso only: hands and feet move independently of where a person is standing,
#: and an outstretched arm would drag the estimate a foot sideways.
TORSO_KEYPOINTS = (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP)


class PersonTrackerNode(Node):
    def __init__(self):
        super().__init__("person_tracker")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("detections_topic", "/perception/people_detections")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

        self.declare_parameter("depth_sample_radius", 4)
        # Keypoints are single points on a curved body part, so their depth
        # patch is sampled tighter than the torso's -- widening it starts
        # picking up whatever is behind the shoulder.
        self.declare_parameter("keypoint_sample_radius", 2)
        self.declare_parameter("keypoint_min_confidence", 0.5)
        self.declare_parameter("min_depth_m", 0.3)
        self.declare_parameter("max_depth_m", 8.0)
        self.declare_parameter("sync_slop", 0.15)
        self.declare_parameter("sync_queue_size", 60)

        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("publish_markers", True)
        # Safety net only. Both detector nodes publish an empty array when they
        # see nobody, which is what normally ages tracks out. This catches the
        # other case: the detector process dying, where no message ever arrives
        # again and the last person seen would otherwise coast forever.
        self.declare_parameter("detection_timeout", 1.0)

        self.declare_parameter("match_radius", 1.0)
        self.declare_parameter("max_coast", 1.5)
        self.declare_parameter("max_predict", 0.6)
        self.declare_parameter("min_hits", 3)
        self.declare_parameter("min_confidence", 0.35)
        self.declare_parameter("min_speed_for_heading", 0.25)

        self.declare_parameter("face_identities_topic", "/perception/face_identities")
        self.declare_parameter("identity_vote_window", 8.0)
        self.declare_parameter("identity_min_votes", 3)
        # Pixels a recognised face's box centre may sit from a person's box
        # centre and still be the same person. Generous, because the two boxes
        # bound different things -- a head and a whole body -- and their centres
        # are half a torso apart by construction. What actually keeps this
        # unambiguous is that people are rarely close enough together in frame
        # for the wrong pairing to be nearer.
        self.declare_parameter("identity_match_pixels", 220.0)

        self._map_frame = self.get_parameter("map_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value
        self._depth_radius = int(self.get_parameter("depth_sample_radius").value)
        self._kp_radius = int(self.get_parameter("keypoint_sample_radius").value)
        self._kp_min_conf = float(self.get_parameter("keypoint_min_confidence").value)
        self._min_depth = float(self.get_parameter("min_depth_m").value)
        self._max_depth = float(self.get_parameter("max_depth_m").value)

        self._tracker = PersonTracker(
            match_radius=float(self.get_parameter("match_radius").value),
            max_coast=float(self.get_parameter("max_coast").value),
            max_predict=float(self.get_parameter("max_predict").value),
            min_hits=int(self.get_parameter("min_hits").value),
            min_confidence=float(self.get_parameter("min_confidence").value),
            min_speed_for_heading=float(
                self.get_parameter("min_speed_for_heading").value
            ),
        )

        self._bridge = CvBridge()
        self._cam_model: Optional[PinholeCameraModel] = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._last_detection_time = 0.0

        self._voter = IdentityVoter(
            window=float(self.get_parameter("identity_vote_window").value),
            min_votes=int(self.get_parameter("identity_min_votes").value),
        )
        # Where each track's bounding box was, per recent frame, so a face
        # recognised on the control PC can be matched back to a track here.
        # A few seconds of history: recognition is slower than detection and
        # its results arrive for frames this node processed a moment ago.
        self._frame_boxes: deque = deque(maxlen=90)

        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self._on_camera_info, 1,
        )

        # Same pairing problem object_projector documents at length: a
        # detection carries the stamp of the image it was computed from, but
        # arrives only once inference has finished. With inference offloaded to
        # another machine that lag is larger still, so the queue has to hold
        # several seconds of depth history or the frame each detection needs
        # has already been evicted when it turns up.
        det_sub = Subscriber(
            self, PersonDetection2DArray, self.get_parameter("detections_topic").value
        )
        depth_sub = Subscriber(self, Image, self.get_parameter("depth_topic").value)
        self._sync = ApproximateTimeSynchronizer(
            [det_sub, depth_sub],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop").value),
        )
        self._sync.registerCallback(self._on_synced)

        self.create_subscription(
            Detection2DArray,
            self.get_parameter("face_identities_topic").value,
            self._on_face_identities,
            10,
        )

        self._people_pub = self.create_publisher(PersonArray, "/perception/people", 10)
        self._marker_pub = self.create_publisher(
            MarkerArray, "/perception/people_markers", 10
        )

        rate = float(self.get_parameter("publish_rate").value)
        self._publish_period = 1.0 / max(rate, 0.1)
        self.create_timer(self._publish_period, self._publish)

        self.get_logger().info("person_tracker up, waiting for camera_info...")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._cam_model is None:
            self._cam_model = PinholeCameraModel()
            self.get_logger().info("camera_info received, person tracking is live")
        self._cam_model.fromCameraInfo(msg)

    # ---- detections -> observations ---------------------------------------

    def _on_synced(self, detections: PersonDetection2DArray, depth_msg: Image) -> None:
        if self._cam_model is None:
            return

        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        scale = depth_scale_for(depth)

        observations = []
        # Boxes of the detections that actually produced an observation, in
        # the same order. Detections whose depth could not be resolved drop
        # out, so this cannot be indexed by position in detections.detections.
        boxes = []
        for detection in detections.detections:
            keypoints = self._reshape_keypoints(detection.keypoints)
            point_cam = self._locate(detection, keypoints, depth, scale)
            if point_cam is None:
                continue

            point_map = self._to_map(point_cam, depth_msg.header)
            if point_map is None:
                continue

            facing = self._facing(keypoints, depth, scale, depth_msg.header)
            observations.append(
                PersonObservation(
                    x=point_map[0], y=point_map[1], z=point_map[2],
                    confidence=float(detection.score),
                    facing=facing,
                )
            )
            boxes.append((float(detection.center_x), float(detection.center_y)))

        t = _stamp_seconds(detections.header.stamp)
        self._last_detection_time = self._now()
        self._tracker.update(observations, t)

        associations = self._tracker.associations()
        self._frame_boxes.append(
            (t, [(boxes[i][0], boxes[i][1], track_id)
                 for i, track_id in associations.items()])
        )

    # ---- names ------------------------------------------------------------

    def _on_face_identities(self, msg: Detection2DArray) -> None:
        """Attach recognised names to tracks, by vote.

        The message arrives from face_identity on the control PC, stamped with
        the frame it was computed from -- so the first job is finding which of
        this node's own recently processed frames that was, and where each
        track's bounding box sat in it.
        """
        if not msg.detections:
            return

        stamp = _stamp_seconds(msg.header.stamp)
        frame = self._frame_for(stamp)
        if frame is None:
            self.get_logger().debug(
                "face identities for a frame this node never processed; ignoring"
            )
            return

        limit = float(self.get_parameter("identity_match_pixels").value)
        now = self._now()
        for detection in msg.detections:
            if not detection.results:
                continue
            best = max(detection.results, key=lambda r: r.hypothesis.score)
            face_x = detection.bbox.center.position.x
            face_y = detection.bbox.center.position.y

            nearest, distance = None, math.inf
            for box_x, box_y, track_id in frame:
                d = math.hypot(box_x - face_x, box_y - face_y)
                if d < distance:
                    nearest, distance = track_id, d

            if nearest is None or distance > limit:
                continue

            self._voter.vote(nearest, best.hypothesis.class_id,
                             float(best.hypothesis.score), now)
            name, confidence = self._voter.best(nearest, now)
            if name and self._tracker.assign_name(nearest, name, confidence):
                self.get_logger().info(
                    f"{nearest} is {name} ({confidence:.2f})"
                )

    def _frame_for(self, stamp: float):
        """The recorded frame closest in time to `stamp`, if any is close enough."""
        best, best_gap = None, math.inf
        for frame_stamp, boxes in self._frame_boxes:
            gap = abs(frame_stamp - stamp)
            if gap < best_gap:
                best, best_gap = boxes, gap
        # Half a frame interval at the slowest detector rate this pipeline
        # runs at. Beyond that the robot has moved and the boxes recorded then
        # no longer say where anybody is now.
        return best if best_gap <= 0.35 else None

    @staticmethod
    def _reshape_keypoints(flat) -> Optional[np.ndarray]:
        """Flat x,y,confidence triples -> (N, 3), or None if there are none."""
        if flat is None or len(flat) < 3:
            return None
        array = np.asarray(flat, dtype=np.float32)
        return array.reshape(-1, 3)

    def _locate(self, detection, keypoints, depth, scale) -> Optional[Tuple[float, float, float]]:
        """Where the person is, in the camera optical frame.

        Prefers the torso keypoints over the bounding-box centre when there are
        any. A person's bbox centre often falls between their arms or legs and
        onto whatever is behind them, which places them a metre or two too far
        away; the shoulders and hips are reliably *on* the person.
        """
        pixels = self._torso_pixels(keypoints)
        if not pixels:
            pixels = [(detection.center_x, detection.center_y)]
            radius = self._depth_radius
        else:
            radius = self._kp_radius

        depths = []
        for px, py in pixels:
            value = sample_depth(
                depth, px, py, radius, self._min_depth, self._max_depth, scale
            )
            if value is not None:
                depths.append((value, px, py))

        if not depths:
            return None

        # Nearest valid torso point, not the median of them: on a partly
        # occluded person some keypoints land on the object in front, and those
        # read as *closer*, never further. Taking the minimum keeps the estimate
        # on the near surface consistently instead of drifting between the
        # person and the furniture depending on which keypoints resolved.
        z, px, py = min(depths, key=lambda d: d[0])
        return backproject(
            px, py, z,
            self._cam_model.fx(), self._cam_model.fy(),
            self._cam_model.cx(), self._cam_model.cy(),
        )

    def _torso_pixels(self, keypoints) -> List[Tuple[float, float]]:
        if keypoints is None:
            return []
        pixels = []
        for index in TORSO_KEYPOINTS:
            if index >= len(keypoints):
                continue
            x, y, confidence = keypoints[index]
            if confidence >= self._kp_min_conf:
                pixels.append((float(x), float(y)))
        return pixels

    def _facing(self, keypoints, depth, scale, header) -> Optional[float]:
        """Map-frame yaw the person faces, from their shoulders. None if unknown."""
        if keypoints is None or len(keypoints) <= KP_RIGHT_SHOULDER:
            return None

        shoulders = []
        for index in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER):
            x, y, confidence = keypoints[index]
            if confidence < self._kp_min_conf:
                return None
            z = sample_depth(
                depth, float(x), float(y), self._kp_radius,
                self._min_depth, self._max_depth, scale,
            )
            if z is None:
                return None
            point_cam = backproject(
                float(x), float(y), z,
                self._cam_model.fx(), self._cam_model.fy(),
                self._cam_model.cx(), self._cam_model.cy(),
            )
            point_map = self._to_map(point_cam, header)
            if point_map is None:
                return None
            shoulders.append(point_map)

        return facing_from_shoulders(shoulders[0], shoulders[1])

    def _to_map(self, point_cam, header) -> Optional[Tuple[float, float, float]]:
        stamped = PointStamped()
        stamped.header.frame_id = self._cam_model.tf_frame
        stamped.header.stamp = header.stamp
        stamped.point.x, stamped.point.y, stamped.point.z = point_cam
        try:
            out = self._tf_buffer.transform(
                stamped, self._map_frame, timeout=rclpy.duration.Duration(seconds=0.2)
            )
        except Exception as exc:
            self.get_logger().warn(
                f"person TF {self._cam_model.tf_frame} -> {self._map_frame} "
                f"failed: {exc}",
                throttle_duration_sec=5.0,
            )
            return None
        return (out.point.x, out.point.y, out.point.z)

    def _robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._robot_frame, rclpy.time.Time()
            )
        except Exception as exc:
            self.get_logger().warn(
                f"robot pose {self._map_frame} -> {self._robot_frame} "
                f"unavailable: {exc}",
                throttle_duration_sec=5.0,
            )
            return None
        q = tf.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        return (tf.transform.translation.x, tf.transform.translation.y, yaw)

    # ---- output -----------------------------------------------------------

    def _publish(self) -> None:
        now = self._now()
        timeout = float(self.get_parameter("detection_timeout").value)
        if self._last_detection_time and (now - self._last_detection_time) > timeout:
            # Detector has gone quiet entirely. Age the tracks out rather than
            # holding somebody's last known position indefinitely.
            self._tracker.update([], now)

        robot = self._robot_pose()
        tracks = self._tracker.tracks()
        # Track ids are handed out afresh as people come and go, so without
        # this the vote table grows for as long as the robot is switched on.
        self._voter.forget_all_except(t.id for t in self._tracker.all_tracks())

        msg = PersonArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame

        for track in tracks:
            msg.people.append(self._to_msg(track, robot))
        self._people_pub.publish(msg)

        if bool(self.get_parameter("publish_markers").value):
            self._marker_pub.publish(self._markers(tracks, msg.header.stamp))

    def _to_msg(self, track: PersonTrack, robot) -> Person:
        person = Person()
        person.track_id = track.id
        person.name = track.name
        person.name_confidence = float(track.name_confidence)

        person.pose.position.x = float(track.x)
        person.pose.position.y = float(track.y)
        person.pose.position.z = float(track.z)
        yaw = track.facing if track.facing is not None else 0.0
        person.pose.orientation.z = math.sin(yaw / 2.0)
        person.pose.orientation.w = math.cos(yaw / 2.0)
        person.has_orientation = track.facing is not None

        person.velocity.x = float(track.vx)
        person.velocity.y = float(track.vy)

        if robot is not None:
            state = relative_to_robot(track, robot[0], robot[1], robot[2])
            person.distance = float(state.distance)
            person.bearing = float(state.bearing)
            person.range_rate = float(state.range_rate)
            person.engagement = float(state.engagement)
        else:
            # No robot pose means every robot-relative field would be a guess.
            # NaN rather than zero: zero distance is a value behaviour would
            # act on, and acting on a fabricated "they are right here" is the
            # worst possible response to not knowing where the robot is.
            person.distance = float("nan")
            person.bearing = float("nan")
            person.range_rate = float("nan")
            person.engagement = float("nan")

        person.confidence = float(track.confidence)
        person.observation_count = int(track.hits)
        person.first_seen = _seconds_to_stamp(track.first_seen)
        person.last_seen = _seconds_to_stamp(track.last_seen)
        person.visible = self._tracker.visible(track.id)
        return person

    def _markers(self, tracks: List[PersonTrack], stamp) -> MarkerArray:
        markers = MarkerArray()
        # Every marker outlives a couple of publish cycles and no more, so a
        # person who leaves takes their marker with them without needing an
        # explicit delete pass (which flickers, because DELETEALL lands before
        # the replacements in the same message).
        lifetime = rclpy.duration.Duration(seconds=3.0 * self._publish_period).to_msg()

        for i, track in enumerate(tracks):
            visible = self._tracker.visible(track.id)

            body = Marker()
            body.header.frame_id = self._map_frame
            body.header.stamp = stamp
            body.ns = "people"
            body.id = i * 3
            body.type = Marker.CYLINDER
            body.action = Marker.ADD
            body.pose.position.x = float(track.x)
            body.pose.position.y = float(track.y)
            body.pose.position.z = 0.85
            body.pose.orientation.w = 1.0
            body.scale.x = body.scale.y = 0.45
            body.scale.z = 1.7
            body.color.r, body.color.g, body.color.b = (0.25, 0.65, 0.95)
            # Coasting tracks draw faint, so a glance at RViz distinguishes a
            # person the robot can actually see from one it is guessing about.
            body.color.a = 0.55 if visible else 0.20
            body.lifetime = lifetime
            markers.markers.append(body)

            label = Marker()
            label.header.frame_id = self._map_frame
            label.header.stamp = stamp
            label.ns = "people_labels"
            label.id = i * 3 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(track.x)
            label.pose.position.y = float(track.y)
            label.pose.position.z = 1.9
            label.pose.orientation.w = 1.0
            label.scale.z = 0.20
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = track.name or track.id
            label.lifetime = lifetime
            markers.markers.append(label)

            if track.facing is not None:
                arrow = Marker()
                arrow.header.frame_id = self._map_frame
                arrow.header.stamp = stamp
                arrow.ns = "people_facing"
                arrow.id = i * 3 + 2
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose.position.x = float(track.x)
                arrow.pose.position.y = float(track.y)
                arrow.pose.position.z = 1.2
                arrow.pose.orientation.z = math.sin(track.facing / 2.0)
                arrow.pose.orientation.w = math.cos(track.facing / 2.0)
                arrow.scale.x, arrow.scale.y, arrow.scale.z = (0.6, 0.08, 0.08)
                arrow.color.r, arrow.color.g, arrow.color.b = (0.95, 0.80, 0.25)
                arrow.color.a = 0.9
                arrow.lifetime = lifetime
                markers.markers.append(arrow)

        return markers


def _stamp_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _seconds_to_stamp(seconds: float) -> TimeMsg:
    msg = TimeMsg()
    msg.sec = int(seconds)
    msg.nanosec = int((seconds - int(seconds)) * 1e9)
    return msg


def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerNode()
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
