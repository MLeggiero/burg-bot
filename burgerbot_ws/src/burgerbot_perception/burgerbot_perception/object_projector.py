"""Turns 2D detections + depth into 3D positions in the map frame.

This is what the D435 buys over a plain RGB camera: a detection's 3D position
falls straight out of the depth image at the same pixel coordinates, rather
than having to be estimated by correlating a bounding box's bearing with the
lidar's range at that angle (which fails for anything above or below the
lidar's fixed scan plane, and degrades for anything off-axis).

Pipeline: Detection2DArray + aligned depth image + camera_info
  -> sample depth at each bbox centre (median of a small patch, not a single
     pixel -- depth sensors are noisy and occasionally return zero at any
     given pixel)
  -> back-project pixel + depth to a 3D point in the camera optical frame
     (image_geometry.PinholeCameraModel, the standard tool for exactly this)
  -> TF-transform that point into the map frame
  -> Detection3DArray
"""

import math
from typing import Optional

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray


class ObjectProjector(Node):
    def __init__(self):
        super().__init__("object_projector")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("detections_topic", "/perception/detections2d")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        # Half-width of the square patch sampled around a bbox centre, in
        # pixels. A single pixel is too easily a depth dropout; too large a
        # patch starts averaging in background behind the object's edges.
        self.declare_parameter("depth_sample_radius", 3)
        # RealSense depth is millimetres as uint16; reject anything outside a
        # plausible indoor range rather than trusting a sensor glitch.
        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 8.0)
        self.declare_parameter("sync_slop", 0.15)
        # How many messages per topic the synchronizer keeps while looking for
        # a timestamp match. This is a duration in disguise: the depth stream
        # runs at the camera's full rate (15 Hz in sim), so message_filters'
        # default of 10 holds well under a second of depth history. A
        # detection carries the stamp of the *image it was computed from*, but
        # is only published once inference finishes -- and inference is slower
        # than a single depth frame interval, so with a short queue the
        # matching depth frame has already been evicted by the time the
        # detection shows up and nothing ever pairs. Confirmed: detections2d
        # was publishing chair hits at 0.88-0.95 confidence while
        # detections3d stayed completely empty. Sized for several seconds of
        # depth history so inference latency can't outrun it.
        self.declare_parameter("sync_queue_size", 60)

        self._map_frame = self.get_parameter("map_frame").value
        self._depth_radius = int(self.get_parameter("depth_sample_radius").value)
        self._min_depth = float(self.get_parameter("min_depth_m").value)
        self._max_depth = float(self.get_parameter("max_depth_m").value)

        self._bridge = CvBridge()
        self._cam_model: Optional[PinholeCameraModel] = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self._on_camera_info, 1,
        )

        # Detections and the depth frame they were computed from have to be
        # matched by timestamp, not just "whatever arrived most recently" --
        # the robot moves, and a stale depth frame paired with a fresh
        # detection silently produces a wrong 3D position with no error to
        # show for it.
        det_sub = Subscriber(self, Detection2DArray, self.get_parameter("detections_topic").value)
        depth_sub = Subscriber(self, Image, self.get_parameter("depth_topic").value)
        self._sync = ApproximateTimeSynchronizer(
            [det_sub, depth_sub],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop").value),
        )
        self._sync.registerCallback(self._on_synced)

        self._pub = self.create_publisher(Detection3DArray, "/perception/detections3d", 10)

        self.get_logger().info("object_projector up, waiting for camera_info...")

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._cam_model is None:
            self._cam_model = PinholeCameraModel()
            self.get_logger().info("camera_info received, projection is live")
        self._cam_model.fromCameraInfo(msg)

    def _on_synced(self, detections: Detection2DArray, depth_msg: Image) -> None:
        if self._cam_model is None:
            return  # No intrinsics yet; nothing to project with.
        if not detections.detections:
            return

        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_scale = 0.001 if depth.dtype == np.uint16 else 1.0  # mm -> m

        out = Detection3DArray()
        out.header = detections.header
        out.header.frame_id = self._map_frame

        for det2d in detections.detections:
            point_cam = self._project_one(det2d, depth, depth_scale)
            if point_cam is None:
                continue

            point_map = self._to_map_frame(point_cam, depth_msg.header)
            if point_map is None:
                continue

            det3d = Detection3D()
            det3d.header = out.header
            det3d.id = det2d.id
            det3d.results = det2d.results
            for result in det3d.results:
                result.pose.pose.position = point_map
                result.pose.pose.orientation.w = 1.0
            det3d.bbox.center.position = point_map
            det3d.bbox.center.orientation.w = 1.0
            out.detections.append(det3d)

        if out.detections:
            self._pub.publish(out)

    def _project_one(self, det2d, depth: np.ndarray, depth_scale: float):
        cx = int(det2d.bbox.center.position.x)
        cy = int(det2d.bbox.center.position.y)
        r = self._depth_radius
        h, w = depth.shape[:2]
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        patch = depth[y0:y1, x0:x1].astype(np.float32) * depth_scale

        valid = patch[(patch >= self._min_depth) & (patch <= self._max_depth)]
        if valid.size == 0:
            return None  # No plausible depth in the patch -- likely a
            # reflective/transparent surface or the object edge; skip rather
            # than fabricate a position from noise.

        z = float(np.median(valid))
        # PinholeCameraModel.projectPixelTo3dRay gives a unit ray through the
        # pixel in the optical frame (+Z forward); scaling by z along that
        # ray is the standard pinhole back-projection, not an approximation.
        ray = self._cam_model.projectPixelTo3dRay((cx, cy))
        return (ray[0] * z, ray[1] * z, ray[2] * z)

    def _to_map_frame(self, point_cam, header):
        stamped = PointStamped()
        stamped.header.frame_id = self._cam_model.tf_frame
        stamped.header.stamp = header.stamp
        stamped.point.x, stamped.point.y, stamped.point.z = point_cam
        try:
            out = self._tf_buffer.transform(
                stamped, self._map_frame, timeout=rclpy.duration.Duration(seconds=0.2)
            )
            return out.point
        except Exception as exc:
            self.get_logger().warn(
                f"projection TF {self._cam_model.tf_frame} -> {self._map_frame} failed: {exc}",
                throttle_duration_sec=5.0,
            )
            return None


def main(args=None):
    rclpy.init(args=args)
    node = ObjectProjector()
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
