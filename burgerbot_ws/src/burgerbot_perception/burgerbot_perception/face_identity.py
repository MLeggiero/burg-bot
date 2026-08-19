"""Puts names to faces, so the companion can remember individual people.

Runs on the control PC next to person_detector_gpu, on the same compressed
colour stream, and publishes what it recognises back as a Detection2DArray
whose class_id is a person's name and whose bbox locates them in the frame.
person_tracker matches those boxes against the person detections it already
processed for that same frame and attaches the name to the right track.

Reusing Detection2DArray rather than inventing a message is not laziness -- it
is exactly what the type is: a thing found in an image, with a class and a
score. Matching on bounding box instead of on array index also means the two
nodes stay correct if one of them ever drops a detection the other kept.

Recognition here never decides anything on its own. A single frame's match is
unreliable at 2 m, and the failure mode is asymmetric: no name costs nothing,
while a wrong name means the robot greets the wrong person and starts filing
its memories of them under somebody else. So matches are votes, and
identity.py's IdentityVoter decides only once one name has won repeatedly.
"""

import os
import threading
from typing import List, Optional

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_srvs.srv import Trigger
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from burgerbot_msgs.srv import EnrollPerson

from .identity import IdentityGallery, sharpness


class FaceIdentity(Node):
    def __init__(self):
        super().__init__("face_identity")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("use_compressed", True)
        self.declare_parameter("output_topic", "/perception/face_identities")

        self.declare_parameter("model_pack", "buffalo_l")
        self.declare_parameter("use_gpu", True)
        self.declare_parameter("det_size", 640)
        self.declare_parameter("max_rate_hz", 5.0)

        self.declare_parameter("match_threshold", 0.42)
        self.declare_parameter("match_margin", 0.05)
        self.declare_parameter("max_embeddings", 12)

        # A face smaller than this is a handful of pixels of cheekbone and
        # produces an embedding that matches everybody equally badly. Rejecting
        # it outright is better than feeding noise into the vote.
        self.declare_parameter("min_face_pixels", 48)
        self.declare_parameter("min_sharpness", 40.0)
        self.declare_parameter("enroll_views", 6)
        self.declare_parameter("enroll_timeout", 20.0)

        # Deliberately outside the repo and outside the map directory. Face
        # embeddings are not map data and have no business being committed
        # alongside one; keeping them in the robot's own state directory means
        # a `git add -A` cannot sweep somebody's biometrics into a public
        # repository by accident.
        self.declare_parameter(
            "state_directory", os.path.join(os.path.expanduser("~"), ".burgerbot")
        )
        self.declare_parameter("gallery_file", "faces.yaml")
        self.declare_parameter("autoload", True)
        self.declare_parameter("autosave", True)

        self._bridge = CvBridge()
        self._min_face = int(self.get_parameter("min_face_pixels").value)
        self._min_sharpness = float(self.get_parameter("min_sharpness").value)
        self._min_period = 1.0 / max(float(self.get_parameter("max_rate_hz").value), 0.01)
        self._last_run = 0.0

        self._gallery = IdentityGallery(
            match_threshold=float(self.get_parameter("match_threshold").value),
            margin=float(self.get_parameter("match_margin").value),
            max_embeddings=int(self.get_parameter("max_embeddings").value),
        )
        # No vote accumulation here on purpose. Deciding a name means deciding
        # it *for a track*, and tracks live in person_tracker -- this node only
        # ever sees pixels and has no way to tell one frame's face from the
        # same person's face in the next. So it reports per-frame matches and
        # person_tracker, which does know, does the voting.
        self._lock = threading.Lock()
        #: Set to a name while an enrolment is in progress; the image callback
        #: files good captures against it instead of matching them.
        self._enrolling: Optional[str] = None
        self._enrolled_views: List[np.ndarray] = []

        self._app = self._load_model()
        if bool(self.get_parameter("autoload").value):
            self._load_gallery()

        group = ReentrantCallbackGroup()
        self._pub = self.create_publisher(
            Detection2DArray, self.get_parameter("output_topic").value, 10
        )

        topic = self.get_parameter("image_topic").value
        if bool(self.get_parameter("use_compressed").value):
            self.create_subscription(
                CompressedImage, f"{topic}/compressed", self._on_compressed,
                qos_profile_sensor_data, callback_group=group,
            )
        else:
            self.create_subscription(
                Image, topic, self._on_raw, qos_profile_sensor_data, callback_group=group
            )

        self.create_service(
            EnrollPerson, "~/enroll", self._on_enroll, callback_group=group
        )
        self.create_service(Trigger, "~/save", self._on_save, callback_group=group)
        self.create_service(Trigger, "~/load", self._on_load, callback_group=group)

        self.get_logger().info(
            f"face_identity up: {len(self._gallery.names())} enrolled "
            f"({', '.join(self._gallery.names()) or 'nobody yet'})"
        )

    # ---- model ------------------------------------------------------------

    def _load_model(self):
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise RuntimeError(
                "face_identity needs insightface and onnxruntime-gpu:\n"
                "  pip install insightface onnxruntime-gpu\n"
                "This node runs on the control PC, not the Pi."
            )

        use_gpu = bool(self.get_parameter("use_gpu").value)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        pack = self.get_parameter("model_pack").value
        app = FaceAnalysis(name=pack, providers=providers)
        size = int(self.get_parameter("det_size").value)
        # ctx_id 0 selects the first GPU; -1 forces CPU. Passed separately from
        # `providers` because insightface uses each for a different thing, and
        # setting only one of them silently runs half the pipeline on the CPU.
        app.prepare(ctx_id=0 if use_gpu else -1, det_size=(size, size))
        self.get_logger().info(f"insightface '{pack}' ready ({'GPU' if use_gpu else 'CPU'})")
        return app

    # ---- recognition --------------------------------------------------------

    def _on_compressed(self, msg: CompressedImage) -> None:
        if self._throttled():
            return
        self._process(
            self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8"), msg.header
        )

    def _on_raw(self, msg: Image) -> None:
        if self._throttled():
            return
        self._process(self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), msg.header)

    def _throttled(self) -> bool:
        # Slower than person detection on purpose. A person's identity does not
        # change between frames, so recognition only has to be fast enough to
        # accumulate a few votes while somebody is nearby -- and the votes are
        # what decide, not any single frame.
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_run < self._min_period:
            return True
        self._last_run = now
        return False

    def _process(self, frame: np.ndarray, header) -> None:
        faces = self._app.get(frame)

        out = Detection2DArray()
        out.header = header  # the source frame's stamp, for association upstream

        for face in faces:
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            width, height = x2 - x1, y2 - y1
            if min(width, height) < self._min_face:
                continue

            with self._lock:
                enrolling = self._enrolling

            if enrolling is not None:
                self._maybe_capture(frame, face, x1, y1, x2, y2)
                continue

            name, score = self._gallery.match(face.normed_embedding)
            if not name:
                continue

            detection = Detection2D()
            detection.header = header
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = name
            hypothesis.hypothesis.score = float(score)
            detection.results.append(hypothesis)

            bbox = BoundingBox2D()
            bbox.center.position.x = x1 + width / 2.0
            bbox.center.position.y = y1 + height / 2.0
            bbox.size_x = width
            bbox.size_y = height
            detection.bbox = bbox
            out.detections.append(detection)

        self._pub.publish(out)

    # ---- enrolment ----------------------------------------------------------

    def _maybe_capture(self, frame, face, x1, y1, x2, y2) -> None:
        """Store this view if it is good enough to be worth keeping.

        Quality gating matters more here than anywhere else in the pipeline.
        A blurred or badly-lit enrolment does not merely fail to help; it
        drags the stored set toward a smeared average that then matches
        everybody a little, which is worse than having no view at all.
        """
        crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        if crop.size == 0:
            return
        gray = crop.mean(axis=2) if crop.ndim == 3 else crop
        if sharpness(gray) < self._min_sharpness:
            return

        embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        with self._lock:
            if self._enrolling is None:
                return
            # Skip a view nearly identical to one already captured: six copies
            # of the same straight-on angle are worth about as much as one.
            # The point of capturing several is coverage.
            if any(float(np.dot(embedding, e)) > 0.92 for e in self._enrolled_views):
                return
            self._enrolled_views.append(embedding)
            captured = len(self._enrolled_views)
        self.get_logger().info(f"captured view {captured}")

    def _on_enroll(self, request, response):
        name = request.name.strip()
        if not name:
            response.success = False
            response.message = "a name is required"
            return response

        if request.forget:
            with self._lock:
                removed = self._gallery.forget(name)
            if removed and bool(self.get_parameter("autosave").value):
                self._save_gallery()
            response.success = removed
            response.message = (
                f"forgot {name}" if removed else f"{name} was not enrolled"
            )
            response.views = 0
            return response

        with self._lock:
            if self._enrolling is not None:
                response.success = False
                response.message = f"already enrolling {self._enrolling}"
                return response
            self._enrolling = name
            self._enrolled_views = []

        wanted = int(self.get_parameter("enroll_views").value)
        timeout = float(self.get_parameter("enroll_timeout").value)
        self.get_logger().info(
            f"enrolling '{name}': look at the camera and turn your head slowly "
            f"({wanted} views, {timeout:.0f}s)"
        )

        # Blocks the caller on purpose -- "did it work?" is the only thing the
        # person standing in front of the camera wants to know, and a service
        # that returned immediately would leave them guessing. Safe because
        # this node runs on a MultiThreadedExecutor with a reentrant group, so
        # the image callback keeps capturing throughout.
        rate = self.create_rate(10.0)
        waited = 0.0
        while rclpy.ok() and waited < timeout:
            with self._lock:
                if len(self._enrolled_views) >= wanted:
                    break
            rate.sleep()
            waited += 0.1

        with self._lock:
            views = list(self._enrolled_views)
            self._enrolling = None
            self._enrolled_views = []
            for embedding in views:
                identity = self._gallery.enroll(name, embedding)
            stored = len(identity.embeddings) if views else 0

        if not views:
            response.success = False
            response.message = (
                f"no face good enough to enrol in {timeout:.0f}s -- get closer, "
                f"face the camera, and check the lighting"
            )
            response.views = 0
            return response

        if bool(self.get_parameter("autosave").value):
            self._save_gallery()

        response.success = True
        response.views = stored
        response.message = f"enrolled {name} from {len(views)} view(s), {stored} stored"
        self.get_logger().info(response.message)
        return response

    # ---- persistence ---------------------------------------------------------

    def _gallery_path(self) -> str:
        return os.path.join(
            self.get_parameter("state_directory").value,
            self.get_parameter("gallery_file").value,
        )

    def _save_gallery(self) -> str:
        path = self._gallery_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self._lock:
            data = self._gallery.to_dict()
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return path

    def _load_gallery(self) -> int:
        path = self._gallery_path()
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except OSError:
            # A missing gallery on a fresh robot is the normal case, not a
            # problem worth a warning every startup.
            return 0
        return self._gallery.load_dict(data)

    def _on_save(self, request, response):
        try:
            path = self._save_gallery()
            response.success = True
            response.message = f"saved {len(self._gallery.names())} person(s) to {path}"
        except OSError as exc:
            response.success = False
            response.message = str(exc)
        self.get_logger().info(response.message)
        return response

    def _on_load(self, request, response):
        count = self._load_gallery()
        response.success = True
        response.message = f"loaded {count} person(s) from {self._gallery_path()}"
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = FaceIdentity()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
