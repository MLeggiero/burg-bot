"""ROS 2 node: drives the face panel from robot state.

Deliberately a *renderer*, not a decision maker. It owns animation timing,
gaze geometry and drawing; what mood to be in is decided upstream in
burgerbot_expressions and arrives as ExpressionCommand messages. Keeping that
line clean means the mood logic stays testable with no display attached, and
this node stays testable with no robot attached.

Two things are handled here rather than upstream, both because they have to run
at frame rate to look right: inertia derived from the velocity command, and
gaze anticipation derived from the planned path.

Threading: the SDL render loop owns the main thread and the frame clock; rclpy
spins in a background thread and callbacks only ever mutate state behind a
lock. Rendering inside a subscription callback would tie frame rate to message
arrival, which is exactly how a face ends up stuttering whenever the robot goes
quiet.
"""

import math
import threading

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Path
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from burgerbot_msgs.msg import ExpressionCommand, EyeState, FaceState as FaceStateMsg
from burgerbot_msgs.msg import GazeTarget, TouchEvent

from . import expressions
from .animator import Animator
from .easing import clamp
from .layers import Compositor


class FaceNode(Node):
    def __init__(self):
        super().__init__("face_node")

        self.declare_parameter("width", 0)
        self.declare_parameter("height", 0)
        self.declare_parameter("fullscreen", True)
        self.declare_parameter("video_driver", "")
        self.declare_parameter("fps", 45)
        self.declare_parameter("supersample", 2)
        self.declare_parameter("background", [0.0, 0.0, 0.0])
        self.declare_parameter("cmd_vel_topic", "/burgerbot_controller/cmd_vel")
        self.declare_parameter("plan_topic", "/plan")
        self.declare_parameter("state_publish_rate", 10.0)
        self.declare_parameter("screen_frame", "screen_optical_frame")
        self.declare_parameter("robot_frame", "base_footprint")
        # Target this far off the screen normal deflects the eyes fully.
        self.declare_parameter("gaze_max_angle_deg", 60.0)
        self.declare_parameter("path_lookahead", 0.6)
        self.declare_parameter("enable_touch", True)

        self.fps = int(self.get_parameter("fps").value)
        self.supersample = int(self.get_parameter("supersample").value)
        self._gaze_max = math.radians(float(self.get_parameter("gaze_max_angle_deg").value))
        self._lookahead = float(self.get_parameter("path_lookahead").value)
        self._screen_frame = self.get_parameter("screen_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value

        self.animator = Animator()
        self.compositor = Compositor()
        self._lock = threading.Lock()
        self._last_state = None

        # TF is optional. The face is a nicety; if the tree is not up yet it
        # must not take the node down with it, so gaze in a robot frame simply
        # degrades to idle drift until transforms appear.
        self._tf_buffer = None
        self._tf_listener = None
        try:
            from tf2_ros import Buffer, TransformListener

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.get_logger().warn(f"TF unavailable, gaze targets limited: {exc}")

        self.create_subscription(
            ExpressionCommand, "/face/expression", self._on_expression, 10
        )
        self.create_subscription(GazeTarget, "/face/gaze", self._on_gaze, 10)
        self.create_subscription(
            TwistStamped,
            self.get_parameter("cmd_vel_topic").value,
            self._on_cmd_vel,
            10,
        )
        # Nav2 latches the plan; match it or the first plan after startup is
        # missed and anticipation stays dead until the next replan.
        plan_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            Path, self.get_parameter("plan_topic").value, self._on_plan, plan_qos
        )

        self._state_pub = self.create_publisher(FaceStateMsg, "/face/state", 10)
        self._touch_pub = self.create_publisher(TouchEvent, "/face/touch", 10)

        rate = float(self.get_parameter("state_publish_rate").value)
        if rate > 0.0:
            self.create_timer(1.0 / rate, self._publish_state)

        self.get_logger().info(
            f"face_node up: {len(expressions.names())} expressions, "
            f"target {self.fps} fps, supersample {self.supersample}x"
        )

    # ---- subscriptions -------------------------------------------------

    def _on_expression(self, msg: ExpressionCommand) -> None:
        if msg.expression and msg.expression not in expressions.EXPRESSIONS:
            self.get_logger().warn(
                f"unknown expression '{msg.expression}' from "
                f"'{msg.source}', using {expressions.DEFAULT}",
                throttle_duration_sec=10.0,
            )
        with self._lock:
            self.animator.set_expression(
                msg.expression or expressions.DEFAULT,
                intensity=msg.intensity if msg.intensity > 0.0 else 1.0,
                blend_time=msg.blend_time if msg.blend_time > 0.0 else None,
                source=msg.source,
            )

    def _on_gaze(self, msg: GazeTarget) -> None:
        with self._lock:
            if msg.mode == GazeTarget.MODE_IDLE:
                self.compositor.gaze.release()
                return
            if msg.mode == GazeTarget.MODE_SCREEN:
                self.compositor.gaze.look_at(msg.point.x, msg.point.y, msg.weight)
                return

            point = self._to_screen_frame(msg)
            if point is None:
                return
            x, y, z = point
            if msg.mode == GazeTarget.MODE_DIRECTION:
                norm = math.sqrt(x * x + y * y + z * z)
                if norm < 1e-6:
                    return
            if z <= 0.01:
                # Behind the panel. Peg the gaze to the correct side rather
                # than letting atan2 wrap it to the wrong one.
                gx = 1.0 if x > 0 else -1.0
                gy = 0.0
            else:
                # In the optical frame +X is right and +Y is *down*, so the
                # vertical term negates to give screen-up.
                gx = clamp(math.atan2(x, z) / self._gaze_max, -1.0, 1.0)
                gy = clamp(math.atan2(-y, z) / self._gaze_max, -1.0, 1.0)
            self.compositor.gaze.look_at(gx, gy, msg.weight)

    def _to_screen_frame(self, msg: GazeTarget):
        """Transform a gaze point into the screen's optical frame."""
        frame = msg.header.frame_id or self._screen_frame
        if frame == self._screen_frame:
            return msg.point.x, msg.point.y, msg.point.z
        if self._tf_buffer is None:
            return None
        try:
            from geometry_msgs.msg import PointStamped
            import tf2_geometry_msgs  # noqa: F401  (registers the PointStamped conversion)

            stamped = PointStamped()
            stamped.header.frame_id = frame
            # Latest available rather than the message stamp: a face lagging
            # to match a stale transform looks worse than one that is a few
            # milliseconds ahead.
            stamped.header.stamp = rclpy.time.Time().to_msg()
            stamped.point = msg.point
            out = self._tf_buffer.transform(stamped, self._screen_frame)
            return out.point.x, out.point.y, out.point.z
        except Exception as exc:
            self.get_logger().warn(
                f"gaze transform {frame} -> {self._screen_frame} failed: {exc}",
                throttle_duration_sec=5.0,
            )
            return None

    def _on_cmd_vel(self, msg: TwistStamped) -> None:
        with self._lock:
            self.compositor.motion.set_velocity(
                msg.twist.linear.x, msg.twist.angular.z
            )

    def _on_plan(self, msg: Path) -> None:
        curvature = self._path_curvature(msg)
        if curvature is None:
            return
        with self._lock:
            self.compositor.motion.path_curvature = curvature

    def _path_curvature(self, path: Path):
        """Signed curvature of the plan a short way ahead. Positive = left.

        Standard pure-pursuit curvature, 2*y/L^2, against the lookahead point
        expressed in the robot frame. Using the plan rather than the commanded
        angular velocity is what makes the gaze *lead* the turn: the path knows
        about the corner before the controller acts on it.
        """
        if not path.poses or self._tf_buffer is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                self._robot_frame, path.header.frame_id, rclpy.time.Time()
            )
        except Exception:
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        for pose in path.poses:
            p = pose.pose.position
            # Into the robot frame.
            lx = p.x * cos_y - p.y * sin_y + t.x
            ly = p.x * sin_y + p.y * cos_y + t.y
            dist = math.hypot(lx, ly)
            if dist >= self._lookahead:
                return clamp(2.0 * ly / max(dist * dist, 1e-3), -2.0, 2.0)
        return 0.0

    # ---- publishing ----------------------------------------------------

    def _publish_state(self) -> None:
        with self._lock:
            state = self._last_state
            name, source, intensity = (
                self.animator.name,
                self.animator.source,
                self.animator.intensity,
            )
            blinking = self.compositor.blink.amount > 0.01
        if state is None:
            return

        msg = FaceStateMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._screen_frame
        msg.expression = name
        msg.source = source
        msg.intensity = float(intensity)
        msg.left_eye = _eye_msg(state.left)
        msg.right_eye = _eye_msg(state.right)
        msg.face_offset_x = float(state.face_offset_x)
        msg.face_offset_y = float(state.face_offset_y)
        msg.face_tilt = float(state.face_tilt)
        msg.face_scale_x = float(state.face_scale_x)
        msg.face_scale_y = float(state.face_scale_y)
        msg.color = [float(c) for c in state.color]
        msg.blinking = blinking
        self._state_pub.publish(msg)

    def publish_touch(self, kind: int, region: int, x: float, y: float) -> None:
        msg = TouchEvent()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._screen_frame
        msg.type = kind
        msg.region = region
        msg.x = float(x)
        msg.y = float(y)
        self._touch_pub.publish(msg)

    # ---- render loop ----------------------------------------------------

    def run(self) -> None:
        """Own the main thread until shutdown. Blocking."""
        import pygame

        from .renderer import Renderer, create_display

        bg = [int(clamp(float(c)) * 255) for c in self.get_parameter("background").value]
        surface, size = create_display(
            width=int(self.get_parameter("width").value),
            height=int(self.get_parameter("height").value),
            fullscreen=bool(self.get_parameter("fullscreen").value),
            video_driver=self.get_parameter("video_driver").value,
        )
        renderer = Renderer(surface, background=tuple(bg), supersample=self.supersample)
        self.get_logger().info(f"display {size[0]}x{size[1]}")

        touch_enabled = bool(self.get_parameter("enable_touch").value)
        clock = pygame.time.Clock()
        running = True

        while running and rclpy.ok():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q,
                ):
                    running = False
                elif touch_enabled:
                    running = self._handle_touch(event, renderer, pygame) and running

            dt = clock.tick(self.fps) / 1000.0
            # A long stall (scheduler hiccup, SD-card contention) must not be
            # integrated as one huge step, or every layer lurches at once.
            dt = min(dt, 0.1)

            with self._lock:
                self.animator.update(dt)
                self.compositor.set_ambient(self.animator.ambient)
                self.compositor.update(dt)
                state = self.compositor.compose(self.animator.state)
                self._last_state = state

            renderer.draw(state)
            renderer.present()

        pygame.quit()

    def _handle_touch(self, event, renderer, pygame) -> bool:
        """Report touches. Returns False only if the loop should stop."""
        kind = None
        pos = None
        if event.type == pygame.FINGERDOWN:
            kind, pos = TouchEvent.TYPE_DOWN, self._finger_pos(event, renderer)
        elif event.type == pygame.FINGERUP:
            kind, pos = TouchEvent.TYPE_UP, self._finger_pos(event, renderer)
        elif event.type == pygame.FINGERMOTION:
            kind, pos = TouchEvent.TYPE_MOVE, self._finger_pos(event, renderer)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            kind, pos = TouchEvent.TYPE_DOWN, event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            kind, pos = TouchEvent.TYPE_UP, event.pos

        if kind is None or pos is None:
            return True

        region = renderer.hit_test(*pos)
        nx, ny = renderer.to_norm(*pos)
        self.publish_touch(kind, region, nx, ny)
        if kind == TouchEvent.TYPE_DOWN:
            with self._lock:
                # Acknowledge the poke immediately. Waiting for the arbiter to
                # round-trip an expression makes the panel feel unresponsive,
                # and physical contact is the one input that must feel instant.
                self.compositor.reaction.fire("bounce", 0.85)
        return True

    @staticmethod
    def _finger_pos(event, renderer):
        # SDL reports touches normalised to the window; the renderer works in
        # pixels.
        return event.x * renderer.width, event.y * renderer.height


def _eye_msg(eye) -> EyeState:
    msg = EyeState()
    msg.center_x = float(eye.center_x)
    msg.center_y = float(eye.center_y)
    msg.width = float(eye.width)
    msg.height = float(eye.height)
    msg.corner_radius = float(eye.corner_radius)
    msg.rotation = float(eye.rotation)
    msg.pupil_x = float(eye.pupil_x)
    msg.pupil_y = float(eye.pupil_y)
    msg.pupil_radius = float(eye.pupil_radius)
    msg.lid_upper = float(eye.lid_upper)
    msg.lid_lower = float(eye.lid_lower)
    msg.lid_angle = float(eye.lid_angle)
    return msg


def main(args=None):
    rclpy.init(args=args)
    node = FaceNode()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
