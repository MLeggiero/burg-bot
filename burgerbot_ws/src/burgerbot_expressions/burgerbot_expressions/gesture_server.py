"""Plays body gestures, and refuses to when it would be a bad idea.

This node is the feasibility gate. Gestures in gestures.py are authored as pure
character motion with no awareness of the world; everything that makes them
safe to actually execute lives here:

  1. Output goes to `gesture_vel`, the *lowest* priority twist_mux input. Both
     navigation and the operator's joystick outrank it, so a gesture can never
     take the robot away from someone who is steering it.
  2. The safety-stop lock in twist_mux already overrides every input, gesture
     included.
  3. This node additionally checks lidar clearance before and during a gesture,
     so it declines rather than merely being overridden.

Layer 1 alone would technically be enough to be safe, but a gesture that gets
silently suppressed mid-motion looks like a malfunction. Checking first means
the robot either performs the whole gesture or visibly does not start it.

That division -- expressive intent authored upstream, physical constraints
enforced downstream, neither reaching into the other -- is the transferable
idea from Disney's expressive-robot work. They reconcile an animator's motion
with balance using a learned policy; a differential-drive base on a flat floor
needs nothing that clever, but the architecture is the same one.
"""

import math
import threading

import rclpy
from geometry_msgs.msg import Twist
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from burgerbot_msgs.action import PlayGesture

from . import gestures


class GestureServer(Node):
    def __init__(self):
        super().__init__("gesture_server")

        self.declare_parameter("publish_rate", 20.0)
        # Clearance required in the direction a translating gesture moves.
        self.declare_parameter("translate_clearance", 0.35)
        # Clearance required all round before turning on the spot.
        self.declare_parameter("rotate_clearance", 0.22)
        self.declare_parameter("sector_half_width_deg", 50.0)
        self.declare_parameter("robot_frame", "base_footprint")
        # How long a paused gesture waits for the way to clear before it gives
        # up, when abort_if_blocked is false.
        self.declare_parameter("max_gate_wait", 3.0)

        self._rate = float(self.get_parameter("publish_rate").value)
        self._sector = math.radians(float(self.get_parameter("sector_half_width_deg").value))
        self._robot_frame = self.get_parameter("robot_frame").value

        self._lock = threading.Lock()
        self._scan = None
        self._safety_stop = False
        #: Yaw of the laser frame in the robot frame. The lidar on this robot
        #: is mounted rotated, so "straight ahead" is not angle 0 in the scan.
        self._laser_yaw = None

        self._tf_buffer = None
        try:
            from tf2_ros import Buffer, TransformListener

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f"TF unavailable: {exc}")

        group = ReentrantCallbackGroup()
        self._cmd_pub = self.create_publisher(Twist, "gesture_vel", 10)
        self.create_subscription(
            LaserScan, "/scan", self._on_scan, qos_profile_sensor_data, callback_group=group
        )
        self.create_subscription(
            Bool, "/safety_stop", self._on_safety_stop, 10, callback_group=group
        )
        # Fire-and-forget trigger, for testing from the command line:
        #   ros2 topic pub --once /gesture std_msgs/String "{data: wiggle}"
        self.create_subscription(String, "/gesture", self._on_trigger, 10, callback_group=group)

        self._server = ActionServer(
            self,
            PlayGesture,
            "play_gesture",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=group,
        )

        self.get_logger().info(f"gesture_server up: {', '.join(gestures.names())}")

    # ---- inputs ---------------------------------------------------------

    def _on_scan(self, msg: LaserScan) -> None:
        with self._lock:
            self._scan = msg

    def _on_safety_stop(self, msg: Bool) -> None:
        with self._lock:
            self._safety_stop = bool(msg.data)

    def _on_trigger(self, msg: String) -> None:
        name = msg.data.strip()
        if gestures.get(name) is None:
            self.get_logger().warn(f"unknown gesture '{name}'")
            return
        thread = threading.Thread(target=self._play_blocking, args=(name,), daemon=True)
        thread.start()

    def _on_goal(self, goal_request) -> GoalResponse:
        if gestures.get(goal_request.gesture) is None:
            self.get_logger().warn(f"rejecting unknown gesture '{goal_request.gesture}'")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    # ---- the gate -------------------------------------------------------

    def _laser_yaw_offset(self):
        """Yaw of the laser frame within the robot frame, cached."""
        if self._laser_yaw is not None:
            return self._laser_yaw
        with self._lock:
            scan = self._scan
        if scan is None or self._tf_buffer is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                self._robot_frame, scan.header.frame_id, rclpy.time.Time()
            )
        except Exception:
            return None
        q = tf.transform.rotation
        self._laser_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        return self._laser_yaw

    def _min_range(self, center_robot_angle=None) -> float:
        """Closest return, optionally restricted to a sector of the robot frame.

        With no TF the sector cannot be located in the scan, so this falls back
        to the full-circle minimum. That is conservative -- it will refuse a
        gesture that would have been fine -- which is the correct direction to
        fail in for something purely decorative.
        """
        with self._lock:
            scan = self._scan
        if scan is None:
            return math.inf

        offset = self._laser_yaw_offset() if center_robot_angle is not None else None
        use_sector = center_robot_angle is not None and offset is not None
        if use_sector:
            center_in_scan = _wrap(center_robot_angle - offset)

        nearest = math.inf
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            if use_sector:
                angle = scan.angle_min + i * scan.angle_increment
                if abs(_wrap(angle - center_in_scan)) > self._sector:
                    continue
            if r < nearest:
                nearest = r
        return nearest

    def _blocked(self, spec, velocity) -> str:
        """Return a reason string if the gesture must not run now, else ''."""
        with self._lock:
            if self._safety_stop:
                return "safety stop active"

        linear, angular = velocity
        if spec.translates and abs(linear) > 1e-3:
            required = float(self.get_parameter("translate_clearance").value)
            heading = 0.0 if linear > 0.0 else math.pi
            if self._min_range(heading) < required:
                return f"less than {required:.2f} m clear in the direction of travel"

        if abs(angular) > 1e-3:
            required = float(self.get_parameter("rotate_clearance").value)
            if self._min_range() < required:
                return f"less than {required:.2f} m clear to turn"

        return ""

    # ---- execution ------------------------------------------------------

    def _play_blocking(self, name: str, scale: float = 1.0, repeat: int = 1) -> None:
        """Run a gesture to completion. Used by the fire-and-forget topic."""
        spec = gestures.get(name)
        if spec is None:
            return
        period = 1.0 / self._rate
        elapsed = 0.0
        while rclpy.ok():
            velocity, _progress, finished = gestures.sample(name, elapsed, scale, repeat)
            if finished or self._blocked(spec, velocity):
                break
            self._publish(velocity)
            self._sleep(period)
            elapsed += period
        self._publish((0.0, 0.0))

    def _execute(self, goal_handle):
        request = goal_handle.request
        spec = gestures.get(request.gesture)
        result = PlayGesture.Result()

        period = 1.0 / self._rate
        elapsed = 0.0
        gate_wait = 0.0
        max_gate_wait = float(self.get_parameter("max_gate_wait").value)

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._publish((0.0, 0.0))
                goal_handle.canceled()
                result.completed = False
                result.message = "canceled"
                return result

            velocity, progress, finished = gestures.sample(
                request.gesture, elapsed, request.scale, max(1, request.repeat)
            )
            if finished:
                break

            reason = self._blocked(spec, velocity)
            if reason:
                self._publish((0.0, 0.0))
                if request.abort_if_blocked:
                    goal_handle.abort()
                    result.completed = False
                    result.message = reason
                    self.get_logger().info(f"gesture '{request.gesture}' gated: {reason}")
                    return result

                # Hold rather than abandon: wait for the way to clear, but do
                # not advance the gesture clock, so it resumes where it paused
                # instead of jumping forward mid-motion.
                gate_wait += period
                if gate_wait >= max_gate_wait:
                    goal_handle.abort()
                    result.completed = False
                    result.message = f"still blocked after {max_gate_wait:.1f}s: {reason}"
                    return result
                self._publish_feedback(goal_handle, progress, True)
                self._sleep(period)
                continue

            gate_wait = 0.0
            self._publish(velocity)
            self._publish_feedback(goal_handle, progress, False)
            self._sleep(period)
            elapsed += period

        self._publish((0.0, 0.0))
        goal_handle.succeed()
        result.completed = True
        result.message = ""
        return result

    def _publish(self, velocity) -> None:
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.angular.z = float(velocity[1])
        self._cmd_pub.publish(msg)

    @staticmethod
    def _publish_feedback(goal_handle, progress: float, gated: bool) -> None:
        feedback = PlayGesture.Feedback()
        feedback.progress = float(progress)
        feedback.gated = gated
        goal_handle.publish_feedback(feedback)

    def _sleep(self, seconds: float) -> None:
        # A plain time.sleep would block this callback group; the ROS rate
        # yields so the scan and safety-stop callbacks keep updating the gate
        # while a gesture is running.
        self.create_rate(1.0 / seconds).sleep()


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def main(args=None):
    rclpy.init(args=args)
    node = GestureServer()
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
