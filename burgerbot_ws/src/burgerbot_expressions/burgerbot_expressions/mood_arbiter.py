"""Turns real robot state into a single expression command.

Everything the face does traces back to something actually true about the
robot. Nothing here is decorative: if the eyes look uncertain it is because the
pose covariance genuinely grew, and if they look tired the battery genuinely
is. That makes the face a piece of honest telemetry you can read across a room,
which is far more useful than a mood that is merely charming.

Sources bid; `arbiter.MoodArbiter` decides. See arbiter.py for why the
arbitration exists at all.
"""

import math

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import BatteryState, LaserScan
from std_msgs.msg import Bool

from burgerbot_msgs.msg import ExpressionCommand, GazeTarget, TouchEvent

from .arbiter import (
    PRIORITY_ALERT,
    PRIORITY_AMBIENT,
    PRIORITY_CONCERN,
    PRIORITY_TASK,
    Candidate,
    MoodArbiter,
    bid_source,
    candidate_from_bid,
)

# rcl_action's status topic QoS. Getting this wrong means silently receiving
# nothing, which looks identical to "navigation is not running".
ACTION_STATUS_QOS = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class MoodArbiterNode(Node):
    def __init__(self):
        super().__init__("mood_arbiter")

        self.declare_parameter("update_rate", 10.0)
        self.declare_parameter("min_hold", 0.6)
        # Republish the winner this often even when unchanged, so a face node
        # that started late or restarted picks up the current mood.
        self.declare_parameter("refresh_period", 2.0)

        self.declare_parameter("danger_distance", 0.22)
        # Below this but outside danger_distance, something is close enough
        # to be worth a wary look -- while mapping, this is what drives
        # neutral (clear path) vs nervous (hugging a wall or obstacle).
        self.declare_parameter("caution_distance", 0.5)
        self.declare_parameter("gaze_attention_distance", 1.2)
        # Height of the gaze point above the floor. The lidar sees obstacles at
        # its own height only; lifting the target a little stops the robot
        # staring at the skirting board when something is beside it.
        self.declare_parameter("gaze_height", 0.15)
        self.declare_parameter("enable_gaze", True)

        self.declare_parameter("battery_low_fraction", 0.20)
        self.declare_parameter("battery_critical_fraction", 0.07)
        # Trace of the x/y/yaw pose covariance above which AMCL is considered
        # to have lost confidence. Tune against `ros2 topic echo /amcl_pose`
        # while driving -- it is very robot and map specific.
        self.declare_parameter("pose_covariance_threshold", 0.35)
        self.declare_parameter("mapping_mode", False)

        self.arbiter = MoodArbiter(min_hold=float(self.get_parameter("min_hold").value))
        self._danger = float(self.get_parameter("danger_distance").value)
        self._caution = float(self.get_parameter("caution_distance").value)
        self._attention = float(self.get_parameter("gaze_attention_distance").value)
        self._mapping_mode = bool(self.get_parameter("mapping_mode").value)
        self._gaze_height = float(self.get_parameter("gaze_height").value)
        self._enable_gaze = bool(self.get_parameter("enable_gaze").value)
        self._last_published = None
        self._last_publish_time = 0.0
        self._nav_active = False

        self._expr_pub = self.create_publisher(ExpressionCommand, "/face/expression", 10)
        self._gaze_pub = self.create_publisher(GazeTarget, "/face/gaze", 10)

        self.create_subscription(
            GoalStatusArray,
            "/navigate_to_pose/_action/status",
            self._on_nav_status,
            ACTION_STATUS_QOS,
        )
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Bool, "/safety_stop", self._on_safety_stop, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl_pose, 10
        )
        self.create_subscription(BatteryState, "/battery_state", self._on_battery, 10)
        self.create_subscription(TouchEvent, "/face/touch", self._on_touch, 10)

        # The inlet for sources that live in other packages -- the companion
        # behaviour, the explorer, anything added later. Without it the only
        # way for another node to affect the face is to publish straight to
        # /face/expression, which does not join the arbitration, it competes
        # with its result: two publishers on one topic, each winning whenever
        # it happened to publish last. The face then flickers between them,
        # which reads as broken rather than as conflicted -- the exact failure
        # this whole file exists to prevent, reintroduced one package over.
        self.create_subscription(
            ExpressionCommand, "/face/expression_bid", self._on_bid, 10
        )

        # The floor. Something must always be bidding, or the face has no
        # defined state between events.
        self.arbiter.submit(
            Candidate(
                source="ambient",
                expression=(
                    ExpressionCommand.CURIOUS
                    if bool(self.get_parameter("mapping_mode").value)
                    else ExpressionCommand.NEUTRAL
                ),
                priority=PRIORITY_AMBIENT,
                stamp=self._now(),
            )
        )

        rate = float(self.get_parameter("update_rate").value)
        self.create_timer(1.0 / max(rate, 1.0), self._tick)
        self.get_logger().info("mood_arbiter up")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- sources --------------------------------------------------------

    def _on_nav_status(self, msg: GoalStatusArray) -> None:
        if not msg.status_list:
            return
        # The list accumulates completed goals; only the newest says anything
        # about what the robot is doing now.
        latest = max(msg.status_list, key=lambda s: _stamp_seconds(s.goal_info.stamp))
        now = self._now()
        status = latest.status

        if status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING):
            self._nav_active = True
            # While mapping, a clean run to the next frontier reads as
            # neutral -- "focused" is reserved for goal-directed navigation
            # outside exploration, where there's an actual destination that
            # warrants visible concentration rather than routine sweeping.
            expression = (
                ExpressionCommand.NEUTRAL if self._mapping_mode else ExpressionCommand.FOCUSED
            )
            self.arbiter.submit(
                Candidate("nav", expression, 1.0, PRIORITY_TASK, stamp=now)
            )
        elif status == GoalStatus.STATUS_SUCCEEDED:
            self._nav_active = False
            self.arbiter.submit(
                Candidate(
                    "nav", ExpressionCommand.HAPPY, 1.0, PRIORITY_TASK,
                    expires_at=now + 4.0, stamp=now,
                )
            )
        elif status in (GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELING,
                        GoalStatus.STATUS_CANCELED):
            self._nav_active = False
            expression = (
                ExpressionCommand.SAD
                if status == GoalStatus.STATUS_ABORTED
                else ExpressionCommand.NEUTRAL
            )
            self.arbiter.submit(
                Candidate(
                    "nav", expression, 1.0, PRIORITY_CONCERN,
                    expires_at=now + 5.0, stamp=now,
                )
            )

    def _on_scan(self, msg: LaserScan) -> None:
        nearest_range = math.inf
        nearest_angle = 0.0
        for i, r in enumerate(msg.ranges):
            # Discard the infinities, NaNs and sub-minimum returns a cheap
            # lidar produces in quantity; treating those as obstacles makes the
            # robot permanently startled.
            if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue
            if r < nearest_range:
                nearest_range = r
                nearest_angle = msg.angle_min + i * msg.angle_increment

        if not math.isfinite(nearest_range):
            self.arbiter.clear("proximity")
            return

        now = self._now()
        if nearest_range < self._danger:
            self.arbiter.submit(
                Candidate(
                    "proximity", ExpressionCommand.STARTLED, 1.0, PRIORITY_ALERT,
                    blend_time=0.10, expires_at=now + 1.2, stamp=now,
                )
            )
        elif self._mapping_mode and nearest_range < self._caution:
            # Below startled, but still close enough to be wary of -- this is
            # what makes the sweep read as neutral-while-clear,
            # nervous-while-hugging-a-wall rather than one flat expression
            # for the whole drive. Gated to mapping_mode so ordinary
            # point-to-point navigation keeps its existing focused/startled
            # behaviour unchanged.
            self.arbiter.submit(
                Candidate(
                    "proximity", ExpressionCommand.NERVOUS, 1.0, PRIORITY_CONCERN,
                    blend_time=0.18, expires_at=now + 1.0, stamp=now,
                )
            )
        else:
            self.arbiter.clear("proximity")

        if self._enable_gaze:
            self._publish_gaze(msg.header.frame_id, nearest_range, nearest_angle)

    def _publish_gaze(self, frame: str, distance: float, angle: float) -> None:
        gaze = GazeTarget()
        gaze.header.stamp = self.get_clock().now().to_msg()
        gaze.header.frame_id = frame
        if distance > self._attention:
            # Nothing worth watching; hand the eyes back to their idle drift.
            gaze.mode = GazeTarget.MODE_IDLE
        else:
            gaze.mode = GazeTarget.MODE_POINT
            gaze.point.x = distance * math.cos(angle)
            gaze.point.y = distance * math.sin(angle)
            gaze.point.z = self._gaze_height
            # Commit harder the closer it is, so distant things get a glance
            # and near things get a stare.
            gaze.weight = float(
                min(1.0, max(0.0, (self._attention - distance) / max(self._attention, 1e-3)))
            )
        self._gaze_pub.publish(gaze)

    def _on_safety_stop(self, msg: Bool) -> None:
        now = self._now()
        if msg.data:
            self.arbiter.submit(
                Candidate(
                    "safety_stop", ExpressionCommand.STARTLED, 1.0, PRIORITY_ALERT,
                    blend_time=0.08, expires_at=now + 1.5, stamp=now,
                )
            )
        else:
            self.arbiter.clear("safety_stop")

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        cov = msg.pose.covariance
        # Diagonal entries for x, y and yaw in the 6x6 row-major covariance.
        trace = cov[0] + cov[7] + cov[35]
        threshold = float(self.get_parameter("pose_covariance_threshold").value)
        now = self._now()
        if trace > threshold:
            # Scale the intensity so mild uncertainty is a flicker of doubt
            # rather than full confusion.
            intensity = min(1.0, (trace - threshold) / max(threshold, 1e-3))
            self.arbiter.submit(
                Candidate(
                    "localization", ExpressionCommand.CONFUSED,
                    max(0.35, intensity), PRIORITY_CONCERN,
                    expires_at=now + 3.0, stamp=now,
                )
            )
        else:
            self.arbiter.clear("localization")

    def _on_battery(self, msg: BatteryState) -> None:
        fraction = msg.percentage
        if fraction > 1.5:  # Some drivers report 0-100 rather than 0-1.
            fraction /= 100.0
        if not math.isfinite(fraction) or fraction <= 0.0:
            return

        now = self._now()
        critical = float(self.get_parameter("battery_critical_fraction").value)
        low = float(self.get_parameter("battery_low_fraction").value)
        if fraction <= critical:
            self.arbiter.submit(
                Candidate("battery", ExpressionCommand.ERROR, 1.0, PRIORITY_CONCERN + 20,
                          stamp=now)
            )
        elif fraction <= low:
            self.arbiter.submit(
                Candidate("battery", ExpressionCommand.SLEEPY, 1.0, PRIORITY_CONCERN,
                          stamp=now)
            )
        else:
            self.arbiter.clear("battery")

    def _on_bid(self, msg: ExpressionCommand) -> None:
        """A bid from another package, submitted like any internal source."""
        if not msg.expression:
            # An empty expression is how a source stands down. Clearing is
            # explicit rather than inferred from silence, because a source that
            # simply stops publishing may equally have crashed, and those two
            # want opposite treatment.
            self.arbiter.clear(bid_source(msg.source))
            return

        self.arbiter.submit(
            candidate_from_bid(
                source=msg.source,
                expression=msg.expression,
                intensity=msg.intensity,
                priority=msg.priority,
                blend_time=msg.blend_time,
                duration=msg.duration.sec + msg.duration.nanosec * 1e-9,
                now=self._now(),
            )
        )

    def _on_touch(self, msg: TouchEvent) -> None:
        if msg.type != TouchEvent.TYPE_DOWN:
            return
        now = self._now()
        # Being touched always shows. Physical contact is the one input where
        # an unresponsive face is actively unpleasant.
        self.arbiter.submit(
            Candidate(
                "touch", ExpressionCommand.HAPPY, 1.0, PRIORITY_ALERT,
                blend_time=0.15, expires_at=now + 2.5, stamp=now,
            )
        )

    # ---- output ---------------------------------------------------------

    def _tick(self) -> None:
        now = self._now()
        winner = self.arbiter.evaluate(now)
        if winner is None:
            return

        refresh = float(self.get_parameter("refresh_period").value)
        changed = (
            self._last_published is None
            or self._last_published[0] != winner.expression
            or self._last_published[1] != winner.source
            or abs(self._last_published[2] - winner.intensity) > 0.05
        )
        if not changed and (now - self._last_publish_time) < refresh:
            return

        msg = ExpressionCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = winner.source
        msg.expression = winner.expression
        msg.intensity = float(winner.intensity)
        msg.priority = int(min(255, max(0, winner.priority)))
        msg.blend_time = float(winner.blend_time)
        self._expr_pub.publish(msg)

        if changed:
            self.get_logger().debug(
                f"face -> {winner.expression} (source={winner.source}, "
                f"priority={winner.priority}, intensity={winner.intensity:.2f})"
            )
        self._last_published = (winner.expression, winner.source, winner.intensity)
        self._last_publish_time = now


def _stamp_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = MoodArbiterNode()
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
