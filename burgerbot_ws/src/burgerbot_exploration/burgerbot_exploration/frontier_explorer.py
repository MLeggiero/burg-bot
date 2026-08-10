"""Autonomous area mapping: drive to frontiers until none remain.

This is the whole answer to "can the robot map an area on its own" -- it
needs nothing from a camera. slam_toolbox is already building the occupancy
grid from the lidar as the robot moves; this node just has to keep picking
somewhere useful to move to. Detection and scoring live in frontier.py as
pure functions; this node is the thin ROS wrapper: read the map and the
robot's pose, call into that module, dispatch the winner to Nav2, repeat.

Goal dispatch goes through nav2_simple_commander's BasicNavigator rather than
a hand-rolled action client -- it already handles the action lifecycle
(waiting for the server, tracking feedback, timeouts) correctly, and using it
means this file is about exploration policy, not action-client bookkeeping.

Loosely coupled to the expression system: this node publishes ExpressionCommand
directly only for "happy" on a completed sweep -- a whole-sweep event with no
equivalent elsewhere. Everything else the face does while exploring (neutral
on a clean run, nervous near a wall, focused/startled as appropriate) is
mood_arbiter's own nav-status and proximity handling in burgerbot_expressions,
which already arbitrates between sources properly; duplicating any of that
here would just race it on the shared /face/expression topic. If
burgerbot_expressions isn't running, the "happy" publish simply has no
subscriber -- this node has no hard dependency on it.
"""

import math
import threading
import time
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros import Buffer, TransformListener

from .frontier import Frontier, find_frontiers, select_frontier

try:
    from burgerbot_msgs.msg import ExpressionCommand

    _HAVE_EXPRESSIONS = True
except ImportError:  # pragma: no cover - burgerbot_msgs is a hard workspace dep
    _HAVE_EXPRESSIONS = False

MAP_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
)


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("planning_period", 3.0)
        self.declare_parameter("occupied_threshold", 65)
        self.declare_parameter("min_cluster_size", 6)
        self.declare_parameter("size_weight", 1.0)
        self.declare_parameter("distance_weight", 1.5)
        self.declare_parameter("blacklist_radius", 0.6)
        self.declare_parameter("goal_timeout", 60.0)
        # Below this, a frontier isn't worth a special trip -- it is either
        # noise that slipped past min_cluster_size or something the robot
        # already grazes as a side effect of reaching better goals.
        self.declare_parameter("min_frontier_size", 6)
        # Nav2's controller_server general_goal_checker uses xy_goal_tolerance
        # 0.25m (see burgerbot_navigation/config/controller_server.yaml). A
        # frontier inside that radius is a trap: goToPose reports SUCCEEDED
        # on its very first internal check, before the control loop ever
        # runs or publishes a single Twist, so the robot never actually
        # moves -- and since it "succeeded" it is never blacklisted, so the
        # same point (typically the lidar's own self-shadow right behind the
        # robot) gets re-picked forever. Keep this comfortably above that
        # tolerance so every dispatched goal requires real driving.
        self.declare_parameter("min_frontier_distance", 0.4)
        # How long to wait with literally nothing found before declaring the
        # sweep complete. Guards against giving up during a transient gap
        # right after SLAM starts, before the first map has arrived.
        self.declare_parameter("empty_grace_period", 10.0)
        self.declare_parameter("publish_expressions", True)

        self._map_frame = self.get_parameter("map_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value
        self._latest_map: Optional[OccupancyGrid] = None
        self._blacklist: List[Tuple[float, float]] = []
        self._exploring = False
        self._empty_since: Optional[float] = None
        self._done = False

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(OccupancyGrid, "/map", self._on_map, MAP_QOS)

        self._expr_pub = None
        if _HAVE_EXPRESSIONS and bool(self.get_parameter("publish_expressions").value):
            self._expr_pub = self.create_publisher(ExpressionCommand, "/face/expression", 10)

        self._navigator = BasicNavigator()

        self.get_logger().info("frontier_explorer up, waiting for Nav2 and a map...")

    # ---- inputs -----------------------------------------------------------

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._latest_map = msg

    def _robot_pose(self) -> Optional[Tuple[float, float]]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._robot_frame, rclpy.time.Time()
            )
            return tf.transform.translation.x, tf.transform.translation.y
        except Exception as exc:
            self.get_logger().warn(f"no {self._robot_frame}->{self._map_frame} TF yet: {exc}",
                                    throttle_duration_sec=5.0)
            return None

    # ---- expression side-channel -------------------------------------------

    def _publish_expression(self, expression: str, intensity: float = 1.0,
                             duration_s: float = 0.0) -> None:
        if self._expr_pub is None:
            return
        msg = ExpressionCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = "exploration"
        msg.expression = expression
        msg.intensity = float(intensity)
        msg.priority = ExpressionCommand.PRIORITY_TASK
        if duration_s > 0.0:
            msg.duration = Duration(seconds=duration_s).to_msg()
        self._expr_pub.publish(msg)

    # ---- exploration loop ---------------------------------------------------

    def _tick(self) -> None:
        if self._exploring:
            return  # A goal is already in flight; let it finish first.

        if self._latest_map is None:
            return
        pose = self._robot_pose()
        if pose is None:
            return

        grid = self._latest_map
        frontiers = find_frontiers(
            grid.data,
            grid.info.width,
            grid.info.height,
            grid.info.resolution,
            grid.info.origin.position.x,
            grid.info.origin.position.y,
            pose[0],
            pose[1],
            occupied_threshold=int(self.get_parameter("occupied_threshold").value),
            min_cluster_size=int(self.get_parameter("min_cluster_size").value),
        )
        min_size = int(self.get_parameter("min_frontier_size").value)
        min_distance = float(self.get_parameter("min_frontier_distance").value)
        frontiers = [f for f in frontiers if f.size >= min_size and f.distance >= min_distance]

        choice = select_frontier(
            frontiers,
            self._blacklist,
            float(self.get_parameter("blacklist_radius").value),
            size_weight=float(self.get_parameter("size_weight").value),
            distance_weight=float(self.get_parameter("distance_weight").value),
        )

        if choice is None:
            self._handle_nothing_found()
            return

        self._empty_since = None
        self._drive_to(choice)

    def _handle_nothing_found(self) -> None:
        grace = float(self.get_parameter("empty_grace_period").value)
        now = time.monotonic()
        if self._empty_since is None:
            self._empty_since = now
            return
        if now - self._empty_since >= grace:
            self.get_logger().info(
                "no reachable frontiers left -- exploration sweep complete"
            )
            self._publish_expression("happy", duration_s=6.0)
            # Stop re-declaring completion every tick; a manual restart (new
            # goal, or clearing costmaps after moving furniture) is what
            # would legitimately reopen the search.
            self._empty_since = now - grace  # holds at "just declared" state
            self._done = True

    def _drive_to(self, frontier: Frontier) -> None:
        goal = PoseStamped()
        goal.header.frame_id = self._map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = frontier.x
        goal.pose.position.y = frontier.y
        # Face the direction of travel isn't knowable yet -- identity
        # orientation is fine, Nav2's regulated pure pursuit rotates onto the
        # path itself rather than needing a meaningful goal heading here.
        goal.pose.orientation.w = 1.0

        self.get_logger().info(
            f"exploring -> ({frontier.x:.2f}, {frontier.y:.2f})  "
            f"size={frontier.size} dist={frontier.distance:.2f}"
        )
        # No expression published here: mood_arbiter's own nav-status and
        # proximity handling (neutral-while-clear, nervous-near-an-obstacle
        # in mapping_mode) now owns this moment. Publishing "curious" here
        # too raced it on the shared /face/expression topic -- face_node
        # takes whichever message arrives last with no priority arbitration
        # of its own, so the two would flicker against each other on every
        # goal dispatch instead of one coherent state winning.
        self._exploring = True
        self._navigator.goToPose(goal)

        timeout = float(self.get_parameter("goal_timeout").value)
        deadline = time.monotonic() + timeout
        result = TaskResult.UNKNOWN
        while not self._navigator.isTaskComplete():
            if time.monotonic() > deadline:
                self._navigator.cancelTask()
                self.get_logger().warn("frontier goal timed out, blacklisting")
                break
            time.sleep(0.2)
        else:
            result = self._navigator.getResult()

        self._exploring = False
        if result != TaskResult.SUCCEEDED:
            self._blacklist.append((frontier.x, frontier.y))
            self.get_logger().info(
                f"frontier not reached (result={result}), blacklisted "
                f"({len(self._blacklist)} total)"
            )


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()

    # BasicNavigator's blocking helpers (goToPose, isTaskComplete, ...) each
    # spin their own node internally, so _tick can never run as a timer
    # callback on an executor that's already spinning this node -- that's a
    # re-entrant rclpy.spin_until_future_complete() call and rclpy raises
    # "Executor is already spinning". Instead this node's own subscriptions
    # (map, TF) are spun on a background thread, and the exploration loop
    # runs as a plain loop on the main thread, where it's free to block on
    # the navigator without colliding with anything.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    period = float(node.get_parameter("planning_period").value)
    try:
        while rclpy.ok():
            if not node._done:
                node._tick()
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
