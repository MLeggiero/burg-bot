"""Executes what social.py decides: goals, gestures, gaze and face bids.

Every judgement about people lives in social.py, which has no ROS in it. This
node is the half that knows about action servers, TF and topics, and it makes
no social decisions of its own -- the same split gesture_server.py has from
gestures.py, and for the same reason: the interesting logic then becomes
testable without a robot, a person, or a graph.

It bids for the face rather than publishing to it. burgerbot_expressions'
mood_arbiter already exists to pick one winner among competing sources, and
going around it by publishing straight to /face/expression -- as
frontier_explorer historically did -- means two publishers racing on one topic
at whatever rate each happens to run. Bidding means a low battery still beats
being pleased to see somebody, which is the correct priority and not one this
node should be able to override.
"""

import math
import os
import random
from typing import List

import rclpy
import yaml
from geometry_msgs.msg import PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from burgerbot_msgs.action import PlayGesture
from burgerbot_msgs.msg import (
    ExpressionCommand,
    GazeTarget,
    PersonArray,
    SocialState,
)

from .heatmap import best_target
from .social import (
    EXPR_NONE,
    SEEKING,
    Decision,
    PersonView,
    SocialBrain,
    SocialConfig,
)


class SocialBehaviorNode(Node):
    def __init__(self):
        super().__init__("social_behavior")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("people_topic", "/perception/people")
        self.declare_parameter("update_rate", 10.0)
        self.declare_parameter("enabled", True)

        # Every SocialConfig field, so the whole behaviour is tunable from
        # companion.yaml without touching code. Named identically to the
        # dataclass fields and read back by name below, so adding a knob to
        # SocialConfig means adding it to one list here rather than to three
        # places that can drift apart.
        self._config_fields = [
            "standoff", "engage_distance", "disengage_distance", "personal_space",
            "max_front_offset", "max_approach_distance",
            "min_engagement_to_approach", "target_stickiness", "seek_after",
            "watch_before_approach",
            "rejection_travel", "rejections_for_sad", "withdraw_cooldown",
            "sad_duration", "approach_timeout",
            "engage_reward", "reject_penalty", "engage_rate", "affinity_half_life",
            "dance_cooldown", "dance_probability", "dance_min_affinity",
            "dance_min_engagement", "dance_timeout", "greet_cooldown",
            "replan_distance", "gaze_height", "commanded_timeout",
        ]
        defaults = SocialConfig()
        for name in self._config_fields:
            self.declare_parameter(name, getattr(defaults, name))
        self.declare_parameter("dance_gestures", list(defaults.dance_gestures))
        self.declare_parameter("greet_gesture", defaults.greet_gesture)

        # ---- seeking ----
        self.declare_parameter("enable_seeking", True)
        self.declare_parameter("seek_min_distance", 1.5)
        self.declare_parameter("seek_distance_scale", 8.0)
        # Somewhere the robot has already waited recently and found nobody is
        # not somewhere to go back to immediately.
        self.declare_parameter("seek_revisit_radius", 1.2)
        self.declare_parameter("seek_memory", 3)

        self.declare_parameter(
            "state_directory", os.path.join(os.path.expanduser("~"), ".burgerbot")
        )
        self.declare_parameter("state_file", "companion_state.yaml")
        self.declare_parameter("autoload", True)
        self.declare_parameter("autosave_period", 120.0)

        self._map_frame = self.get_parameter("map_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value
        self._enabled = bool(self.get_parameter("enabled").value)

        self._brain = SocialBrain(config=self._build_config(), rng=random.Random())
        if bool(self.get_parameter("autoload").value):
            self._load_state()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._people: List[PersonView] = []
        self._hotspots: List[tuple] = []
        self._recent_seek_targets: List[tuple] = []
        self._nav_goal_handle = None
        self._seek_goal_active = False

        group = ReentrantCallbackGroup()
        self.create_subscription(
            PersonArray, self.get_parameter("people_topic").value,
            self._on_people, 10, callback_group=group,
        )
        self.create_subscription(
            PoseArray, "/companion/hotspots", self._on_hotspots, 1, callback_group=group
        )
        self.create_subscription(
            Bool, "/companion/enable", self._on_enable, 10, callback_group=group
        )
        # Somewhere else -- burgerbot_dialog, today -- has resolved a request
        # into a pose and wants the robot to go there. It deliberately does not
        # call navigate_to_pose itself: that action is last-goal-wins, so a
        # second client would mean two nodes silently preempting each other
        # with nothing in any log to say which one won. This node keeps sole
        # ownership of the client, exactly as mood_arbiter keeps sole ownership
        # of the face.
        self.create_subscription(
            PoseStamped, "/dialog/goal_request", self._on_goal_request, 10,
            callback_group=group,
        )

        # The arbiter's bid inlet, not /face/expression itself.
        self._bid_pub = self.create_publisher(ExpressionCommand, "/face/expression_bid", 10)
        self._gaze_pub = self.create_publisher(GazeTarget, "/face/gaze", 10)
        self._state_pub = self.create_publisher(SocialState, "/companion/social_state", 10)

        self._nav = ActionClient(
            self, NavigateToPose, "navigate_to_pose", callback_group=group
        )
        self._gesture = ActionClient(
            self, PlayGesture, "play_gesture", callback_group=group
        )

        self.create_service(Trigger, "~/save", self._on_save, callback_group=group)
        self.create_service(Trigger, "~/load", self._on_load, callback_group=group)

        rate = float(self.get_parameter("update_rate").value)
        self.create_timer(1.0 / max(rate, 1.0), self._tick, callback_group=group)

        period = float(self.get_parameter("autosave_period").value)
        if period > 0.0:
            self.create_timer(period, self._autosave, callback_group=group)

        self.get_logger().info(
            f"social_behavior up ({'enabled' if self._enabled else 'disabled'})"
        )

    def _build_config(self) -> SocialConfig:
        values = {name: self.get_parameter(name).value for name in self._config_fields}
        values["dance_gestures"] = tuple(self.get_parameter("dance_gestures").value)
        values["greet_gesture"] = self.get_parameter("greet_gesture").value
        # rejections_for_sad is declared from an int default so it arrives as
        # an int; everything else is a float. Cast rather than trust, because a
        # yaml file written as `rejections_for_sad: 3.0` would otherwise make
        # the comparison against an int count silently never fire.
        values["rejections_for_sad"] = int(values["rejections_for_sad"])
        return SocialConfig(**values)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- inputs -----------------------------------------------------------

    def _on_people(self, msg: PersonArray) -> None:
        views = []
        for person in msg.people:
            # person_tracker publishes NaN for the robot-relative fields when
            # it has no robot pose. Acting on those would mean driving to a
            # goal computed from a distance that is not a number.
            if not math.isfinite(person.distance) or not math.isfinite(person.engagement):
                continue
            views.append(
                PersonView(
                    id=person.track_id,
                    x=person.pose.position.x,
                    y=person.pose.position.y,
                    distance=person.distance,
                    bearing=person.bearing,
                    range_rate=person.range_rate,
                    engagement=person.engagement,
                    visible=person.visible,
                    name=person.name,
                    facing=_yaw_of(person.pose) if person.has_orientation else None,
                )
            )
        self._people = views

    def _on_hotspots(self, msg: PoseArray) -> None:
        # z carries the heat -- see person_heatmap._publish for why.
        self._hotspots = [
            (p.position.x, p.position.y, p.position.z) for p in msg.poses
        ]

    def _on_enable(self, msg: Bool) -> None:
        if bool(msg.data) == self._enabled:
            return
        self._enabled = bool(msg.data)
        self.get_logger().info(f"companion {'enabled' if self._enabled else 'disabled'}")
        if not self._enabled:
            self._cancel_nav()
            self._release_gaze()

    def _on_goal_request(self, msg: PoseStamped) -> None:
        if not self._enabled:
            self.get_logger().info("ignoring a commanded goal; companion is disabled")
            return
        q = msg.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self._brain.command(
            msg.pose.position.x, msg.pose.position.y, yaw,
            msg.header.frame_id or "there", self._now(),
        )
        self.get_logger().info(
            f"commanded to ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )

    def _robot_pose(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._robot_frame, rclpy.time.Time()
            )
        except Exception as exc:
            self.get_logger().warn(
                f"robot pose unavailable: {exc}", throttle_duration_sec=5.0
            )
            return None
        q = tf.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        return (tf.transform.translation.x, tf.transform.translation.y, yaw)

    # ---- the tick ---------------------------------------------------------

    def _tick(self) -> None:
        if not self._enabled:
            return
        robot = self._robot_pose()
        if robot is None:
            return

        decision = self._brain.update(self._people, self._now(), robot)
        self._act(decision, robot)
        self._publish_state(decision)

    def _act(self, decision: Decision, robot) -> None:
        if decision.cancel_goal:
            self._cancel_nav()

        if decision.expression != EXPR_NONE:
            self._bid(decision)
        # No else-clause clearing the bid. An ExpressionCommand with a duration
        # expires in the arbiter on its own, so simply not bidding is how a
        # source stands down -- and it avoids a "clear" message racing the
        # next bid when somebody walks in and out of view.

        if decision.gaze is not None:
            self._look_at(*decision.gaze)
        else:
            self._release_gaze()

        if decision.approach_goal is not None:
            self._send_nav_goal(*decision.approach_goal)
            self._seek_goal_active = False

        if decision.gesture is not None:
            self._play(decision.gesture)

        if decision.state == SEEKING:
            self._seek(robot)

    def _bid(self, decision: Decision) -> None:
        msg = ExpressionCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = "companion"
        msg.expression = decision.expression
        msg.intensity = float(decision.expression_intensity)
        msg.priority = int(min(255, max(0, decision.expression_priority)))
        # Outlives a couple of ticks and no more. A social bid that never
        # expired would pin the face to "happy" the moment the node stopped
        # publishing -- which is exactly when it should stop claiming anything.
        msg.duration.sec = 1
        self._bid_pub.publish(msg)

    def _look_at(self, x: float, y: float, z: float) -> None:
        gaze = GazeTarget()
        gaze.header.stamp = self.get_clock().now().to_msg()
        gaze.header.frame_id = self._map_frame
        gaze.mode = GazeTarget.MODE_POINT
        gaze.point.x, gaze.point.y, gaze.point.z = float(x), float(y), float(z)
        gaze.weight = 1.0
        self._gaze_pub.publish(gaze)

    def _release_gaze(self) -> None:
        gaze = GazeTarget()
        gaze.header.stamp = self.get_clock().now().to_msg()
        gaze.header.frame_id = self._map_frame
        gaze.mode = GazeTarget.MODE_IDLE
        self._gaze_pub.publish(gaze)

    # ---- navigation --------------------------------------------------------

    def _send_nav_goal(self, x: float, y: float, yaw: float) -> None:
        if not self._nav.server_is_ready():
            self.get_logger().warn(
                "navigate_to_pose not available; companion cannot approach",
                throttle_duration_sec=10.0,
            )
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self._map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        future = self._nav.send_goal_async(goal)
        future.add_done_callback(self._on_nav_accepted)

    def _on_nav_accepted(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f"goal failed to send: {exc}")
            self._seek_goal_active = False
            return
        if handle is None or not handle.accepted:
            self.get_logger().warn("goal rejected by Nav2")
            self._seek_goal_active = False
            return
        self._nav_goal_handle = handle
        # Every goal, not just seek goals. A seek goal that finishes and leaves
        # its in-flight flag set means the robot drives to one hotspot and then
        # never looks anywhere else for the rest of the session -- it is idle,
        # it believes it is still travelling, and nothing ever contradicts it.
        handle.get_result_async().add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future) -> None:
        try:
            future.result()
        except Exception as exc:
            self.get_logger().warn(f"goal result failed: {exc}")
        # Whatever the outcome -- arrived, aborted, preempted by the next goal
        # -- nothing is in flight on the robot's behalf any more.
        self._seek_goal_active = False
        # Tell the brain too, or a commanded trip only ends on its timeout and
        # the robot stands about for a minute after it has already arrived.
        if self._brain.commanded():
            self._brain.command_finished()

    def _cancel_nav(self) -> None:
        handle, self._nav_goal_handle = self._nav_goal_handle, None
        self._seek_goal_active = False
        if handle is not None:
            handle.cancel_goal_async()

    def _seek(self, robot) -> None:
        """Drive to somewhere people are usually found."""
        if not bool(self.get_parameter("enable_seeking").value):
            return
        if self._seek_goal_active or not self._hotspots:
            return

        target = best_target(
            self._hotspots, robot[0], robot[1],
            min_distance=float(self.get_parameter("seek_min_distance").value),
            exclude=self._recent_seek_targets,
            exclude_radius=float(self.get_parameter("seek_revisit_radius").value),
            distance_scale=float(self.get_parameter("seek_distance_scale").value),
        )
        if target is None:
            # Everywhere worth going has been tried recently. Forget the oldest
            # so the robot eventually circles back rather than parking forever.
            if self._recent_seek_targets:
                self._recent_seek_targets.pop(0)
            return

        yaw = math.atan2(target[1] - robot[1], target[0] - robot[0])
        self._send_nav_goal(target[0], target[1], yaw)
        self._seek_goal_active = True
        self._recent_seek_targets.append(target)
        limit = max(1, int(self.get_parameter("seek_memory").value))
        self._recent_seek_targets = self._recent_seek_targets[-limit:]
        self.get_logger().info(
            f"nobody around; waiting where people usually are "
            f"({target[0]:.1f}, {target[1]:.1f})"
        )

    # ---- gestures ----------------------------------------------------------

    def _play(self, name: str) -> None:
        if not self._gesture.server_is_ready():
            self.get_logger().warn(
                f"gesture server not available; cannot play '{name}'",
                throttle_duration_sec=10.0,
            )
            # Tell the brain immediately rather than leaving it in DANCING
            # until the timeout: there is no server, so nothing will ever
            # report back on its own.
            self._brain.dance_finished()
            return

        goal = PlayGesture.Goal()
        goal.gesture = name
        goal.scale = 1.0
        goal.repeat = 1
        # Aborts rather than waiting for space to clear. A dance held pending
        # for three seconds and then performed is worse than one that simply
        # did not happen -- by then the moment it was a response to has passed.
        goal.abort_if_blocked = True

        future = self._gesture.send_goal_async(goal)
        future.add_done_callback(self._on_gesture_accepted)

    def _on_gesture_accepted(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f"gesture failed to send: {exc}")
            self._brain.dance_finished()
            return
        if handle is None or not handle.accepted:
            self._brain.dance_finished()
            return
        handle.get_result_async().add_done_callback(self._on_gesture_result)

    def _on_gesture_result(self, future) -> None:
        try:
            result = future.result().result
            if not result.completed and result.message:
                self.get_logger().info(f"gesture gated: {result.message}")
        except Exception as exc:
            self.get_logger().warn(f"gesture result failed: {exc}")
        self._brain.dance_finished()

    # ---- telemetry ---------------------------------------------------------

    def _publish_state(self, decision: Decision) -> None:
        msg = SocialState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.state = decision.state
        msg.target_id = decision.target_id
        msg.target_name = decision.target_name
        msg.target_distance = float(decision.target_distance)
        msg.affinity = float(decision.affinity)
        msg.rejections = int(decision.rejections)
        msg.people_visible = int(decision.people_visible)
        msg.reason = decision.reason
        self._state_pub.publish(msg)

    # ---- persistence --------------------------------------------------------

    def _path(self) -> str:
        return os.path.join(
            self.get_parameter("state_directory").value,
            self.get_parameter("state_file").value,
        )

    def _save_state(self) -> str:
        path = self._path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = self._brain.to_dict()
        data["saved_at"] = self._now()
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return path

    def _load_state(self) -> int:
        try:
            with open(self._path()) as f:
                data = yaml.safe_load(f) or {}
        except OSError:
            return 0  # No state yet on a fresh robot; not a problem.

        count = self._brain.load_dict(data)
        # Affinity fades with wall-clock time, not with time the robot spent
        # switched on -- otherwise a robot left off for a month comes back
        # convinced of everything it thought before, and a robot left on
        # forgets its friends at a rate nobody can predict.
        saved_at = float(data.get("saved_at", 0.0))
        if saved_at > 0.0:
            self._brain.decay_affinity(max(0.0, self._now() - saved_at))
        if count:
            self.get_logger().info(f"remembered {count} person(s)")
        return count

    def _autosave(self) -> None:
        try:
            self._save_state()
        except OSError as exc:
            self.get_logger().warn(f"companion state autosave failed: {exc}")

    def _on_save(self, request, response):
        try:
            path = self._save_state()
            response.success = True
            response.message = f"saved companion state to {path}"
        except OSError as exc:
            response.success = False
            response.message = str(exc)
        self.get_logger().info(response.message)
        return response

    def _on_load(self, request, response):
        count = self._load_state()
        response.success = True
        response.message = f"loaded {count} person(s) from {self._path()}"
        return response


def _yaw_of(pose) -> float:
    q = pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def main(args=None):
    rclpy.init(args=args)
    node = SocialBehaviorNode()
    # Multi-threaded because the tick timer, two action clients and the
    # subscriptions all have to make progress concurrently -- a single-threaded
    # executor deadlocks the moment the tick waits on an action result.
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
