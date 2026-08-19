"""Runs a conversation: gathers state, asks the model, executes the reply.

Every judgement lives in the pure modules -- what to say is the model's, what
counts as a valid reply is schema.py's, when to give up is conversation.py's,
where a place is is places.py's. This node does topics, actions and threads.

It does not drive the robot. Everything it wants goes through machinery that
already exists and already refuses when it should:

  * the face, by bidding on mood_arbiter's inlet, so a flat battery still wins
  * the body, through the gesture server, so its lidar gate still declines a
    dance there is no room for
  * navigation, by publishing a pose for social_behavior to execute, because
    navigate_to_pose is last-goal-wins and two clients means two nodes
    silently preempting each other with nothing in any log to say which won

The model call runs on a worker thread and never inside a callback. That is not
a performance choice: a callback that blocks for eight seconds stops the tick
timer, which stops the face bid being refreshed, which means the one moment the
robot most needs to look like it is thinking is the moment it freezes.
"""

import math
import os
import threading
from typing import List, Optional, Tuple

import rclpy
import yaml
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from burgerbot_msgs.action import PlayGesture
from burgerbot_msgs.msg import (
    DialogTurn,
    ExpressionCommand,
    GazeTarget,
    PersonArray,
    SemanticMap,
    SocialState,
)
from burgerbot_msgs.srv import NamePlace

from . import schema
from .backend import OpenAICompatBackend
from .conversation import Conversation, ConversationConfig
from .places import PlaceBook
from .prompt import DEFAULT_PERSONA, RobotContext, build_messages


class DialogManager(Node):
    def __init__(self):
        super().__init__("dialog_manager")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("update_rate", 10.0)
        self.declare_parameter("enabled", True)

        self.declare_parameter("base_url", "http://localhost:11434/v1")
        self.declare_parameter("model", "qwen3:14b")
        self.declare_parameter("temperature", 0.4)
        self.declare_parameter("max_tokens", 200)
        self.declare_parameter("use_schema", True)
        self.declare_parameter("request_timeout", 12.0)
        self.declare_parameter("check_backend_at_startup", True)

        self.declare_parameter("persona", DEFAULT_PERSONA)
        self.declare_parameter("max_history_turns", 8)
        self.declare_parameter("max_objects", 10)
        self.declare_parameter("max_say_chars", schema.MAX_SAY_CHARS)

        self.declare_parameter("thinking_after", 0.8)
        self.declare_parameter("stall_after", 3.5)
        self.declare_parameter("give_up_after", 12.0)
        self.declare_parameter("conversation_timeout", 90.0)
        self.declare_parameter("failures_to_open", 3)
        self.declare_parameter("circuit_cooldown", 60.0)
        self.declare_parameter("gesture_cooldown", 12.0)
        self.declare_parameter("nav_cooldown", 20.0)

        self.declare_parameter("enable_navigation", True)
        self.declare_parameter("enable_gestures", True)
        self.declare_parameter("place_standoff", 0.9)
        # Above the companion's 55 so what the robot is saying shows through,
        # below CONCERN (120) so a flat battery still outranks conversation.
        # Not a number the model gets to choose.
        self.declare_parameter("expression_priority", 60)
        self.declare_parameter("gaze_height", 1.5)

        self.declare_parameter(
            "state_directory", os.path.join(os.path.expanduser("~"), ".burgerbot")
        )
        self.declare_parameter("places_file", "places.yaml")

        self._map_frame = self.get_parameter("map_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value
        self._enabled = bool(self.get_parameter("enabled").value)

        self._conversation = Conversation(
            config=ConversationConfig(
                thinking_after=float(self.get_parameter("thinking_after").value),
                stall_after=float(self.get_parameter("stall_after").value),
                give_up_after=float(self.get_parameter("give_up_after").value),
                conversation_timeout=float(
                    self.get_parameter("conversation_timeout").value),
                failures_to_open=int(self.get_parameter("failures_to_open").value),
                circuit_cooldown=float(self.get_parameter("circuit_cooldown").value),
                max_history_turns=int(self.get_parameter("max_history_turns").value),
                gesture_cooldown=float(self.get_parameter("gesture_cooldown").value),
                nav_cooldown=float(self.get_parameter("nav_cooldown").value),
            )
        )
        self._backend = OpenAICompatBackend(
            base_url=self.get_parameter("base_url").value,
            model=self.get_parameter("model").value,
            temperature=float(self.get_parameter("temperature").value),
            max_tokens=int(self.get_parameter("max_tokens").value),
            use_schema=bool(self.get_parameter("use_schema").value),
            schema=schema.RESPONSE_SCHEMA,
        )
        self._places = PlaceBook()
        self._load_places()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._lock = threading.Lock()
        self._social: Optional[SocialState] = None
        self._people: Optional[PersonArray] = None
        self._objects: List[Tuple[str, float, float]] = []
        self._hotspots: List[Tuple[float, float, float]] = []
        self._battery: Optional[float] = None
        self._worker: Optional[threading.Thread] = None

        group = ReentrantCallbackGroup()
        self.create_subscription(String, "/dialog/say", self._on_said, 10,
                                 callback_group=group)
        self.create_subscription(SocialState, "/companion/social_state",
                                 self._on_social, 10, callback_group=group)
        self.create_subscription(PersonArray, "/perception/people",
                                 self._on_people, 10, callback_group=group)
        self.create_subscription(SemanticMap, "/perception/semantic_map_msg",
                                 self._on_objects, 10, callback_group=group)
        self.create_subscription(PoseArray, "/companion/hotspots",
                                 self._on_hotspots, 1, callback_group=group)
        self.create_subscription(BatteryState, "/battery_state",
                                 self._on_battery, 10, callback_group=group)

        self._bid_pub = self.create_publisher(ExpressionCommand, "/face/expression_bid", 10)
        self._gaze_pub = self.create_publisher(GazeTarget, "/face/gaze", 10)
        self._turn_pub = self.create_publisher(DialogTurn, "/dialog/turn", 10)
        self._reply_pub = self.create_publisher(String, "/dialog/reply", 10)
        # Not navigate_to_pose. social_behavior owns that client; see the
        # module docstring.
        self._goal_pub = self.create_publisher(PoseStamped, "/dialog/goal_request", 10)

        self._gesture = ActionClient(self, PlayGesture, "play_gesture",
                                     callback_group=group)
        self.create_service(NamePlace, "~/name_place", self._on_name_place,
                            callback_group=group)

        rate = float(self.get_parameter("update_rate").value)
        self.create_timer(1.0 / max(rate, 1.0), self._tick, callback_group=group)

        if bool(self.get_parameter("check_backend_at_startup").value):
            threading.Thread(target=self._check_backend, daemon=True).start()

        self.get_logger().info(
            f"dialog_manager up: {self._backend.model} at {self._backend.base_url}; "
            f"{len(self._places.names())} place(s) known"
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _check_backend(self) -> None:
        """Say at startup whether the model is reachable.

        Off the main thread, because a server that is down takes the full
        timeout to say so and a node that hangs for ten seconds during
        construction looks like a crash.
        """
        result = self._backend.health()
        if result.ok:
            self.get_logger().info(
                f"model reachable ({result.model}, {result.latency:.1f}s)")
        else:
            self.get_logger().warn(
                f"model NOT reachable ({result.error}: {result.detail}). "
                f"The robot will still run and will say so when spoken to."
            )

    # ---- inputs -----------------------------------------------------------

    def _on_social(self, msg: SocialState) -> None:
        with self._lock:
            self._social = msg

    def _on_people(self, msg: PersonArray) -> None:
        with self._lock:
            self._people = msg

    def _on_objects(self, msg: SemanticMap) -> None:
        with self._lock:
            self._objects = [
                (o.label, o.pose.position.x, o.pose.position.y) for o in msg.objects
            ]

    def _on_hotspots(self, msg: PoseArray) -> None:
        with self._lock:
            # z carries the heat; see person_heatmap for why.
            self._hotspots = [
                (p.position.x, p.position.y, p.position.z) for p in msg.poses
            ]

    def _on_battery(self, msg: BatteryState) -> None:
        fraction = msg.percentage
        if fraction > 1.5:  # some drivers report 0-100
            fraction /= 100.0
        if math.isfinite(fraction) and fraction > 0.0:
            with self._lock:
                self._battery = fraction

    def _robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._robot_frame, rclpy.time.Time()
            )
        except Exception:
            return None
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (tf.transform.translation.x, tf.transform.translation.y, yaw)

    # ---- a turn -----------------------------------------------------------

    def _on_said(self, msg: String) -> None:
        text = msg.data.strip()
        if not text or not self._enabled:
            return

        now = self._now()
        if self._conversation.circuit_open(now):
            # Answer instantly from the canned list rather than spending
            # another timeout discovering the server is still down.
            self._speak(self._conversation.fallback_line("circuit"), heard=text,
                        error="circuit open")
            return

        turn = self._conversation.start_turn(text, now)
        context = self._build_context()
        messages = build_messages(
            context,
            self._conversation.history(),
            text,
            persona=self.get_parameter("persona").value,
            max_history=int(self.get_parameter("max_history_turns").value),
            max_objects=int(self.get_parameter("max_objects").value),
        )

        # One request in flight at a time. A second utterance supersedes rather
        # than queueing, and the superseded turn's reply is dropped by id when
        # it lands -- see Conversation.is_current.
        self._worker = threading.Thread(
            target=self._run_turn, args=(turn.id, text, messages), daemon=True
        )
        self._worker.start()

    def _run_turn(self, turn_id: int, heard: str, messages) -> None:
        timeout = float(self.get_parameter("request_timeout").value)
        result = self._backend.complete(messages, deadline=timeout)
        now = self._now()

        if not self._conversation.is_current(turn_id):
            self.get_logger().debug(
                f"dropped a reply for turn {turn_id}; it was superseded")
            return

        if not result.ok:
            self._conversation.fail(turn_id, now)
            self._speak(self._conversation.fallback_line(result.error or "invalid"),
                        heard=heard, error=f"{result.error}: {result.detail}",
                        latency=result.latency)
            self.get_logger().warn(f"turn failed ({result.error}): {result.detail}")
            return

        parsed = schema.parse_reply(
            result.text, max_say=int(self.get_parameter("max_say_chars").value)
        )
        for problem in parsed.problems:
            self.get_logger().info(f"model reply: {problem}")

        if not parsed.ok:
            self._conversation.fail(turn_id, now)
            self._speak(self._conversation.fallback_line("invalid"), heard=heard,
                        error="; ".join(parsed.problems), latency=result.latency)
            return

        self._conversation.complete(turn_id, heard, parsed.reply.say, now)
        self._execute(parsed.reply, heard, result.latency)

    def _build_context(self) -> RobotContext:
        with self._lock:
            social = self._social
            objects = list(self._objects)
            battery = self._battery
            people = self._people

        robot = self._robot_pose() or (0.0, 0.0, 0.0)
        relative = []
        for label, x, y in objects:
            dx, dy = x - robot[0], y - robot[1]
            bearing = math.atan2(dy, dx) - robot[2]
            relative.append(
                (label, math.hypot(dx, dy),
                 math.atan2(math.sin(bearing), math.cos(bearing)))
            )

        return RobotContext(
            partner_name=social.target_name if social else "",
            partner_affinity=social.affinity if social else 0.0,
            partner_rejections=social.rejections if social else 0,
            # SocialState has no encounter count; a positive affinity only
            # arises from having spent time with somebody, so it stands in.
            partner_encounters=1 if (social and social.affinity > 0.0) else 0,
            social_state=social.state if social else "",
            social_reason=social.reason if social else "",
            people_visible=len(people.people) if people else 0,
            objects=tuple(relative),
            places=tuple(self._places.names()),
            battery=battery,
            mood="",
            can_navigate=bool(self.get_parameter("enable_navigation").value),
        )

    # ---- acting on a reply -------------------------------------------------

    def _execute(self, reply: schema.Reply, heard: str, latency: float) -> None:
        now = self._now()
        goal_name, resolved = "", False

        if reply.go_to and bool(self.get_parameter("enable_navigation").value):
            goal_name = reply.go_to
            resolved = self._try_navigate(reply.go_to, now)

        gesture = ""
        if reply.gesture and bool(self.get_parameter("enable_gestures").value):
            if self._conversation.allow_gesture(now):
                gesture = reply.gesture
                self._play(gesture)
            else:
                # Expected rather than exceptional: small models fill in every
                # optional field, so most turns arrive carrying a gesture.
                self.get_logger().debug(f"gesture '{reply.gesture}' still on cooldown")

        self._speak(reply.say, heard=heard, expression=reply.expression,
                    gesture=gesture, goal=goal_name, goal_resolved=resolved,
                    latency=latency)

    def _try_navigate(self, name: str, now: float) -> bool:
        if not self._conversation.allow_nav(now):
            self.get_logger().info(f"ignoring '{name}': navigation is rate limited")
            return False

        robot = self._robot_pose()
        if robot is None:
            self.get_logger().warn("no robot pose; cannot resolve a place")
            return False

        with self._lock:
            objects = list(self._objects)
            hotspots = list(self._hotspots)

        result = self._places.resolve(
            name, objects=objects, hotspots=hotspots, robot=(robot[0], robot[1]),
            standoff=float(self.get_parameter("place_standoff").value),
        )
        if not result.ok:
            self.get_logger().info(f"'{name}' did not resolve: {result.reason}")
            return False

        x, y, yaw = result.goal
        goal = PoseStamped()
        goal.header.frame_id = self._map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        self._goal_pub.publish(goal)
        self.get_logger().info(
            f"'{name}' -> ({x:.2f}, {y:.2f}) via {result.source}; "
            f"handed to social_behavior"
        )
        return True

    def _play(self, name: str) -> None:
        if not self._gesture.server_is_ready():
            self.get_logger().warn("gesture server not available",
                                   throttle_duration_sec=30.0)
            return
        goal = PlayGesture.Goal()
        goal.gesture = name
        goal.scale = 1.0
        goal.repeat = 1
        # Aborts rather than waiting. A gesture held pending for three seconds
        # and then performed lands after the sentence it belonged to.
        goal.abort_if_blocked = True
        self._gesture.send_goal_async(goal)

    def _speak(self, text: str, heard: str = "", expression: str = "",
               gesture: str = "", goal: str = "", goal_resolved: bool = False,
               latency: float = 0.0, error: str = "") -> None:
        """Publish what the robot says, and the record of why.

        Text only for now: there is no speaker on this robot. When there is,
        this is the one function that changes -- everything upstream already
        produces an utterance and a face to hold while saying it.
        """
        if text:
            self._reply_pub.publish(String(data=text))
            self.get_logger().info(f"robot: {text}")

        if expression:
            self._bid(expression, 1.0)
        elif error:
            self._bid("confused", 0.8)

        turn = DialogTurn()
        turn.header.stamp = self.get_clock().now().to_msg()
        with self._lock:
            turn.speaker = self._social.target_name if self._social else ""
        turn.heard = heard
        turn.said = text
        turn.expression = expression
        turn.gesture = gesture
        turn.goal = goal
        turn.goal_resolved = goal_resolved
        turn.latency = float(latency)
        turn.error = error
        self._turn_pub.publish(turn)

    def _bid(self, expression: str, intensity: float) -> None:
        msg = ExpressionCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = "dialog"
        msg.expression = expression
        msg.intensity = float(intensity)
        msg.priority = int(self.get_parameter("expression_priority").value)
        # Long enough to outlast the sentence being read, short enough that the
        # face is not pinned to it if this node stops publishing -- which is
        # exactly when it should stop claiming anything.
        msg.duration.sec = 4
        self._bid_pub.publish(msg)

    # ---- the tick ----------------------------------------------------------

    def _tick(self) -> None:
        if not self._enabled:
            return
        now = self._now()

        if self._conversation.thinking_visibly(now):
            self._bid("focused", 1.0)
            self._look_away()

        filler = self._conversation.due_filler(now)
        if filler is not None:
            # No model involved, which is the point: this has to work when the
            # model is the thing that is broken.
            self._reply_pub.publish(String(data=filler))
            self.get_logger().info(f"robot: {filler}")

        turn = self._conversation.turn()
        if turn is not None and self._conversation.expired(now):
            self._conversation.fail(turn.id, now)
            self._speak(self._conversation.fallback_line("timeout"),
                        heard=turn.text, error="gave up waiting")

        if self._conversation.idle_too_long(now):
            self._conversation.end_conversation(now)
            self.get_logger().debug("conversation ended; history dropped")

    def _look_away(self) -> None:
        """Break gaze while thinking.

        People look away while formulating an answer and back when they are
        ready to yield the floor. On a face with no mouth this is the strongest
        "working on it" signal available, and it is what stops a two-second
        round trip reading as a hang.
        """
        gaze = GazeTarget()
        gaze.header.stamp = self.get_clock().now().to_msg()
        gaze.header.frame_id = self._robot_frame
        gaze.mode = GazeTarget.MODE_DIRECTION
        gaze.point.x = 1.0
        gaze.point.y = 0.4
        gaze.point.z = 0.6
        gaze.weight = 0.5
        self._gaze_pub.publish(gaze)

    # ---- places ------------------------------------------------------------

    def _places_path(self) -> str:
        return os.path.join(self.get_parameter("state_directory").value,
                            self.get_parameter("places_file").value)

    def _load_places(self) -> int:
        try:
            with open(self._places_path()) as f:
                data = yaml.safe_load(f) or {}
        except OSError:
            return 0  # No places yet on a fresh robot; not a problem.
        return self._places.load_dict(data)

    def _save_places(self) -> str:
        path = self._places_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self._places.to_dict(), f, sort_keys=False)
        return path

    def _on_name_place(self, request, response):
        name = request.name.strip()
        if not name:
            response.success = False
            response.message = "a name is required"
            return response

        if request.forget:
            removed = self._places.forget(name)
            if removed:
                self._save_places()
            response.success = removed
            response.message = (f"forgot {name}" if removed
                                else f"{name} was not a place I knew")
            return response

        robot = self._robot_pose()
        if robot is None:
            # Naming a place while localization is lost stores a
            # confident-looking pose that is simply wrong, and nothing later
            # can tell it apart from a good one.
            response.success = False
            response.message = (
                "I do not know where I am, so I cannot name this place. Check "
                "that localization is running."
            )
            return response

        place = self._places.teach(name, robot[0], robot[1], robot[2])
        if place is None:
            response.success = False
            response.message = f"'{name}' has nothing in it I can use as a name"
            return response

        self._save_places()
        response.success = True
        response.pose.position.x = robot[0]
        response.pose.position.y = robot[1]
        response.pose.orientation.z = math.sin(robot[2] / 2.0)
        response.pose.orientation.w = math.cos(robot[2] / 2.0)
        response.message = (
            f"this is now '{place.name}', at ({robot[0]:.2f}, {robot[1]:.2f})"
        )
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DialogManager()
    # Multi-threaded because the tick timer, the subscriptions and the service
    # all have to keep running while a model call is outstanding.
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
