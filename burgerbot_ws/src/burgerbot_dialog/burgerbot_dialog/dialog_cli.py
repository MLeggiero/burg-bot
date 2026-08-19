"""Type at the robot from a terminal, and see what it says back.

The text-first interface. There is no microphone or speaker on this robot, and
building the conversation against a keyboard first is not a compromise -- it is
how you tune a prompt without waiting four seconds to hear each result. When
audio arrives, it publishes to and subscribes from these same two topics and
nothing else changes.

    ros2 run burgerbot_dialog dialog_cli

Ctrl-D or 'quit' to leave. It only ever prints and publishes, so running two of
them, or running one while something else drives /dialog/say, is fine.
"""

import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DialogCLI(Node):
    def __init__(self):
        super().__init__("dialog_cli")
        self._pub = self.create_publisher(String, "/dialog/say", 10)
        self.create_subscription(String, "/dialog/reply", self._on_reply, 10)
        print("Talking to the robot. Ctrl-D or 'quit' to leave.\n", flush=True)

    def _on_reply(self, msg: String) -> None:
        # Carriage return first so a reply arriving while you are mid-word does
        # not interleave with what you are typing. Not perfect -- a proper
        # readline integration would be -- but enough that a conversation is
        # readable.
        print(f"\r  robot: {msg.data}\n> ", end="", flush=True)

    def read_loop(self) -> None:
        while rclpy.ok():
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit"):
                break
            self._pub.publish(String(data=line))
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = DialogCLI()
    # stdin blocks, so it gets its own thread and rclpy keeps the main one.
    threading.Thread(target=node.read_loop, daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
