import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """The conversation layer.

    Needs a model server reachable at `base_url`. Ollama and vLLM both work;
    see the control-PC section of docs/INSTALL.md, and in particular the VRAM
    note -- vLLM's default memory fraction will starve the perception nodes.

        ros2 launch burgerbot_dialog dialog.launch.py

    Then talk to it:

        ros2 run burgerbot_dialog dialog_cli

    Teach it somewhere by driving there and naming it. This is what makes "go
    to the kitchen" mean anything; nothing here infers room labels:

        ros2 service call /dialog_manager/name_place \
            burgerbot_msgs/srv/NamePlace "{name: kitchen}"

    The node starts and runs normally with no model server at all -- it says so
    at startup and says so again, in character, when spoken to. That is
    deliberate: an unreachable model must not be the difference between a robot
    that works and one that does not.
    """
    use_sim_time = LaunchConfiguration("use_sim_time")
    base_url = LaunchConfiguration("base_url")
    model = LaunchConfiguration("model")
    enable_cli = LaunchConfiguration("enable_cli")

    config = os.path.join(
        get_package_share_directory("burgerbot_dialog"), "config", "dialog.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "base_url",
                default_value="http://localhost:11434/v1",
                description="OpenAI-compatible endpoint. Ollama defaults to "
                "port 11434, vLLM to 8000. Point this at the control PC's "
                "address when the model runs there.",
            ),
            DeclareLaunchArgument("model", default_value="qwen3:14b"),
            DeclareLaunchArgument(
                "enable_cli",
                default_value="false",
                description="Also start the typing console. Usually better run "
                "in its own terminal with `ros2 run`, since it wants stdin.",
            ),
            Node(
                package="burgerbot_dialog",
                executable="dialog_manager",
                name="dialog_manager",
                output="screen",
                parameters=[
                    config,
                    {
                        "use_sim_time": use_sim_time,
                        "base_url": base_url,
                        "model": model,
                    },
                ],
            ),
            Node(
                package="burgerbot_dialog",
                executable="dialog_cli",
                name="dialog_cli",
                output="screen",
                condition=IfCondition(enable_cli),
            ),
        ]
    )
