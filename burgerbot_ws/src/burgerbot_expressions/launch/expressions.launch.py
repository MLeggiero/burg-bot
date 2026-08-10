import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Mood arbiter + gesture server.

    Neither node touches the display, so this is safe to run in simulation or
    on a headless robot. The face renderer is launched separately.
    """
    use_sim_time = LaunchConfiguration("use_sim_time")
    mapping_mode = LaunchConfiguration("mapping_mode")
    enable_gestures = LaunchConfiguration("enable_gestures")

    config = os.path.join(
        get_package_share_directory("burgerbot_expressions"),
        "config",
        "expressions.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "mapping_mode",
                default_value="false",
                description="Resting face is curious rather than neutral while SLAM runs.",
            ),
            DeclareLaunchArgument(
                "enable_gestures",
                default_value="true",
                description="Set false to keep the face but stop the robot moving expressively.",
            ),
            Node(
                package="burgerbot_expressions",
                executable="mood_arbiter",
                name="mood_arbiter",
                output="screen",
                parameters=[
                    config,
                    {"use_sim_time": use_sim_time, "mapping_mode": mapping_mode},
                ],
            ),
            Node(
                package="burgerbot_expressions",
                executable="gesture_server",
                name="gesture_server",
                output="screen",
                parameters=[config, {"use_sim_time": use_sim_time}],
                condition=IfCondition(enable_gestures),
            ),
        ]
    )
