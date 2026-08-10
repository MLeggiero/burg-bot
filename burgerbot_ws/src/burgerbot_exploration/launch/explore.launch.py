import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Autonomous mapping: SLAM + Nav2 must already be running.

        ros2 launch burgerbot_bringup real_robot.launch.py use_slam:=true
        ros2 launch burgerbot_exploration explore.launch.py

    This intentionally does not bring up SLAM or Nav2 itself -- exploration
    is something you turn on against an already-running mapping session, not
    a mode the base bringup switches into. Stop it any time with Ctrl-C or
    `ros2 lifecycle` on Nav2; the robot simply stops requesting new goals and
    finishes whatever it was already doing.
    """
    use_sim_time = LaunchConfiguration("use_sim_time")

    config = os.path.join(
        get_package_share_directory("burgerbot_exploration"),
        "config",
        "exploration.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="burgerbot_exploration",
                executable="frontier_explorer",
                name="frontier_explorer",
                output="screen",
                parameters=[config, {"use_sim_time": use_sim_time}],
            ),
        ]
    )
