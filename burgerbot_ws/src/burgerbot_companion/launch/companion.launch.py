import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Companion behaviour and the person heatmap.

    Assumes people are already being tracked (burgerbot_perception's
    people.launch.py), the face and mood arbiter are up
    (burgerbot_expressions), and Nav2 is running -- this node approaches
    people through navigate_to_pose and will say so and do nothing if that
    action server is missing.

        ros2 launch burgerbot_companion companion.launch.py

    Turn the whole behaviour off and on again without restarting anything:

        ros2 topic pub --once /companion/enable std_msgs/Bool "{data: false}"

    which cancels any approach in progress and hands the eyes back. Useful
    while tuning something else, and the first thing to try when the robot is
    doing something around people that you would rather it stopped doing.
    """
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_heatmap = LaunchConfiguration("enable_heatmap")

    config = os.path.join(
        get_package_share_directory("burgerbot_companion"), "config", "companion.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "enable_heatmap",
                default_value="true",
                description="Accumulate where people are seen and publish it "
                "as a map layer. Without it the robot still does everything "
                "else, it just has nowhere in mind to go when it is alone.",
            ),
            Node(
                package="burgerbot_companion",
                executable="social_behavior",
                name="social_behavior",
                output="screen",
                parameters=[config, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="burgerbot_companion",
                executable="person_heatmap",
                name="person_heatmap",
                output="screen",
                parameters=[config, {"use_sim_time": use_sim_time}],
                condition=IfCondition(enable_heatmap),
            ),
        ]
    )
