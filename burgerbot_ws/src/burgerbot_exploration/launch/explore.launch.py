import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
    goal_timeout = LaunchConfiguration("goal_timeout")
    map_topic = LaunchConfiguration("map_topic")

    config = os.path.join(
        get_package_share_directory("burgerbot_exploration"),
        "config",
        "exploration.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            # Overridable because it scales with the size of the space, not
            # with the robot. exploration.yaml's default suits a room; a
            # frontier across a warehouse can be 20m+ away, which is longer
            # than that default at this robot's speed, so every distant
            # goal would time out and be blacklisted as unreachable even
            # though the robot was making fine progress toward it.
            DeclareLaunchArgument(
                "goal_timeout",
                default_value="60.0",
                description="Seconds to wait for a frontier goal before "
                "giving up on it and blacklisting it. Raise for large spaces.",
            ),
            DeclareLaunchArgument(
                "map_topic",
                default_value="/map",
                description="Occupancy grid to find frontiers in. Use "
                "/global_costmap/costmap to explore from Nav2's live "
                "scan-fused costmap instead of slam_toolbox's map.",
            ),
            Node(
                package="burgerbot_exploration",
                executable="frontier_explorer",
                name="frontier_explorer",
                output="screen",
                # The dict comes after the yaml, so it wins for the keys it
                # sets and leaves everything else in the file alone.
                parameters=[
                    config,
                    {
                        "use_sim_time": use_sim_time,
                        "goal_timeout": ParameterValue(goal_timeout, value_type=float),
                        "map_topic": map_topic,
                    },
                ],
            ),
        ]
    )
