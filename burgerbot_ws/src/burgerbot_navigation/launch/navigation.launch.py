import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time")
    lifecycle_nodes = ["controller_server", "planner_server", "smoother_server", "bt_navigator", "behavior_server"]
    burgerbot_navigation_pkg = get_package_share_directory("burgerbot_navigation")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true"
    )

    nav2_controller_server = Node(
        package="nav2_controller",
        executable="controller_server",
        output="screen",
        parameters=[
            os.path.join(
                burgerbot_navigation_pkg,
                "config",
                "controller_server.yaml"),
            {"use_sim_time": use_sim_time}
        ],
    )
    
    nav2_planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[
            os.path.join(
                burgerbot_navigation_pkg,
                "config",
                "planner_server.yaml"),
            {"use_sim_time": use_sim_time}
        ],
    )

    nav2_behaviors = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[
            os.path.join(
                burgerbot_navigation_pkg,
                "config",
                "behavior_server.yaml"),
            {"use_sim_time": use_sim_time}
        ],
    )
    
    # bt_navigator.yaml carries default_nav_to_pose_bt_xml as a bare filename
    # (see below) rather than the absolute path Antonio's own machine had
    # hardcoded there -- an install-time absolute path can't be baked into a
    # config file that's supposed to work on both the dev machine and the
    # robot. The launch-time override here computes it correctly for
    # whichever machine this actually runs on.
    behavior_tree_xml = os.path.join(
        burgerbot_navigation_pkg,
        "behavior_tree",
        "simple_navigation_w_replanning_and_recovery.xml",
    )

    nav2_bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[
            os.path.join(
                burgerbot_navigation_pkg,
                "config",
                "bt_navigator.yaml"),
            {
                "use_sim_time": use_sim_time,
                "default_nav_to_pose_bt_xml": behavior_tree_xml,
                "default_nav_through_poses_bt_xml": behavior_tree_xml,
            }
        ],
    )

    nav2_smoother_server = Node(
        package="nav2_smoother",
        executable="smoother_server",
        name="smoother_server",
        output="screen",
        parameters=[
            os.path.join(
                burgerbot_navigation_pkg,
                "config",
                "smoother_server.yaml"),
            {"use_sim_time": use_sim_time}
        ],
    )

    nav2_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[
            {"node_names": lifecycle_nodes},
            {"use_sim_time": use_sim_time},
            {"autostart": True},
            # See the matching comment in burgerbot_mapping/launch/slam.launch.py
            # -- the default 4.0s bond_timeout is tight enough that a heavily
            # loaded startup (e.g. testbed.launch.py use_perception:=true)
            # can abort bringup here too, even with every one of these nodes
            # (controller/planner/behavior/bt_navigator/...) actually fine.
            {"bond_timeout": 12.0},
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        nav2_controller_server,
        nav2_planner_server,
        nav2_smoother_server,
        nav2_behaviors,
        nav2_bt_navigator,
        nav2_lifecycle_manager,
    ])