import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """twist_mux + twist_relay: the only path from any cmd_vel source (Nav2,
    a joystick, expressive gestures) to the wheels.

    Split out of joystick_teleop.launch.py, which used to bundle this in with
    the joy_node/joy_teleop input devices -- a real bug, not a style choice:
    a launch file that skips joystick_teleop.launch.py because it doesn't
    want to require a physical joystick (testbed.launch.py, autonomous
    exploration with no human in the loop) was also silently skipping the
    only thing that gets Nav2's /cmd_vel output onto the robot at all. Found
    by the robot never moving despite Nav2 accepting and processing goals.

    diff_drive_controller has use_stamped_vel: true, so it needs
    TwistStamped on burgerbot_controller/cmd_vel -- twist_relay is what
    converts twist_mux's plain Twist output into that.
    """
    burgerbot_controller_pkg = get_package_share_directory("burgerbot_controller")
    use_sim_time = LaunchConfiguration("use_sim_time")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="True", description="Use simulated time"
    )

    twist_mux_launch = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("twist_mux"), "launch", "twist_mux_launch.py"
        ),
        launch_arguments={
            "cmd_vel_out": "burgerbot_controller/cmd_vel_unstamped",
            "config_locks": os.path.join(
                burgerbot_controller_pkg, "config", "twist_mux_locks.yaml"
            ),
            "config_topics": os.path.join(
                burgerbot_controller_pkg, "config", "twist_mux_topics.yaml"
            ),
            "config_joy": os.path.join(
                burgerbot_controller_pkg, "config", "twist_mux_joy.yaml"
            ),
            "use_sim_time": use_sim_time,
        }.items(),
    )

    twist_relay_node = Node(
        package="burgerbot_controller",
        executable="twist_relay",
        name="twist_relay",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            twist_mux_launch,
            twist_relay_node,
        ]
    )
