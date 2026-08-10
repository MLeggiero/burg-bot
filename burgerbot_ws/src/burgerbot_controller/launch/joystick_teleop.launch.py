import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Joystick input devices, plus the twist_mux/twist_relay bridge they need
    to reach the wheels through. If you don't have a physical joystick, you
    still need burgerbot_controller/launch/twist_mux.launch.py on its own --
    see that file's docstring for why it's a separate file now.
    """

    use_sim_time_arg = DeclareLaunchArgument(name="use_sim_time", default_value="True",
                                      description="Use simulated time"
    )

    joy_teleop = Node(
        package="joy_teleop",
        executable="joy_teleop",
        parameters=[os.path.join(get_package_share_directory("burgerbot_controller"), "config", "joy_teleop.yaml"),
                    {"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joystick",
        parameters=[os.path.join(get_package_share_directory("burgerbot_controller"), "config", "joy_config.yaml"),
                    {"use_sim_time": LaunchConfiguration("use_sim_time")}]
    )

    twist_mux = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_controller"),
            "launch",
            "twist_mux.launch.py",
        ),
        launch_arguments={"use_sim_time": LaunchConfiguration("use_sim_time")}.items(),
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            joy_teleop,
            joy_node,
            twist_mux,
        ]
    )
