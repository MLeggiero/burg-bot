import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Bring up the face renderer.

    On the robot the defaults are correct. On a dev machine run it windowed:

        ros2 launch burgerbot_face face.launch.py fullscreen:=false video_driver:=''
    """
    fullscreen = LaunchConfiguration("fullscreen")
    video_driver = LaunchConfiguration("video_driver")
    use_sim_time = LaunchConfiguration("use_sim_time")

    config = os.path.join(
        get_package_share_directory("burgerbot_face"), "config", "face.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("fullscreen", default_value="true"),
            DeclareLaunchArgument(
                "video_driver",
                default_value="kmsdrm",
                description=(
                    "SDL video driver. 'kmsdrm' renders straight to the DSI "
                    "panel with no X or Wayland session. Set empty on a dev "
                    "machine to use the desktop."
                ),
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="burgerbot_face",
                executable="face_node",
                name="face_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "fullscreen": fullscreen,
                        "video_driver": video_driver,
                        "use_sim_time": use_sim_time,
                    },
                ],
                # The face is strictly additive. If it dies, navigation carries
                # on without it -- so do not let it take the launch down.
                respawn=True,
                respawn_delay=3.0,
            ),
        ]
    )
