import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    """Bring up the D435 with the settings burgerbot_perception needs.

    align_depth.enable is the one setting that actually matters here: it
    resamples the depth image onto the color image's pixel grid, so a
    detection's bounding box (in color-image pixel coordinates) can index
    straight into the depth image at the same coordinates. Without it,
    depth and color pixels don't correspond to the same physical point and
    every projected object position would be wrong.

    Resolution/fps (640x480 @ 15) is a deliberate concession to the Pi 4:
    the detector, not the camera, is the actual bottleneck (see
    object_detector.py), and there's no benefit running the camera faster
    than the pipeline consuming it can use.
    """
    rs_launch = os.path.join(
        get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py"
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch),
                launch_arguments={
                    # Matches the frame names baked into burgerbot_camera.xacro
                    # (name:="camera") -- keep these in sync if either changes.
                    "camera_name": "camera",
                    "camera_namespace": "camera",
                    "enable_color": "true",
                    "enable_depth": "true",
                    "align_depth.enable": "true",
                    "rgb_camera.color_profile": "640,480,15",
                    "depth_module.depth_profile": "640,480,15",
                    # Nothing here consumes infrared or a point cloud; leaving
                    # them on on a Pi 4 is pure wasted USB bandwidth and CPU.
                    "enable_infra1": "false",
                    "enable_infra2": "false",
                    "pointcloud.enable": "false",
                    # The camera's own extrinsics come from the physical
                    # unit's factory calibration, not the URDF's estimate --
                    # this is what actually publishes camera_depth_optical_frame
                    # -> camera_color_optical_frame with the real numbers.
                    "publish_tf": "true",
                }.items(),
            ),
        ]
    )
