import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """D435 + detection + 3D projection + semantic map, all four pieces.

    Entirely additive: nothing in navigation, localization, mapping, or
    exploration depends on this running. Bring it up alongside the normal
    robot bringup once the camera is physically mounted and a model has been
    exported (scripts/export_detection_model.sh):

        ros2 launch burgerbot_bringup real_robot.launch.py
        ros2 launch burgerbot_perception perception.launch.py

    Or, in Gazebo, with sim:=true to bridge the simulated camera sensor
    (camera_sim.launch.py) instead of talking to real D435 hardware that
    doesn't exist in a container:

        ros2 launch burgerbot_bringup testbed.launch.py use_perception:=true
    """
    use_sim_time = LaunchConfiguration("use_sim_time")
    sim = LaunchConfiguration("sim")
    perception_pkg = get_package_share_directory("burgerbot_perception")

    config = os.path.join(perception_pkg, "config", "perception.yaml")
    # Computed here, not hardcoded in the yaml -- see perception.yaml's own
    # comment on why (the same bug class as the bt_navigator path fix).
    # float32, not the int8 file sitting next to it -- the int8 graph cannot
    # be executed by either TFLite path (XNNPACK fails on its quantized
    # Transpose ops, the reference kernels fail on its sigmoid output scale).
    # scripts/export_detection_model.sh documents this in full.
    model_path = os.path.join(perception_pkg, "models", "yolov8n_float32.tflite")
    labels_path = os.path.join(perception_pkg, "models", "labels.txt")

    camera = IncludeLaunchDescription(
        os.path.join(perception_pkg, "launch", "camera.launch.py"),
        condition=UnlessCondition(sim),
    )
    camera_sim = IncludeLaunchDescription(
        os.path.join(perception_pkg, "launch", "camera_sim.launch.py"),
        condition=IfCondition(sim),
    )

    detector = Node(
        package="burgerbot_perception",
        executable="object_detector",
        name="object_detector",
        output="screen",
        parameters=[
            config,
            {
                "use_sim_time": use_sim_time,
                "model_path": model_path,
                "labels_path": labels_path,
            },
        ],
    )

    projector = Node(
        package="burgerbot_perception",
        executable="object_projector",
        name="object_projector",
        output="screen",
        parameters=[config, {"use_sim_time": use_sim_time}],
    )

    semantic_map = Node(
        package="burgerbot_perception",
        executable="semantic_map",
        name="semantic_map",
        output="screen",
        parameters=[config, {"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "sim", default_value="false",
                description="Bridge Gazebo's simulated camera sensor instead "
                "of launching the real D435 driver.",
            ),
            camera,
            camera_sim,
            detector,
            projector,
            semantic_map,
        ]
    )
