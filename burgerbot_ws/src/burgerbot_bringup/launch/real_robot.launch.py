import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_slam = LaunchConfiguration("use_slam")
    use_face = LaunchConfiguration("use_face")
    use_expressions = LaunchConfiguration("use_expressions")

    use_slam_arg = DeclareLaunchArgument(
        "use_slam",
        default_value="false"
    )

    use_face_arg = DeclareLaunchArgument(
        "use_face",
        default_value="true",
        description="Render the face on the DSI panel. Set false for a headless run."
    )

    use_expressions_arg = DeclareLaunchArgument(
        "use_expressions",
        default_value="true",
        description="Run the mood arbiter and gesture server."
    )

    hardware_interface = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_firmware"),
            "launch",
            "hardware_interface.launch.py"
        ),
    )

    laser_driver = Node(
            package="rplidar_ros",
            executable="rplidar_node",
            name="rplidar_node",
            parameters=[os.path.join(
                get_package_share_directory("burgerbot_bringup"),
                "config",
                "rplidar_a1.yaml"
            )],
            output="screen"
    )
    
    controller = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_controller"),
            "launch",
            "controller.launch.py"
        ),
        launch_arguments={
            "use_simple_controller": "False",
            "use_python": "False"
        }.items(),
    )
    
    joystick = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_controller"),
            "launch",
            "joystick_teleop.launch.py"
        ),
        launch_arguments={
            "use_sim_time": "False"
        }.items()
    )

    imu_driver_node = Node(
        package="burgerbot_firmware",
        executable="mpu6050_driver.py"
    )

    localization = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_localization"),
            "launch",
            "global_localization.launch.py"
        ),
        condition=UnlessCondition(use_slam)
    )

    slam = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_mapping"),
            "launch",
            "slam.launch.py"
        ),
        condition=IfCondition(use_slam)
    )

    navigation = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_navigation"),
            "launch",
            "navigation.launch.py"
        ),
    )
    
    # Straight to the DSI panel via KMS/DRM -- no X, no Wayland, no desktop
    # session on the Pi.
    face = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_face"),
            "launch",
            "face.launch.py"
        ),
        launch_arguments={
            "fullscreen": "true",
            "video_driver": "kmsdrm",
            "use_sim_time": "false",
        }.items(),
        condition=IfCondition(use_face)
    )

    expressions = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_expressions"),
            "launch",
            "expressions.launch.py"
        ),
        launch_arguments={
            "use_sim_time": "false",
            "mapping_mode": use_slam,
        }.items(),
        condition=IfCondition(use_expressions)
    )

    return LaunchDescription([
        use_slam_arg,
        use_face_arg,
        use_expressions_arg,
        hardware_interface,
        laser_driver,
        controller,
        joystick,
        imu_driver_node,
        localization,
        slam,
        navigation,
        face,
        expressions,
    ])