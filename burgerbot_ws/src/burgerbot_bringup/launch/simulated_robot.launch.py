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
        description="Render the face in a desktop window alongside the sim."
    )

    use_expressions_arg = DeclareLaunchArgument(
        "use_expressions",
        default_value="true",
        description="Run the mood arbiter and gesture server."
    )

    gazebo = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_description"),
            "launch",
            "gazebo.launch.py"
        ),
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
            "use_sim_time": "True"
        }.items()
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

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(
                get_package_share_directory("nav2_bringup"),
                "rviz",
                "nav2_default_view.rviz"
            )
        ],
        output="screen",
        parameters=[{"use_sim_time": True}]
    )

    # Windowed, not fullscreen, and no kmsdrm -- the point of running the face
    # in simulation is to iterate on expressions next to RViz, not to take over
    # the display.
    face = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_face"),
            "launch",
            "face.launch.py"
        ),
        launch_arguments={
            "fullscreen": "false",
            "video_driver": "",
            "use_sim_time": "true",
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
            "use_sim_time": "true",
            # Resting face is curious while exploring rather than neutral.
            "mapping_mode": use_slam,
        }.items(),
        condition=IfCondition(use_expressions)
    )

    return LaunchDescription([
        use_slam_arg,
        use_face_arg,
        use_expressions_arg,
        gazebo,
        controller,
        joystick,
        localization,
        slam,
        navigation,
        face,
        expressions,
        rviz,
    ])