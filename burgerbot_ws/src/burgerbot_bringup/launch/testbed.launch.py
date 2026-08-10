import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Autonomous-mapping testbed: an enclosed room, one obstacle, no human
    driving. This is the integration test for burgerbot_exploration --
    everything downstream of it (frontier detection, clustering, blacklist)
    was only ever exercised against synthetic occupancy grids before this.

        ros2 launch burgerbot_bringup testbed.launch.py

    Brings up Gazebo (test_room.world), the controller, SLAM, Nav2, the
    frontier explorer, and RViz together -- unlike simulated_robot.launch.py,
    which leaves mapping strategy and exploration as separate opt-in pieces
    you compose yourself, this file is specifically "map this room with no
    human in the loop" as a single command.
    """
    use_face = LaunchConfiguration("use_face")
    use_expressions = LaunchConfiguration("use_expressions")
    use_perception = LaunchConfiguration("use_perception")

    use_face_arg = DeclareLaunchArgument(
        "use_face",
        default_value="true",
        description="Show the face alongside Gazebo/RViz during the demo.",
    )
    use_expressions_arg = DeclareLaunchArgument(
        "use_expressions",
        default_value="true",
        description="Mood arbiter -- frontier_explorer publishes curious/happy "
        "as it explores, so this is what actually makes that visible.",
    )
    use_perception_arg = DeclareLaunchArgument(
        "use_perception",
        default_value="false",
        description="Bring up burgerbot_perception (simulated camera + "
        "detector + semantic map) alongside the mapping sweep. Off by "
        "default: it needs an exported model (scripts/export_detection_model.sh) "
        "and adds real CPU cost the base exploration testbed doesn't need.",
    )

    gazebo = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_description"),
            "launch",
            "gazebo.launch.py",
        ),
        launch_arguments={"world_name": "test_room"}.items(),
    )

    controller = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_controller"),
            "launch",
            "controller.launch.py",
        ),
        launch_arguments={
            "use_simple_controller": "False",
            "use_python": "False",
        }.items(),
    )

    # The only path from Nav2's /cmd_vel output to the wheels. Deliberately
    # not joystick_teleop.launch.py -- that also starts joy_node/joy_teleop,
    # which want an actual joystick device. This is the twist_mux+twist_relay
    # half on its own, with no hardware dependency. Omitting this entirely
    # (the first version of this file did) means Nav2 accepts and processes
    # goals normally, computes velocity commands, publishes them to
    # /cmd_vel -- and the robot never moves, because nothing is listening to
    # relay them into the stamped topic diff_drive_controller actually reads.
    twist_mux = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_controller"),
            "launch",
            "twist_mux.launch.py",
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    slam = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_mapping"),
            "launch",
            "slam.launch.py",
        ),
    )

    navigation = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_navigation"),
            "launch",
            "navigation.launch.py",
        ),
    )

    # explore.launch.py defaults use_sim_time to "false" -- it's designed to
    # be bolted onto an already-running robot, real or simulated, and has no
    # opinion of its own about which. Here it's unambiguously simulated;
    # leaving this at its default would drift the explorer's TF lookups
    # against Gazebo's /clock and break frontier goal dispatch.
    exploration = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_exploration"),
            "launch",
            "explore.launch.py",
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=[
            "-d",
            os.path.join(
                get_package_share_directory("nav2_bringup"),
                "rviz",
                "nav2_default_view.rviz",
            ),
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    face = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_face"), "launch", "face.launch.py"
        ),
        launch_arguments={
            "fullscreen": "false",
            "video_driver": "",
            "use_sim_time": "true",
        }.items(),
        condition=IfCondition(use_face),
    )

    expressions = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_expressions"),
            "launch",
            "expressions.launch.py",
        ),
        launch_arguments={
            "use_sim_time": "true",
            "mapping_mode": "true",
        }.items(),
        condition=IfCondition(use_expressions),
    )

    perception = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_perception"),
            "launch",
            "perception.launch.py",
        ),
        launch_arguments={"use_sim_time": "true", "sim": "true"}.items(),
        condition=IfCondition(use_perception),
    )

    return LaunchDescription(
        [
            use_face_arg,
            use_expressions_arg,
            use_perception_arg,
            gazebo,
            controller,
            twist_mux,
            slam,
            navigation,
            exploration,
            face,
            expressions,
            perception,
            rviz,
        ]
    )
