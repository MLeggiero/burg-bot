import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

#: How a launch argument spells "yes". Compared case-insensitively because
#: users type all of these and every one of them means the same thing.
_TRUTHY = "('true', '1', 'yes', 'on')"


def _truthy(config):
    return ["'", config, "'.lower() in ", _TRUTHY]


def _any_true(*configs):
    parts = []
    for config in configs:
        if parts:
            parts.append(" or ")
        parts.extend(config if isinstance(config, list) else _truthy(config))
    return PythonExpression(parts)


def _all_true(*configs):
    parts = []
    for config in configs:
        if parts:
            parts.append(" and ")
        parts.extend(config if isinstance(config, list) else _truthy(config))
    return PythonExpression(parts)


def _not(config):
    return ["not ("] + _truthy(config) + [")"]


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

    It also hosts the companion testbed, which is the same idea with the
    opposite premise -- a human very much in the loop:

        ros2 launch burgerbot_bringup testbed.launch.py \\
            world_name:=social_room use_companion:=true

    That world has two walking actors in it, one who comes over and one who
    keeps leaving, so the whole behaviour is demoable from one command. Note
    that use_companion turns exploration off: both dispatch goals to Nav2 and
    the last one to send wins, so running them together is not a blend of two
    behaviours, it is two behaviours taking turns at random.
    """
    use_face = LaunchConfiguration("use_face")
    use_expressions = LaunchConfiguration("use_expressions")
    use_perception = LaunchConfiguration("use_perception")
    use_companion = LaunchConfiguration("use_companion")
    use_dialog = LaunchConfiguration("use_dialog")
    person_detector = LaunchConfiguration("person_detector")
    use_exploration = LaunchConfiguration("use_exploration")
    world_name = LaunchConfiguration("world_name")
    goal_timeout = LaunchConfiguration("goal_timeout")
    spawn_z = LaunchConfiguration("spawn_z")
    map_topic = LaunchConfiguration("map_topic")

    world_name_arg = DeclareLaunchArgument(
        "world_name",
        default_value="test_room",
        description="Gazebo world from burgerbot_description/worlds. "
        "'test_room' is a 4x4m room; 'tugbot_warehouse' is a ~25x40m "
        "warehouse whose models are fetched from Fuel on first launch; "
        "'social_room' is an 8x6m room with two walking actors, for the "
        "companion behaviour.",
    )
    # Scales with the space, not the robot: a warehouse frontier can be 20m+
    # away, far longer than the room-sized default allows, and a goal that
    # times out gets blacklisted as unreachable even when the robot was
    # driving toward it perfectly well.
    goal_timeout_arg = DeclareLaunchArgument(
        "goal_timeout",
        default_value="60.0",
        description="Exploration goal timeout in seconds. Raise it for "
        "large worlds; 240 is a reasonable starting point for the warehouse.",
    )

    spawn_z_arg = DeclareLaunchArgument(
        "spawn_z",
        default_value="0.0",
        description="Height (m) to spawn the robot at. Set to e.g. 0.15 to "
        "watch it drop and settle, which distinguishes a robot stuck in "
        "geometry from one that is simply idle.",
    )

    map_topic_arg = DeclareLaunchArgument(
        "map_topic",
        default_value="/map",
        description="Occupancy grid the frontier explorer reads. Set to "
        "/global_costmap/costmap to explore from Nav2's live scan-fused "
        "costmap when slam_toolbox's own map is not growing.",
    )

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
    use_companion_arg = DeclareLaunchArgument(
        "use_companion",
        default_value="false",
        description="Bring up people tracking and the companion behaviour. "
        "Implies use_perception (the detections come from there). Pair it "
        "with world_name:=social_room, which has people in it to react to.",
    )
    person_detector_arg = DeclareLaunchArgument(
        "person_detector",
        default_value="lite",
        description="'lite' re-labels the onboard detector's output at no "
        "extra cost; 'gpu' runs a pose model on a CUDA device; 'none' if a "
        "control PC is publishing detections instead.",
    )
    use_dialog_arg = DeclareLaunchArgument(
        "use_dialog",
        default_value="false",
        description="Bring up the conversation layer. Needs a model server "
        "reachable at its base_url (see docs/INSTALL.md); it starts and runs "
        "without one, and says so. Implies use_companion, which owns the Nav2 "
        "client the dialog layer hands resolved goals to.",
    )
    # Exploration and the companion both dispatch goals to Nav2, and the last
    # one to send wins. Running them together is not a compromise between two
    # behaviours, it is two behaviours taking turns at random -- so bringing up
    # the companion (or the dialog layer, which needs it) turns the explorer
    # off unless it is asked for explicitly.
    use_exploration_arg = DeclareLaunchArgument(
        "use_exploration",
        default_value="true",
        description="Run frontier exploration. Turn this off when running "
        "the companion, or the two fight over navigate_to_pose.",
    )

    # Launch arguments arrive as strings and users write them every which way
    # ("true", "True", "1"), so these normalise rather than compare literally.
    # A condition that quietly evaluates false because somebody capitalised an
    # argument leaves a node simply not running, with nothing in the log to
    # explain it -- the worst kind of launch bug to chase.
    companion_wanted = _any_true(use_companion, use_dialog)
    perception_wanted = _any_true(use_perception, use_companion, use_dialog)
    exploration_wanted = _all_true(use_exploration, _not(use_companion),
                                   _not(use_dialog))

    gazebo = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_description"),
            "launch",
            "gazebo.launch.py",
        ),
        launch_arguments={"world_name": world_name, "spawn_z": spawn_z}.items(),
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
        launch_arguments={
            "use_sim_time": "true",
            "goal_timeout": goal_timeout,
            "map_topic": map_topic,
        }.items(),
        condition=IfCondition(exploration_wanted),
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
        # use_companion implies perception: the person detections it needs
        # come from the object detector this brings up. Asking for the
        # companion and silently getting a robot that can see nobody would be
        # a confusing way to spend ten minutes.
        condition=IfCondition(perception_wanted),
    )

    people = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_perception"),
            "launch",
            "people.launch.py",
        ),
        launch_arguments={
            "use_sim_time": "true",
            "detector": person_detector,
            "tracker": "true",
        }.items(),
        condition=IfCondition(companion_wanted),
    )

    companion = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_companion"),
            "launch",
            "companion.launch.py",
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
        condition=IfCondition(companion_wanted),
    )

    dialog = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("burgerbot_dialog"),
            "launch",
            "dialog.launch.py",
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
        condition=IfCondition(use_dialog),
    )

    return LaunchDescription(
        [
            world_name_arg,
            goal_timeout_arg,
            spawn_z_arg,
            map_topic_arg,
            use_face_arg,
            use_expressions_arg,
            use_perception_arg,
            use_companion_arg,
            person_detector_arg,
            use_dialog_arg,
            use_exploration_arg,
            gazebo,
            controller,
            twist_mux,
            slam,
            navigation,
            exploration,
            face,
            expressions,
            perception,
            people,
            companion,
            dialog,
            rviz,
        ]
    )
