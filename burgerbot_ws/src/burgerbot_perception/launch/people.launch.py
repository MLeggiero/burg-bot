import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _detector_is(detector, name):
    """Condition on a launch argument equalling a literal string."""
    return IfCondition(PythonExpression(["'", detector, "' == '", name, "'"]))


def generate_launch_description():
    """People detection and tracking, with the heavy half optionally offloaded.

    The robot always runs person_tracker: it needs the depth image and the TF
    tree, both of which live on the Pi, and neither of which is worth shipping
    over a network. What varies is where the detections come from.

    On the robot alone (no GPU anywhere):

        ros2 launch burgerbot_perception people.launch.py

    person_detector_lite re-labels the detections the onboard TFLite model is
    already producing, so this costs essentially nothing on top of
    perception.launch.py -- but it inherits that model's 1.5 Hz throttle and
    has no keypoints, so orientation can only be inferred from which way
    somebody is walking.

    With a control PC. On the robot:

        ros2 launch burgerbot_perception people.launch.py detector:=none

    and on the PC, in the same ROS_DOMAIN_ID:

        ros2 launch burgerbot_perception people.launch.py \\
            detector:=gpu tracker:=false use_identity:=true

    The split is deliberately asymmetric. Compressed colour goes up to the PC;
    small detection messages come back. Depth never leaves the robot -- it is
    the largest stream, the least compressible, and the only thing that needs
    it is already running next to it.
    """
    detector = LaunchConfiguration("detector")
    tracker = LaunchConfiguration("tracker")
    use_identity = LaunchConfiguration("use_identity")
    use_sim_time = LaunchConfiguration("use_sim_time")
    publish_compressed = LaunchConfiguration("publish_compressed")
    image_topic = LaunchConfiguration("image_topic")

    config = os.path.join(
        get_package_share_directory("burgerbot_perception"), "config", "people.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "detector",
                default_value="lite",
                description="Where person detections come from: 'lite' "
                "(re-label the Pi's existing detections, no extra cost, no "
                "keypoints), 'gpu' (pose model on a CUDA device), or 'none' "
                "(something else is publishing them -- use this on the robot "
                "when the PC is running the gpu detector).",
            ),
            DeclareLaunchArgument(
                "tracker",
                default_value="true",
                description="Run person_tracker. Set false on the control PC: "
                "tracking needs the depth image and the robot's TF tree, so it "
                "belongs on the robot even when detection does not.",
            ),
            DeclareLaunchArgument(
                "use_identity",
                default_value="false",
                description="Run face recognition (needs insightface and a "
                "GPU). Off by default: it is the one part of this stack that "
                "stores biometric data, so it should be a deliberate choice.",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/camera/camera/color/image_raw",
                description="Raw colour stream to compress and/or detect on.",
            ),
            DeclareLaunchArgument(
                "publish_compressed",
                default_value="false",
                description="Run image_transport republish to create the "
                "<image_topic>/compressed stream the offloaded detector reads. "
                "Needed in Gazebo, where the ros_gz bridge publishes raw "
                "images only. The real RealSense driver already provides it "
                "through image_transport's plugins, so this is a no-op there "
                "and off by default.",
            ),
            # Runs on the robot, not the PC: compressing where the raw frames
            # already are is the entire point. Republishing on the far side
            # would mean shipping the uncompressed stream first.
            Node(
                package="image_transport",
                executable="republish",
                name="color_compressor",
                arguments=["raw", "compressed"],
                remappings=[
                    ("in", image_topic),
                    ("out/compressed", [image_topic, "/compressed"]),
                ],
                output="screen",
                condition=IfCondition(publish_compressed),
            ),
            Node(
                package="burgerbot_perception",
                executable="person_detector_lite",
                name="person_detector_lite",
                output="screen",
                parameters=[config, {"use_sim_time": use_sim_time}],
                condition=_detector_is(detector, "lite"),
            ),
            Node(
                package="burgerbot_perception",
                executable="person_detector_gpu",
                name="person_detector_gpu",
                output="screen",
                parameters=[config, {"use_sim_time": use_sim_time}],
                condition=_detector_is(detector, "gpu"),
            ),
            Node(
                package="burgerbot_perception",
                executable="face_identity",
                name="face_identity",
                output="screen",
                parameters=[config, {"use_sim_time": use_sim_time}],
                condition=IfCondition(use_identity),
            ),
            Node(
                package="burgerbot_perception",
                executable="person_tracker",
                name="person_tracker",
                output="screen",
                parameters=[config, {"use_sim_time": use_sim_time}],
                condition=IfCondition(tracker),
            ),
        ]
    )
