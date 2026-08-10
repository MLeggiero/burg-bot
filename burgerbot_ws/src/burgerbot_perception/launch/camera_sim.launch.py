from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Stand-in for camera.launch.py when running in Gazebo.

    camera.launch.py talks to the real D435 driver, which has no simulated
    equivalent -- there's no hardware for it to open in a container. This
    bridges the rgbd_camera sensor added to camera_color_optical_frame in
    burgerbot_camera.xacro (see that file's comment) onto the exact topic
    names object_detector.py and object_projector.py already expect, so the
    rest of the perception pipeline runs unmodified in sim.

    The gz-side topic names (camera/image, camera/depth_image,
    camera/camera_info) come from the <topic>camera</topic> set on that
    sensor -- keep the two in sync if either changes.
    """
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="camera_sim_bridge",
        arguments=[
            "camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        remappings=[
            ("camera/image", "/camera/camera/color/image_raw"),
            ("camera/depth_image", "/camera/camera/aligned_depth_to_color/image_raw"),
            ("camera/camera_info", "/camera/camera/color/camera_info"),
        ],
        output="screen",
    )

    return LaunchDescription([bridge])
