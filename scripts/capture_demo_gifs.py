#!/usr/bin/env python3
"""Record the running robot to animated GIFs, straight off the ROS topics.

Run this alongside a live `testbed.launch.py` session; it subscribes, samples
on a timer, and writes GIFs when it's done:

    ros2 launch burgerbot_bringup testbed.launch.py use_perception:=true &
    python3 scripts/capture_demo_gifs.py --duration 180 --out docs/media

Produces:
    mapping.gif      the occupancy grid filling in, robot pose + path drawn on
    camera_pov.gif   what the robot's camera sees while it explores

Subscribing to topics rather than screen-recording the Gazebo/RViz windows is
deliberate. Desktop capture (ffmpeg x11grab) records solid black under WSLg:
its compositor never populates the legacy X11 root-window buffer, so there is
nothing there to grab, GPU or software rendered. Reading /map and the camera
stream needs no display at all, works headless and in CI, and produces a
tighter picture of what the robot is actually doing than a window would.

Pillow is the only extra dependency (pip install pillow); it is a
capture-time tool, not something the robot needs at runtime.
"""

import argparse
import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

MAP_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
)

# Occupancy grid values are -1 unknown, 0..100 probability of being occupied.
UNKNOWN_RGB = (58, 58, 64)
FREE_RGB = (236, 238, 242)
OCCUPIED_RGB = (24, 24, 28)
ROBOT_RGB = (110, 200, 255)
PATH_RGB = (255, 150, 60)


class DemoCapture(Node):
    def __init__(self, scale: int, fps: float, map_topic: str = "/map"):
        super().__init__("demo_capture")
        self._bridge = CvBridge()
        self._scale = scale

        self._map: OccupancyGrid = None
        self._path: Path = None
        self._frame = None

        self.map_frames = []
        self.cam_frames = []
        self.trail = []

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(OccupancyGrid, map_topic, self._on_map, MAP_QOS)
        self.create_subscription(Path, "/plan", self._on_path, 10)
        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self._on_image, 1
        )
        self.create_timer(1.0 / fps, self._sample)

    # ---- inputs ----------------------------------------------------------

    def _on_map(self, msg):
        self._map = msg

    def _on_path(self, msg):
        self._path = msg

    def _on_image(self, msg):
        self._frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _robot_xy(self):
        try:
            tf = self._tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
            return tf.transform.translation.x, tf.transform.translation.y
        except Exception:
            return None

    # ---- sampling --------------------------------------------------------

    def _sample(self):
        from PIL import Image as PILImage

        pose = self._robot_xy()
        if pose is not None:
            self.trail.append(pose)

        if self._map is not None:
            self.map_frames.append(PILImage.fromarray(self._render_map(pose)))

        if self._frame is not None:
            img = PILImage.fromarray(self._frame)
            img.thumbnail((400, 400))
            self.cam_frames.append(img)

    def _render_map(self, pose):
        """Occupancy grid -> RGB, with the driven trail and current pose on top."""
        grid = self._map
        info = grid.info
        data = np.asarray(grid.data, dtype=np.int16).reshape(info.height, info.width)

        rgb = np.zeros((info.height, info.width, 3), dtype=np.uint8)
        rgb[data < 0] = UNKNOWN_RGB
        rgb[(data >= 0) & (data < 50)] = FREE_RGB
        rgb[data >= 50] = OCCUPIED_RGB

        def to_cell(wx, wy):
            gx = int((wx - info.origin.position.x) / info.resolution)
            gy = int((wy - info.origin.position.y) / info.resolution)
            if 0 <= gx < info.width and 0 <= gy < info.height:
                return gx, gy
            return None

        for wx, wy in self.trail:
            cell = to_cell(wx, wy)
            if cell:
                rgb[cell[1], cell[0]] = PATH_RGB

        if pose is not None:
            cell = to_cell(*pose)
            if cell:
                gx, gy = cell
                r = 2
                y0, y1 = max(0, gy - r), min(info.height, gy + r + 1)
                x0, x1 = max(0, gx - r), min(info.width, gx + r + 1)
                rgb[y0:y1, x0:x1] = ROBOT_RGB

        # OccupancyGrid rows run +Y up; images run +Y down.
        rgb = np.flipud(rgb)
        return np.kron(rgb, np.ones((self._scale, self._scale, 1), dtype=np.uint8))


def save_gif(frames, path, fps):
    if not frames:
        print(f"  no frames captured for {path}")
        return
    # Normalise to the first frame's size: the occupancy grid grows as SLAM
    # discovers space, so mid-run frames are physically larger than early ones
    # and GIF requires every frame to share one canvas size.
    target = frames[-1].size
    frames = [f if f.size == target else f.resize(target) for f in frames]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=max(20, round(1000.0 / fps)),
        loop=0,
        optimize=True,
    )
    print(f"  wrote {path}  ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=float, default=180.0, help="seconds to record")
    p.add_argument("--fps", type=float, default=2.0, help="sampling rate")
    p.add_argument("--playback-fps", type=float, default=12.0, help="GIF playback rate")
    p.add_argument("--scale", type=int, default=4, help="map pixel magnification")
    p.add_argument("--out", default="docs/media", help="output directory")
    p.add_argument(
        "--map-topic", default="/map",
        help="Occupancy grid to record. Use /global_costmap/costmap when "
             "exploring from the costmap rather than slam_toolbox's map.",
    )
    args = p.parse_args()

    import os

    os.makedirs(args.out, exist_ok=True)

    rclpy.init()
    node = DemoCapture(args.scale, args.fps, args.map_topic)
    print(f"recording {args.duration:.0f}s ...")

    import time

    end = time.time() + args.duration
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    print(f"captured {len(node.map_frames)} map / {len(node.cam_frames)} camera frames")
    save_gif(node.map_frames, os.path.join(args.out, "mapping.gif"), args.playback_fps)
    save_gif(node.cam_frames, os.path.join(args.out, "camera_pov.gif"), args.playback_fps)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
