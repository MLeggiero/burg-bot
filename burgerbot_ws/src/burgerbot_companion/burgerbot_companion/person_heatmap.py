"""Accumulates where people are seen, and publishes it as a map layer.

Parallel to burgerbot_perception's semantic_map in both purpose and shape: a
layer alongside the occupancy grid rather than inside it, persisted next to the
map it belongs with. A probability grid is not a place to store "people stand
here", any more than it was a place to store "this cell is a chair".

What it buys is the difference between exploring and looking for somebody. A
kitchen and a corridor are the same thing on an occupancy grid and could not be
less alike socially, and once the robot knows which is which, "nobody is around"
stops meaning "wander" and starts meaning "go and wait where people turn up".
"""

import os
from typing import Optional

import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_srvs.srv import Trigger

from burgerbot_msgs.msg import PersonArray

from .heatmap import PersonHeatmap


class PersonHeatmapNode(Node):
    def __init__(self):
        super().__init__("person_heatmap")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("people_topic", "/perception/people")
        self.declare_parameter("resolution", 0.3)
        self.declare_parameter("half_life", 1800.0)
        self.declare_parameter("prune_below", 0.02)
        # Fraction of the busiest cell a cell must reach to be published as a
        # hotspot. Relative rather than absolute because the units here are
        # arbitrary -- person-seconds at whatever rate the detector ran -- so
        # an absolute threshold would need retuning every time the pipeline
        # got faster.
        self.declare_parameter("hotspot_fraction", 0.35)
        self.declare_parameter("max_hotspots", 8)
        self.declare_parameter("publish_rate", 1.0)

        self.declare_parameter(
            "state_directory", os.path.join(os.path.expanduser("~"), ".burgerbot")
        )
        self.declare_parameter("heatmap_file", "people_heatmap.yaml")
        self.declare_parameter("autoload", True)
        # Periodic, because the interesting failure is not a clean shutdown --
        # it is the robot's battery going flat, which is exactly when nothing
        # gets a chance to run a save handler.
        self.declare_parameter("autosave_period", 120.0)

        self._map_frame = self.get_parameter("map_frame").value
        self._heatmap = PersonHeatmap(
            resolution=float(self.get_parameter("resolution").value),
            half_life=float(self.get_parameter("half_life").value),
            prune_below=float(self.get_parameter("prune_below").value),
        )
        self._last_people_stamp: Optional[float] = None

        if bool(self.get_parameter("autoload").value):
            loaded = self._load()
            if loaded:
                self.get_logger().info(f"loaded {loaded} heatmap cell(s)")

        self.create_subscription(
            PersonArray, self.get_parameter("people_topic").value, self._on_people, 10
        )
        self._grid_pub = self.create_publisher(OccupancyGrid, "/companion/heatmap", 1)
        self._hotspot_pub = self.create_publisher(PoseArray, "/companion/hotspots", 1)

        rate = float(self.get_parameter("publish_rate").value)
        self.create_timer(1.0 / max(rate, 0.05), self._publish)

        period = float(self.get_parameter("autosave_period").value)
        if period > 0.0:
            self.create_timer(period, self._autosave)

        self.create_service(Trigger, "~/save", self._on_save)
        self.create_service(Trigger, "~/load", self._on_load)
        self.get_logger().info("person_heatmap up")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- accumulation ----------------------------------------------------

    def _on_people(self, msg: PersonArray) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_people_stamp is None:
            self._last_people_stamp = stamp
            return

        dt = stamp - self._last_people_stamp
        self._last_people_stamp = stamp
        # A gap this long means the tracker restarted or the clock jumped.
        # Crediting it as time somebody stood still would dump minutes of
        # weight into one cell from a single message.
        if dt <= 0.0 or dt > 5.0:
            return

        for person in msg.people:
            if not person.visible:
                # Coasting tracks are a guess about where somebody is. Feeding
                # guesses into a long-lived accumulation bakes them in.
                continue
            self._heatmap.observe(person.pose.position.x, person.pose.position.y, dt)

    # ---- output ----------------------------------------------------------

    def _publish(self) -> None:
        self._heatmap.decay_to(self._now())
        stamp = self.get_clock().now().to_msg()

        hotspots = self._heatmap.hotspots(
            float(self.get_parameter("hotspot_fraction").value)
        )[: int(self.get_parameter("max_hotspots").value)]

        poses = PoseArray()
        poses.header.stamp = stamp
        poses.header.frame_id = self._map_frame
        for x, y, value in hotspots:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            # Heat has nowhere natural to live in a Pose, and inventing a
            # message for one float is not worth it -- so it rides in z, which
            # is otherwise meaningless for a point on the floor. Documented
            # here because it is the sort of thing that is baffling later.
            pose.position.z = float(value)
            pose.orientation.w = 1.0
            poses.poses.append(pose)
        self._hotspot_pub.publish(poses)

        grid = self._to_grid(stamp)
        if grid is not None:
            self._grid_pub.publish(grid)

    def _to_grid(self, stamp) -> Optional[OccupancyGrid]:
        """Render the sparse cells as an OccupancyGrid for RViz.

        Sized to whatever is currently occupied rather than to the SLAM map:
        this layer has no fixed extent, and matching the map's would mean
        publishing a mostly-empty grid the size of the building every second.
        """
        bounds = self._heatmap.bounds()
        peak = self._heatmap.peak()
        if bounds is None or peak <= 0.0:
            return None

        min_x, min_y, max_x, max_y = bounds
        resolution = self._heatmap.resolution
        width = max(1, int(round((max_x - min_x) / resolution)))
        height = max(1, int(round((max_y - min_y) / resolution)))

        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = self._map_frame
        grid.info.resolution = resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = min_x
        grid.info.origin.position.y = min_y
        grid.info.origin.orientation.w = 1.0

        # -1 is "unknown", which renders as transparent rather than as cold --
        # so the layer shows where people have been without painting a solid
        # rectangle over the map everywhere they have not.
        data = [-1] * (width * height)
        origin_cell = self._heatmap.cell_of(min_x + 0.5 * resolution,
                                            min_y + 0.5 * resolution)
        for (cx, cy), value in self._heatmap.cells().items():
            ix = cx - origin_cell[0]
            iy = cy - origin_cell[1]
            if 0 <= ix < width and 0 <= iy < height:
                data[iy * width + ix] = int(max(1, min(100, round(100.0 * value / peak))))
        grid.data = data
        return grid

    # ---- persistence -----------------------------------------------------

    def _path(self) -> str:
        return os.path.join(
            self.get_parameter("state_directory").value,
            self.get_parameter("heatmap_file").value,
        )

    def _save(self) -> str:
        path = self._path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self._heatmap.to_dict(), f, sort_keys=False)
        return path

    def _load(self) -> int:
        try:
            with open(self._path()) as f:
                data = yaml.safe_load(f) or {}
        except OSError:
            return 0  # No heatmap yet is the normal case on a fresh robot.
        return self._heatmap.load_dict(data)

    def _autosave(self) -> None:
        if not self._heatmap.cells():
            return
        try:
            self._save()
        except OSError as exc:
            self.get_logger().warn(f"heatmap autosave failed: {exc}")

    def _on_save(self, request, response):
        try:
            path = self._save()
            response.success = True
            response.message = f"saved {len(self._heatmap.cells())} cell(s) to {path}"
        except OSError as exc:
            response.success = False
            response.message = str(exc)
        self.get_logger().info(response.message)
        return response

    def _on_load(self, request, response):
        count = self._load()
        response.success = True
        response.message = f"loaded {count} cell(s) from {self._path()}"
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PersonHeatmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
