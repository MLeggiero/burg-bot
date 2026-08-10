"""Frontier detection: pure grid math, no ROS dependency.

A frontier is the boundary between explored free space and the unknown --
finding and driving to one is the entire idea behind autonomous exploration.
Keeping the geometry here pure (plain arrays in, plain dataclasses out) means
the algorithm is unit-testable on small synthetic grids without a ROS graph,
map server, or SLAM running, the same pattern used for the expression
arbiter and gesture library.

Grid convention matches nav_msgs/OccupancyGrid: row-major, index = y*width+x,
cell values are -1 (unknown), 0-100 (occupancy probability, usually near-binary
for slam_toolbox's own output).
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

UNKNOWN = -1


@dataclass
class GridInfo:
    """Just enough of OccupancyGrid.info to do the geometry."""

    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float

    def to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Grid cell -> world (map frame) coordinates, cell centre."""
        return (
            self.origin_x + (gx + 0.5) * self.resolution,
            self.origin_y + (gy + 0.5) * self.resolution,
        )

    def to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """World coordinates -> grid cell. Not bounds-checked."""
        return (
            int((wx - self.origin_x) / self.resolution),
            int((wy - self.origin_y) / self.resolution),
        )


@dataclass
class Frontier:
    """One candidate exploration target."""

    #: World-frame goal point -- the cluster's actual nearest frontier cell to
    #: its own centroid, not the centroid itself (which can land off the
    #: frontier entirely for a concave or banana-shaped cluster).
    x: float
    y: float
    #: Cell count. Larger frontiers border more unexplored area, so they carry
    #: more information value per Nav2 goal spent reaching them.
    size: int
    #: Straight-line distance from the robot's current position, world units.
    distance: float

    def score(self, size_weight: float, distance_weight: float) -> float:
        """Higher is better. Rewards large frontiers, penalises distance.

        This is the standard simple utility used by frontier-exploration
        implementations (e.g. explore_lite): no information-theoretic
        modelling, just a linear trade-off with tunable weights. It is not
        the most sophisticated policy available, but it is transparent,
        cheap to compute every planning cycle, and easy to retune from the
        two weights alone if the robot is being too timid (raise
        distance_weight) or wasting time on tiny slivers (raise size_weight).
        """
        return size_weight * self.size - distance_weight * self.distance


def _grid_from_occupancy(
    data, width: int, height: int, occupied_threshold: int
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Split the flat grid into (unknown, occupied, free) boolean masks."""
    arr = np.asarray(data, dtype=np.int16).reshape(height, width)
    unknown = arr == UNKNOWN
    occupied = (~unknown) & (arr >= occupied_threshold)
    free = (~unknown) & (~occupied)
    return unknown, occupied, free


def _frontier_mask(unknown: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Free cells with at least one unknown cell among their 8 neighbours.

    8-connectivity (not 4) is what lets this catch a frontier that only
    touches unknown space diagonally, which happens constantly along a
    coarsely-resolved boundary.
    """
    height, width = free.shape
    touches_unknown = np.zeros_like(free, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(unknown)
            src_y0, src_y1 = max(0, -dy), height - max(0, dy)
            dst_y0, dst_y1 = max(0, dy), height - max(0, -dy)
            src_x0, src_x1 = max(0, -dx), width - max(0, dx)
            dst_x0, dst_x1 = max(0, dx), width - max(0, -dx)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = unknown[src_y0:src_y1, src_x0:src_x1]
            touches_unknown |= shifted
    return free & touches_unknown


def _connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """8-connected components of True cells, via iterative flood fill.

    Iterative (an explicit stack), not recursive -- a recursive flood fill on
    a grid the size of a real SLAM map blows Python's recursion limit almost
    immediately.
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[List[Tuple[int, int]]] = []

    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((x, y))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
        components.append(cells)
    return components


def find_frontiers(
    data,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    robot_x: float,
    robot_y: float,
    occupied_threshold: int = 65,
    min_cluster_size: int = 4,
) -> List[Frontier]:
    """Find candidate exploration frontiers in an occupancy grid.

    `min_cluster_size` filters single-cell and pair-cell frontiers, which are
    overwhelmingly SLAM noise along an already-explored wall rather than a
    genuine opening into unmapped space -- without this filter the explorer
    spends most of its time twitching at wall texture instead of covering new
    area.
    """
    if width <= 0 or height <= 0 or len(data) != width * height:
        return []

    info = GridInfo(width, height, resolution, origin_x, origin_y)
    unknown, occupied, free = _grid_from_occupancy(data, width, height, occupied_threshold)
    frontier_mask = _frontier_mask(unknown, free)

    frontiers = []
    for cells in _connected_components(frontier_mask):
        if len(cells) < min_cluster_size:
            continue

        cx = sum(x for x, _ in cells) / len(cells)
        cy = sum(y for _, y in cells) / len(cells)
        # Snap to the cluster's own nearest actual frontier cell rather than
        # goal-ing to the raw centroid, which can land in occupied or unknown
        # space for a concave cluster (an L-shaped or crescent frontier is
        # common around a doorway or the corner of a room).
        gx, gy = min(cells, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)

        wx, wy = info.to_world(gx, gy)
        distance = ((wx - robot_x) ** 2 + (wy - robot_y) ** 2) ** 0.5
        frontiers.append(Frontier(x=wx, y=wy, size=len(cells), distance=distance))

    return frontiers


def select_frontier(
    frontiers: List[Frontier],
    blacklist: List[Tuple[float, float]],
    blacklist_radius: float,
    size_weight: float = 1.0,
    distance_weight: float = 1.5,
) -> Optional[Frontier]:
    """Pick the best-scoring frontier that isn't near a blacklisted point.

    The blacklist is a radius match rather than an exact-point match because
    the "same" frontier's centroid shifts by a cell or two between scans as
    the map fills in -- an exact match would let the explorer retry a goal
    Nav2 just failed to reach, over and over, forever.
    """
    candidates = [
        f
        for f in frontiers
        if not any(
            (f.x - bx) ** 2 + (f.y - by) ** 2 <= blacklist_radius**2
            for bx, by in blacklist
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.score(size_weight, distance_weight))
