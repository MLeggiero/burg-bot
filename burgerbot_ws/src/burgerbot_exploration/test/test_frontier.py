"""Frontier detection on small synthetic grids. Pure logic, no ROS graph."""

import pytest

from burgerbot_exploration.frontier import GridInfo, find_frontiers, select_frontier

CELL = {"U": -1, "F": 0, "O": 100}


def _grid(rows):
    """Build (data, width, height) from a list of same-length strings.

    Each character is one cell: U=unknown, F=free, O=occupied. Row 0 is the
    grid's y=0 row, matching OccupancyGrid's row-major bottom-up convention
    closely enough for these tests (only relative geometry matters here, not
    which physical edge is "up").
    """
    width = len(rows[0])
    assert all(len(r) == width for r in rows), "ragged test grid"
    height = len(rows)
    data = [CELL[c] for row in rows for c in row]
    return data, width, height


def _find(rows, robot=(0.0, 0.0), **kwargs):
    data, w, h = _grid(rows)
    return find_frontiers(
        data, w, h, resolution=1.0, origin_x=0.0, origin_y=0.0,
        robot_x=robot[0], robot_y=robot[1], **kwargs
    )


def test_no_frontier_in_fully_unknown_grid():
    # Nothing has been explored yet -- no free cell exists to stand a
    # frontier on, so there is nothing to detect yet either.
    assert _find(["UUUU", "UUUU", "UUUU", "UUUU"]) == []


def test_no_frontier_in_fully_explored_grid():
    assert _find(["FFFF", "FFFF", "FFFF", "FFFF"]) == []


def test_finds_frontier_at_free_unknown_boundary():
    frontiers = _find(
        [
            "UUUUU",
            "UFFFU",
            "UFFFU",
            "UFFFU",
            "UUUUU",
        ],
        min_cluster_size=1,
    )
    assert len(frontiers) >= 1
    # Every returned goal point must itself be inside the free region, or
    # Nav2 would be sent straight at a wall.
    for f in frontiers:
        gx, gy = int(f.x), int(f.y)
        assert 1 <= gx <= 3 and 1 <= gy <= 3


def test_occupied_cells_are_never_frontiers():
    # An occupied wall sits directly against unknown space, but a wall is not
    # somewhere the robot can stand, let alone something worth "exploring".
    frontiers = _find(
        [
            "UUUUU",
            "UFFFU",
            "UFOOU",
            "UUUUU",
        ],
        min_cluster_size=1,
    )
    for f in frontiers:
        assert (int(f.x), int(f.y)) != (2, 2)
        assert (int(f.x), int(f.y)) != (3, 2)


def test_min_cluster_size_filters_noise():
    rows = [
        "UUUUU",
        "UFFFU",
        "UFFFU",
        "UUUUU",
    ]
    permissive = _find(rows, min_cluster_size=1)
    strict = _find(rows, min_cluster_size=1000)
    assert len(permissive) > 0
    assert len(strict) == 0


def test_goal_point_lands_on_an_actual_frontier_cell():
    # An L-shaped free region: the centroid of the frontier cluster falls in
    # the concave corner, which is unknown space, not free space. The goal
    # must snap to a real cell in the cluster instead of using the raw mean.
    rows = [
        "UUUUUU",
        "UFFUUU",
        "UFFUUU",
        "UFFFFU",
        "UFFFFU",
        "UUUUUU",
    ]
    frontiers = _find(rows, min_cluster_size=1)
    data, w, h = _grid(rows)
    free_cells = {(x, y) for y in range(h) for x in range(w) if rows[y][x] == "F"}
    for f in frontiers:
        assert (int(f.x), int(f.y)) in free_cells


def test_distance_is_measured_from_robot_position():
    rows = [
        "UUUUUUUUU",
        "UFFUUUFFU",
        "UFFUUUFFU",
        "UUUUUUUUU",
    ]
    near = _find(rows, robot=(1.5, 1.5), min_cluster_size=1)
    far = _find(rows, robot=(20.0, 20.0), min_cluster_size=1)
    # Same frontiers exist either way; only the reported distance changes.
    assert len(near) == len(far)
    assert sum(f.distance for f in near) < sum(f.distance for f in far)


def test_select_frontier_prefers_higher_score():
    from burgerbot_exploration.frontier import Frontier

    big_far = Frontier(x=10.0, y=0.0, size=50, distance=10.0)
    small_near = Frontier(x=1.0, y=0.0, size=4, distance=1.0)
    chosen = select_frontier(
        [big_far, small_near], blacklist=[], blacklist_radius=0.5,
        size_weight=1.0, distance_weight=1.0,
    )
    # 50*1 - 10*1 = 40  vs  4*1 - 1*1 = 3 -- big_far wins clearly.
    assert chosen is big_far


def test_select_frontier_respects_blacklist_radius():
    from burgerbot_exploration.frontier import Frontier

    only_option = Frontier(x=5.0, y=5.0, size=10, distance=1.0)
    chosen = select_frontier(
        [only_option], blacklist=[(5.1, 5.1)], blacklist_radius=1.0,
    )
    assert chosen is None


def test_select_frontier_returns_none_for_empty_input():
    assert select_frontier([], blacklist=[], blacklist_radius=1.0) is None


def test_grid_info_world_round_trip():
    info = GridInfo(width=10, height=10, resolution=0.05, origin_x=-1.0, origin_y=-2.0)
    for gx, gy in [(0, 0), (5, 5), (9, 9)]:
        wx, wy = info.to_world(gx, gy)
        bx, by = info.to_grid(wx, wy)
        assert bx == gx and by == gy


def test_mismatched_data_length_returns_empty_rather_than_raising():
    # A malformed or mid-update grid message should degrade to "nothing
    # found" rather than crash the exploration node.
    assert find_frontiers([0, 0, 0], width=10, height=10,
                           resolution=1.0, origin_x=0.0, origin_y=0.0,
                           robot_x=0.0, robot_y=0.0) == []
