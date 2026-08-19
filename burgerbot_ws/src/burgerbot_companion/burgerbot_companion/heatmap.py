"""Where people usually are. Pure logic, no ROS dependency.

The robot already builds a map of where the *walls* are. This is the other half
of what a companion needs to know about a space: which parts of it people
actually occupy. A kitchen and a corridor look identical on an occupancy grid
and could not be less alike socially, and knowing the difference is what turns
"drive somewhere unexplored" into "go and wait where somebody will turn up".

Deliberately not a fixed-size grid pinned to the occupancy map. That would have
to be resampled every time SLAM grew or shifted its map, and resampling a
diffuse accumulation like this loses exactly the long-tail counts it exists to
accumulate. A sparse dictionary keyed by world-frame cell index has no bounds
to outgrow, no origin to track, and no resampling step -- and a room's worth of
30 cm cells is a few thousand floats, which is nothing.

Old sightings fade. Without that the heatmap converges on wherever the robot
happened to spend its first afternoon and never updates, which is worse than
having none: a confidently wrong idea of where people are sends the robot to
stand hopefully in an empty room.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Cell = Tuple[int, int]


def best_target(
    hotspots: Sequence[Tuple[float, float, float]],
    robot_x: float,
    robot_y: float,
    min_distance: float = 1.5,
    exclude: Sequence[Tuple[float, float]] = (),
    exclude_radius: float = 1.0,
    distance_scale: float = 8.0,
) -> Optional[Tuple[float, float]]:
    """Pick somewhere to go and wait, from a list of (x, y, heat).

    A free function rather than a method because the node that accumulates the
    heatmap and the node that decides where to drive are not the same one --
    the behaviour receives hotspots over a topic and has its own opinion about
    which of them it has already tried recently.

    Scored by heat discounted for distance, rather than simply hottest:
    crossing an entire building for a marginally warmer cell is a bad trade,
    and a robot that does it spends its whole day in transit. The minimum
    distance stops it "travelling" to the cell it is already standing in,
    which would otherwise always win.
    """
    best, best_score = None, 0.0
    for x, y, value in hotspots:
        distance = math.hypot(x - robot_x, y - robot_y)
        if distance < min_distance:
            continue
        if any(math.hypot(x - ex, y - ey) <= exclude_radius for ex, ey in exclude):
            continue
        score = value / (1.0 + distance / max(distance_scale, 1e-3))
        if score > best_score:
            best, best_score = (x, y), score
    return best


@dataclass
class PersonHeatmap:
    """Accumulated person-seconds per cell of the map frame."""

    #: Metres per cell. Coarser than the 5 cm occupancy grid on purpose -- this
    #: is answering "which end of the room", not "which square centimetre", and
    #: a fine grid spreads the same evidence over so many cells that nothing
    #: ever accumulates enough to stand out.
    resolution: float = 0.3
    #: Seconds for a cell's weight to halve. Half an hour makes the map track
    #: the current day rather than the whole history; a much longer value makes
    #: it a record of where people used to be.
    half_life: float = 1800.0
    #: Cells below this are dropped on decay, so the dictionary does not fill
    #: with the ghosts of every square metre anybody ever crossed.
    prune_below: float = 0.02

    _cells: Dict[Cell, float] = field(default_factory=dict)
    _last_decay: Optional[float] = None

    # ---- accumulation ---------------------------------------------------

    def cell_of(self, x: float, y: float) -> Cell:
        return (math.floor(x / self.resolution), math.floor(y / self.resolution))

    def center_of(self, cell: Cell) -> Tuple[float, float]:
        return (
            (cell[0] + 0.5) * self.resolution,
            (cell[1] + 0.5) * self.resolution,
        )

    def observe(self, x: float, y: float, dt: float, weight: float = 1.0) -> None:
        """Record somebody standing at (x, y) for dt seconds.

        Weighted by duration rather than counted per detection, so the heatmap
        does not simply record where the detector ran fastest. Somebody
        standing still for a minute is a stronger signal about a place than
        somebody walking through it in two seconds, and a per-frame count says
        the opposite whenever the frame rate changes.
        """
        if dt <= 0.0 or weight <= 0.0:
            return
        cell = self.cell_of(x, y)
        self._cells[cell] = self._cells.get(cell, 0.0) + dt * weight

    def decay_to(self, t: float) -> None:
        """Fade everything by however long has passed since the last call."""
        if self._last_decay is None:
            self._last_decay = t
            return
        elapsed = t - self._last_decay
        if elapsed <= 0.0:
            return
        self._last_decay = t

        if self.half_life <= 0.0:
            return
        factor = 0.5 ** (elapsed / self.half_life)
        self._cells = {
            cell: value * factor
            for cell, value in self._cells.items()
            if value * factor >= self.prune_below
        }

    # ---- reading --------------------------------------------------------

    def value(self, x: float, y: float) -> float:
        return self._cells.get(self.cell_of(x, y), 0.0)

    def peak(self) -> float:
        return max(self._cells.values(), default=0.0)

    def hotspots(self, min_fraction: float = 0.35) -> List[Tuple[float, float, float]]:
        """(x, y, value) for cells at least min_fraction of the peak, hottest first.

        Relative to the peak rather than an absolute threshold, because the
        units here are arbitrary -- person-seconds accumulated at whatever rate
        the detector happened to run -- so any absolute number would need
        retuning the moment the pipeline got faster.
        """
        peak = self.peak()
        if peak <= 0.0:
            return []
        floor = peak * min_fraction
        hot = [
            (*self.center_of(cell), value)
            for cell, value in self._cells.items()
            if value >= floor
        ]
        hot.sort(key=lambda h: h[2], reverse=True)
        return hot

    def best_target(
        self,
        robot_x: float,
        robot_y: float,
        min_distance: float = 1.5,
        exclude: Sequence[Tuple[float, float]] = (),
        exclude_radius: float = 1.0,
        distance_scale: float = 8.0,
        min_fraction: float = 0.35,
    ) -> Optional[Tuple[float, float]]:
        """Where to go and wait for somebody, or None if nowhere is known."""
        return best_target(
            self.hotspots(min_fraction), robot_x, robot_y,
            min_distance, exclude, exclude_radius, distance_scale,
        )

    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """(min_x, min_y, max_x, max_y) covering every occupied cell."""
        if not self._cells:
            return None
        xs = [c[0] for c in self._cells]
        ys = [c[1] for c in self._cells]
        return (
            min(xs) * self.resolution,
            min(ys) * self.resolution,
            (max(xs) + 1) * self.resolution,
            (max(ys) + 1) * self.resolution,
        )

    def cells(self) -> Dict[Cell, float]:
        return dict(self._cells)

    # ---- persistence ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "resolution": self.resolution,
            "half_life": self.half_life,
            # Flat triples rather than nested maps: YAML has no tuple key, and
            # a list of three numbers per cell round-trips through every
            # implementation without needing a custom representer.
            "cells": [[c[0], c[1], round(v, 4)] for c, v in sorted(self._cells.items())],
        }

    def load_dict(self, data: dict) -> int:
        data = data or {}
        saved_resolution = float(data.get("resolution", self.resolution))
        cells = data.get("cells", [])

        if abs(saved_resolution - self.resolution) > 1e-9:
            # Cell indices mean nothing without the resolution that produced
            # them. Rebuilding through world coordinates is approximate --
            # several old cells can land in one new one -- but it is right in
            # the way that matters: the heat stays where it was in the room.
            self._cells = {}
            for entry in cells:
                x = (entry[0] + 0.5) * saved_resolution
                y = (entry[1] + 0.5) * saved_resolution
                cell = self.cell_of(x, y)
                self._cells[cell] = self._cells.get(cell, 0.0) + float(entry[2])
        else:
            self._cells = {(int(e[0]), int(e[1])): float(e[2]) for e in cells}

        self._last_decay = None
        return len(self._cells)
