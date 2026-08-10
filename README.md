# burg-bot

A TurtleBot3-Burger-style differential-drive robot running ROS 2 Jazzy, with a
full autonomy stack, an expressive animated face, and camera-based semantic
mapping.

The robot maps a space on its own — no joystick, no waypoints — while a 7"
display gives it a face that reacts to what it is actually doing, and a depth
camera labels the objects it finds and pins them to the map.

<p align="center">
  <img src="docs/media/mapping.gif" width="45%" alt="Autonomous frontier exploration building an occupancy grid">
  <img src="docs/media/expression_cycle.gif" width="45%" alt="Face expression cycle">
</p>

**Left:** autonomous frontier exploration in Gazebo. The occupancy grid fills
in from the lidar, the orange trail is the robot's actual driven path, the blue
square is the robot, and the dark disc is the obstacle it routes around.
**Right:** the procedural face cycling its expression library.

---

## Credit

The physical robot design and the foundational ROS 2 packages come from
**Antonio Brandi's** excellent course, *Self-Driving and ROS 2 — Learn by Doing!
Plan & Navigation*:

- Course: <https://www.udemy.com/course/self-driving-and-ros-2-learn-by-doing-plan-navigation/>
- Source: <https://github.com/AntoBrandi/Self-Driving-and-ROS-2-Learn-by-Doing-Plan-Navigation>

This repository builds on that foundation. The split is:

| Derived from the course | Original to this repository |
| --- | --- |
| `burgerbot_description` — URDF, meshes, Gazebo | `burgerbot_face` — procedural face renderer |
| `burgerbot_controller` — diff-drive, twist_mux | `burgerbot_expressions` — mood arbiter, gestures |
| `burgerbot_firmware` — Pico interface, IMU | `burgerbot_exploration` — frontier exploration |
| `burgerbot_localization` — EKF, AMCL | `burgerbot_perception` — detection, semantic map |
| `burgerbot_mapping` — slam_toolbox | `burgerbot_msgs` — expression/semantic interfaces |
| `burgerbot_navigation` — Nav2 configuration | Docker dev environment, Gazebo testbed |
| `burgerbot_utils` — lidar safety stop | `screen_link` / camera additions to the URDF |

Course exercise code that the production stack supersedes (hand-written A*,
Dijkstra, pure-pursuit and PD controllers, a from-scratch Kalman filter and
occupancy-grid mapper) has been removed — Nav2, `robot_localization` and
`slam_toolbox` do those jobs here. Those implementations are worth studying in
the original repository.

---

## Hardware

| Part | Notes |
| --- | --- |
| Raspberry Pi 4 (4 GB) | Main compute, Ubuntu 24.04 Server |
| Raspberry Pi Pico | Motor control and encoders over serial |
| DC motors + encoders | Differential drive |
| 2D lidar | SLAM and obstacle avoidance |
| MPU6050 IMU | Fused with wheel odometry via EKF |
| Official Pi 7" touch display | The face, over DSI ribbon |
| Intel RealSense D435 | Object detection and 3D projection |

The Pi 4 has no GPU or NPU, which shapes the perception design: inference is
throttled to ~1.5 Hz rather than run per frame, because objects worth labelling
on a map do not move fast.

---

## What it does

### Autonomous mapping

`burgerbot_exploration` finds the boundaries between known-free and unknown
space, scores each by size and distance, and sends the winner to Nav2. It stops
when no frontier remains — which is also what guarantees no unexplored pockets
are left behind.

Frontier detection and scoring are pure functions over an occupancy grid
(`frontier.py`), unit-tested on synthetic grids with no ROS graph running. The
node is a thin wrapper that reads the map, calls into that module, and
dispatches goals.

### Expressive face

Two light-blue ovals on black. No mouth — people read eyes far more strongly
than mouths, and a face with no mouth never lands in the uncanny valley of a
bad one. Every emotion comes out of eye geometry, timing, and motion.

<p align="center">
  <img src="docs/media/nervous_idle.gif" width="40%" alt="Nervous expression idle motion">
</p>

Expression is honest telemetry, not decoration. Each mood traces back to
something true about the robot: navigation status, lidar proximity, pose
covariance, battery level, touch. Sources bid with a priority and an expiry and
exactly one wins, the same shape as `twist_mux` arbitrating velocity commands —
without that, the face flickers between competing sources and reads as broken
rather than alive.

During a mapping sweep the face is neutral on a clean run and **nervous** —
narrowed, with fast sway and quick blinks — when something comes within half a
metre.

### Semantic mapping

<p align="center">
  <img src="docs/media/camera_pov.gif" width="55%" alt="Robot camera point of view while exploring">
</p>

A YOLOv8n detector runs on the colour stream; detections are back-projected to
3D using the aligned depth image and a pinhole model, transformed into the map
frame, and folded into a persistent de-duplicated object layer that saves as
`objects.yaml` alongside the SLAM map.

The object layer is deliberately parallel to the occupancy grid rather than
part of it — a probability grid is not a place to store "this cell is a chair."

---

## Verified results

From the Gazebo testbed (`test_room.world`, a 4×4 m room with one obstacle and
a chair), all committed under `burgerbot_ws/src/burgerbot_mapping/maps/test_room/`:

- **Mapping:** 81×81 cells at 5 cm resolution — a 4.05 m square, matching the
  room as designed.
- **Detection:** the chair recognised at 0.88–0.95 confidence.
- **Semantic map:** tracked to `(-1.42, -1.32, 0.42)` against a true placement
  of `(-1.2, -1.2)`, from 9 observations.

The ~0.2 m offset is expected: depth is sampled at the bounding-box centre, so
the point lands on whichever surface faces the robot, not the object's centroid.

---

## Quick start

Everything runs in a container — the workspace targets Ubuntu 24.04 / ROS 2
Jazzy, which need not match your host.

```bash
docker compose run --rm dev bash
```

Then inside:

```bash
cd /workspace/burgerbot_ws && colcon build --symlink-install && source install/setup.bash
```

Autonomous mapping demo (Gazebo + SLAM + Nav2 + exploration + face):

```bash
ros2 launch burgerbot_bringup testbed.launch.py
```

Add object detection and semantic labelling:

```bash
ros2 launch burgerbot_bringup testbed.launch.py use_perception:=true
```

On the real robot:

```bash
ros2 launch burgerbot_bringup real_robot.launch.py
```

Tune the face with no robot and no ROS graph attached:

```bash
ros2 run burgerbot_face demo_expressions
```

### Object detection model

The detector needs a TFLite model, which is generated rather than committed:

```bash
./scripts/export_detection_model.sh
```

This exports YOLOv8n through ONNX and `onnx2tf`. It ships the **float32**
variant. The int8 model is also produced but is not loadable: XNNPACK cannot
delegate its quantized Transpose ops, and with the delegate disabled the
reference sigmoid kernel rejects its output scale (it requires exactly 1/256).
No delegate setting satisfies both. At the pipeline's 1.5 Hz throttle, float32
fits a Pi 4 comfortably.

---

## Repository layout

```
burgerbot_ws/src/
  burgerbot_bringup       Top-level launches: real robot, simulation, testbed
  burgerbot_controller    Diff-drive control, twist_mux, noisy-odometry demo
  burgerbot_description   URDF, meshes, Gazebo worlds
  burgerbot_exploration   Frontier-based autonomous mapping
  burgerbot_expressions   Mood arbitration and expressive gestures
  burgerbot_face          Procedural face renderer for the 7" panel
  burgerbot_firmware      Pico serial interface, ros2_control hardware, IMU
  burgerbot_localization  EKF sensor fusion and AMCL
  burgerbot_mapping       slam_toolbox, map saving, saved maps
  burgerbot_msgs          Expression, gaze, touch and semantic interfaces
  burgerbot_navigation    Nav2 configuration and behaviour trees
  burgerbot_perception    Object detection, 3D projection, semantic map
  burgerbot_utils         Lidar-based safety stop
docker/                   Container image and entrypoint
scripts/                  Model export, demo capture, container helper
docs/media/               README animations
```

---

## Notes

The demo GIFs are captured by subscribing to ROS topics
(`scripts/capture_demo_gifs.py`), not by screen-recording Gazebo. Desktop
capture via `ffmpeg x11grab` produces solid black under WSLg — its compositor
never populates the legacy X11 root-window buffer, so there is nothing to grab
regardless of GPU or software rendering. Reading `/map` and the camera stream
needs no display at all and works headless.

`base_footprint_noisy` appearing as a detached frame in RViz is expected: it is
a deliberately noise-corrupted odometry pose from the course's
`noisy_controller`, feeding the EKF so the filtered estimate can be compared
against raw drift. It has no children, so RViz draws it as a lone triad that
wanders as noise accumulates.

## License

Apache-2.0, matching the upstream course code.
