# burg-bot

An autonomous differential-drive robot with a face.

A TurtleBot3-Burger-style robot running the full ROS 2 autonomy stack — SLAM,
AMCL, Nav2 with custom planner and controller plugins — plus an expression
system that gives it a character rather than just a status LED.

The autonomy stack is derived from Antonio Brandi's excellent course
[Self-Driving and ROS 2 — Learn by Doing! Plan & Navigation][course]
([repo][upstream], Apache-2.0), renamed `bumperbot_*` → `burgerbot_*` and
adapted for this robot's hardware. The expression system is new.

[course]: https://www.udemy.com/course/self-driving-and-ros-2-learn-by-doing-plan-navigation/
[upstream]: https://github.com/AntoBrandi/Self-Driving-and-ROS-2-Learn-by-Doing-Plan-Navigation

---

## The expression system

Most robot faces are a screensaver: a loop of pre-rendered clips playing
independently of what the robot is doing. This one is wired to real state, and
it moves the body as well as the screen.

The design follows [Disney Research's expressive-robot work][disney]: an
animator authors motion freely, and a separate physics-aware layer reconciles
it with what the body can actually do. Neither layer compromises for the other.
Here that maps to:

| Disney | burg-bot |
|---|---|
| Animator authors expressive motion | Gesture library + parametric face poses |
| RL policy enforces physical constraints | `twist_mux` priority + safety-stop + lidar clearance gate |
| 4-DOF head, antennae | 800×480 procedural face |
| Expressive gait | Expressive base motion layered under navigation |

[disney]: https://spectrum.ieee.org/disney-robot

### The face

Two light-blue ovals on black. No mouth, no eyebrows — which is a constraint
worth keeping, because it forces every emotion through eye geometry, and eyes
are what people actually read.

Everything is drawn from primitives and driven by parameters, so any two
expressions blend continuously. A sprite-sheet face can only cut between fixed
states, and that is what makes most robot faces look like a slideshow.

Ten expressions (`neutral, happy, curious, focused, confused, startled, sad,
sleepy, determined, error`) are keyframes in
[`expressions.py`](burgerbot_ws/src/burgerbot_face/burgerbot_face/expressions.py) —
edit that one file to retune the robot's personality. On top of the pose sit
five composited animation layers, and these matter more than the keyframes do:

| Layer | What it does |
|---|---|
| **Blink** | Poisson-timed, ~4 s mean, asymmetric close/open. Never metronomic — regularity is what reads as a machine cycling |
| **Idle** | Two incommensurate sine waves, so the "breathing" never visibly loops |
| **Gaze** | Tracks a world-frame point through TF, plus microsaccades — real eyes are never still even when fixating |
| **Motion** | Inertia from `cmd_vel`: turning left slides the face right, braking squashes it |
| **Reaction** | Transient additive impulses (recoil, shake, squash) that always decay to nothing |

The best trick is **anticipation**: gaze leads a turn *before* the body
rotates, read off the curvature of the planned path. It fights the inertia
layer, which pulls the other way — eyes ahead, body trailing. That opposition
is what sells it, and it is nearly free because Nav2 already publishes the plan.

### What the face is reacting to

Nothing here is decorative. If the eyes look unsure it is because the pose
covariance genuinely grew.

| Signal | Reaction |
|---|---|
| `navigate_to_pose` status | `focused` while running, `happy` on success, `sad` on abort |
| `/scan` nearest obstacle | `startled` inside the danger radius; eyes track the nearest thing |
| `/cmd_vel` + `/plan` | squash-stretch, lean, gaze lead |
| `/amcl_pose` covariance | `confused`, scaled by how lost it actually is |
| `/battery_state` | `sleepy` when low, `error` when critical |
| safety-stop | `startled` |
| screen touch | `happy` + a bounce |

Sources bid with a **priority and a TTL**, and one wins. Without that
arbitration the face flickers between competing publishers, which reads as
broken rather than conflicted. It is the same shape as `twist_mux` arbitrating
velocity, applied to expression.

### Body gestures

`nod_yes`, `shake_no`, `wiggle`, `curious_tilt`, `celebrate`, `anticipate`,
`recoil` — short authored `Twist` sequences played through the `PlayGesture`
action.

Gestures are written as pure character motion with **no awareness of the
world**. Three separate things make them safe:

1. They publish to `gesture_vel`, the **lowest-priority** `twist_mux` input.
   Navigation and the operator's joystick both outrank them.
2. The `safety_stop` lock in `twist_mux` overrides every input.
3. `gesture_server` checks lidar clearance before and during, so it *declines*
   rather than merely being overridden — a gesture that gets silently
   suppressed mid-motion looks like a malfunction.

```bash
ros2 topic pub --once /gesture std_msgs/String "{data: wiggle}"
```

---

## Hardware

| Part | Notes |
|---|---|
| Raspberry Pi 4 | Ubuntu 24.04 Server arm64 + ROS 2 Jazzy |
| Raspberry Pi Pico (RP2040) | Motor control + encoders, USB CDC to the Pi |
| Official Pi 7" Touch Display | 800×480, DSI ribbon — the face |
| 2D lidar (RPLidar A1) | SLAM, AMCL, obstacle avoidance, gaze targets |
| MPU6050 IMU | Fused with wheel odometry in the EKF |
| DC gearmotors + quadrature encoders | Driven by an L298N-style H-bridge |

### ⚠️ Before first power-on

**The Pico is 3.3 V and its GPIOs are not 5 V tolerant.** The Arduino Uno this
design came from was. Many geared motors ship with 5 V hall encoders — those
need a level shifter or divider on A and B, or they will damage the Pico.

**Check your power budget.** The 7" panel draws roughly 500 mA on top of the Pi
and the motors. Brownouts under motor load present as random Pi reboots, which
is a genuinely confusing thing to debug if you have not ruled it out first.

---

## Setup

### 1. Development machine (Windows + WSL2) — containerized

The course targets Jazzy on Ubuntu 24.04; your WSL2 distro is likely a
different release with no ROS installed. Rather than adding a second WSL
distro just to get the right Ubuntu version, Jazzy runs in a **container** on
top of whatever WSL2 distro you already have (Docker Engine runs natively in
WSL2 — no Docker Desktop needed):

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER   # log out/in, or `newgrp docker`, for this to take effect
```

Then, from the repo root:

```bash
./scripts/dev_container.sh
```

First run builds the image (Nav2, SLAM, `ros2_control`, `twist_mux`, Gazebo
integration — resolved via `rosdep` straight from this workspace's own
`package.xml` files, so the image can't quietly drift out of sync with what
the packages actually declare). After that you're in a shell with the ROS 2
underlay sourced and `/workspace` bind-mounted to the repo:

```bash
colcon_build          # alias for: colcon build --symlink-install && source install/setup.bash
ros2 launch burgerbot_bringup simulated_robot.launch.py
```

Gazebo, RViz, and the windowed face demo all render through **WSLg** — the
container mounts `/mnt/wslg` (X11 + Wayland sockets) and `/usr/lib/wsl` (the
D3D12/dxcore GPU driver WSLg uses; there's no `/dev/dri` under WSLg, and
that's expected, not a misconfiguration) straight through at the same paths.
Sanity check GPU passthrough with `glxinfo | grep renderer` inside the
container — it should name a `D3D12` device, not `llvmpipe` (software
rendering; RViz and Gazebo will limp along at a few fps if you see this).

`docker-compose.yml` uses `network_mode: host`, which is deliberate: ROS 2's
DDS discovery is multicast-based, and under Docker's default bridge network
each `docker compose run` gets an isolated network namespace and can't see
another terminal's nodes. Host networking puts every container invocation on
one ROS graph, which is the point of running Gazebo in one terminal and Nav2
in another.

Prefer bare-metal WSL instead? A second `Ubuntu-24.04` distro
(`wsl --install -d Ubuntu-24.04`) plus `ros-jazzy-desktop` and the same apt
list as the Pi below works exactly the way it always has — the container is a
convenience, not a requirement. Either way, if you do work outside the
container: **clone into the WSL filesystem (`~/burg-bot`), not `/mnt/c`.**
Building a ROS workspace across the Windows filesystem boundary is
dramatically slower — the container's bind mount pays this cost too, so if
`colcon build` feels sluggish, it's worth cloning into `~/burg-bot` and
pointing the container's volume mount there instead.

### 2. Raspberry Pi 4

Flash **Ubuntu 24.04 Server arm64**, then:

```bash
sudo apt install -y ros-jazzy-ros-base ros-jazzy-nav2-bringup ros-jazzy-navigation2 ros-jazzy-slam-toolbox ros-jazzy-robot-localization ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-twist-mux ros-jazzy-rplidar-ros python3-pygame libserial-dev
```

Match the ROS distro on both sides. Cross-distro DDS is not guaranteed to
interoperate, and the failures are subtle.

### 3. The DSI panel

Add to `/boot/firmware/config.txt`:

```
dtoverlay=vc4-kms-v3d
dtoverlay=vc4-kms-dsi-7inch
```

Reboot, then confirm the panel is there:

```bash
modetest -M vc4 -c | grep -A2 connected
```

No desktop environment is needed. SDL renders straight to the panel through
KMS/DRM, which is why the robot can run Ubuntu Server and still have a face.

Give yourself DRM access:

```bash
sudo usermod -aG video,render,input $USER
```

### 4. Flash the Pico

Open
[`burgerbot_pico.ino`](burgerbot_ws/src/burgerbot_firmware/firmware/burgerbot_pico/burgerbot_pico.ino)
in the Arduino IDE with the **earlephilhower Raspberry Pi Pico/RP2040** core
installed. Select *Raspberry Pi Pico*, hold BOOTSEL while plugging in, Upload.

Pin assignments and the calibration constants are at the top of the sketch.

Pin the serial device so it does not move around:

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="2e8a", SYMLINK+="burgerbot_pico"' | sudo tee /etc/udev/rules.d/99-burgerbot.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 5. Build

Dev machine, in the container (`./scripts/dev_container.sh` drops you into a
shell with this already sourced):

```bash
./scripts/fetch_sim_assets.sh    # dev machine only; ~117MB of Gazebo world assets
colcon_build                     # alias: colcon build --symlink-install && source install/setup.bash
```

Pi, or bare-metal dev machine:

```bash
git clone <this repo> ~/burg-bot && cd ~/burg-bot
cd burgerbot_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

---

## Calibration — do this before trusting anything

Three numbers determine whether the robot goes where it thinks it does.
Everything downstream — odometry, the EKF, AMCL, the costmap — inherits any
error in them.

1. **`ENCODER_TICKS_PER_REV`** in the Pico sketch. Mark a wheel, turn it exactly
   ten revolutions by hand, read the reported position, divide by ten.
2. **`wheel_radius` and `wheel_separation`** in
   [`burgerbot_controllers.yaml`](burgerbot_ws/src/burgerbot_controller/config/burgerbot_controllers.yaml).
   Measure them; do not inherit the upstream `0.033` / `0.17`. A wheel constant
   5% out sends the robot in a slow curve when told to drive straight.
3. **The screen mount** in
   [`burgerbot_screen.xacro`](burgerbot_ws/src/burgerbot_description/urdf/burgerbot_screen.xacro).
   The defaults are a plausible front mount, not your robot. If the panel
   overhangs further than assumed, raise `robot_radius` in the Nav2 costmap
   configs to match — a footprint that excludes the display is exactly how a
   robot clips doorframes it thinks it is clearing.

Sanity checks:

```bash
ros2 control list_hardware_interfaces
ros2 topic echo /burgerbot_controller/odom --field pose.pose.position
```

Push the robot a measured 2 m and spin it 360°; odometry should roughly agree.

---

## Running

### Simulation

```bash
ros2 launch burgerbot_bringup simulated_robot.launch.py
```

Gazebo, Nav2, RViz, and the face in a desktop window. Add `use_slam:=true` to
map instead of localize. `use_face:=false` / `use_expressions:=false` to drop
either half.

### Real robot

```bash
ros2 launch burgerbot_bringup real_robot.launch.py use_slam:=true
```

Drive it around with the joystick to build a map, then save it:

```bash
ros2 run nav2_map_server map_saver_cli -f ~/burg-bot/maps/home
```

Then navigate on the saved map:

```bash
ros2 launch burgerbot_bringup real_robot.launch.py use_slam:=false
```

### Face only, no robot

The iteration loop for animation work — no ROS graph, no hardware:

```bash
ros2 run burgerbot_face demo_expressions
```

Number keys select expressions, arrows steer the gaze, `[` `]` and `,` `.`
simulate turning and accelerating so you can see the inertia layer respond.

Reference renders, headless:

```bash
python3 -m burgerbot_face.demo_expressions --dump-dir /tmp/faces
```

---

## Verification

```bash
colcon test --packages-select burgerbot_expressions && colcon test-result --verbose
```

The gesture tests assert that oscillating gestures integrate to zero net
displacement. That is not a formality — it caught a real bug where `wiggle`'s
decaying envelope left the robot yawed a few degrees every time it played,
which would have slowly corrupted odometry for reasons nothing in the
navigation stack could explain.

End-to-end, in order:

1. **Sim** — SLAM a world, save the map, relocalize, navigate around an
   obstacle, watch the face track it all.
2. **Robot** — same, on the real panel.
3. **Degradation** — kill `face_node` mid-navigation. Navigation must continue
   unaffected. The expression layer is strictly additive and never load-bearing.

---

## Layout

```
docker/Dockerfile           dev/sim container image (Jazzy + WSLg passthrough)
docker-compose.yml          container run config -- network_mode: host for DDS
scripts/dev_container.sh    ./scripts/dev_container.sh  -> shell in the container
scripts/fetch_sim_assets.sh restores the vendored AWS RoboMaker world assets

burgerbot_ws/src/
├── burgerbot_bringup/        top-level launch
├── burgerbot_description/    URDF, meshes, Gazebo worlds, screen_link
├── burgerbot_msgs/           interfaces, including the expression messages
├── burgerbot_firmware/       Pico sketch + ros2_control SystemInterface
├── burgerbot_controller/     diff_drive_controller, twist_mux, joy teleop
├── burgerbot_localization/   EKF (wheel odom + IMU) and AMCL
├── burgerbot_mapping/        slam_toolbox
├── burgerbot_planning/       A* / Dijkstra Nav2 global planner plugins
├── burgerbot_motion/         pure pursuit / PD Nav2 controller plugins
├── burgerbot_navigation/     Nav2 servers, costmaps, behavior trees
├── burgerbot_utils/          safety stop
├── burgerbot_face/           procedural face renderer          [new]
└── burgerbot_expressions/    mood arbiter + gesture server     [new]
```

Nav2 planner and controller plugins are C++ because `pluginlib` has no Python
loader. The expression system is Python, where iteration speed matters more
than microseconds.

### Performance

The face composites and draws in ~2.4 ms/frame at 2× supersampling on a desktop
x86 core. The Pi 4 is roughly 5–10× slower at this, so expect 12–24 ms — viable
at the default 45 fps, but measure. If it cannot hold frame rate, set
`supersample: 1` in
[`face.yaml`](burgerbot_ws/src/burgerbot_face/config/face.yaml); it is the
single biggest lever.

---

## License

Apache-2.0. Portions derived from [AntoBrandi/Self-Driving-and-ROS-2][upstream],
also Apache-2.0, with original authorship retained in each package manifest.
