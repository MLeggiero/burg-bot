# Installation

Three machines can be involved, and they are set up in genuinely different
ways. Work out which ones you need first â€” most people only need the first.

| Machine | What it is for | Needed when |
| --- | --- | --- |
| **Development box** | Gazebo, RViz, `colcon build`, editing | Always. Everything runs here in a container. |
| **The robot** | Raspberry Pi 4 driving real hardware | You have the physical robot. Native install, no container. |
| **Control PC** | GPU work offloaded off the Pi | You want fast person tracking, face recognition, or the dialog layer. |

The three sections below are self-contained. You should never need to read two
of them to set up one machine.

---

## A. Development box (Windows 11 + WSL2 + Docker)

Everything runs in a container targeting Ubuntu 24.04 / ROS 2 Jazzy, so your
host does not have to match. This is the only setup path that is regularly
exercised.

### 1. WSL2

From an elevated PowerShell:

```powershell
wsl --install -d Ubuntu
```

Reboot when it asks. Everything after this runs **inside** the Ubuntu shell,
not in PowerShell.

### 2. Docker Engine, inside WSL â€” not Docker Desktop

Install Docker Engine directly in the distro. Docker Desktop works, but it adds
a second virtualisation layer between the container and WSLg's GPU device, and
GPU passthrough for Gazebo is fiddly enough without it. `scripts/dev_container.sh`
tells you the same thing when it cannot find Docker.

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

Log out and back in â€” group membership is only picked up at login, so `docker`
will keep refusing permission until you do.

### 3. Clone and build

```bash
git clone <your-fork-url> burg-bot && cd burg-bot
./scripts/dev_container.sh
```

The first run builds the image, which takes a while: it runs `rosdep install`
against every `package.xml` in the workspace, so Nav2, slam_toolbox,
ros2_control, twist_mux and the Gazebo bridges all come from that one step
rather than a hand-maintained list that drifts.

Then inside the container:

```bash
colcon_build
```

That alias is baked into the image and expands to `cd /workspace/burgerbot_ws
&& colcon build --symlink-install && source install/setup.bash`.

### 4. Check GPU passthrough before blaming anything else

```bash
glxinfo | grep -i "OpenGL renderer"
```

It must name your actual GPU through D3D12. If it says `llvmpipe`, you are on
software rendering and Gazebo will run at a few frames a second â€” the
`MESA_LOADER_DRIVER_OVERRIDE` and `GALLIUM_DRIVER` settings in
`docker-compose.yml` are what fix that, and they explain why in a comment.

`xeyes` is a faster check that the display socket works at all.

### 5. Run something

```bash
ros2 launch burgerbot_bringup testbed.launch.py
```

First launch pauses while Gazebo downloads models from Fuel. They cache in a
named Docker volume, so it only happens once â€” that volume exists precisely
because re-downloading on every `docker compose run` was slow enough to make
slam_toolbox's startup time out intermittently.

### 6. Object detection model (optional)

The detector needs a TFLite model, generated rather than committed:

```bash
./scripts/export_detection_model.sh
colcon build --packages-select burgerbot_perception
```

The second command is not optional. The model lands in a directory that
`setup.py` globs at build time, and an already-built package will not pick up a
file that did not exist when it was built.

Then:

```bash
ros2 launch burgerbot_bringup testbed.launch.py use_perception:=true
```

---

## B. The robot (Raspberry Pi 4, Ubuntu 24.04 Server)

Native, not containerised. `docker/Dockerfile`'s header explains the reasoning
at length: device passthrough for the DSI panel, the serial port and the I2C
bus, all inside a privileged container, buys nothing for a single-purpose robot
brain and adds a lot of ways to fail.

### 1. ROS 2 Jazzy

Install `ros-jazzy-ros-base` following the official Jazzy instructions for
Ubuntu 24.04 â€” `ros-base`, not `desktop`, since nothing on the robot renders
RViz. Then:

```bash
sudo apt install python3-rosdep python3-colcon-common-extensions
sudo rosdep init && rosdep update
```

### 2. Workspace dependencies

```bash
cd ~/burg-bot/burgerbot_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

That resolves everything the packages declare, including the ones easy to miss
by hand: `python3-pygame` (the face), `libserial-dev` and `python3-serial` (the
Pico link), `python3-smbus` (the IMU), and the `rplidar_ros` and
`realsense2_camera` drivers.

### 3. Serial devices â€” do this before the first launch

Two launch files name device paths that **do not exist until you install a udev
rule**:

- `/dev/burgerbot_pico`, from `burgerbot_description/urdf/burgerbot_ros2_control.xacro`
- `/dev/rplidar`, from `burgerbot_bringup/config/rplidar_a1.yaml`

```bash
sudo cp scripts/udev/99-burgerbot.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER
ls -l /dev/burgerbot_pico /dev/rplidar
```

Log out and back in after the `usermod`. Read the comments in that rules file
before trusting it: vendor and product IDs differ between board revisions and
firmware stacks, and a rule that matches nothing fails silently â€” you get no
symlink and a launch error pointing at a missing device with no hint as to why.
`udevadm info -a -n /dev/ttyACM0` shows your own IDs.

The kernel names are assigned in probe order, which is the actual reason for
all this: plug the lidar in before the Pico and `/dev/ttyACM0` swaps between
them. Neither failure announces itself.

### 4. IMU over I2C

`mpu6050_driver.py` opens bus 1 and expects the device at `0x68`.

```bash
sudo apt install i2c-tools
sudo usermod -aG i2c $USER
i2cdetect -y 1
```

You want `68` in the grid. Nothing there means the bus is disabled â€” add
`dtparam=i2c_arm=on` to `/boot/firmware/config.txt` and reboot â€” or the wiring
is wrong.

### 5. The face panel

`face.launch.py` renders through SDL's `kmsdrm` driver, straight to the DSI
panel with no X and no Wayland session.

```bash
sudo usermod -aG video,render $USER
```

**A black screen here is almost always a desktop session holding DRM master.**
Ubuntu Server should not have one; if you installed a desktop, disable the
display manager with `sudo systemctl disable --now gdm3`, or run the face
windowed instead.

Test it alone before involving the rest of the stack:

```bash
ros2 launch burgerbot_face face.launch.py
```

### 6. Bring the robot up

```bash
ros2 launch burgerbot_bringup real_robot.launch.py
```

That starts the hardware interface, the lidar, `ros2_control`, the IMU,
joystick teleop, AMCL, Nav2, the face and the expression stack. Add
`use_slam:=true` to build a map instead of localising against a saved one, or
`use_face:=false` for a headless run.

---

## C. Control PC (RTX 5070 Ti)

Only needed for the offloaded work: fast person detection, face recognition and
the dialog layer. Everything still works without it, just slower and with fewer
features.

### 1. ROS 2 Jazzy

Install `ros-jazzy-desktop`, set the same `ROS_DOMAIN_ID` as the robot, and
build the workspace the same way as section B.

### 2. CUDA and PyTorch â€” the version rule that bites

A 5070 Ti is Blackwell, compute capability `sm_120`. It needs **CUDA 12.8 or
newer and a matching PyTorch wheel**:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

An older wheel installs and imports perfectly happily, then fails at the first
kernel launch with "no kernel image is available for execution on the device",
which reads like a broken model rather than a version mismatch.
`person_detector_gpu` checks `torch.cuda.get_arch_list()` at startup and refuses
with an actionable message instead of letting that surface mid-inference.

### 3. Perception packages

```bash
pip install ultralytics insightface onnxruntime-gpu
```

Deliberately not rosdep dependencies. They pull in several gigabytes of GPU
stack and are only ever wanted here, so listing them in `package.xml` would make
every `rosdep install` on the Pi try to fetch them.

### 4. Local model server, for the dialog layer

Ollama is the least work. vLLM is faster and considerably better at
schema-constrained JSON output, which is what the dialog layer leans on.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:14b
```

or

```bash
pip install vllm
vllm serve <model> --gpu-memory-utilization 0.55 --max-model-len 8192
```

Either way the dialog layer talks to an OpenAI-compatible endpoint, so point
`base_url` in `burgerbot_dialog/config/dialog.yaml` at whichever you run — or
pass it at launch:

```bash
ros2 launch burgerbot_dialog dialog.launch.py base_url:=http://192.168.1.51:11434/v1
```

Check it before involving the robot. The node probes the endpoint at startup
and logs whether it is reachable, so a wrong URL is a line in the log at boot
rather than a confused face several minutes later:

```bash
ros2 run burgerbot_dialog dialog_cli
```

If the server rejects the request outright, try `use_schema:=false`. That stops
the node sending a JSON-schema `response_format`, which vLLM honours and older
Ollama builds do not. Nothing else changes — the reply is validated the same
way either way, which is why the validator assumes nothing about what the
server promised.

### 5. VRAM budget â€” read this before wondering why things fall over

16 GB does not stretch as far as it looks once several models are resident:

| Component | Roughly |
| --- | --- |
| 12-14B instruct model, 4-bit | 8-10 GB |
| KV cache, grows with context length | 1-2 GB |
| `yolo11n-pose` | 0.5 GB |
| insightface `buffalo_l` | 0.7 GB |
| Headroom | 1 GB |

**`--gpu-memory-utilization` matters more than the model size.** vLLM defaults
to 0.9 and preallocates its KV cache against that, so it takes about 14 GB
before the perception nodes have even started, and they then fail to allocate.
Set it near 0.55 and cap `--max-model-len`: a companion's turns are short, and a
32k context buys nothing but reserved memory.

If you add the vision-language work later, budget for one multimodal model doing
both chat and vision rather than a separate text model and VLM. Those do not
both fit.

### 6. Split the perception pipeline

On the robot:

```bash
ros2 launch burgerbot_perception people.launch.py detector:=none publish_compressed:=true
```

On the PC:

```bash
ros2 launch burgerbot_perception people.launch.py detector:=gpu tracker:=false use_identity:=true
```

Compressed colour goes up, small detection messages come back, depth never
leaves the robot. `people.launch.py`'s docstring explains why the split is
asymmetric.

---

## D. Two machines, one ROS graph

The step most likely to fail and hardest to diagnose, because each machine looks
perfectly healthy on its own.

Both need the same domain:

```bash
export ROS_DOMAIN_ID=0
```

Then check they can actually see each other:

```bash
ros2 topic list
ros2 multicast receive
```

Run `ros2 topic list` on both; they should agree. Run `ros2 multicast receive` on
one and `ros2 multicast send` on the other.

**If the multicast test fails, this is a network problem, not a ROS problem.**
DDS discovers peers by multicast, and a great many WiFi access points drop or
rate-limit multicast between wireless clients. The symptom is distinctive:
everything works over Ethernet and nothing is discovered over WiFi.

The fix is to tell DDS where its peers are instead of asking it to find them.
Write `~/cyclonedds.xml` on both machines:

```xml
<CycloneDDS>
  <Domain id="any">
    <General>
      <Interfaces><NetworkInterface name="wlan0"/></Interfaces>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>10</MaxAutoParticipantIndex>
      <Peers>
        <Peer address="192.168.1.50"/>
        <Peer address="192.168.1.51"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
```

Substitute the robot's and the PC's addresses, and each machine's own interface
name. Then on both:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/YOUR_USER/cyclonedds.xml
```

Set this on **both** machines with **both** addresses listed. Half-configured
unicast discovery is worse than none: one side finds the other and the other
does not, so a topic shows up in one `ros2 topic list` and not the other, which
looks like a QoS bug and is not.

---

## Troubleshooting

**`docker: permission denied`** â€” not in the `docker` group, or in it but not
logged out and back in since.

**Gazebo renders at a few frames a second** â€” `glxinfo` says `llvmpipe`. GPU
passthrough is not working. Section A.4.

**Face is black on the robot** â€” a desktop session holds DRM master, or you are
not in `video` and `render`. Section B.5.

**`could not open port /dev/burgerbot_pico`** â€” the udev rule is not installed or
does not match your hardware. Check `ls -l /dev/burgerbot_pico`, then
`udevadm info -a -n /dev/ttyACM0`. Section B.3.

**Detector runs but the semantic map stays empty** â€” check `/perception/detections2d`
has traffic first. If it does, the problem is downstream in projection, meaning
TF or depth, not detection.

**`no kernel image is available for execution on the device`** â€” wrong PyTorch
build for a 50-series card. Section C.2.

**The model server dies with an out-of-memory error when perception starts** â€”
`--gpu-memory-utilization` is too high. Section C.5.

**Nodes on the PC cannot see the robot's topics** â€” section D. Test multicast
before changing anything else.

**The robot replies "my head is offline"** — the model server is unreachable and
the dialog node has stopped calling it for a minute. Check the endpoint is up
and that `base_url` points at it; the node logs the reason at startup.

**Every turn fails with an HTTP error** — the server is rejecting the
`response_format` field. Set `use_schema: false` in `dialog.yaml`. Section C.4.

**"go to the kitchen" gets "I do not know where that is"** — correct behaviour,
not a fault. Nothing infers room labels. Drive the robot there and name it with
the `name_place` service. Section C.4.
