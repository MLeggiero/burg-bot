#!/usr/bin/env bash
# Run the autonomous-mapping testbed and record a video of it.
#
# Brings up Gazebo (test_room.world) + SLAM + Nav2 + the frontier explorer
# with no human driving, screen-records the WSLg display for a bounded
# duration, saves the resulting map alongside the video, then tears
# everything down. Meant to be run via the dev container:
#
#   ./scripts/dev_container.sh ./scripts/record_testbed.sh
#   ./scripts/dev_container.sh ./scripts/record_testbed.sh 300   # 5 min cap
#
# Output (both under recordings/, gitignored -- these are demo artifacts,
# not source):
#   recordings/testbed.mp4
#   recordings/test_room/map.pgm + map.yaml
set -o pipefail
# Not -e: a late failure (e.g. map save) shouldn't discard a video that
# already recorded successfully. Not -u either, despite that generally being
# good practice: colcon's generated install/setup.bash references several
# variables (COLCON_TRACE among them) without ever assuming nounset, which is
# completely normal bash but a hard error under `set -u` -- confirmed by
# actually hitting it. Sourcing a ROS/colcon workspace under `-u` is a known
# incompatibility, not something specific to this script.

DURATION="${1:-240}"  # seconds. 4 min default -- a 4x4m room explores well
                       # within that; this is a hard cap so the script always
                       # terminates and delivers something even if the sweep
                       # doesn't fully converge.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/recordings"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT/burgerbot_ws"
echo "Building (picks up test_room.world / testbed.launch.py if this is a"
echo "fresh checkout or they changed since the last build)..."
colcon build --symlink-install
source install/setup.bash

echo ""
echo "Launching the testbed (Gazebo + SLAM + Nav2 + frontier_explorer)..."
# Deliberately forcing software rendering (unsetting the GPU-passthrough
# overrides docker-compose.yml normally sets) for this one process, not the
# container as a whole. `ffmpeg -f x11grab` reads the legacy X11 root window
# buffer; GPU-accelerated windows under WSLg's compositor present via direct
# GPU buffer sharing and never touch that buffer, so a hardware-accelerated
# Gazebo/RViz recorded this way comes out solid black -- confirmed by
# extracting and inspecting frames, not assumed. This is a well-documented
# x11grab limitation under any modern compositing window manager, not
# specific to WSLg. Trading render smoothness for the recording actually
# showing content is the right tradeoff for a short demo capture.
env -u MESA_LOADER_DRIVER_OVERRIDE -u GALLIUM_DRIVER \
    ros2 launch burgerbot_bringup testbed.launch.py > "$OUT_DIR/launch.log" 2>&1 &
LAUNCH_PID=$!

cleanup() {
    echo "Shutting down (pid $LAUNCH_PID)..."
    kill -INT "$LAUNCH_PID" 2>/dev/null
    # ROS 2 launch cascades SIGINT to everything it started and gives them a
    # moment to exit cleanly (Gazebo in particular does not appreciate being
    # SIGKILLed). Only escalate if it's still around after a real wait.
    for _ in $(seq 1 10); do
        kill -0 "$LAUNCH_PID" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL "$LAUNCH_PID" 2>/dev/null
}
trap cleanup EXIT

echo "Waiting for Gazebo/RViz to come up (first launch compiles shaders,"
echo "give it real time)..."
sleep 20

echo ""
echo "Recording for ${DURATION}s (Ctrl-C to stop early -- the video and map"
echo "are still saved either way)..."
# No -video_size: x11grab captures the display's actual full resolution when
# it's omitted, which is more robust than hardcoding a size that may not
# match whatever WSLg happens to report on your machine.
ffmpeg -y -f x11grab -framerate 15 -i "${DISPLAY:-:0}" \
    -t "$DURATION" -c:v libx264 -preset fast -pix_fmt yuv420p \
    "$OUT_DIR/testbed.mp4" 2>"$OUT_DIR/ffmpeg.log"

if [ ! -s "$OUT_DIR/testbed.mp4" ]; then
    echo "ffmpeg produced no output -- check $OUT_DIR/ffmpeg.log and" >&2
    echo "$OUT_DIR/launch.log. Is DISPLAY set and X11 reachable (this must" >&2
    echo "run inside the dev container with GUI passthrough, not bare)?" >&2
    exit 1
fi
echo "Wrote $OUT_DIR/testbed.mp4"

echo ""
echo "Saving the resulting map..."
mkdir -p "$OUT_DIR/test_room"
ros2 run nav2_map_server map_saver_cli -f "$OUT_DIR/test_room/map" \
    --ros-args -p save_map_timeout:=10.0 -p use_sim_time:=true \
    || echo "map save failed -- the video is still valid evidence on its own" >&2

echo ""
echo "Done. See:"
echo "  $OUT_DIR/testbed.mp4"
echo "  $OUT_DIR/test_room/map.pgm (+ .yaml)"
