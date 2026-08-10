#!/usr/bin/env bash
# Source the ROS 2 underlay, then the workspace overlay if it has been built
# yet, then hand off to whatever command the container was started with.
set -e

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"

WS_SETUP="/workspace/burgerbot_ws/install/setup.bash"
if [ -f "$WS_SETUP" ]; then
    source "$WS_SETUP"
else
    echo "note: burgerbot_ws is not built yet -- run: colcon_build" >&2
fi

exec "$@"
