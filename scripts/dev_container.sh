#!/usr/bin/env bash
# Enter the burg-bot Jazzy dev container.
#
#   ./scripts/dev_container.sh              interactive shell
#   ./scripts/dev_container.sh colcon_build  run one command and exit
#
# Builds the image on first run (or after docker/Dockerfile changes) and
# reuses it after that -- `docker compose run` only rebuilds when the
# Dockerfile or its build context actually changed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found. On WSL2, install Docker Engine directly inside" >&2
    echo "this distro (no Docker Desktop needed) -- see README.md." >&2
    exit 1
fi

if [ ! -d /mnt/wslg ]; then
    echo "warning: /mnt/wslg not found -- GUI passthrough (Gazebo, RViz, the" >&2
    echo "face demo window) will not work. Are you running this inside WSL2?" >&2
fi

if [ "$#" -eq 0 ]; then
    exec docker compose run --rm dev bash
else
    exec docker compose run --rm dev "$@"
fi
