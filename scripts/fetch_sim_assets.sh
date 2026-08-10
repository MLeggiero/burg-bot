#!/usr/bin/env bash
# Fetch the third-party AWS RoboMaker Gazebo assets used by the small_house and
# small_warehouse worlds.
#
# These are ~117MB of binary meshes and textures owned by AWS (MIT licensed),
# so they are deliberately not vendored into this repo. They are only needed to
# render the furnished demo worlds -- everything else, including empty.world and
# the whole real-robot stack, works without them.
#
# Usage:  ./scripts/fetch_sim_assets.sh
set -euo pipefail

UPSTREAM="https://github.com/AntoBrandi/Self-Driving-and-ROS-2-Learn-by-Doing-Plan-Navigation.git"
BRANCH="jazzy"
ASSET_PATH="Section9_Build_the_Robot/bumperbot_ws/src/bumperbot_description"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/burgerbot_ws/src/burgerbot_description"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ -d "$DEST/models" ] && [ -d "$DEST/photos" ]; then
  echo "Sim assets already present at $DEST -- nothing to do."
  echo "(Delete models/ and photos/ to force a re-fetch.)"
  exit 0
fi

echo "Fetching sim assets from upstream (~117MB, this takes a minute)..."
git clone --depth 1 --branch "$BRANCH" --filter=blob:none --sparse "$UPSTREAM" "$TMP/up" >/dev/null 2>&1
git -C "$TMP/up" sparse-checkout set "$ASSET_PATH/models" "$ASSET_PATH/photos" >/dev/null 2>&1

for d in models photos; do
  if [ -d "$TMP/up/$ASSET_PATH/$d" ]; then
    rm -rf "${DEST:?}/$d"
    cp -r "$TMP/up/$ASSET_PATH/$d" "$DEST/$d"
    echo "  installed $d/  ($(du -sh "$DEST/$d" | cut -f1))"
  else
    echo "  WARNING: upstream has no $d/ -- skipping" >&2
  fi
done

echo
echo "Done. Rebuild so the assets land in the install space:"
echo "  colcon build --packages-select burgerbot_description"
