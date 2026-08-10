# ROS 2 Jazzy dev/sim environment for burg-bot.
#
# This runs on the DEVELOPMENT MACHINE only (Gazebo, RViz, colcon build,
# code editing) -- not on the robot. The Pi 4 stays native Ubuntu 24.04
# Server; containerizing a KMSDRM display pipeline and a hard-real-time
# ros2_control loop for a single-purpose robot brain buys little and adds
# a real amount of fragility (device passthrough for /dev/dri, /dev/input,
# the serial port, all inside a privileged container) for no corresponding
# benefit. See README.md for the Pi setup.
#
# Base: osrf/ros:jazzy-desktop-full includes RViz2, rqt and the rest of the
# desktop tooling; Nav2, SLAM, ros2_control, twist_mux and Gazebo integration
# are not part of it and are installed explicitly below via rosdep against
# this workspace's own package.xml files, so this list never drifts out of
# sync with what the packages actually declare.
FROM osrf/ros:jazzy-desktop-full

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    LANG=en_US.UTF-8

# --- WSLg GUI + GPU passthrough -----------------------------------------
# Gazebo and RViz render through WSLg's virtual GPU (D3D12/dxcore via
# /usr/lib/wsl, device /dev/dxg -- there is no /dev/dri under WSLg), and the
# X11/Wayland sockets under /mnt/wslg are bind-mounted at the identical path
# by docker-compose.yml. mesa-utils and x11-apps are included purely for
# `glxinfo` / `xeyes` as a quick smoke test that passthrough is working.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libglx-mesa0 mesa-utils x11-apps \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg: screen-records the WSLg X11 display for scripts/record_testbed.sh
# (there is nothing else in this image that can capture a video of the
# Gazebo/RViz windows running).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Workspace dependencies ----------------------------------------------
# libserial-dev and python3-smbus are pinned explicitly: they are host-level
# native/system deps some rosdep setups miss depending on the rosdistro
# index snapshot, so belt-and-suspenders here is cheap and re-running
# apt-get install on an already-satisfied package is a no-op.
COPY burgerbot_ws/src /tmp/burgerbot_ws/src
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-rosdep python3-colcon-common-extensions \
        libserial-dev python3-smbus git \
    && (rosdep init 2>/dev/null || true) \
    && rosdep update \
    && rosdep install --from-paths /tmp/burgerbot_ws/src --ignore-src -r -y \
        --rosdistro "${ROS_DISTRO}" \
    && rm -rf /tmp/burgerbot_ws /var/lib/apt/lists/*

# ai-edge-litert (burgerbot_perception's TFLite runtime) has no rosdep key --
# it's a plain pip package, Google's actively maintained successor to the
# abandoned tflite-runtime (whose last release has no Python 3.12 wheel at
# all, which is what Jazzy actually runs). Confirmed to ship real wheels for
# this image's Python/arch before relying on it here.
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir ai-edge-litert

# --- Non-root user matching the WSL host's UID/GID ------------------------
# Bind-mounted colcon build/install/log output has to be writable by (and
# readable back from) the host user, or every build leaves root-owned files
# behind in the mounted workspace.
#
# Ubuntu's official 24.04 base image (which osrf/ros:jazzy-* is built on)
# already ships a default `ubuntu` user/group at 1000:1000 -- the same UID/GID
# WSL2 gives its first Linux user. A plain groupadd/useradd collides with
# that. Rename the existing account instead of creating a new one when it's
# already sitting on the UID/GID we want; fall back to creating fresh
# otherwise, so this still works against a base image that doesn't predefine
# one.
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN if getent passwd "${USER_UID}" >/dev/null 2>&1; then \
        existing_user="$(getent passwd "${USER_UID}" | cut -d: -f1)"; \
        existing_group="$(getent group "${USER_GID}" | cut -d: -f1)"; \
        [ "${existing_group}" != "${USERNAME}" ] && groupmod -n "${USERNAME}" "${existing_group}"; \
        [ "${existing_user}" != "${USERNAME}" ] && usermod -l "${USERNAME}" -d "/home/${USERNAME}" -m "${existing_user}"; \
    else \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}"; \
    fi \
    && usermod -aG sudo "${USERNAME}" \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-dev \
    && chmod 0440 /etc/sudoers.d/90-dev

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Pre-create the Gazebo Fuel cache directory, owned by the non-root user,
# before docker-compose.yml mounts its gz-fuel-cache named volume over this
# path. Docker copies an existing image directory's ownership into a fresh
# named volume on first mount; skipping this step leaves the mount point
# root:root (Docker's default for a path that doesn't already exist in the
# image), and the dev user silently can't write to it -- confirmed: every
# Fuel-hosted model include (world) failed with "Unable to create directory"
# / "Unable to unzip" until this was added.
RUN mkdir -p "/home/${USERNAME}/.gz" && chown "${USERNAME}:${USERNAME}" "/home/${USERNAME}/.gz"

# Convenience shortcut referenced by the entrypoint's build hint.
RUN echo "alias colcon_build='cd /workspace/burgerbot_ws && colcon build --symlink-install && source install/setup.bash'" \
        >> "/home/${USERNAME}/.bashrc"

USER ${USERNAME}
WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
