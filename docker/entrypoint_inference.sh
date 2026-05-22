#!/usr/bin/env bash
# Entrypoint for the inference container.
#
# Sources ROS2, makes the project's Python modules importable, builds the
# ROS2 workspace, then execs whatever command Compose passes.
set -e

# 1. ROS2 environment (rclpy, ros2 CLI, etc.)
source /opt/ros/humble/setup.bash

# 2. Project modules (envs, networks, algos) importable from workspace root
export PYTHONPATH="/workspace:${PYTHONPATH}"

# 3. Build the ROS2 workspace.
#    Build/install go to container-local /tmp dirs, NOT the volume-mounted
#    ros2_ws/build|install — this keeps the host's build artifacts untouched
#    and avoids cross-environment contamination between host and container.
cd /workspace/ros2_ws
echo "[entrypoint] Building ROS2 workspace (container-local build dirs)..."
colcon build --symlink-install \
    --build-base /tmp/colcon_build \
    --install-base /tmp/colcon_install
source /tmp/colcon_install/setup.bash

# 4. Hand off to the container's command
cd /workspace
exec "$@"