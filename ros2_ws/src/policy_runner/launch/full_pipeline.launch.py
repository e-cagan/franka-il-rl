"""
Launch the full sim+inference pipeline: MuJoCo bridge + policy inference.

This is the primary entry point for end-to-end evaluation of a trained
policy in simulation through the ROS2 stack. Both nodes share their
observation/action topics; the bridge handles env stepping, the inference
node handles policy forward passes.

Example:
    # Default: run BC baseline on CPU, headless
    ros2 launch policy_runner full_pipeline.launch.py

    # With viewer
    ros2 launch policy_runner full_pipeline.launch.py render:=true

    # Custom checkpoint
    ros2 launch policy_runner full_pipeline.launch.py \\
        checkpoint:=$HOME/franka-il-rl/data/checkpoints/dagger/.../last.pt

    # GPU inference, sparse-reward env
    ros2 launch policy_runner full_pipeline.launch.py \\
        device:=cuda reward_type:=sparse
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    project_root = os.environ.get(
        "FRANKA_IL_RL_ROOT",
        os.path.expanduser("~/franka-il-rl"),
    )

    # --- Launch arguments ---
    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint",
        default_value=os.path.join(
            project_root, "data", "checkpoints", "bc", "bc_baseline", "last.pt",
        ),
        description="Path to policy checkpoint",
    )
    backend_arg = DeclareLaunchArgument(
        "backend", default_value="torch",
        description="Inference backend: torch | onnx | tensorrt",
    )
    device_arg = DeclareLaunchArgument(
        "device", default_value="cpu",
        description="Inference device: cpu | cuda",
    )
    render_arg = DeclareLaunchArgument(
        "render", default_value="false",
        description="Open MuJoCo viewer (true | false)",
    )
    seed_arg = DeclareLaunchArgument(
        "seed", default_value="42",
        description="Base seed for episode initial conditions",
    )
    reward_type_arg = DeclareLaunchArgument(
        "reward_type", default_value="sparse",
        description="Reward type: sparse | dense",
    )
    max_steps_arg = DeclareLaunchArgument(
        "max_episode_steps", default_value="50",
        description="Episode horizon",
    )

    # --- Nodes ---
    bridge_node = Node(
        package="mujoco_bridge",
        executable="mujoco_bridge_node",
        name="mujoco_bridge",
        output="screen",
        parameters=[{
            "render": ParameterValue(LaunchConfiguration("render"), value_type=bool),
            "max_episode_steps": ParameterValue(
                LaunchConfiguration("max_episode_steps"), value_type=int,
            ),
            "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
            "auto_reset": True,
            "reward_type": LaunchConfiguration("reward_type"),
        }],
    )

    inference_node = Node(
        package="policy_runner",
        executable="policy_inference_node",
        name="policy_inference",
        output="screen",
        parameters=[{
            "checkpoint_path": LaunchConfiguration("checkpoint"),
            "backend": LaunchConfiguration("backend"),
            "device": LaunchConfiguration("device"),
            "deterministic": True,
        }],
    )

    return LaunchDescription([
        checkpoint_arg, backend_arg, device_arg, render_arg,
        seed_arg, reward_type_arg, max_steps_arg,
        bridge_node, inference_node,
    ])