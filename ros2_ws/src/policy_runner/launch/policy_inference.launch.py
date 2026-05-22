"""
Launch the policy inference node alone (no MuJoCo bridge).

Use this when observations come from another source (e.g., a real
robot, a different sim wrapper). For end-to-end sim eval, use
full_pipeline.launch.py instead.

Example:
    ros2 launch policy_runner policy_inference.launch.py \\
        checkpoint:=$HOME/franka-il-rl/data/checkpoints/bc/.../last.pt
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # PYTHONPATH must include the project root so backends can import
    # networks.mlp.MLPPolicy. We let FRANKA_IL_RL_ROOT override; default
    # assumes the standard ~/franka-il-rl layout.
    project_root = os.environ.get(
        "FRANKA_IL_RL_ROOT",
        os.path.expanduser("~/franka-il-rl"),
    )
    env_with_path = {
        "PYTHONPATH": project_root + ":" + os.environ.get("PYTHONPATH", ""),
    }

    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint",
        default_value=os.path.join(
            project_root, "data", "checkpoints", "bc", "bc_baseline", "last.pt",
        ),
        description="Path to BC checkpoint .pt file",
    )
    backend_arg = DeclareLaunchArgument(
        "backend", default_value="torch",
        description="Inference backend: torch | onnx | tensorrt",
    )
    device_arg = DeclareLaunchArgument(
        "device", default_value="cpu",
        description="Inference device: cpu | cuda",
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
        additional_env=env_with_path,
    )

    return LaunchDescription([
        checkpoint_arg, backend_arg, device_arg,
        inference_node,
    ])