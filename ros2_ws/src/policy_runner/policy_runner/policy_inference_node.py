"""
ROS2 node that runs a trained policy: observation in, action out.

Topology:
    /sim/observation  (Float32MultiArray, 28-D) --in-->  [this node]  --out--> /sim/action  (Float32MultiArray, 4-D)

The node is backend-agnostic: it loads a TorchBackend, OnnxBackend, or
TensorRTBackend based on the `backend` parameter and treats them
identically through the InferenceBackend interface.

Configuration parameters (declared via ROS2 params):
    checkpoint_path (str):  Required. Path to model file. Format
                             depends on backend (.pt / .onnx / .engine).
    backend (str):          'torch' | 'onnx' | 'tensorrt'. Default: 'torch'.
    device (str):            'cpu' | 'cuda'. Passed to the backend.
    deterministic (bool):    Use deterministic policy output. Default: True.
    obs_topic (str):         Topic to subscribe to. Default: '/sim/observation'.
    action_topic (str):      Topic to publish to. Default: '/sim/action'.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray

from policy_runner.inference_backends import build_backend


# Observations are latched (TRANSIENT_LOCAL) so the inference node, which
# may subscribe after the bridge publishes its first obs, still receives
# the most recent observation on connection. RELIABLE prevents mid-loop
# drops that would stall the ping-pong control loop.
_OBS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# Actions don't need latching (the bridge subscribes during its own init,
# before the inference node ever publishes), but RELIABLE keeps the loop
# from stalling on a dropped action.
_ACTION_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class PolicyInferenceNode(Node):

    def __init__(self):
        super().__init__("policy_inference")

        # --- Parameters ---
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("backend", "torch")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("deterministic", True)
        self.declare_parameter("obs_topic", "/sim/observation")
        self.declare_parameter("action_topic", "/sim/action")

        checkpoint_path = self.get_parameter("checkpoint_path").value
        backend_name = self.get_parameter("backend").value
        device = self.get_parameter("device").value
        deterministic = self.get_parameter("deterministic").value
        obs_topic = self.get_parameter("obs_topic").value
        action_topic = self.get_parameter("action_topic").value

        if not checkpoint_path:
            self.get_logger().error(
                "`checkpoint_path` parameter is required. "
                "Set it via launch arg or YAML config."
            )
            raise RuntimeError("checkpoint_path empty")

        # --- Backend ---
        self._backend = build_backend(backend_name)
        self.get_logger().info(
            f"Loading {backend_name} backend from {checkpoint_path} "
            f"(device={device})..."
        )
        self._backend.load(
            checkpoint_path=checkpoint_path,
            config={
                "device": device,
                "deterministic": deterministic,
            },
        )
        self.get_logger().info(f"Backend ready: {self._backend.name}")

        # --- Pub/Sub ---
        self._action_pub = self.create_publisher(
            Float32MultiArray, action_topic, _ACTION_QOS
        )
        self._obs_sub = self.create_subscription(
            Float32MultiArray, obs_topic, self._on_obs, _OBS_QOS
        )

        # --- Stats ---
        self._inference_count = 0
        self._log_every = 200  # ~8s at 25Hz control rate

        self.get_logger().info(
            f"Subscribed to {obs_topic}, publishing to {action_topic}. "
            "Waiting for observations..."
        )

    def _on_obs(self, msg: Float32MultiArray) -> None:
        obs = np.asarray(msg.data, dtype=np.float32)

        try:
            action = self._backend.infer(obs)
        except Exception as exc:  # noqa: BLE001
            # Don't kill the node on a single bad inference. Log and
            # skip; the next observation will retry.
            self.get_logger().error(f"Inference failed: {exc}")
            return

        action_msg = Float32MultiArray()
        action_msg.data = action.tolist()
        self._action_pub.publish(action_msg)

        self._inference_count += 1
        if self._inference_count % self._log_every == 0:
            self.get_logger().info(
                f"Inferences: {self._inference_count}  "
                f"last action norm: {np.linalg.norm(action):.3f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()