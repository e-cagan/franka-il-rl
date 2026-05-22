"""
ROS2 node that bridges a MuJoCo-based Fetch env to ROS2 topics.

Topology:
    [env.step] --pub--> /sim/observation                  (Float32MultiArray, 28-D)
                                                          --> consumed by policy_inference
    /sim/action <--sub-- [env.step input]                 (Float32MultiArray, 4-D)
    /sim/episode_info --pub-->                            (Float32MultiArray: [step, episode, success, return])

Control flow is a ping-pong loop, not a fixed-rate timer:
    1. Reset env, publish initial observation.
    2. Inference node receives obs, publishes action.
    3. This node's action callback fires, steps env, publishes next obs.
    4. ...repeat until episode ends, then reset and continue (auto_reset).

This is faster and simpler than a fixed-Hz timer for sim-only setups
(no real-time deadlines, env doesn't care about wall-clock rate).
For real-robot deployment we'd revisit and add rate limiting.

Configuration parameters:
    render (bool):              Open MuJoCo viewer. Default: False.
    max_episode_steps (int):    Episode length cap. Default: 50.
    seed (int):                 Base seed; each episode uses seed + episode_count.
    auto_reset (bool):          Reset env after each episode end. Default: True.
    reward_type (str):          'sparse' | 'dense'. Default: 'sparse'.
    obs_topic (str):            Topic to publish observations on.
    action_topic (str):         Topic to subscribe to for actions.
    info_topic (str):           Topic for episode info broadcasts.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray

# `envs.fetch_pickplace` lives in the project root; PYTHONPATH must include
# franka-il-rl/ for this import to resolve. Launch files set this via
# additional_env; manual runs need it set in the shell.
from envs.fetch_pickplace import FetchPickPlaceWrapper


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


class MujocoBridgeNode(Node):

    def __init__(self):
        super().__init__("mujoco_bridge")

        # --- Parameters ---
        self.declare_parameter("render", False)
        self.declare_parameter("max_episode_steps", 50)
        self.declare_parameter("seed", 42)
        self.declare_parameter("auto_reset", True)
        self.declare_parameter("reward_type", "sparse")
        self.declare_parameter("obs_topic", "/sim/observation")
        self.declare_parameter("action_topic", "/sim/action")
        self.declare_parameter("info_topic", "/sim/episode_info")

        render = self.get_parameter("render").value
        max_steps = self.get_parameter("max_episode_steps").value
        self._seed = self.get_parameter("seed").value
        self._auto_reset = self.get_parameter("auto_reset").value
        reward_type = self.get_parameter("reward_type").value

        obs_topic = self.get_parameter("obs_topic").value
        action_topic = self.get_parameter("action_topic").value
        info_topic = self.get_parameter("info_topic").value

        # --- Environment ---
        self._env = FetchPickPlaceWrapper(
            render_mode="human" if render else None,
            max_episode_steps=max_steps,
            reward_type=reward_type,
        )
        self.get_logger().info(
            f"FetchPickPlaceWrapper ready (render={render}, "
            f"max_steps={max_steps}, reward={reward_type})"
        )

        # --- Pub/Sub ---
        self._obs_pub = self.create_publisher(
            Float32MultiArray, obs_topic, _OBS_QOS
        )
        self._info_pub = self.create_publisher(
            Float32MultiArray, info_topic, _ACTION_QOS
        )
        self._action_sub = self.create_subscription(
            Float32MultiArray, action_topic, self._on_action, _ACTION_QOS
        )

        # --- State ---
        self._episode_step = 0
        self._episode_count = 0
        self._episode_return = 0.0
        self._last_success = 0.0
        self._total_successes = 0

        # Render timer: pump the viewer independently of the control loop
        # so the GUI event loop stays alive (human mode needs regular
        # render() calls; doing it inside the step callback deadlocks).
        self._render = render
        if self._render:
            self._render_timer = self.create_timer(1.0 / 30.0, self._render_tick)

        # Kick off the loop: publish the first observation. The inference
        # node will respond with an action, which lands in _on_action and
        # advances the env one step.
        self._reset_and_publish()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def _reset_and_publish(self) -> None:
        obs, _info = self._env.reset(
            seed=self._seed + self._episode_count,
        )
        self._episode_step = 0
        self._episode_return = 0.0
        self._publish_obs(obs)

    def _render_tick(self):
        try:
            self._env.render()
        except Exception:
            pass

    def _on_action(self, msg: Float32MultiArray) -> None:
        action = np.asarray(msg.data, dtype=np.float32)

        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated

        self._episode_step += 1
        self._episode_return += float(reward)
        self._last_success = float(info.get("is_success", 0.0) > 0.5)

        if done:
            self._total_successes += int(self._last_success)
            self.get_logger().info(
                f"Episode {self._episode_count}: "
                f"steps={self._episode_step}  "
                f"return={self._episode_return:.1f}  "
                f"success={int(self._last_success)}  "
                f"running_rate={self._total_successes}/{self._episode_count + 1}"
            )
            self._publish_info()
            self._episode_count += 1

            if self._auto_reset:
                self._reset_and_publish()
            # If auto_reset is False, we stop publishing observations.
            # The loop halts naturally; user must restart the node for
            # another run.
        else:
            self._publish_obs(obs)
            self._publish_info()

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _publish_obs(self, obs: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = obs.astype(np.float32).tolist()
        self._obs_pub.publish(msg)

    def _publish_info(self) -> None:
        msg = Float32MultiArray()
        # [step, episode_count, success_in_episode, episode_return]
        msg.data = [
            float(self._episode_step),
            float(self._episode_count),
            float(self._last_success),
            float(self._episode_return),
        ]
        self._info_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._env.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()