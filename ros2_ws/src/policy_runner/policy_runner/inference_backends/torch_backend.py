"""
PyTorch inference backend.

Loads a BC checkpoint (saved via BCTrainer.save_checkpoint) and runs
the MLP policy forward pass on each observation. This is the
reference backend used in Week 11; ONNX and TensorRT backends in
Week 12 should produce identical numerical output (within fp tolerance).
"""

import numpy as np
import torch

from policy_runner.inference_backends.base import InferenceBackend


class TorchBackend(InferenceBackend):
    """
    PyTorch forward-pass backend.

    Expects a checkpoint with the BCTrainer format:
        {
            "policy_state_dict": ...,
            "optimizer_state_dict": ...,
            "config": {                  # used to rebuild architecture
                "obs_dim": int,
                "action_dim": int,
                "hidden_sizes": [int, ...],
                ...
            },
            "best_val_loss": float,
            "best_success_rate": float,
        }

    The architecture is rebuilt from the checkpoint's `config` field —
    not from any external config passed at load time. This guarantees
    the network matches the saved weights regardless of what the
    running system thinks the defaults should be.
    """

    _name = "torch"

    def __init__(self):
        self._policy = None
        self._device = None
        self._obs_dim = None
        self._action_dim = None

    def load(self, checkpoint_path: str, config: dict) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Architecture comes from the checkpoint itself, not the runtime config
        if "config" not in ckpt:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no 'config' field; "
                "cannot rebuild architecture. Was this saved by BCTrainer?"
            )

        ckpt_cfg = ckpt["config"]
        self._obs_dim = ckpt_cfg.get("obs_dim", 28)
        self._action_dim = ckpt_cfg.get("action_dim", 4)
        hidden_sizes = tuple(ckpt_cfg.get("hidden_sizes", [256, 256, 256]))

        # Runtime config controls device / precision, not architecture
        device_str = config.get("device", "cpu")
        self._device = torch.device(device_str)

        # Import here so this module loads cleanly even when networks.mlp
        # is not on the path (e.g., ROS2 tooling that just inspects the
        # backend class without intending to run inference).
        from networks.mlp import MLPPolicy

        self._policy = MLPPolicy(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
        ).to(self._device)
        self._policy.load_state_dict(ckpt["policy_state_dict"])
        self._policy.eval()

    @torch.no_grad()
    def infer(self, obs: np.ndarray) -> np.ndarray:
        if self._policy is None:
            raise RuntimeError("TorchBackend.load() was not called before infer()")

        if obs.shape != (self._obs_dim,):
            raise ValueError(
                f"Expected obs of shape ({self._obs_dim},), got {obs.shape}"
            )

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self._device)
        action_t = self._policy(obs_t)
        action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Fetch action space is [-1, 1]^4; clip defensively in case the
        # policy emits slightly outside (it shouldn't, but cheap insurance).
        return np.clip(action, -1.0, 1.0)

    @property
    def name(self) -> str:
        return self._name