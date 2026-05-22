"""
Abstract base class for inference backends.

Strategy Pattern: the policy inference node delegates the actual
"obs -> action" computation to a pluggable backend. Backends are
swapped via config (`backend: torch | onnx | tensorrt`) without
changing the node code.

Week 11: TorchBackend (PyTorch forward pass)
Week 12: OnnxBackend, TensorRTBackend (deployment optimizations)
"""

from abc import ABC, abstractmethod
import numpy as np


class InferenceBackend(ABC):
    """
    Interface contract for all inference backends.

    Lifecycle:
        1. __init__()      — cheap, no model loading
        2. load(path, cfg) — load weights, build runtime
        3. infer(obs)      — called per env step
        4. (no explicit close needed; backends free resources via GC)

    Backends should be:
      - Stateless across infer() calls (MLP policies are; recurrent
        policies would need internal state, not used in this project).
      - Thread-unsafe by default; the inference node calls infer()
        from a single subscriber callback, no locking needed.
    """

    @abstractmethod
    def load(self, checkpoint_path: str, config: dict) -> None:
        """
        Load model weights and prepare runtime.

        Args:
            checkpoint_path: path to model file. Format is backend-specific
                (.pt for torch, .onnx for onnx, .engine for tensorrt).
            config: runtime config (device, deterministic flag, etc.).
                Architecture hyperparameters should come from the
                checkpoint itself, not this dict.
        """

    @abstractmethod
    def infer(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute one action from one observation.

        Args:
            obs: (obs_dim,) float32 observation.

        Returns:
            (action_dim,) float32 action, clipped to environment bounds.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'torch', 'onnx', 'tensorrt') for logging."""