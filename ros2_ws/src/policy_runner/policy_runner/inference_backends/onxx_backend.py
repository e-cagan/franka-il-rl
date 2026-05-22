"""
ONNX Runtime inference backend (Week 12).

Stub for now. Implementation in Week 12 alongside the export script
(scripts/export_onnx.py). Same load/infer interface as TorchBackend
so the inference node can swap with a config change.
"""

import numpy as np

from policy_runner.inference_backends.base import InferenceBackend


class OnnxBackend(InferenceBackend):
    _name = "onnx"

    def load(self, checkpoint_path: str, config: dict) -> None:
        raise NotImplementedError(
            "OnnxBackend is scheduled for Week 12. "
            "Use backend='torch' in the meantime."
        )

    def infer(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name