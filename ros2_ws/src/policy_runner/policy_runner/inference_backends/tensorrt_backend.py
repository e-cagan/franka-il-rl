"""
TensorRT FP16 inference backend (Week 12).

Stub for now. Implementation in Week 12 alongside the engine builder
(scripts/build_trt_engine.py). Same load/infer interface as
TorchBackend so the inference node can swap with a config change.

FP16 was selected because the BC policy is small (~140k params) and
FP16 cuts memory bandwidth roughly in half on the deployment target
(Jetson Orin Nano per the broader project plan, RTX 3050 Ti laptop
during dev). INT8 quantization is out of scope; the latency win is
not worth the calibration overhead at this model size.
"""

import numpy as np

from policy_runner.inference_backends.base import InferenceBackend


class TensorRTBackend(InferenceBackend):
    _name = "tensorrt"

    def load(self, checkpoint_path: str, config: dict) -> None:
        raise NotImplementedError(
            "TensorRTBackend is scheduled for Week 12. "
            "Use backend='torch' in the meantime."
        )

    def infer(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name