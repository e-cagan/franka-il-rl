"""
ONNX Runtime inference backend.

Loads a .onnx model (produced by scripts/export_onnx.py) and runs it via
ONNX Runtime. Device selection picks the execution provider: CPU by
default, CUDA when device='cuda'. Numerically matches TorchBackend
within fp tolerance (verified at export time).
"""

import numpy as np

from policy_runner.inference_backends.base import InferenceBackend


class OnnxBackend(InferenceBackend):
    _name = "onnx"

    def __init__(self):
        self._sess = None
        self._input_name = None
        self._output_name = None

    def load(self, checkpoint_path: str, config: dict) -> None:
        # checkpoint_path here is a .onnx file, not a .pt
        import onnxruntime as ort

        device = config.get("device", "cpu")
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._sess = ort.InferenceSession(checkpoint_path, providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name

        # Report which provider actually got selected (CUDA may silently
        # fall back to CPU if the GPU EP isn't available).
        active = self._sess.get_providers()[0]
        print(f"[OnnxBackend] active provider: {active}")

    def infer(self, obs: np.ndarray) -> np.ndarray:
        if self._sess is None:
            raise RuntimeError("OnnxBackend.load() was not called before infer()")
        x = obs.astype(np.float32).reshape(1, -1)
        out = self._sess.run([self._output_name], {self._input_name: x})[0]
        action = out.reshape(-1).astype(np.float32)
        return np.clip(action, -1.0, 1.0)

    @property
    def name(self) -> str:
        return self._name