"""
TensorRT inference backend via ONNX Runtime's TensorRT Execution Provider.

Runs a .onnx model through ONNX Runtime with the TensorrtExecutionProvider,
which builds a TensorRT engine from the ONNX graph on first run and caches
it to disk. FP16 is enabled through provider options. The provider list
falls back to CUDA then CPU, so if the TRT EP fails to initialize (a common
version-alignment issue between onnxruntime-gpu, CUDA, and TensorRT), the
model still runs on GPU/CPU rather than crashing.

Why ORT-TRT EP instead of a native .engine:
  - On x86 dev hardware, a native engine's only value is the latency
    benchmark; ORT-TRT EP gives the same TensorRT execution with no manual
    engine build or CUDA buffer management.
  - TensorRT engines are not portable across architectures, so an x86
    engine wouldn't transfer to the Jetson target anyway. Jetson deployment
    builds its own engine on-device (e.g. via trtexec).
"""

import os
import numpy as np

from policy_runner.inference_backends.base import InferenceBackend


class TensorRTBackend(InferenceBackend):
    _name = "tensorrt"

    def __init__(self):
        self._sess = None
        self._input_name = None
        self._output_name = None

    def load(self, checkpoint_path: str, config: dict) -> None:
        # checkpoint_path here is a .onnx file
        import onnxruntime as ort

        cache_dir = config.get("trt_cache_dir", "/tmp/trt_cache")
        os.makedirs(cache_dir, exist_ok=True)

        trt_options = {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
        }
        providers = [
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        # First load triggers engine build — can take tens of seconds.
        # Subsequent loads reuse the cached engine.
        self._sess = ort.InferenceSession(checkpoint_path, providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name

        active = self._sess.get_providers()[0]
        print(f"[TensorRTBackend] active provider: {active}")
        if active != "TensorrtExecutionProvider":
            print("[TensorRTBackend] WARNING: TensorRT EP not active; fell back "
                  f"to {active}. Check onnxruntime-gpu / tensorrt versions.")

    def infer(self, obs: np.ndarray) -> np.ndarray:
        if self._sess is None:
            raise RuntimeError("TensorRTBackend.load() was not called before infer()")
        x = obs.astype(np.float32).reshape(1, -1)
        out = self._sess.run([self._output_name], {self._input_name: x})[0]
        return np.clip(out.reshape(-1).astype(np.float32), -1.0, 1.0)

    @property
    def name(self) -> str:
        return self._name