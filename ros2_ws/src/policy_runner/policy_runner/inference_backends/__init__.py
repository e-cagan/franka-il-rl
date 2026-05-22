"""Inference backend implementations for policy_runner."""

from policy_runner.inference_backends.base import InferenceBackend

__all__ = ["InferenceBackend", "build_backend"]


def build_backend(name: str) -> InferenceBackend:
    """
    Factory: return a fresh (unloaded) backend instance by name.

    Imports are lazy so a system that only has, e.g., PyTorch installed
    can still build a TorchBackend without ONNX/TensorRT dependencies
    being present.
    """
    name = name.lower()
    if name == "torch":
        from policy_runner.inference_backends.torch_backend import TorchBackend
        return TorchBackend()
    if name == "onnx":
        from policy_runner.inference_backends.onxx_backend import OnnxBackend
        return OnnxBackend()
    if name == "tensorrt":
        from policy_runner.inference_backends.tensorrt_backend import TensorRTBackend
        return TensorRTBackend()
    raise ValueError(
        f"Unknown backend '{name}'. Choices: torch, onnx, tensorrt."
    )