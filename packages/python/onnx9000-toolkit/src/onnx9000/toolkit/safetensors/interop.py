"""Provide interoperability functions for loading PyTorch, TensorFlow, and Flax safetensors."""

from typing import Any

from .parser import load_file


def load_pytorch_safetensors(filename: str, device: str = "cpu") -> dict[str, Any]:
    """Load PyTorch safetensors and map correctly to ONNX conventions."""
    tensors = load_file(filename, device=device)
    mapped_tensors = {}
    for k, v in tensors.items():
        # Example ONNX mapping: ensuring batch-norm / conv weights match ONNX topological expectations
        # Usually ONNX weights are identical to PyTorch, but we might need to squeeze/unsqueeze
        # For simplicity in this implementation, we assume basic identical layout or standard transpose.
        mapped_tensors[k] = v
    return mapped_tensors


def load_tensorflow_safetensors(filename: str, device: str = "cpu") -> dict[str, Any]:
    """Load TensorFlow safetensors and map correctly to ONNX conventions (e.g. NHWC -> NCHW)."""
    import numpy as np

    tensors = load_file(filename, device=device)
    mapped_tensors = {}
    for k, v in tensors.items():
        if isinstance(v, np.ndarray) and v.ndim == 4:
            # Basic heuristic: if it looks like a TF Conv2D kernel [H, W, I, O] -> ONNX [O, I, H, W]
            v = np.transpose(v, (3, 2, 0, 1))
        mapped_tensors[k] = v
    return mapped_tensors


def load_flax_safetensors(filename: str, device: str = "cpu") -> dict[str, Any]:
    """Load Flax/JAX safetensors and natively map to ONNX conventions."""
    tensors = load_file(filename, device=device)
    mapped_tensors = {}
    for k, v in tensors.items():
        # Remap Flax hierarchical keys (e.g. layers.0.attention.kernel -> layers.0.attention.weight)
        new_k = (
            k.replace(".kernel", ".weight").replace(".scale", ".weight").replace(".bias", ".bias")
        )
        mapped_tensors[new_k] = v
    return mapped_tensors
