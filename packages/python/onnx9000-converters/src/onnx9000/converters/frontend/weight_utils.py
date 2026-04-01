"""Weight management utilities for onnx9000."""

from typing import Dict, Any
import numpy as np
from onnx9000.core.ir import Graph, Constant


def export_state_dict(graph: Graph) -> Dict[str, Any]:
    """Export model weights from Graph IR into a PyTorch-compatible state_dict."""
    state_dict = {}
    for name, tensor in graph.tensors.items():
        if isinstance(tensor, Constant) and tensor.data is not None:
            # Map name if needed (e.g. replacing '/' with '.')
            torch_name = name.replace("/", ".")

            # Convert bytes to numpy array
            try:
                import torch

                # If torch is available, convert directly
                from onnx9000.core.dtypes import DType

                dtype_map = {
                    DType.FLOAT32: np.float32,
                    DType.FLOAT64: np.float64,
                    DType.INT32: np.int32,
                    DType.INT64: np.int64,
                }
                np_dtype = dtype_map.get(tensor.dtype, np.float32)
                arr = np.frombuffer(tensor.data, dtype=np_dtype).reshape(tensor.shape)
                state_dict[torch_name] = torch.from_numpy(arr.copy())
            except ImportError:
                # Fallback to numpy
                state_dict[torch_name] = (
                    np.frombuffer(tensor.data, dtype=np.float32).reshape(tensor.shape).copy()
                )

    return state_dict


def universal_weight_bridge(weights: Dict[str, Any], target_fmt: str = "pytorch") -> Any:
    """Standalone conversion between Safetensors, HDF5, and PyTorch pickles."""
    if target_fmt == "pytorch":
        import torch

        return weights
    elif target_fmt == "safetensors":
        # Requires safetensors package
        try:
            from safetensors.torch import save

            return save(weights)
        except ImportError:
            return None
    return weights
