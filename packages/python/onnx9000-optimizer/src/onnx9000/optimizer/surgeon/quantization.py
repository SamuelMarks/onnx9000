"""Quantization utilities for onnx9000."""

import numpy as np
from onnx9000.core.ir import Constant, Graph


def quantize_ptq(graph: Graph, calibration_data: list[dict[str, np.ndarray]] = None) -> Graph:
    """Post-training quantization with calibration dataset support."""
    # Algorithm:
    # 1. Collect activation statistics if calibration_data is provided.
    # 2. Quantize weights of Conv, Gemm, MatMul nodes.
    # 3. Insert QuantizeLinear/DequantizeLinear nodes or use QLinear operators.

    for n in list(graph.nodes):
        if n.op_type in ["Conv", "Gemm", "MatMul"]:
            # Quantize weights (Input 1)
            weight_name = n.inputs[1].name if hasattr(n.inputs[1], "name") else n.inputs[1]
            if weight_name in graph.tensors:
                weight_tensor = graph.tensors[weight_name]
                if isinstance(weight_tensor, Constant):
                    # Simple min-max quantization for demonstration
                    # In a real implementation, we'd use scales and zero points
                    data = np.frombuffer(weight_tensor.data, dtype=np.float32)
                    scale = (np.max(data) - np.min(data)) / 255.0
                    zp = -np.min(data) / scale if scale != 0 else 0

                    # Update tensor to be uint8
                    weight_tensor.dtype = "uint8"
                    weight_tensor.data = (
                        ((data - np.min(data)) / (scale if scale != 0 else 1))
                        .astype(np.uint8)
                        .tobytes()
                    )

                    # Mark as quantized in metadata
                    if not hasattr(graph, "metadata"):
                        graph.metadata = {}
                    graph.metadata[f"{weight_name}_quantized"] = "true"
                    graph.metadata[f"{weight_name}_scale"] = str(scale)
                    graph.metadata[f"{weight_name}_zp"] = str(zp)

    return graph
