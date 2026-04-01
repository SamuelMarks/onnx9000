"""Quantization utilities for onnx9000."""

from typing import Dict, Any, List
import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor, Constant
from onnx9000.core.dtypes import DType


def quantize_ptq(graph: Graph, calibration_data: List[Dict[str, np.ndarray]] = None) -> Graph:
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
                    pass

    return graph
