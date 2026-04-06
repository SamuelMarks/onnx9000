"""Quantization utilities for onnx9000."""

import struct
import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node


def quantize_ptq(graph: Graph, calibration_data: list[dict[str, np.ndarray]] = None) -> Graph:
    """Post-training quantization with calibration dataset support."""
    for n in list(graph.nodes):
        if n.op_type in ["Conv", "Gemm", "MatMul"]:
            weight_name = n.inputs[1].name if hasattr(n.inputs[1], "name") else n.inputs[1]
            if weight_name in graph.tensors:
                weight_tensor = graph.tensors[weight_name]
                if isinstance(weight_tensor, Constant) and weight_tensor.data is not None:
                    try:
                        data = np.frombuffer(weight_tensor.data, dtype=np.float32)
                    except ValueError:
                        continue

                    if len(data) == 0:
                        continue

                    data_min = np.min(data)
                    data_max = np.max(data)
                    scale = (data_max - data_min) / 255.0
                    zp = int(round(-data_min / scale)) if scale != 0 else 0
                    zp = max(0, min(255, zp))

                    quantized_data = np.clip(
                        np.round(data / (scale if scale != 0 else 1) + zp), 0, 255
                    ).astype(np.uint8)

                    weight_tensor.dtype = "uint8"
                    weight_tensor.data = quantized_data.tobytes()

                    scale_name = f"{weight_name}_scale"
                    scale_tensor = Constant(
                        name=scale_name,
                        values=struct.pack("<f", float(scale)),
                        shape=(),
                        dtype=DType.FLOAT32,
                    )
                    graph.add_tensor(scale_tensor)

                    zp_name = f"{weight_name}_zero_point"
                    zp_tensor = Constant(
                        name=zp_name,
                        values=struct.pack("<B", zp),
                        shape=(),
                        dtype=DType.UINT8,
                    )
                    graph.add_tensor(zp_tensor)

                    dequant_out_name = f"{weight_name}_dequantized"
                    dequant_node = Node(
                        op_type="DequantizeLinear",
                        inputs=[weight_name, scale_name, zp_name],
                        outputs=[dequant_out_name],
                        name=f"DequantizeLinear_{weight_name}",
                    )

                    idx = graph.nodes.index(n)
                    graph.nodes.insert(idx, dequant_node)
                    n.inputs[1] = dequant_out_name

    return graph
