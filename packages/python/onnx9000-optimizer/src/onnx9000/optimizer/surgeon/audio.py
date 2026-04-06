"""Audio-specific graph optimizations for onnx9000."""

import math
import struct

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph


def _unpack_scalar(c: Constant, is_float: bool = False) -> float:
    if c.data is None:
        return 0.0
    val = c.data
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, (bytes, bytearray, memoryview)):
        b = bytes(val)
        if len(b) == 4:
            return struct.unpack("<f" if is_float else "<i", b)[0]
        elif len(b) == 8:
            return struct.unpack("<d" if is_float else "<q", b)[0]
    return 0.0


def fold_mel_weights(graph: Graph) -> Graph:
    """Implement Mel-scale conversion as a constant-folded initializer pass."""
    nodes_to_remove = []
    for n in graph.nodes:
        if n.op_type == "MelWeightMatrix":
            # MelWeightMatrix inputs: num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz
            if len(n.inputs) == 5 and all(
                isinstance(graph.tensors.get(i.name if hasattr(i, "name") else i), Constant)
                for i in n.inputs
            ):
                try:
                    c_num_mel_bins = graph.tensors.get(
                        n.inputs[0].name if hasattr(n.inputs[0], "name") else n.inputs[0]
                    )
                    c_dft_length = graph.tensors.get(
                        n.inputs[1].name if hasattr(n.inputs[1], "name") else n.inputs[1]
                    )
                    c_sample_rate = graph.tensors.get(
                        n.inputs[2].name if hasattr(n.inputs[2], "name") else n.inputs[2]
                    )
                    c_lower_edge = graph.tensors.get(
                        n.inputs[3].name if hasattr(n.inputs[3], "name") else n.inputs[3]
                    )
                    c_upper_edge = graph.tensors.get(
                        n.inputs[4].name if hasattr(n.inputs[4], "name") else n.inputs[4]
                    )

                    num_mel_bins = int(_unpack_scalar(c_num_mel_bins, False))
                    dft_length = int(_unpack_scalar(c_dft_length, False))
                    sample_rate = int(_unpack_scalar(c_sample_rate, False))
                    if sample_rate == 0:
                        sample_rate = 1
                    lower_edge_hertz = _unpack_scalar(c_lower_edge, True)
                    upper_edge_hertz = _unpack_scalar(c_upper_edge, True)

                    num_spectrogram_bins = dft_length // 2 + 1

                    mel_low = 2595.0 * math.log10(1.0 + lower_edge_hertz / 700.0)
                    mel_high = 2595.0 * math.log10(1.0 + upper_edge_hertz / 700.0)
                    mel_points = [
                        mel_low + i * (mel_high - mel_low) / (num_mel_bins + 1)
                        for i in range(num_mel_bins + 2)
                    ]

                    hz_points = [700.0 * (10.0 ** (m / 2595.0) - 1.0) for m in mel_points]
                    bin_points = [(dft_length * hz) / sample_rate for hz in hz_points]

                    weights = []
                    for i in range(num_spectrogram_bins):
                        for j in range(num_mel_bins):
                            lower_bin = bin_points[j]
                            center_bin = bin_points[j + 1]
                            upper_bin = bin_points[j + 2]

                            if i < lower_bin or i > upper_bin:
                                weight = 0.0
                            elif i < center_bin:
                                weight = (i - lower_bin) / (center_bin - lower_bin)
                            else:
                                weight = (upper_bin - i) / (upper_bin - center_bin)
                            weights.append(weight)

                    packed_data = struct.pack(f"<{len(weights)}f", *weights)

                    out_name = n.outputs[0].name if hasattr(n.outputs[0], "name") else n.outputs[0]

                    res_c = Constant(
                        name=f"{out_name}_folded",
                        values=packed_data,
                        shape=(num_spectrogram_bins, num_mel_bins),
                        dtype=DType.FLOAT32,
                    )
                    graph.tensors[out_name] = res_c
                    nodes_to_remove.append(n)
                except Exception as e:
                    import logging

                    logging.warning("Failed to fold MelWeightMatrix: %s", e)
                    continue

    for n in nodes_to_remove:
        graph.remove_node(n)

    return graph
