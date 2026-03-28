"""TFLite quantization utilities and weight quantizers."""

import logging
import struct
from typing import Dict, List
from onnx9000.core.ir import Graph, Tensor
from ..flatbuffer.schema import QuantizationParameters

logger = logging.getLogger(__name__)


class TensorQuantization:
    """Container for tensor quantization parameters."""

    def __init__(
        self,
        min: List[float],
        max: List[float],
        scale: List[float],
        zero_point: List[int],
        quantized_dimension: int,
    ):
        """Initialize TensorQuantization parameters."""
        self.min = min
        self.max = max
        self.scale = scale
        self.zero_point = zero_point
        self.quantized_dimension = quantized_dimension


class Quantizer:
    """TFLite model quantizer."""

    def __init__(self, graph: Graph, mode: str = "none"):
        """Initialize the Quantizer with a graph and quantization mode."""
        self.graph = graph
        self.mode = mode
        self.quantization_map: Dict[str, TensorQuantization] = {}

    def get_quantization_offset(self, builder, tensor: Tensor) -> int:
        """Create and return the FlatBuffer offset for a tensor's quantization parameters."""
        q = self.quantization_map.get(tensor.name)
        if not q:
            return 0

        min_offset = 0
        max_offset = 0
        scale_offset = 0
        zp_offset = 0

        if q.min:
            builder.start_vector(4, len(q.min), 4)
            for val in reversed(q.min):
                builder.add_float32(val)
            min_offset = builder.end_vector(len(q.min))

        if q.max:
            builder.start_vector(4, len(q.max), 4)
            for val in reversed(q.max):
                builder.add_float32(val)
            max_offset = builder.end_vector(len(q.max))

        if q.scale:
            builder.start_vector(4, len(q.scale), 4)
            for val in reversed(q.scale):
                builder.add_float32(val)
            scale_offset = builder.end_vector(len(q.scale))

        if q.zero_point:
            builder.start_vector(8, len(q.zero_point), 8)
            for val in reversed(q.zero_point):
                lower = val & 0xFFFFFFFF
                if lower > 2147483647:
                    lower -= 4294967296
                upper = (val >> 32) & 0xFFFFFFFF
                if upper > 2147483647:
                    upper -= 4294967296
                builder.add_int32(upper)
                builder.add_int32(lower)
            zp_offset = builder.end_vector(len(q.zero_point))

        return QuantizationParameters.create(
            builder, min_offset, max_offset, scale_offset, zp_offset, 0, 0, q.quantized_dimension
        )

    def quantize(self) -> None:
        """Apply quantization to the graph based on the selected mode."""
        if self.mode == "none":
            return
        if self.mode == "fp16":
            self.quantize_fp16()
        elif self.mode == "int8":
            self.quantize_int8()

    def quantize_fp16(self) -> None:
        """Quantize the graph to FLOAT16."""
        # 241. Downcast FLOAT32 FlatBuffer arrays entirely to FLOAT16 bytes explicitly for FP16 models.
        for name, tensor in self.graph.tensors.items():
            if tensor.dtype == "float32" and tensor.is_initializer and tensor.data is not None:
                tensor.dtype = "float16"
                tensor.data = self._float32_to_float16_bytes(tensor.data)

    def _float32_to_float16_bytes(self, f32_bytes: bytes) -> bytes:
        """Convert float32 bytes to float16 bytes."""
        num_floats = len(f32_bytes) // 4
        f32_array = struct.unpack(f"<{num_floats}f", f32_bytes)

        f16_array = []
        for val in f32_array:
            f16_array.append(self._to_half(val))

        return struct.pack(f"<{num_floats}H", *f16_array)

    def _to_half(self, val: float) -> int:
        """Convert a single float32 to float16 (represented as an int)."""
        # Simplistic IEEE 754 float32 to float16 packing
        x = struct.unpack("<I", struct.pack("<f", val))[0]

        bits = (x >> 16) & 0x8000
        m = (x >> 12) & 0x07FF
        e = (x >> 23) & 0xFF

        if e < 103:
            return bits
        if e > 142:
            bits |= 0x7C00
            bits |= 1 if (e == 255 and (x & 0x007FFFFF)) else 0
            return bits & 0xFFFF
        if e < 113:
            m |= 0x0800
            bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1)
            return bits & 0xFFFF

        bits |= ((e - 112) << 10) | (m >> 1)
        bits += m & 1
        return bits & 0xFFFF

    def quantize_int8(self) -> None:
        """Quantize the graph to INT8."""
        has_uint8 = False
        has_int16 = False
        minmax_extracted = 0

        for i in range(len(self.graph.nodes)):
            node = self.graph.nodes[i]
            if node.op_type in ("QuantizeLinear", "DynamicQuantizeLinear"):
                x = node.inputs[0] if len(node.inputs) > 0 else None
                y_scale = node.inputs[1] if len(node.inputs) > 1 else None
                y_zero_point = node.inputs[2] if len(node.inputs) > 2 else None
                y = node.outputs[0] if node.outputs else None

                if x and y_scale and y_zero_point and y:
                    scale_tensor = self.graph.tensors.get(y_scale)
                    zp_tensor = self.graph.tensors.get(y_zero_point)

                    if scale_tensor and zp_tensor and scale_tensor.data and zp_tensor.data:
                        scale_data = list(
                            struct.unpack(f"<{len(scale_tensor.data) // 4}f", scale_tensor.data)
                        )

                        zp_type = zp_tensor.dtype
                        if zp_type == "uint8":
                            has_uint8 = True
                        elif zp_type == "int16":
                            has_int16 = True

                        zp_format = (
                            "b" if zp_type == "int8" else ("h" if zp_type == "int16" else "B")
                        )
                        zp_data = list(
                            struct.unpack(
                                f"<{len(zp_tensor.data) // struct.calcsize(zp_format)}{zp_format}",
                                zp_tensor.data,
                            )
                        )

                        axis_attr = node.attributes.get("axis")
                        axis = axis_attr.value if axis_attr else 0

                        q = TensorQuantization(
                            min=[],
                            max=[],
                            scale=scale_data,
                            zero_point=zp_data,
                            quantized_dimension=axis,
                        )

                        if len(q.scale) == 1 and len(q.zero_point) == 1:
                            s = q.scale[0]
                            z = q.zero_point[0]
                            q_min = (
                                0
                                if zp_type == "uint8"
                                else (-32768 if zp_type == "int16" else -128)
                            )
                            q_max = (
                                255
                                if zp_type == "uint8"
                                else (32767 if zp_type == "int16" else 127)
                            )

                            min_bound = (q_min - z) * s
                            max_bound = (q_max - z) * s

                            fused = node.attributes.get("fused_activation")
                            if fused and fused.value == "Relu":
                                min_bound = max(0.0, min_bound)
                            elif fused and fused.value == "Relu6":
                                min_bound = max(0.0, min_bound)
                                max_bound = min(6.0, max_bound)

                            q.min = [min_bound]
                            q.max = [max_bound]
                            minmax_extracted += 1

                        if len(q.scale) > 1 and node.op_type != "QuantizeLinear":
                            logger.warning(
                                f"[onnx2tf] EdgeTPU Warning: Per-channel quantization on node {node.name} might cause compilation failures if not aligned correctly."
                            )

                        self.quantization_map[y] = q

        if has_uint8:
            logger.info("[onnx2tf] Notice: Generating legacy UINT8 quantization schema.")
        if has_int16:
            logger.info("[onnx2tf] Notice: Generating INT16x8 mixed precision quantization schema.")
        if minmax_extracted > 0:
            logger.info(
                f"[onnx2tf] Notice: Embedded MinMax quantization fallbacks for {minmax_extracted} tensors."
            )

        logger.warning(
            "[onnx2tf] Warning: INT8 AST lowering is experimental. Ensure your model uses standard QuantizeLinear/DequantizeLinear."
        )
