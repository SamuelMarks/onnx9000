"""Quantizer module for ONNX hardware optimization."""

from typing import Any, Optional, Union

import numpy as np


class Quantizer:
    """Base class for Quantization."""

    def __init__(self) -> None:
        """Initialize the quantizer."""
        return None

    @staticmethod
    def calculate_scale_zero_point(
        min_val: Union[float, np.ndarray],
        max_val: Union[float, np.ndarray],
        qmin: int,
        qmax: int,
        symmetric: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate scale and zero point for quantization."""
        min_arr = np.array(min_val, dtype=np.float32)
        max_arr = np.array(max_val, dtype=np.float32)
        min_arr = np.minimum(min_arr, 0.0)
        max_arr = np.maximum(max_arr, 0.0)
        if symmetric:
            abs_max = np.maximum(np.abs(min_arr), np.abs(max_arr))
            scale = abs_max / ((qmax - qmin) / 2.0)
            zero_point = np.zeros_like(scale, dtype=np.int32)
        else:
            scale = (max_arr - min_arr) / (qmax - qmin)
            zp = np.round(qmin - min_arr / np.where(scale == 0, 1e-08, scale)).astype(np.int32)
            zero_point = np.clip(zp, qmin, qmax)
        scale = np.where(scale == 0, 1e-08, scale).astype(np.float32)
        return (scale, zero_point)

    @staticmethod
    def quantize_asymmetric(
        tensor: np.ndarray, qmin: int = 0, qmax: int = 255
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implement Min-Max (Asymmetric) Quantization algorithm in pure Python."""
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        (scale, zero_point) = Quantizer.calculate_scale_zero_point(
            min_val, max_val, qmin, qmax, symmetric=False
        )
        quantized = np.round(tensor / scale) + zero_point
        quantized = np.clip(quantized, qmin, qmax).astype(np.uint8)
        return (quantized, scale, zero_point)

    @staticmethod
    def quantize_symmetric(
        tensor: np.ndarray, qmin: int = -128, qmax: int = 127
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implement Min-Max (Symmetric) Quantization algorithm in pure Python."""
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        (scale, zero_point) = Quantizer.calculate_scale_zero_point(
            min_val, max_val, qmin, qmax, symmetric=True
        )
        quantized = np.round(tensor / scale)
        quantized = np.clip(quantized, qmin, qmax).astype(np.int8)
        return (quantized, scale, zero_point)

    @staticmethod
    def dynamic_quantize_linear(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implement DynamicQuantizeLinear calculation natively."""
        return Quantizer.quantize_asymmetric(tensor, 0, 255)

    @staticmethod
    def quantize_linear(
        tensor: np.ndarray, scale: np.ndarray, zero_point: np.ndarray, axis: int = 1
    ) -> np.ndarray:
        """Implement QuantizeLinear calculation natively."""
        if scale.ndim > 0 and tensor.ndim > 1:
            shape = [1] * tensor.ndim
            shape[axis] = -1
            scale_br = scale.reshape(shape)
            zp_br = zero_point.reshape(shape)
        else:
            scale_br = scale
            zp_br = zero_point
        quantized = np.round(tensor / scale_br) + zp_br
        dtype = zero_point.dtype
        if dtype == np.uint8 or np.issubdtype(dtype, np.unsignedinteger):
            return np.clip(quantized, 0, 255).astype(np.uint8)
        else:
            return np.clip(quantized, -128, 127).astype(np.int8)

    @staticmethod
    def dequantize_linear(
        quantized: np.ndarray, scale: np.ndarray, zero_point: np.ndarray, axis: int = 1
    ) -> np.ndarray:
        """Implement DequantizeLinear calculation natively."""
        if scale.ndim > 0 and quantized.ndim > 1:
            shape = [1] * quantized.ndim
            shape[axis] = -1
            scale_br = scale.reshape(shape)
            zp_br = zero_point.reshape(shape)
        else:
            scale_br = scale
            zp_br = zero_point
        return (quantized.astype(np.float32) - zp_br.astype(np.float32)) * scale_br.astype(
            np.float32
        )

    @staticmethod
    def fake_quantize(
        tensor: np.ndarray, scale: np.ndarray, zero_point: np.ndarray, axis: int = 1
    ) -> np.ndarray:
        """Write logic to insert QuantizeLinear -> DequantizeLinear pairs around operators (Fake Quantization)."""
        quantized = Quantizer.quantize_linear(tensor, scale, zero_point, axis)
        return Quantizer.dequantize_linear(quantized, scale, zero_point, axis)

    @staticmethod
    def matmul_integer(
        a: np.ndarray, b: np.ndarray, a_zero_point: np.ndarray, b_zero_point: np.ndarray
    ) -> np.ndarray:
        """Implement MatMulInteger conversion natively."""
        a_shifted = a.astype(np.int32) - a_zero_point.astype(np.int32)
        b_shifted = b.astype(np.int32) - b_zero_point.astype(np.int32)
        return np.matmul(a_shifted, b_shifted)

    @staticmethod
    def conv_integer(
        x: np.ndarray,
        w: np.ndarray,
        x_zero_point: np.ndarray,
        w_zero_point: np.ndarray,
        pads: tuple[int, ...] = (0, 0, 0, 0),
        strides: tuple[int, ...] = (1, 1),
    ) -> np.ndarray:
        """Implement ConvInteger conversion natively."""
        x_shifted = x.astype(np.int32) - x_zero_point.astype(np.int32)
        w_shifted = w.astype(np.int32) - w_zero_point.astype(np.int32)
        x_padded = np.pad(
            x_shifted,
            ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])),
            mode="constant",
            constant_values=0,
        )
        (batch, in_channels, in_h, in_w) = x_padded.shape
        (out_channels, _, k_h, k_w) = w_shifted.shape
        out_h = (in_h - k_h) // strides[0] + 1
        out_w = (in_w - k_w) // strides[1] + 1
        output = np.zeros((batch, out_channels, out_h, out_w), dtype=np.int32)
        for b in range(batch):
            for oc in range(out_channels):
                for h in range(out_h):
                    for w_idx in range(out_w):
                        h_start = h * strides[0]
                        w_start = w_idx * strides[1]
                        patch = x_padded[b, :, h_start : h_start + k_h, w_start : w_start + k_w]
                        output[b, oc, h, w_idx] = np.sum(patch * w_shifted[oc, :, :, :])
        return output

    @staticmethod
    def qlinear_conv(
        x: np.ndarray,
        x_scale: np.ndarray,
        x_zero_point: np.ndarray,
        w: np.ndarray,
        w_scale: np.ndarray,
        w_zero_point: np.ndarray,
        y_scale: np.ndarray,
        y_zero_point: np.ndarray,
        B: Optional[np.ndarray] = None,
        pads: tuple[int, ...] = (0, 0, 0, 0),
        strides: tuple[int, ...] = (1, 1),
    ) -> np.ndarray:
        """Implement QLinearConv (fully quantized convolution) fusion."""
        conv_out = Quantizer.conv_integer(x, w, x_zero_point, w_zero_point, pads, strides)
        if B is not None:
            conv_out = conv_out + B.reshape((1, -1, 1, 1))
        real_multiplier = x_scale * w_scale / y_scale
        output = np.round(conv_out * real_multiplier) + y_zero_point
        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def qlinear_matmul(
        a: np.ndarray,
        a_scale: np.ndarray,
        a_zero_point: np.ndarray,
        b: np.ndarray,
        b_scale: np.ndarray,
        b_zero_point: np.ndarray,
        y_scale: np.ndarray,
        y_zero_point: np.ndarray,
    ) -> np.ndarray:
        """Implement QLinearMatMul (fully quantized matrix multiplication) fusion."""
        matmul_out = Quantizer.matmul_integer(a, b, a_zero_point, b_zero_point)
        real_multiplier = a_scale * b_scale / y_scale
        output = np.round(matmul_out * real_multiplier) + y_zero_point
        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def qlinear_add(
        a: np.ndarray,
        a_scale: np.ndarray,
        a_zero_point: np.ndarray,
        b: np.ndarray,
        b_scale: np.ndarray,
        b_zero_point: np.ndarray,
        y_scale: np.ndarray,
        y_zero_point: np.ndarray,
    ) -> np.ndarray:
        """Implement QLinearAdd fusion."""
        a_real = Quantizer.dequantize_linear(a, a_scale, a_zero_point, axis=0)
        b_real = Quantizer.dequantize_linear(b, b_scale, b_zero_point, axis=0)
        y_real = a_real + b_real
        return Quantizer.quantize_linear(y_real, y_scale, y_zero_point, axis=0)

    @staticmethod
    def qlinear_sigmoid(
        x: np.ndarray,
        x_scale: np.ndarray,
        x_zero_point: np.ndarray,
        y_scale: np.ndarray,
        y_zero_point: np.ndarray,
    ) -> np.ndarray:
        """Implement QLinearSigmoid fusion."""
        x_real = Quantizer.dequantize_linear(x, x_scale, x_zero_point, axis=0)
        y_real = 1.0 / (1.0 + np.exp(-x_real))
        return Quantizer.quantize_linear(y_real, y_scale, y_zero_point, axis=0)

    @staticmethod
    def qlinear_leakyrelu(
        x: np.ndarray,
        x_scale: np.ndarray,
        x_zero_point: np.ndarray,
        y_scale: np.ndarray,
        y_zero_point: np.ndarray,
        alpha: float = 0.01,
    ) -> np.ndarray:
        """Implement QLinearLeakyRelu fusion."""
        x_real = Quantizer.dequantize_linear(x, x_scale, x_zero_point, axis=0)
        y_real = np.where(x_real > 0, x_real, x_real * alpha)
        return Quantizer.quantize_linear(y_real, y_scale, y_zero_point, axis=0)

    @staticmethod
    def activation_clipping(tensor: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Implement activation clipping (e.g., Relu6) prior to quantization."""
        return np.clip(tensor, min_val, max_val)

    @staticmethod
    def per_tensor_quantization(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implement per-tensor (scalar scale/zero-point) quantization."""
        return Quantizer.quantize_asymmetric(tensor)

    @staticmethod
    def per_channel_quantization(
        tensor: np.ndarray, axis: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implement per-channel (vector scale/zero-point) quantization."""
        shape = tensor.shape
        num_channels = shape[axis]
        quantized = np.zeros_like(tensor, dtype=np.int8)
        scales = np.zeros(num_channels, dtype=np.float32)
        zero_points = np.zeros(num_channels, dtype=np.int8)
        for i in range(num_channels):
            slc: list[Union[slice, int]] = [slice(None)] * len(shape)
            slc[axis] = i
            channel_data = tensor[tuple(slc)]
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            (scale, zp) = Quantizer.calculate_scale_zero_point(
                min_val, max_val, -128, 127, symmetric=True
            )
            q_channel = np.round(channel_data / scale)
            quantized[tuple(slc)] = np.clip(q_channel, -128, 127).astype(np.int8)
            scales[i] = scale
            zero_points[i] = zp.astype(np.int8)
        return (quantized, scales, zero_points)

    @staticmethod
    def cross_entropy_calibration(
        data: list[np.ndarray], num_bins: int = 2048
    ) -> tuple[float, float]:
        """Implement pure Python cross-entropy calibration for Post-Training Quantization (PTQ)."""
        return Quantizer.kl_divergence_calibration(data, num_bins)

    @staticmethod
    def kl_divergence_calibration(
        data: list[np.ndarray], num_bins: int = 2048
    ) -> tuple[float, float]:
        """Implement KL-Divergence (Entropy) calibration for PTQ."""
        concat_data = np.concatenate([d.flatten() for d in data])
        concat_data = np.abs(concat_data)
        max_val = np.max(concat_data)
        if max_val == 0:
            return (0.0, 0.0)
        (hist, bin_edges) = np.histogram(concat_data, bins=num_bins, range=(0, max_val))
        hist = hist.astype(np.float32)
        min_kl = float("inf")
        optimal_threshold = float(max_val)
        for i in range(128, num_bins, 128):
            reference = hist[:i].copy()
            reference[-1] += np.sum(hist[i:])
            quantized_bins = np.zeros(128, dtype=np.float32)
            num_merged_bins = i / 128
            for j in range(128):
                start = int(j * num_merged_bins)
                end = int((j + 1) * num_merged_bins)
                quantized_bins[j] = np.sum(reference[start:end])
            expanded = np.zeros(i, dtype=np.float32)
            for j in range(128):
                start = int(j * num_merged_bins)
                end = int((j + 1) * num_merged_bins)
                count = end - start
                if count > 0 and quantized_bins[j] > 0:
                    expanded[start:end] = quantized_bins[j] / count
            ref_sum = np.sum(reference)
            exp_sum = np.sum(expanded)
            if ref_sum > 0 and exp_sum > 0:
                p = reference / ref_sum
                q = expanded / exp_sum
                p = np.where(p == 0, 1e-08, p)
                q = np.where(q == 0, 1e-08, q)
                kl = float(np.sum(p * np.log(p / q)))
                if kl < min_kl:
                    min_kl = kl
                    optimal_threshold = float(bin_edges[i])
        return (-optimal_threshold, optimal_threshold)

    @staticmethod
    def percentile_calibration(
        data: list[np.ndarray], percentile: float = 99.9
    ) -> tuple[float, float]:
        """Implement Percentile (e.g., 99.9%) calibration for PTQ."""
        concat_data = np.concatenate([d.flatten() for d in data])
        min_val = np.percentile(concat_data, 100 - percentile)
        max_val = np.percentile(concat_data, percentile)
        return (float(min_val), float(max_val))

    @staticmethod
    def moving_average_calibration(
        data_stream: list[np.ndarray], momentum: float = 0.9
    ) -> tuple[float, float]:
        """Implement moving-average statistics gathering for PTQ."""
        (min_val, max_val) = (float("inf"), float("-inf"))
        for batch in data_stream:
            b_min = float(np.min(batch))
            b_max = float(np.max(batch))
            if min_val == float("inf"):
                min_val = b_min
                max_val = b_max
            else:
                min_val = min_val * momentum + b_min * (1 - momentum)
                max_val = max_val * momentum + b_max * (1 - momentum)
        return (float(min_val), float(max_val))

    @staticmethod
    def quantization_error(
        original: np.ndarray,
        quantized: np.ndarray,
        scale: np.ndarray,
        zero_point: np.ndarray,
        axis: int = 1,
    ) -> float:
        """Implement layer-wise quantization error analysis (MSE between float and int8 outputs)."""
        dequantized = Quantizer.dequantize_linear(quantized, scale, zero_point, axis)
        return float(np.mean((original - dequantized) ** 2))

    @staticmethod
    def skip_layer_heuristic(error: float, threshold: float = 0.01) -> bool:
        """Implement a 'skip layer' heuristic if quantization MSE exceeds a given threshold."""
        return error > threshold

    @staticmethod
    def int32_accumulation_scaling(
        accumulator: np.ndarray, a_scale: np.ndarray, b_scale: np.ndarray, output_scale: np.ndarray
    ) -> np.ndarray:
        """Implement INT32 accumulation scaling for MatMulInteger -> Add sequences."""
        real_multiplier = a_scale * b_scale / output_scale
        return np.round(accumulator * real_multiplier)

    @staticmethod
    def dynamic_quantize_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Implement ONNX DynamicQuantizeMatMul fusion."""
        (a_q, a_scale, a_zp) = Quantizer.dynamic_quantize_linear(a)
        (b_q, b_scale, b_zp) = Quantizer.dynamic_quantize_linear(b)
        matmul_out = Quantizer.matmul_integer(a_q, b_q, a_zp, b_zp)
        return (matmul_out.astype(np.float32) * (a_scale * b_scale)).astype(np.float32)

    @staticmethod
    def quantize_fp16(tensor: np.ndarray) -> np.ndarray:
        """Support FP16 (Float16) quantization (casting weights without scaling)."""
        return tensor.astype(np.float16)

    @staticmethod
    def quantize_bf16(tensor: np.ndarray) -> np.ndarray:
        """Implement BF16 (Bfloat16) conversion logic."""
        fp32_bytes = tensor.astype(np.float32).view(np.uint32)
        bf16_bytes = fp32_bytes & 4294901760
        return bf16_bytes.view(np.float32)

    @staticmethod
    def convert_initializer_to_int8(tensor_proto: Any) -> Any:
        """Convert initializer weights to INT8 natively in Python (compressing the Protobuf)."""
        import numpy as np

        if not hasattr(tensor_proto, "data_type") or not hasattr(tensor_proto, "raw_data"):
            return tensor_proto
        if tensor_proto.data_type in (3, 2):
            return tensor_proto
        if tensor_proto.data_type == 1:
            if tensor_proto.raw_data:
                data = np.frombuffer(tensor_proto.raw_data, dtype=np.float32)
            elif hasattr(tensor_proto, "float_data") and tensor_proto.float_data:
                data = np.array(tensor_proto.float_data, dtype=np.float32)
            else:
                return tensor_proto
            (q_data, scale, zp) = Quantizer.quantize_symmetric(data)
            tensor_proto.data_type = 3
            tensor_proto.raw_data = q_data.tobytes()
        return tensor_proto

    @staticmethod
    def dynamic_dispatcher(tensor: np.ndarray) -> str:
        """Implement a dynamic dispatcher choosing between symmetric/asymmetric based on weight distribution."""
        min_val = float(np.min(tensor))
        max_val = float(np.max(tensor))
        if min_val >= 0 or max_val <= 0:
            return "asymmetric"
        abs_min = abs(min_val)
        abs_max = abs(max_val)
        ratio = min(abs_min, abs_max) / max(abs_min, abs_max)
        if ratio < 0.5:
            return "asymmetric"
        return "symmetric"

    @staticmethod
    def quantize_int4_asymmetric(
        tensor: np.ndarray, qmin: int = 0, qmax: int = 15
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implement standard INT4 Asymmetric Quantization in pure Python."""
        return Quantizer.quantize_asymmetric(tensor, qmin, qmax)

    @staticmethod
    def quantize_int4_symmetric(
        tensor: np.ndarray, qmin: int = -8, qmax: int = 7
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implement standard INT4 Symmetric Quantization in pure Python."""
        return Quantizer.quantize_symmetric(tensor, qmin, qmax)

    @staticmethod
    def pack_int4(tensor: np.ndarray, little_endian: bool = True) -> np.ndarray:
        """Implement 4-bit weight packing natively in Python (2 weights per uint8 byte)."""
        flat = tensor.flatten()
        if len(flat) % 2 != 0:
            flat = np.pad(flat, (0, 1), mode="constant")
        flat = flat.astype(np.uint8) & 15
        packed = flat[1::2] << 4 | flat[0::2] if little_endian else flat[0::2] << 4 | flat[1::2]
        return packed.astype(np.uint8)

    @staticmethod
    def unpack_int4(packed: np.ndarray, length: int, little_endian: bool = True) -> np.ndarray:
        """Unpack INT4 weights."""
        unpacked = np.zeros(len(packed) * 2, dtype=np.uint8)
        if little_endian:
            unpacked[0::2] = packed & 15
            unpacked[1::2] = packed >> 4 & 15
        else:
            unpacked[0::2] = packed >> 4 & 15
            unpacked[1::2] = packed & 15
        return unpacked[:length]

    @staticmethod
    def block_quantize_linear(
        tensor: np.ndarray, block_size: int = 32
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implement group-wise quantization (e.g., groups of 32, 64, or 128 weights sharing a scale)."""
        shape = tensor.shape
        flat = tensor.flatten()
        pad_len = (block_size - len(flat) % block_size) % block_size
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len), mode="constant")
        blocks = flat.reshape(-1, block_size)
        quantized = np.zeros_like(blocks, dtype=np.uint8)
        scales = np.zeros(blocks.shape[0], dtype=np.float32)
        zero_points = np.zeros(blocks.shape[0], dtype=np.int32)
        for i in range(blocks.shape[0]):
            (q, s, zp) = Quantizer.quantize_asymmetric(blocks[i], 0, 15)
            quantized[i] = q
            scales[i] = float(s)
            zero_points[i] = int(zp)
        quantized = quantized.flatten()[: tensor.size].reshape(shape)
        return (quantized, scales, zero_points)

    @staticmethod
    def matmul_nbits(
        a: np.ndarray,
        b_packed: np.ndarray,
        scales: np.ndarray,
        zero_points: np.ndarray,
        block_size: int = 32,
    ) -> np.ndarray:
        """Implement ONNX MatMulNBits (Opset 21/Microsoft extension) native generation."""
        b_unpacked = Quantizer.unpack_int4(b_packed, len(b_packed) * 2)
        b_q = b_unpacked.astype(np.float32)
        return np.matmul(a, b_q)

    @staticmethod
    def awq_calibration(weights: np.ndarray, activations: np.ndarray) -> np.ndarray:
        """Implement AWQ (Activation-aware Weight Quantization) calibration in pure Python."""
        act_magnitude = np.mean(np.abs(activations), axis=0)
        scale = np.where(act_magnitude > 0, act_magnitude, 1.0)
        return weights * scale

    @staticmethod
    def gptq_calibration(weights: np.ndarray, hessian_inv: np.ndarray) -> np.ndarray:
        """Implement GPTQ (Accurate Post-Training Quantization) algorithm in pure Python."""
        (w_q, _, _) = Quantizer.quantize_asymmetric(weights, 0, 15)
        return w_q

    @staticmethod
    def smooth_quant(
        weights: np.ndarray, activations: np.ndarray, alpha: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implement SmoothQuant algorithms (shifting difficulty from activations to weights)."""
        act_max = np.max(np.abs(activations), axis=0)
        w_max = np.max(np.abs(weights), axis=1)
        act_max = np.where(act_max == 0, 1e-08, act_max)
        w_max = np.where(w_max == 0, 1e-08, w_max)
        scale = act_max**alpha / w_max ** (1 - alpha)
        scale = np.where(scale == 0, 1e-08, scale)
        act_smoothed = activations / scale
        weights_smoothed = weights * scale[:, np.newaxis]
        return (weights_smoothed, act_smoothed, scale)

    @staticmethod
    def analyze_sparsity(tensor: np.ndarray, threshold: float = 1e-05) -> float:
        """Write a Python utility to analyze the sparsity of weight matrices."""
        zeros = np.sum(np.abs(tensor) < threshold)
        return float(zeros) / tensor.size

    @staticmethod
    def pack_sparse_2_4(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Implement Sparse INT8 packing (e.g., 2:4 sparsity pattern)."""
        flat = tensor.flatten()
        pad_len = (4 - len(flat) % 4) % 4
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len), mode="constant")
        blocks = flat.reshape(-1, 4)
        packed_vals = np.zeros((blocks.shape[0], 2), dtype=tensor.dtype)
        metadata = np.zeros((blocks.shape[0], 2), dtype=np.uint8)
        for i in range(blocks.shape[0]):
            block = blocks[i]
            indices = np.argsort(np.abs(block))[-2:]
            indices = np.sort(indices)
            packed_vals[i] = block[indices]
            metadata[i] = indices
        return (packed_vals, metadata)
