"""Utilities for unpacking raw ONNX tensor bytes into C-compatible string literals."""

import struct

from onnx9000.core.dtypes import DType


def dtype_to_struct_fmt(dtype: DType) -> str:
    """Map an ONNX DType to a Python struct format character.

    Args:
        dtype: The ONNX DType to map.

    Returns:
        The corresponding struct format character.
    """
    mapping = {
        DType.FLOAT32: "f",
        DType.FLOAT64: "d",
        DType.INT8: "b",
        DType.UINT8: "B",
        DType.INT16: "h",
        DType.UINT16: "H",
        DType.INT32: "i",
        DType.UINT32: "I",
        DType.INT64: "q",
        DType.UINT64: "Q",
        DType.BOOL: "B",
    }
    return mapping.get(dtype, "B")


def unpack_bytes_to_str(data: bytes, dtype: DType, force_float32: bool = True) -> str:
    """Convert raw bytes into a comma-separated string of C literals.

    This function handles various data types, bit-packing for booleans,
    bfloat16 to float32 conversion, and clamps subnormal floats to zero.

    Args:
        data: Raw byte data from a tensor.
        dtype: The ONNX DType of the data.
        force_float32: Whether to downcast double precision floats to single precision.

    Returns:
        A formatted string of C literals suitable for inclusion in a C array.
    """
    # 219: Bit-packed boolean arrays
    if dtype == DType.BOOL and len(data) > 0:
        values = list(struct.unpack("<" + ("B" * len(data)), data))
        packed = []
        for i in range(0, len(values), 8):
            chunk = values[i : i + 8]
            byte = 0
            for j, bit in enumerate(chunk):
                if bit:
                    byte |= 1 << j
            packed.append(f"0x{byte:02X}")
        return ",\n".join([", ".join(packed[i : i + 16]) for i in range(0, len(packed), 16)])

    # 249: Extract and parse bfloat16 to float32
    if dtype == DType.BFLOAT16:
        values = []
        for i in range(0, len(data), 2):
            bf16_val = struct.unpack("<H", data[i : i + 2])[0]
            # Convert bfloat16 to float32 by padding with 16 zeros
            f32_val = struct.unpack("<f", struct.pack("<I", bf16_val << 16))[0]
            values.append(f32_val)
        dtype = DType.FLOAT32
    # 231: Handle float64 fallback cleanly
    elif dtype == DType.FLOAT64 and force_float32:
        num_elements = len(data) // 8
        values = list(struct.unpack("<" + ("d" * num_elements), data))
        # downcast in python memory
        dtype = DType.FLOAT32
    else:
        fmt_char = dtype_to_struct_fmt(dtype)
        if len(data) % struct.calcsize(fmt_char) != 0:
            values = list(struct.unpack("<" + ("B" * len(data)), data))
            str_values = [f"0x{v:02X}" for v in values]
            return ",\n".join(
                [", ".join(str_values[i : i + 16]) for i in range(0, len(str_values), 16)]
            )
        else:
            fmt_str = "<" + (fmt_char * (len(data) // struct.calcsize(fmt_char)))
            values = list(struct.unpack(fmt_str, data))

    # 217: Identify sub-normal float ranges and clamp to zero
    if dtype in (DType.FLOAT32, DType.FLOAT64):
        values = list(values)
        for i in range(len(values)):
            if -1e-32 < values[i] < 1e-32:
                values[i] = 0.0

    if dtype == DType.FLOAT32:
        # 265: Ensure deterministic float formatting
        str_values = [
            f"{v:g}f" if "." in f"{v:g}" or "e" in f"{v:g}" else f"{v:g}.0f" for v in values
        ]
    elif dtype == DType.FLOAT64:
        str_values = [
            f"{v:g}" if "." in f"{v:g}" or "e" in f"{v:g}" else f"{v:g}.0" for v in values
        ]
    else:
        str_values = [str(v) for v in values]

    lines = []
    chunk_size = 16
    for i in range(0, len(str_values), chunk_size):
        lines.append(", ".join(str_values[i : i + chunk_size]))

    return ",\n".join(lines)
