"""WebAssembly binary encoding and module emission utilities for ONNX9000.

Provides unsigned LEB128 variable-length integer encoding, IEEE-754 float16 and float32
binary packing, and standalone WebAssembly binary module emission.
"""

import struct

from onnx9000.core.exceptions import CompilationError
from onnx9000.core.ir import Graph


def leb128_u(value: int) -> bytes:
    """Encode an unsigned integer into Variable-Length LEB128 binary representation.

    Args:
        value: Non-negative integer to encode.

    Returns:
        LEB128-encoded bytes.

    Raises:
        ValueError: If `value` is negative.
    """
    if value < 0:
        raise CompilationError(f"leb128_u requires a non-negative integer, got {value}")
    result = bytearray()
    n = value
    while True:
        byte = n & 0x7F
        n >>= 7
        if n == 0:
            result.append(byte)
            return bytes(result)
        result.append(byte | 0x80)


def float32_to_bytes(value: float) -> bytes:
    """Pack a Python float into little-endian IEEE-754 single-precision (float32) bytes.

    Args:
        value: Floating-point scalar to pack.

    Returns:
        4-byte little-endian IEEE-754 binary representation.
    """
    return struct.pack("<f", value)


def float16_to_bytes(value: float) -> bytes:
    """Pack a Python float into little-endian IEEE-754 half-precision (float16) bytes.

    Args:
        value: Floating-point scalar to pack.

    Returns:
        2-byte little-endian IEEE-754 binary representation.
    """
    return struct.pack("<e", value)


def emit_wasm_module(graph: Graph | None = None) -> bytes:
    r"""Generate a valid WebAssembly binary module header and custom graph metadata section.

    Args:
        graph: Optional ONNX9000 Graph to embed as a custom section name.

    Returns:
        WebAssembly binary module bytes starting with `\x00asm\x01\x00\x00\x00`.
    """
    magic = b"\x00asm"
    version = b"\x01\x00\x00\x00"
    if graph is None:
        return magic + version
    name_bytes = graph.name.encode("utf-8")
    payload = leb128_u(len(name_bytes)) + name_bytes
    section = b"\x00" + leb128_u(len(payload)) + payload
    return magic + version + section


__all__ = ["emit_wasm_module", "float16_to_bytes", "float32_to_bytes", "leb128_u"]
