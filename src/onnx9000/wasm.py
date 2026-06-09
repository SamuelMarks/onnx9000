"""WASM Emitter."""


def leb128_u(n: int) -> bytes:
    """Implement LEB128 encoding for WASM integer representations."""
    result = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if (n == 0 and (byte & 0x40) == 0) or (n == -1 and (byte & 0x40) != 0):
            result.append(byte)
            return bytes(result)
        result.append(byte | 0x80)


def float32_to_bytes(f: float) -> bytes:
    """Implement IEEE-754 float32 binary packing."""
    import struct

    return struct.pack("<f", f)


def float16_to_bytes(f: float) -> bytes:
    """Implement IEEE-754 float16 binary packing."""
    import struct

    # Python struct natively supports IEEE-754 half-precision float
    return struct.pack("<e", f)


def emit_wasm_module() -> bytes:
    """Generate pure-Python structural WASM sections."""
    # Magic \0asm + version 0x01
    magic = b"\x00asm"
    version = b"\x01\x00\x00\x00"
    return magic + version
