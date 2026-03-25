import struct
import numpy as np


def f32_to_f16(data: bytes) -> bytes:
    # 123. Implement C-equivalent Float32 to Float16 downcasting loop natively in JS/Python.
    floats = np.frombuffer(data, dtype=np.float32)
    f16s = floats.astype(np.float16)
    return f16s.tobytes()


def quantize_q4_0(data: bytes) -> bytes:
    # 125. Implement Q4_0 quantization math (block size 32, scale, 4-bit nibbles).
    floats = np.frombuffer(data, dtype=np.float32)
    assert len(floats) % 32 == 0, "Length must be a multiple of 32 for Q4_0"
    num_blocks = len(floats) // 32

    out = bytearray(num_blocks * 18)
    for i in range(num_blocks):
        block = floats[i * 32 : (i + 1) * 32]
        amax = np.max(np.abs(block))
        d = np.float32(amax / 7.0) if amax != 0 else np.float32(0.0)
        id_val = 1.0 / d if d != 0 else 0.0

        # Write d (float16)
        struct.pack_into("<e", out, i * 18, d.astype(np.float16))

        for j in range(16):
            v0 = int(np.round(block[j] * id_val)) + 8
            v1 = int(np.round(block[j + 16] * id_val)) + 8
            v0 = max(0, min(15, v0))
            v1 = max(0, min(15, v1))
            out[i * 18 + 2 + j] = v0 | (v1 << 4)

    return bytes(out)


def quantize_q4_1(data: bytes) -> bytes:
    # 127. Implement Q4_1 quantization math (block size 32, scale, min_val, 4-bit nibbles).
    floats = np.frombuffer(data, dtype=np.float32)
    assert len(floats) % 32 == 0, "Length must be a multiple of 32 for Q4_1"
    num_blocks = len(floats) // 32

    out = bytearray(num_blocks * 20)
    for i in range(num_blocks):
        block = floats[i * 32 : (i + 1) * 32]
        vmin = np.min(block)
        vmax = np.max(block)
        d = np.float32((vmax - vmin) / 15.0) if vmax != vmin else np.float32(0.0)
        id_val = 1.0 / d if d != 0 else 0.0

        # Write d, m (float16)
        struct.pack_into("<e", out, i * 20, d.astype(np.float16))
        struct.pack_into("<e", out, i * 20 + 2, vmin.astype(np.float16))

        for j in range(16):
            v0 = int(np.round((block[j] - vmin) * id_val))
            v1 = int(np.round((block[j + 16] - vmin) * id_val))
            v0 = max(0, min(15, v0))
            v1 = max(0, min(15, v1))
            out[i * 20 + 4 + j] = v0 | (v1 << 4)

    return bytes(out)


def quantize_q8_0(data: bytes) -> bytes:
    # 129. Implement Q8_0 quantization math (block size 32, scale, 8-bit values).
    floats = np.frombuffer(data, dtype=np.float32)
    assert len(floats) % 32 == 0, "Length must be a multiple of 32 for Q8_0"
    num_blocks = len(floats) // 32

    out = bytearray(num_blocks * 34)
    for i in range(num_blocks):
        block = floats[i * 32 : (i + 1) * 32]
        amax = np.max(np.abs(block))
        d = np.float32(amax / 127.0) if amax != 0 else np.float32(0.0)
        id_val = 1.0 / d if d != 0 else 0.0

        # Write d (float16)
        struct.pack_into("<e", out, i * 34, d.astype(np.float16))

        for j in range(32):
            v0 = int(np.round(block[j] * id_val))
            v0 = max(-128, min(127, v0))
            out[i * 34 + 2 + j] = v0 & 0xFF

    return bytes(out)
