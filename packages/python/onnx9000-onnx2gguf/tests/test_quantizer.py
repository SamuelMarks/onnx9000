"""Module docstring."""

import numpy as np
from onnx9000.onnx2gguf.quantizer import f32_to_f16, quantize_q4_0, quantize_q4_1, quantize_q8_0


def test_quantizer():
    """Provides functional implementation."""
    # 32 floats (1 block)
    data = np.arange(32, dtype=np.float32).tobytes()

    f16 = f32_to_f16(data)
    assert len(f16) == 64

    q4_0 = quantize_q4_0(data)
    assert len(q4_0) == 18

    q4_1 = quantize_q4_1(data)
    assert len(q4_1) == 20

    q8_0 = quantize_q8_0(data)
    assert len(q8_0) == 34
