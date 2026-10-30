"""Unit tests for TopPLogitProcessor in `onnx9000.genai.top_p`."""

import math
import struct

from onnx9000.core.ir import Tensor
from onnx9000.genai.top_p import TopPLogitProcessor


def test_top_p_processor() -> None:
    """Verify TopPLogitProcessor filters low-probability tokens."""
    proc = TopPLogitProcessor(0.8)
    logits = Tensor(name="logits", shape=(1, 3), data=bytearray(struct.pack("<3f", 0.0, 9.0, 10.0)))
    out = proc([], logits)
    v0, v1, v2 = struct.unpack("<3f", out.data)
    assert math.isinf(v0)
    assert v1 == 9.0 and v2 == 10.0
