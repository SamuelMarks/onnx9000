"""Unit tests for LogitProcessorList in `onnx9000.genai.logit_processor_list`."""

import struct

from onnx9000.core.ir import Tensor
from onnx9000.genai.logit_processor_list import LogitProcessorList
from onnx9000.genai.logit_processors import TemperatureLogitProcessor


def test_logit_processor_list() -> None:
    """Verify LogitProcessorList chains processors sequentially."""
    buf = bytearray(struct.pack("<2f", 2.0, 4.0))
    logits = Tensor(name="logits", shape=(1, 2), data=buf)
    proc_list = LogitProcessorList([TemperatureLogitProcessor(2.0)])
    out = proc_list([], logits)
    v0, v1 = struct.unpack("<2f", out.data)
    assert v0 == 1.0 and v1 == 2.0
