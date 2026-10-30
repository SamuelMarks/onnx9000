"""Unit tests for SequenceTensorUtils in `onnx9000.genai.tensor_utils`."""

import pytest
from onnx9000.core.ir import Tensor
from onnx9000.genai.tensor_utils import SequenceTensorUtils


def test_sequence_tensor_utils() -> None:
    """Verify SequenceTensorUtils.expand_sequence_dimension expands sequence axis and validates rank."""
    t = Tensor(name="kv", shape=(1, 2, 4), data=bytearray(32))
    expanded = SequenceTensorUtils.expand_sequence_dimension(t, 4)
    assert expanded.shape == (1, 4, 4)
    assert len(expanded.data) == 64

    with pytest.raises(ValueError):
        SequenceTensorUtils.expand_sequence_dimension(Tensor(name="bad", shape=(4,)), 8)
