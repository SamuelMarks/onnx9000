"""Module docstring."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType
import pytest


def test_sequence_ops(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def seq_model(x, pos):
        """seq_model docstring."""
        seq = onnx9000.ops.sequence_construct([x])
        out = onnx9000.ops.sequence_at(seq, pos)
        s_e = onnx9000.ops.sequence_empty(1)
        s_del = onnx9000.ops.sequence_erase(seq, pos)
        s_ins = onnx9000.ops.sequence_insert(seq, x, pos)
        s_len = onnx9000.ops.sequence_length(seq)
        s_map = onnx9000.ops.sequence_map(seq)
        c = onnx9000.ops.concat_from_sequence(seq, axis=0)
        s = onnx9000.ops.split_to_sequence(x, axis=0)
        return out, s_e, s_del, s_ins, s_len, s_map, c, s

    x = onnx9000.Tensor(shape=(4,), dtype=DType.FLOAT32, name="x")
    pos = onnx9000.Tensor(shape=(1,), dtype=DType.INT64, name="pos")
    builder = seq_model(x, pos)
    out_path = temp_dir / "sequence.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    pos_val = np.array([0], dtype=np.int64)
    outputs = model(x_val, pos_val)
    assert len(outputs) == 8
    assert outputs[0].dtype == np.float32
    assert outputs[4].dtype == np.int64
