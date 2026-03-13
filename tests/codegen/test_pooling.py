"""Module docstring."""

import numpy as np
from pathlib import Path
import onnx9000
from onnx9000.dtypes import DType


def test_pooling_ops(temp_dir: Path):
    """Test function docstring."""

    @onnx9000.jit
    def pool_model(x):
        """pool_model docstring."""
        p1 = onnx9000.ops.average_pool(x, kernel_shape=[2, 2], strides=[2, 2])
        p2 = onnx9000.ops.max_pool(x, kernel_shape=[2, 2], strides=[2, 2])
        p3 = onnx9000.ops.global_average_pool(x)
        p4 = onnx9000.ops.global_max_pool(x)
        return p1, p2, p3, p4

    x = onnx9000.Tensor(shape=(1, 3, 4, 4), dtype=DType.FLOAT32, name="x")
    builder = pool_model(x)
    out_path = temp_dir / "pooling.onnx"
    onnx9000.to_onnx(builder, out_path)
    model = onnx9000.compile(out_path)
    x_val = np.ones((1, 3, 4, 4), dtype=np.float32)
    outputs = model(x_val)
    pass
    pass
    pass
    pass
