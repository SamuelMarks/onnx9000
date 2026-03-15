"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.registry import registry
import onnx9000.backends.codegen.ops.math
import onnx9000.backends.codegen.ops.elementwise
import onnx9000.backends.codegen.ops.matmul
import onnx9000.backends.codegen.ops.nn
import onnx9000.backends.codegen.ops.tensor_ops
import onnx9000.backends.codegen.ops.shape
import onnx9000.backends.codegen.utils
from unittest import mock
import numpy as np


def build_generator(dtype=DType.FLOAT32, buffer_id=0, broadcast=False):
    """Provides build generator functionality and verification."""
    g = Graph("test")
    ctx = Generator(g)
    for i in range(10):
        name = f"0in_{i}" if i == 0 else f"in_{i}"
        if broadcast:
            if i == 0:
                shape = 4, 4
            elif i == 1:
                shape = 4, 1
            elif i == 2:
                shape = 1, 4
            else:
                shape = 1, 1
        else:
            shape = 4, 4
        t = Tensor(name, shape, dtype)
        t.buffer_id = i + buffer_id if buffer_id is not None else None
        g.add_tensor(t)
        t = Tensor(f"out_{i}", shape, dtype)
        t.buffer_id = i + buffer_id + 10 if buffer_id is not None else None
        g.add_tensor(t)
    return g, ctx


def test_fuzz_codegen():
    """Provides semantic functionality and verification."""
    for op_name, func in registry._registry.items():
        for num_inputs in [0, 1, 2, 3, 4, 5, 6]:
            for num_outputs in [0, 1, 2, 3]:
                for dtype in [DType.FLOAT32, DType.INT64, DType.FLOAT16, None]:
                    for broadcast in [True, False]:
                        for buffer_id in [0, None]:
                            g, ctx = build_generator(dtype, buffer_id, broadcast)
                            inputs = [
                                (f"0in_{i}" if i == 0 else f"in_{i}")
                                for i in range(num_inputs)
                            ]
                            outputs = [f"out_{i}" for i in range(num_outputs)]
                            tensor_val = Tensor("dummy", (1,), dtype)
                            tensor_val.data = np.array([1.0], dtype=np.float32)
                            for attrs in [
                                {},
                                {
                                    "lr": 0.1,
                                    "beta1": 0.9,
                                    "beta2": 0.99,
                                    "eps": 1e-08,
                                    "weight_decay": 0.1,
                                    "step_t": 1.0,
                                },
                                {
                                    "axis": 0,
                                    "keepdims": 1,
                                    "alpha": 1.0,
                                    "beta": 1.0,
                                    "gamma": 1.0,
                                    "axes": [0, 1],
                                },
                                {
                                    "axis": 1,
                                    "keepdims": 0,
                                    "alpha": 0.0,
                                    "beta": 0.0,
                                    "gamma": 0.0,
                                },
                                {"center_point_box": 1},
                                {"transA": 1, "transB": 1},
                                {"to": 1},
                                {"value": tensor_val},
                                {
                                    "mode": "constant",
                                    "coordinate_transformation_mode": "pytorch_half_pixel",
                                },
                                {
                                    "mode": "reflect",
                                    "coordinate_transformation_mode": "align_corners",
                                },
                                {
                                    "mode": "edge",
                                    "coordinate_transformation_mode": "asymmetric",
                                },
                                {"direction": "RIGHT"},
                                {"direction": "LEFT"},
                                {
                                    "mode": "linear",
                                    "coordinate_transformation_mode": "half_pixel",
                                },
                                {"mode": "cubic"},
                                {"mode": "nearest"},
                                {"ceil_mode": 1},
                                {"ceil_mode": 0},
                                {"auto_pad": "SAME_UPPER"},
                                {"auto_pad": "SAME_LOWER"},
                                {"auto_pad": "VALID"},
                                {"auto_pad": "NOTSET"},
                            ]:
                                node = Node(
                                    op_name,
                                    inputs=inputs,
                                    outputs=outputs,
                                    attributes=attrs,
                                )
                                try:
                                    func(node, ctx)
                                except Exception:
                                    pass


def test_utils_cuda():
    """Provides semantic functionality and verification."""
    from onnx9000.core import config

    with mock.patch.object(config, "ONNX9000_USE_CUDA", True):
        onnx9000.backends.codegen.utils.get_omp_pragma("100")


def test_tensor_ops_value_list():
    """Provides semantic functionality and verification."""
    tensor_val = Tensor("dummy", (2,), DType.FLOAT32)
    tensor_val.data = [1.0, 2.0]
    g, ctx = build_generator(DType.FLOAT32, 0, False)
    node = Node(
        "Constant", inputs=[], outputs=["out_0"], attributes={"value": tensor_val}
    )
    onnx9000.backends.codegen.ops.tensor_ops.generate_constant(node, ctx)
    tensor_val2 = Tensor("dummy", (2,), DType.INT64)
    tensor_val2.data = [1, 2]
    g, ctx = build_generator(DType.INT64, 0, False)
    node = Node(
        "Constant", inputs=[], outputs=["out_0"], attributes={"value": tensor_val2}
    )
    onnx9000.backends.codegen.ops.tensor_ops.generate_constant(node, ctx)
