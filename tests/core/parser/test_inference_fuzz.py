"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.core.parser.inference import _INFERENCE_RULES
import numpy as np


def build_graph(dtype=DType.FLOAT32, pre_add_out=False, out_name="out_1"):
    """Provides build graph functionality and verification."""
    g = Graph("test")
    shapes = {
        "scalar": (1,),
        "vec": (4,),
        "mat": (4, 4),
        "mat2": (4, 1),
        "mat3": (1, 4),
        "mat4": (3, 4),
        "bad_mat": (5, 5),
        "none": None,
        "empty": (0,),
        "1d": (2,),
        "2d": (2, 2),
        "3d": (2, 2, 2),
        "4d": (2, 2, 2, 2),
        "5d": (2, 2, 2, 2, 2),
    }
    for name, shp in shapes.items():
        t = Tensor(name, shp, dtype)
        if shp is not None:
            t.data = np.zeros(shp, dtype=np.float32)
        g.add_tensor(t)
    if pre_add_out:
        g.add_tensor(Tensor(out_name, None, None))
    return g


def test_inference_fuzz():
    """Provides semantic functionality and verification."""
    for op_name, func in _INFERENCE_RULES.items():
        for pre_add_out in [True, False]:
            for input_names in [
                ["mat", "mat"],
                ["mat", "mat2"],
                ["mat2", "mat"],
                ["mat", "bad_mat"],
                ["mat", "scalar"],
                ["scalar", "mat"],
                ["vec", "mat"],
                ["mat", "vec"],
                ["4d", "4d"],
                ["4d", "1d"],
                ["none", "none"],
                ["mat", "none"],
                ["none", "mat"],
                [],
                ["mat"],
                ["mat", "mat", "mat"],
                ["4d", "2d"],
                ["3d", "2d"],
                ["5d", "5d"],
                ["mat", "mat", "mat", "mat", "mat"],
            ]:
                for out_name in ["out_1"]:
                    for attrs in [
                        {},
                        {"axis": 0},
                        {"axis": 1},
                        {"axis": -1},
                        {"axis": 2},
                        {"axes": [0, 1]},
                        {"axes": [1]},
                        {"axes": [-1]},
                        {"axes": [2]},
                        {"keepdims": 0},
                        {"keepdims": 1},
                        {"perm": [1, 0]},
                        {"perm": [0, 2, 1, 3]},
                        {"dilations": [1, 1]},
                        {"strides": [1, 1], "pads": [0, 0, 0, 0]},
                        {"kernel_shape": [3, 3]},
                        {"group": 1},
                        {"group": 2},
                        {"transA": 1, "transB": 1},
                        {"direction": "forward"},
                        {"direction": "reverse"},
                        {"direction": "bidirectional"},
                        {"to": DType.INT64},
                        {"to": DType.FLOAT16},
                        {"blocksize": 2, "mode": "DCR"},
                        {"center_point_box": 1},
                        {"value": 1.0},
                    ]:
                        g = build_graph(pre_add_out=pre_add_out, out_name=out_name)
                        node = Node(
                            op_name,
                            inputs=input_names,
                            outputs=[out_name],
                            attributes=attrs,
                        )
                        try:
                            func(node, g)
                        except Exception:
                            pass
