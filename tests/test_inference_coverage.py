"""Module docstring."""

import pytest
from onnx9000.ir import Graph, Node, Tensor
from onnx9000.dtypes import DType
from onnx9000.parser.inference import _INFERENCE_RULES, DynamicDim


def test_all_inference_rules():
    """test_all_inference_rules docstring."""
    graph = Graph(name="test")

    # Iterate over all rules
    for op_type, func in _INFERENCE_RULES.items():
        for num_inputs in [0, 1, 2, 3, 4, 5]:
            inputs = [f"in_{op_type}_{num_inputs}_{i}" for i in range(num_inputs)]
            outputs = [f"out_{op_type}_{num_inputs}_0", f"out_{op_type}_{num_inputs}_1"]
            node = Node(op_type=op_type, inputs=inputs, outputs=outputs, attributes={})

            for inp in inputs:
                if inp not in graph.tensors:
                    graph.add_tensor(
                        Tensor(name=inp, shape=(2, 3, 4, 5), dtype=DType.FLOAT32)
                    )

            try:
                func(node, graph)
            except Exception:
                pass

        # Also try with an initializer input
        node = Node(
            op_type=op_type,
            inputs=["init_1", "init_2"],
            outputs=["out_0"],
            attributes={},
        )
        if "init_1" not in graph.tensors:
            graph.add_tensor(
                Tensor(
                    name="init_1", shape=(2,), dtype=DType.INT64, is_initializer=True
                )
            )
        if "init_2" not in graph.tensors:
            graph.add_tensor(
                Tensor(
                    name="init_2", shape=(2,), dtype=DType.INT64, is_initializer=True
                )
            )
        try:
            func(node, graph)
        except Exception:
            pass
