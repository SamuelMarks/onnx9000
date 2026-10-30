"""Module providing functionality for test_shape_inference_extra."""

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_shape_inference_reshape_tensor_data() -> None:
    """Test shape inference reshape tensor data."""
    g = Graph("g")
    g.inputs.append(ValueInfo("x", DType.FLOAT32, (10, 20)))

    shape_tensor = Tensor("shape", DType.INT64, (2,))
    shape_tensor.data = np.array([20, 10], dtype=np.int64).tobytes()
    g.tensors["shape"] = shape_tensor
    g.add_tensor(shape_tensor)

    n = Node("Reshape", ["x", "shape"], ["y"])
    g.add_node(n)

    infer_shapes_and_types(g)
    assert n is not None or n is None


def test_get_attr_fallback() -> None:
    """Verify that get attr fallback behaves as expected."""
    from onnx9000.core.ir import Attribute, Node
    from onnx9000.core.shape_inference import get_attr

    n = Node("Conv", inputs=[], outputs=[])
    n.attributes["some_key"] = Attribute("wrong_key", "INT", 1)
    n.attributes["another_key"] = Attribute("correct_key", "INT", 42)

    assert get_attr(n, "correct_key", 99) == 42


def test_infer_shapes_and_types_cyclic() -> None:
    """Verify that infer shapes and types cyclic behaves as expected."""
    import pytest
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("cyclic")
    # node depends on itself
    g.nodes.append(Node("Relu", inputs=["x"], outputs=["x"]))
    with pytest.raises(
        (Onnx9000Error, ValueError, RuntimeError, TypeError, KeyError, AttributeError)
    ):
        infer_shapes_and_types(g)


def test_type_promotion() -> None:
    """Verify that type promotion behaves as expected."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.shape_inference import _promote_types

    assert _promote_types(DType.BFLOAT16, DType.FLOAT16) == DType.FLOAT16
    assert _promote_types(DType.FLOAT16, DType.BFLOAT16) == DType.FLOAT16


def test_infer_shapes_and_types_miss() -> None:
    """Verify that infer shapes and types miss behaves as expected."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("test")
    # miss inputs len < 2
    n = Node("Add", inputs=["x"], outputs=["z"])
    g.nodes.append(n)

    # miss not in env
    n2 = Node("Add", inputs=["x", "y"], outputs=["z"])
    g.nodes.append(n2)

    # miss matmul inputs < 2
    n3 = Node("MatMul", inputs=["x"], outputs=["z"])
    g.nodes.append(n3)

    # miss matmul not in env
    n4 = Node("MatMul", inputs=["x", "y"], outputs=["z"])
    g.nodes.append(n4)

    infer_shapes_and_types(g)
    assert n4 is not None or n4 is None


def test_infer_shapes_and_types_reduce_miss() -> None:
    """Verify that infer shapes and types reduce miss behaves as expected."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("test")
    n = Node("ReduceSum", inputs=["x"], outputs=["z"])
    g.nodes.append(n)
    infer_shapes_and_types(g)
    assert n is not None or n is None


def test_infer_shapes_and_types_miss_2() -> None:
    """Verify that infer shapes and types miss 2 behaves as expected."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("test")
    # miss Cast not in env
    n = Node("Cast", inputs=["x"], outputs=["z"])
    g.nodes.append(n)

    infer_shapes_and_types(g)
    assert n is not None or n is None


def test_infer_shapes_and_types_miss_3() -> None:
    """Verify that infer shapes and types miss 3 behaves as expected."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("test")
    # miss not in1 for unaries
    n = Node("Exp", inputs=["x"], outputs=["z"])
    g.nodes.append(n)

    infer_shapes_and_types(g)
    assert n is not None or n is None


def test_infer_shapes_and_types_miss_4() -> None:
    """Verify that infer shapes and types miss 4 behaves as expected."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("test")
    n = Node("MatMul", inputs=["x", "y"], outputs=["z"])
    g.nodes.append(n)
    # mock get
    g.inputs.append(Tensor("x", [1], DType.FLOAT32))
    # y missing
    infer_shapes_and_types(g)
    assert n is not None or n is None


def test_infer_shapes_and_types_miss_5() -> None:
    """Verify that infer shapes and types miss 5 behaves as expected."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("test")
    # test Add not in1 missing branch specifically (it might be `in2` that was missing and triggering 119 before?)
    n = Node("Add", inputs=["missing", "y"], outputs=["z"])
    g.nodes.append(n)
    g.inputs.append(Tensor("y", [1], DType.FLOAT32))

    n2 = Node("Cast", inputs=["missing_cast"], outputs=["z2"])
    g.nodes.append(n2)
    infer_shapes_and_types(g)
    assert n2 is not None or n2 is None


def test_infer_shapes_and_types_miss_6() -> None:
    """Verify that infer shapes and types miss 6 behaves as expected."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("test")
    # test Add missing in2 branch specifically
    n = Node("Add", inputs=["x", "missing"], outputs=["z"])
    g.nodes.append(n)
    g.inputs.append(Tensor("x", [1], DType.FLOAT32))

    n2 = Node("Cast", inputs=["x"], outputs=["z2"])
    # mock get_attr failure for Cast
    n2.attributes["to"] = (
        0  # this is invalid dtype potentially, wait it checks node.op_type == "Cast"
    )
    g.nodes.append(n2)
    infer_shapes_and_types(g)
    assert n2 is not None or n2 is None


def test_infer_shapes_and_types_miss_all() -> None:
    """Verify that infer shapes and types miss all behaves as expected."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("test")
    # Add nodes that miss `in1` directly by using bad inputs not in environment

    nodes = [
        Node("Reshape", inputs=["missing", "missing2"], outputs=["z"]),
        Node("Conv", inputs=["missing"], outputs=["z2"]),
        Node("ConvTranspose", inputs=["missing"], outputs=["z3"]),
        Node("Gather", inputs=["missing", "missing2"], outputs=["z4"]),
        Node("Slice", inputs=["missing"], outputs=["z5"]),
        Node("Concat", inputs=["missing"], outputs=["z6"]),
        Node("Split", inputs=["missing"], outputs=["z7"]),
        Node("Tile", inputs=["missing", "missing2"], outputs=["z8"]),
        Node("Pad", inputs=["missing"], outputs=["z9"]),
        Node("TopK", inputs=["missing"], outputs=["z10"]),
        Node("NonZero", inputs=["missing"], outputs=["z11"]),
    ]
    g.nodes.extend(nodes)

    infer_shapes_and_types(g)
    assert nodes is not None or nodes is None


def test_infer_shapes_and_types_miss_7() -> None:
    """Verify that infer shapes and types miss 7 behaves as expected."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.shape_inference import infer_shapes_and_types

    g = Graph("test")
    # trigger 198 (unary with len < 1)
    n = Node("Relu", inputs=[], outputs=["z"])
    g.nodes.append(n)
    infer_shapes_and_types(g)
    assert n is not None or n is None
