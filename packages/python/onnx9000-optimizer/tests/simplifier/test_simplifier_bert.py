import numpy as np
from onnx9000.core.ir import Graph, Node, Constant, Variable, ValueInfo
from onnx9000.optimizer.simplifier.api import simplify


def test_lock_batch_size_on_bert_and_evaluate_shape_constant_folding():
    graph = Graph("BERT_Simulated")

    # Input has dynamic batch size: (N, 512)
    x = Variable("input_ids", shape=("N", 512), dtype=np.dtype("int64"))
    graph.add_tensor(x)
    graph.inputs.append(ValueInfo("input_ids", shape=("N", 512), dtype=np.dtype("int64")))

    # Shape of input
    shape_node = Node(op_type="Shape", inputs=["input_ids"], outputs=["shape_out"], name="shape1")
    graph.add_node(shape_node)

    # We want to extract batch size (index 0) using Slice
    starts = Constant(
        "starts", values=np.array([0], dtype=np.int64), shape=(1,), dtype=np.dtype("int64")
    )
    ends = Constant(
        "ends", values=np.array([1], dtype=np.int64), shape=(1,), dtype=np.dtype("int64")
    )
    axes = Constant(
        "axes", values=np.array([0], dtype=np.int64), shape=(1,), dtype=np.dtype("int64")
    )
    steps = Constant(
        "steps", values=np.array([1], dtype=np.int64), shape=(1,), dtype=np.dtype("int64")
    )

    graph.add_tensor(starts)
    graph.add_tensor(ends)
    graph.add_tensor(axes)
    graph.add_tensor(steps)
    graph.initializers.extend(["starts", "ends", "axes", "steps"])

    slice_node = Node(
        op_type="Slice",
        inputs=["shape_out", "starts", "ends", "axes", "steps"],
        outputs=["batch_size_tensor"],
        name="slice1",
    )
    graph.add_node(slice_node)

    # And maybe we expand a constant vector to batch_size
    v = Constant(
        "v", values=np.array([1.5], dtype=np.float32), shape=(1,), dtype=np.dtype("float32")
    )
    graph.add_tensor(v)
    graph.initializers.append("v")

    # Expand expects a shape tensor: e.g. [batch_size, 512]
    # Concat batch_size_tensor with a fixed [512] tensor
    c512 = Constant(
        "c512", values=np.array([512], dtype=np.int64), shape=(1,), dtype=np.dtype("int64")
    )
    graph.add_tensor(c512)
    graph.initializers.append("c512")

    concat_node = Node(
        op_type="Concat",
        inputs=["batch_size_tensor", "c512"],
        outputs=["expanded_shape"],
        attributes={},  # need to add axis=0 manually
        name="concat1",
    )
    from onnx9000.core.ir import Attribute

    concat_node.attributes["axis"] = Attribute("axis", "INT", 0)
    graph.add_node(concat_node)

    expand_node = Node(
        op_type="Expand", inputs=["v", "expanded_shape"], outputs=["out"], name="expand1"
    )
    graph.add_node(expand_node)

    graph.outputs.append(ValueInfo("out", shape=("N", 512), dtype=np.dtype("float32")))

    # Simplifier with dynamic batch size. Shape cannot fold completely because N is string.
    import copy

    graph2 = copy.deepcopy(graph)
    sim1 = simplify(graph)

    # Now explicitly lock the batch size by simulating the CLI --input-shape 'input_ids:1,512'
    # This should cascade fold Shape -> Slice -> Concat -> Expanded_Shape into a Constant(1, 512)
    # Then Expand just becomes Expand(v, Constant(1, 512)) and even Expand folds if max size allows!

    input_shapes = {"input_ids": [1, 512]}

    sim2 = simplify(graph2, input_shapes=input_shapes)

    # Validate cascading! Shape -> Slice -> Concat should all be folded into a single constant.
    # The Expand node should also evaluate to a constant of shape [1, 512].
    assert any(n.op_type == "Constant" for n in sim2.nodes)
    assert not any(n.op_type == "Shape" for n in sim2.nodes)
    assert not any(n.op_type == "Slice" for n in sim2.nodes)
    assert not any(n.op_type == "Concat" for n in sim2.nodes)
    # Expand should also be gone, completely folded
    assert not any(n.op_type == "Expand" for n in sim2.nodes)
