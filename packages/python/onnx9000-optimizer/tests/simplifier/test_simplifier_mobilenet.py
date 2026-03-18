import numpy as np
from onnx9000.core.ir import Constant, Graph, Node, ValueInfo, Variable
from onnx9000.optimizer.simplifier.api import simplify


def test_fold_batch_normalization_natively_on_standard_mobilenet():
    graph = Graph("MobileNet_Simulated")

    x = Variable("x", shape=(1, 32, 112, 112), dtype=np.dtype("float32"))
    graph.add_tensor(x)
    graph.inputs.append(x)

    # BN inputs
    scale = Constant(
        "scale", values=np.ones(32, dtype=np.float32), shape=(32,), dtype=np.dtype("float32")
    )
    b = Constant("b", values=np.zeros(32, dtype=np.float32), shape=(32,), dtype=np.dtype("float32"))
    mean = Constant(
        "mean", values=np.full(32, 0.5, dtype=np.float32), shape=(32,), dtype=np.dtype("float32")
    )
    var = Constant(
        "var", values=np.full(32, 0.1, dtype=np.float32), shape=(32,), dtype=np.dtype("float32")
    )

    graph.add_tensor(scale)
    graph.add_tensor(b)
    graph.add_tensor(mean)
    graph.add_tensor(var)
    graph.initializers.extend(["scale", "b", "mean", "var"])

    # Actually, the checkmark is for BN *fusion*? Or folding?
    # "Fold BatchNormalization natively (If all inputs are constants)"
    # MobileNet BN is folded natively if inputs are constant, or FUSED into Conv!

    bn_node = Node(
        op_type="BatchNormalization",
        inputs=["x", "scale", "b", "mean", "var"],
        outputs=["y"],
        name="bn",
    )
    graph.add_node(bn_node)

    graph.outputs.append(ValueInfo("y", shape=(1, 32, 112, 112), dtype=np.dtype("float32")))

    # Simplify
    graph_sim = simplify(graph)
    assert len(graph_sim.nodes) == 1

    # If X was constant, it would fold completely.
    x_const = Constant(
        "x_const",
        values=np.ones((1, 32, 112, 112), dtype=np.float32),
        shape=(1, 32, 112, 112),
        dtype=np.dtype("float32"),
    )
    graph2 = Graph("MobileNet_Constant")
    graph2.add_tensor(x_const)
    graph2.add_tensor(scale)
    graph2.add_tensor(b)
    graph2.add_tensor(mean)
    graph2.add_tensor(var)
    graph2.initializers.extend(["x_const", "scale", "b", "mean", "var"])

    bn_node2 = Node(
        op_type="BatchNormalization",
        inputs=["x_const", "scale", "b", "mean", "var"],
        outputs=["y_const"],
        name="bn2",
    )
    graph2.add_node(bn_node2)
    graph2.outputs.append(ValueInfo("y_const", shape=(1, 32, 112, 112), dtype=np.dtype("float32")))

    graph_sim2 = simplify(graph2)
    # Since all inputs are constants, BN should be folded completely, replaced by a constant!
    # The node should be removed (DCE or Identity) and the output fed directly.
    # Wait, the output 'y_const' should become an initializer!
    assert len(graph_sim2.nodes) == 1
    assert graph_sim2.nodes[0].op_type == "Constant"
