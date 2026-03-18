import numpy as np
from onnx9000.core.ir import Constant, DynamicDim, Graph, Node, ValueInfo, Variable
from onnx9000.optimizer.simplifier.api import simplify


def test_dynamic_axes_preservation():
    graph = Graph("Dynamic_Axes_Test")
    x = Variable("x", shape=(DynamicDim("batch"), 3, 224, 224), dtype=np.dtype("float32"))
    graph.add_tensor(x)
    graph.inputs.append(
        ValueInfo("x", shape=(DynamicDim("batch"), 3, 224, 224), dtype=np.dtype("float32"))
    )

    # Subgraph for If node
    subgraph = Graph("If_Subgraph")
    subgraph.add_tensor(
        Variable("x_sub", shape=(DynamicDim("batch"), 3, 224, 224), dtype=np.dtype("float32"))
    )
    # some dummy op
    n = Node("Relu", ["x_sub"], ["y_sub"], name="relu1")
    subgraph.add_node(n)
    subgraph.add_tensor(
        Variable("y_sub", shape=(DynamicDim("batch"), 3, 224, 224), dtype=np.dtype("float32"))
    )
    subgraph.outputs.append(
        ValueInfo("y_sub", shape=(DynamicDim("batch"), 3, 224, 224), dtype=np.dtype("float32"))
    )

    from onnx9000.core.ir import Attribute

    if_node = Node(
        op_type="If",
        inputs=["cond"],
        outputs=["y"],
        attributes={
            "then_branch": Attribute("then_branch", "GRAPH", subgraph),
            "else_branch": Attribute("else_branch", "GRAPH", subgraph),
        },
        name="if1",
    )
    graph.add_node(if_node)

    # Condition
    cond = Constant("cond", values=np.array(True), shape=(), dtype=np.dtype("bool"))
    graph.add_tensor(cond)
    graph.initializers.append("cond")

    graph.outputs.append(
        ValueInfo("y", shape=(DynamicDim("batch"), 3, 224, 224), dtype=np.dtype("float32"))
    )

    # Simplifying the graph should fold the If node because cond is constant!
    sim = simplify(graph)

    # After folding, Relu is injected. The dynamic dim "batch" MUST be preserved on the output!
    y_out = next(o for o in sim.outputs if "y_sub" in o.name)
    assert any(isinstance(d, DynamicDim) and d.value == "batch" for d in y_out.shape)
