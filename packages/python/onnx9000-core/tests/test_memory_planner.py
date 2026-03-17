from onnx9000.core.ir import Graph, Node, Tensor, Constant, Attribute
from onnx9000.core.dtypes import DType
from onnx9000.core.memory_planner import simulate_memory_plan


def test_memory_planner_first_fit():
    g = Graph("test")
    # A -> B -> C -> D
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.inputs.append("A")

    n1 = Node("Add", inputs=["A", "A"], outputs=["B"])
    g.add_tensor(Tensor("B", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_node(n1)

    n2 = Node("Relu", inputs=["B"], outputs=["C"])
    g.add_tensor(Tensor("C", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_node(n2)

    n3 = Node("Sigmoid", inputs=["C"], outputs=["D"])
    g.add_tensor(Tensor("D", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_node(n3)

    # Run first fit simulation
    arena = simulate_memory_plan(g, strategy="first_fit")

    # B is an intermediate, dies at Relu, so Relu can be in-place
    assert arena.tensor_offsets["B"] == arena.tensor_offsets["C"]
    assert arena.tensor_offsets["C"] == arena.tensor_offsets["D"]


def test_memory_planner_reshape_view():
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(10, 20), dtype=DType.FLOAT32))
    g.inputs.append("X")
    n1 = Node("Reshape", inputs=["X", "shape"], outputs=["Y"])
    g.add_tensor(Tensor("Y", shape=(200,), dtype=DType.FLOAT32))
    g.add_node(n1)

    arena = simulate_memory_plan(g)
    assert arena.tensor_offsets["X"] == arena.tensor_offsets["Y"]
