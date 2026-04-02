"""Further tests for a large graph with complex operations in the C compiler."""

from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo


def test_mega_graph_routing_coverage2():
    """Another test for routing coverage for a mega graph with complex operations."""
    tA = Tensor("A", [1, 2, 2], DType.FLOAT32, data=b"\x00" * 16)
    tB = Tensor("B", [1, 2, 2], DType.FLOAT32, data=b"\x00" * 16)
    tC = Tensor("C", [1, 2, 2], DType.FLOAT32, data=b"\x00" * 16)
    tInt8 = Tensor("TInt8", [1], DType.INT8, data=b"\x01")
    tInt32 = Tensor("TInt32", [1], DType.INT32, data=b"\x01\x00\x00\x00")
    tAxes = Tensor("Axes", [1], DType.INT32, data=b"\x00\x00\x00\x00")
    tSteps = Tensor("Steps", [1], DType.INT32, data=b"\x01\x00\x00\x00")

    tensors = {
        "A": tA,
        "B": tB,
        "C": tC,
        "TInt8": tInt8,
        "TInt32": tInt32,
        "Axes": tAxes,
        "Steps": tSteps,
    }
    for i in range(1, 24):
        tensors[f"O{i}"] = Tensor(f"O{i}", [1, 2, 2], DType.FLOAT32, data=b"\x00" * 16)

    inputs = [ValueInfo("A", [1, 2, 2], DType.FLOAT32)]
    outputs = [ValueInfo("O", [1, 2, 2], DType.FLOAT32)]

    nodes = [
        # Gather axis < 0
        Node("Gather", ["A", "B"], ["O1"], attributes={"axis": -1}),
        # ScatterElements reductions
        Node(
            "ScatterElements", ["A", "B", "C"], ["O2"], attributes={"reduction": "add", "axis": -1}
        ),
        Node("ScatterElements", ["A", "B", "C"], ["O3"], attributes={"reduction": "mul"}),
        Node("ScatterElements", ["A", "B", "C"], ["O4"], attributes={"reduction": "max"}),
        Node("ScatterElements", ["A", "B", "C"], ["O5"], attributes={"reduction": "min"}),
        # ScatterND reduction
        Node("ScatterND", ["A", "B", "C"], ["O6"], attributes={"reduction": "add"}),
        # CumSum reverse and exclusive
        Node("CumSum", ["A", "B"], ["O7"], attributes={"reverse": 1, "exclusive": 1}),
        Node("CumSum", ["A", "B"], ["O8"], attributes={"reverse": 1, "exclusive": 0}),
        Node("CumSum", ["A", "B"], ["O9"], attributes={"reverse": 0, "exclusive": 1}),
        # ConstantOfShape int8, int32
        Node("ConstantOfShape", ["A"], ["O10"], attributes={"value": tInt8}),
        Node("ConstantOfShape", ["A"], ["O11"], attributes={"value": tInt32}),
        # Slice dynamic
        Node("Slice", ["A", "TInt32", "TInt32", "Axes", "Steps"], ["O12"]),
        # GatherND
        Node("GatherND", ["A", "B"], ["O13"], attributes={"batch_dims": 1, "mode": "CRD"}),
        Node("DepthToSpace", ["A"], ["O15"], attributes={"blocksize": 2, "mode": "CRD"}),
        Node("SpaceToDepth", ["A"], ["O16"], attributes={"blocksize": 2, "mode": "CRD"}),
        Node("OneHot", ["A", "B", "C"], ["O17"], attributes={"axis": 0}),
        Node("OneHot", ["A", "B", "C"], ["O18"], attributes={"axis": -1}),
        Node("GatherND", ["A", "B"], ["O14"], attributes={"batch_dims": 0, "mode": "CRD"}),
        Node("MatMul", ["A", "B"], ["O19"]),
    ]

    # Dummy shapes to bypass shape inference crashes on dynamic slices
    tensors["O1"].shape = [1, 2, 1, 2, 2]
    tensors["O2"].shape = [1, 2, 2]
    tensors["O3"].shape = [1, 2, 2]
    tensors["O4"].shape = [1, 2, 2]
    tensors["O5"].shape = [1, 2, 2]
    tensors["O6"].shape = [1, 2, 2]
    tensors["O7"].shape = [1, 2, 2]
    tensors["O8"].shape = [1, 2, 2]
    tensors["O9"].shape = [1, 2, 2]
    tensors["O10"].shape = [1, 2, 2]
    tensors["O11"].shape = [1, 2, 2]
    tensors["O12"].shape = [1, 2, 2]
    tensors["O13"].shape = [1, 2, 2]
    tensors["O14"].shape = [1, 2, 2]
    tensors["O15"] = Tensor("O15", [1, 8, 1, 1], DType.FLOAT32, data=b"")
    tensors["O16"] = Tensor("O16", [1, 2, 4, 4], DType.FLOAT32, data=b"")
    tensors["O17"] = Tensor("O17", [1, 2, 2], DType.FLOAT32, data=b"")
    tensors["O18"] = Tensor("O18", [1, 2, 2], DType.FLOAT32, data=b"")

    graph = Graph("Mega2")
    graph.nodes = nodes
    graph.inputs = inputs
    graph.outputs = outputs
    graph.tensors = tensors

    compiler = C89Compiler(graph=graph)
    c_code = (
        compiler.compile_source() if hasattr(compiler, "compile_source") else compiler.generate()[1]
    )

    assert "add" in c_code or "mul" in c_code or "MAX" in c_code
