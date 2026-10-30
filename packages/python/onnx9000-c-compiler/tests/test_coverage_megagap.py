"""Unit tests and verification suite for coverage megagap functionality."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.c_compiler.mlir import MLIRCompiler
from onnx9000.c_compiler.simd_macros import emit_simd_macros
from onnx9000.core.ir import Graph, Node


def test_mlir_compiler() -> None:
    """Verify that mlir compiler behaves as expected."""
    graph = Graph("test")
    graph.nodes.append(Node(op_type="Conv", inputs=[], outputs=[]))
    graph.nodes.append(Node(op_type="Relu", inputs=[], outputs=[]))
    c = MLIRCompiler(graph)
    assert "tosa.conv2d" in c.generate_tosa()

    c = MLIRCompiler(graph)
    assert "module {" in c.generate_linalg()

    c = MLIRCompiler(graph)
    assert "module {" in c.generate_stablehlo()


def test_simd_macros() -> None:
    """Verify that simd macros behaves as expected."""
    b = C89Builder()
    emit_simd_macros(b)
    code = b.get_code()
    assert "SIMD Macros" in code
    assert "block_q4_0" in code
    assert "ggml_vec_dot_q4_0_q8_0" in code


def test_pooling_multi_axis_reduction() -> None:
    """Verify that pooling multi axis reduction behaves as expected."""
    from onnx9000.c_compiler.ast_builder import C89Builder
    from onnx9000.c_compiler.pooling import generate_reduction
    from onnx9000.core.ir import Node, Tensor

    b = C89Builder()
    node = Node("ReduceMean", inputs=["a"], outputs=["b"])

    class MockAttr:
        """Represents mock attr."""

        def __init__(
            self, val: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> None:
            """Initialize MockAttr with configured parameters and state.

            Args:
                val: Parameter specifying val.
            """
            self.value = val

    node.attributes = {"axes": MockAttr([0, 1])}
    in_tensor = Tensor("a", shape=[1, 2, 3, 4], dtype=1)
    out_tensor = Tensor("b", shape=[1, 4], dtype=1)
    generate_reduction(b, node, out_tensor, in_tensor, "in_name", "out_name", "reduce_mean")
    assert "Multi-axis reduction requires chaining or higher dimensional loop" in b.get_code()
