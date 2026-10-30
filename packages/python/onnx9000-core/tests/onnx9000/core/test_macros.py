"""Unit tests for IR macros in `onnx9000.core.macros`."""

from onnx9000.core.ir import Graph, Tensor
from onnx9000.core.macros import (
    MacroExpander,
    MacroMatcher,
    ir_macro,
    moe_layer_macro,
    transformer_block_macro,
)


def test_macros_module() -> None:
    """Verify MacroExpander, MacroMatcher, ir_macro, transformer_block_macro, and moe_layer_macro."""
    g = Graph("macro_g")
    assert MacroExpander().apply(g) is g
    assert MacroMatcher().apply(g) is g

    @ir_macro("TestCustomMacro")
    def my_macro(x: Tensor) -> Tensor:
        """Execute my macro operation.

        Args:
            x: Parameter specifying x (Tensor).

        Returns:
            Result of my macro (Tensor).
        """
        return x

    x = Tensor("x", (2, 2), 1)
    assert isinstance(my_macro(x), Tensor)
    assert isinstance(transformer_block_macro(x, x, x), Tensor)
    assert isinstance(moe_layer_macro(x, x, [x]), Tensor)
