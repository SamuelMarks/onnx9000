"""Module providing core logic and structural definitions."""

from onnx9000.core.dtypes import DType
from onnx9000.converters.frontend.ast_parser import ScriptCompiler
from onnx9000.converters.frontend.tensor import Tensor


def test_ast_parser_miss() -> None:
    """Tests the test_ast_parser_miss functionality."""

    def f1(x, y=None):
        """Tests the f1 functionality."""
        return (x, y)

    def f2(x):
        """Tests the f2 functionality."""
        return [x]

    sc1 = ScriptCompiler(f1)
    sc1.compile(Tensor((1,), DType.FLOAT32))
    sc2 = ScriptCompiler(f2)
    sc2.compile(Tensor((1,), DType.FLOAT32))
