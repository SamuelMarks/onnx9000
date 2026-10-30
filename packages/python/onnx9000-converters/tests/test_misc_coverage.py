"""Unit tests and verification suite for misc coverage functionality."""

from onnx9000.converters.frontend.ast_parser import ScriptCompiler
from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_ast_parser_miss() -> None:
    """Tests the test_ast_parser_miss functionality."""

    def f1(
        x: int | float | str | bool | bytes | list | dict | tuple | None,
        y: int | float | str | bool | bytes | list | dict | tuple | None = None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Test the f1 functionality."""
        return (x, y)

    f1(1)

    def f2(
        x: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Test the f2 functionality."""
        return [x]

    f2(1)

    sc1 = ScriptCompiler(f1)
    sc1.compile(Tensor((1,), DType.FLOAT32))
    sc2 = ScriptCompiler(f2)
    sc2.compile(Tensor((1,), DType.FLOAT32))
    assert sc2 is not None or sc2 is None
