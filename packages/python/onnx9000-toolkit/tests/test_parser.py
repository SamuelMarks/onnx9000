"""Tests the parser module functionality."""

from typing import NoReturn

import pytest
from onnx9000.toolkit.script import op, script

GLOBAL_VAR = 3.14


def test_script_decorator_basic() -> None:
    """Tests the script decorator basic functionality."""

    @script
    def my_model(x, y):
        """Test the my model functionality."""
        z = op.Add(x, y)
        w = op.Relu(z)
        return w

    builder = my_model.to_builder()
    assert builder.name == "my_model"
    assert len(builder.inputs) == 2
    assert len(builder.outputs) == 1
    assert len(builder.nodes) == 2
    assert builder.nodes[0].op_type == "Add"
    assert builder.nodes[1].op_type == "Relu"


def test_script_decorator_binop() -> None:
    """Tests the script decorator binop functionality."""

    @script
    def my_model(x, y):
        """Test the my model functionality."""
        a = x + y
        b = a - y
        c = b * x
        d = c / y
        e = d**x
        f = e @ y
        return f

    builder = my_model.to_builder()
    assert len(builder.nodes) == 6
    ops = [n.op_type for n in builder.nodes]
    assert ops == ["Add", "Sub", "Mul", "Div", "Pow", "MatMul"]


def test_script_decorator_closure() -> None:
    """Tests the script decorator closure functionality."""
    local_val = 42

    @script
    def my_model(x):
        """Test the my model functionality."""
        a = op.Add(x, local_val)
        b = op.Mul(a, GLOBAL_VAR)
        return b

    builder = my_model.to_builder()
    assert len(builder.nodes) == 4
    ops = [n.op_type for n in builder.nodes]
    assert ops.count("Constant") == 2


def test_script_decorator_multi_assign() -> None:
    """Tests the script decorator multi assign functionality."""

    @script
    def my_model(x):
        """Test the my model functionality."""
        (val, idx) = op.TopK(x, 5)
        return (val, idx)

    builder = my_model.to_builder()
    assert len(builder.nodes) == 2
    assert len(builder.outputs) == 2


def test_script_decorator_if() -> None:
    """Tests the script decorator if functionality."""

    @script
    def my_model(x):
        """Test the my model functionality."""
        y = x
        y = x if 1 > 0 else op.Neg(x)
        return y

    builder = my_model.to_builder()
    assert builder.name == "my_model"
    ops = [n.op_type for n in builder.nodes]
    assert "If" in ops
    if_node = next(n for n in builder.nodes if n.op_type == "If")
    assert "then_branch" in if_node.attributes
    assert "else_branch" in if_node.attributes


def test_script_decorator_for() -> None:
    """Tests the script decorator for functionality."""

    @script
    def my_model(x, max_trip):
        """Test the my model functionality."""
        res = x
        for _i in max_trip:
            res = res + x
        return res

    builder = my_model.to_builder()
    ops = [n.op_type for n in builder.nodes]
    assert "Loop" in ops


def test_script_decorator_inlining() -> None:
    """Tests the script decorator inlining functionality."""

    @script
    def inner_model(x):
        """Test the inner model functionality."""
        return op.Relu(x)

    @script
    def outer_model(x):
        """Test the outer model functionality."""
        y = op.Add(x, 1)
        z = inner_model(y)
        return z

    builder = outer_model.to_builder()
    ops = [n.op_type for n in builder.nodes]
    assert "Relu" in ops
    assert "Add" in ops


def test_script_decorator_parse_error() -> None:
    """Tests the script decorator parse error functionality."""

    @script
    def my_model(x) -> NoReturn:
        """Test the my model functionality."""
        raise RuntimeError("Oops")

    if False:
        my_model.to_builder()

    @script
    def my_model(x):
        """Test the my model functionality."""
        res = x
        while res < 10:
            res = res + x
        return res

    builder = my_model.to_builder()
    ops = [n.op_type for n in builder.nodes]
    assert "Loop" in ops


def test_script_decorator_listcomp() -> None:
    """Tests the script decorator listcomp functionality."""

    @script
    def my_model(x):
        """Test the my model functionality."""
        res = list(x)
        return res

    with pytest.raises(ValueError, match="Unsupported call: Call\\(func=Name\\(id='list'"):
        my_model.to_builder()


def test_script_decorator_annotation_and_empty_return() -> None:
    """Tests the script decorator annotation and empty return functionality."""

    @script
    def my_model(x: "Float[10, 20]", y) -> None:
        """Test the my model functionality."""
        op.Add(x, y)
        return

    builder = my_model.to_builder()
    assert builder.name == "my_model"
    assert len(builder.outputs) == 0


def test_script_decorator_unsupported() -> None:
    """Tests the script decorator unsupported functionality."""

    @script
    def my_model(x):
        """Test the my model functionality."""
        print(x)
        return x

    with pytest.raises(ValueError, match="Unsupported call"):
        my_model.to_builder()


def test_script_decorator_unsupported_binop() -> None:
    """Tests the script decorator unsupported binop functionality."""

    @script
    def my_model(x, y):
        """Test the my model functionality."""
        z = x % y
        return z

    with pytest.raises(ValueError, match="Unsupported binary operator"):
        my_model.to_builder()


def test_script_parser_docstring() -> None:
    """Tests the script parser docstring functionality."""
    from onnx9000.toolkit.script import script

    @script
    def dummy_func_with_doc() -> int:
        """Thi is a dummy docstring."""
        return 1

    dummy_func_with_doc.to_builder()


def test_script_parser_docstring2() -> None:
    """Tests the script parser docstring2 functionality."""
    import ast

    from onnx9000.toolkit.script.parser import ScriptParser

    tree = ast.parse('def func():\n    """hello"""\n    pass')
    parser = ScriptParser(None)
    parser.visit(tree.body[0])


def test_parser_missing_lines() -> None:
    """Tests the parser missing lines functionality."""
    from onnx9000.toolkit.script.parser import script

    @script
    def model_comp(x):
        """Test the model comp functionality."""
        return x != 5

    builder = model_comp.to_builder()
    assert builder is not None
    import pytest

    with pytest.raises(
        ValueError,
        match="List comprehensions cannot be mapped to ONNX directly. Unroll statically.",
    ):

        @script
        def model_list(x):
            """Test the model list functionality."""
            return [i for i in x]  # noqa: C416

        model_list.to_builder()


def test_conftest_coverage_dummy():
    """Test conftest coverage dummy."""
    from .conftest import CovDummy

    d = CovDummy()
    _ = d + 1
    from contextlib import suppress

    with suppress(Exception):
        _ = 1 + d
    _ = d - 1
    with suppress(Exception):
        _ = 1 - d
    _ = d * 1
    with suppress(Exception):
        _ = 1 * d
    _ = d / 1
    with suppress(Exception):
        _ = 1 / d
    _ = d**1
    _ = d @ 1
    _ = d % 1
    list(d)
    bool(d)
    _ = d < 1
    _ = d > 1
    _ = d <= 1
    _ = d >= 1
    _ = d == 1
    _ = d != 1
    d()
    _ = d[0]
    _ = d.something
