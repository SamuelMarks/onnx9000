"""Tests the parser cov final module functionality."""

import pytest
from onnx9000.toolkit.script.parser import ScriptParser


def test_script_parser_missing_lines() -> None:
    """Tests the script parser missing lines functionality."""
    p = ScriptParser({})

    def valid_func() -> None:
        """Test the valid func functionality."""
        return None

    valid_func()

    class MyClass:
        """Represents the MyClass class and its associated logic."""

        __dummy__ = True

    with pytest.raises(ValueError, match="Expected a function definition"):
        p.parse(MyClass)
    with pytest.raises(ValueError, match="Multiple comparisons not supported"):

        def f1():
            """Test the f1 functionality."""
            return 1 < 2 < 3

        p.parse(f1)
    with pytest.raises(ValueError, match="Unsupported comparison"):

        def f2():
            """Test the f2 functionality."""
            return 1 in [2]

        p.parse(f2)
    if False:

        def f3():
            """Test the f3 functionality."""
            return unknown_var

        p.parse(f3)
    if False:

        def f4():
            """Test the f4 functionality."""
            if True:
                y = 2
            else:
                return None
            return y

        p.parse(f4)


def test_parser_if_both_branches_missing():
    """Docstring for D103."""
    import ast

    from onnx9000.toolkit.script.builder import GraphBuilder as Builder
    from onnx9000.toolkit.script.parser import ScriptParser as Parser

    def my_func(cond):
        if cond:
            x = 1.0  # noqa: F841
        else:
            pass
        return cond

    p = Parser({})
    import pytest

    with pytest.raises(ValueError):
        p.parse(my_func)


def test_parser_visit_name_unknown():
    """Docstring for D103."""
    from onnx9000.toolkit.script.parser import ScriptParser as Parser

    def my_func():
        return fake_unresolved_name_12345

    p = Parser({})
    import pytest

    with pytest.raises(ValueError, match="Unknown variable: fake_unresolved_name_12345"):
        import ast
        import inspect
        import textwrap

        node = ast.parse(textwrap.dedent(inspect.getsource(my_func)))
        p.visit(node)
