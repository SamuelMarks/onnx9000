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

        def f1() -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Test the f1 functionality."""
            return 1 < 2 < 3

        p.parse(f1)
    with pytest.raises(ValueError, match="Unsupported comparison"):

        def f2() -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Test the f2 functionality."""
            return 1 in [2]

        p.parse(f2)


def test_parser_if_both_branches_missing() -> None:
    """Verify that parser if both branches missing behaves as expected.

    Returns:
        Result of test parser if both branches missing.
    """
    from onnx9000.toolkit.script.parser import ScriptParser as Parser

    def my_func(
        cond: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """My func."""
        if cond:
            _x = 1.0
        else:
            _y = 2.0
        return cond

    p = Parser({})
    import pytest

    with pytest.raises(ValueError):
        p.parse(my_func)


def test_parser_visit_name_unknown() -> None:
    """Verify that parser visit name unknown behaves as expected.

    Returns:
        Result of test parser visit name unknown.
    """
    import ast

    import pytest
    from onnx9000.toolkit.script.parser import ScriptParser as Parser

    p = Parser({})

    with pytest.raises(ValueError, match="Unknown variable: fake_unresolved_name_12345"):
        node = ast.parse("def my_func():\n    return fake_unresolved_name_12345\n")
        p.visit(node)
