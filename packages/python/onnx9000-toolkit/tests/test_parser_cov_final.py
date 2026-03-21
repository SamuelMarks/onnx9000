"""Tests the parser cov final module functionality."""

import pytest
from onnx9000.toolkit.script.parser import ScriptParser


def test_script_parser_missing_lines() -> None:
    """Tests the script parser missing lines functionality."""
    p = ScriptParser({})

    def valid_func() -> None:
        """Tests the valid func functionality."""
        pass

    class MyClass:
        """Represents the MyClass class and its associated logic."""

        pass

    with pytest.raises(ValueError, match="Expected a function definition"):
        p.parse(MyClass)
    with pytest.raises(ValueError, match="Multiple comparisons not supported"):

        def f1():
            """Tests the f1 functionality."""
            return 1 < 2 < 3

        p.parse(f1)
    with pytest.raises(ValueError, match="Unsupported comparison"):

        def f2():
            """Tests the f2 functionality."""
            return 1 in [2]

        p.parse(f2)
    with pytest.raises(ValueError, match="Unknown variable: unknown_var"):

        def f3():
            """Tests the f3 functionality."""
            return unknown_var

        p.parse(f3)
    with pytest.raises(ValueError, match="Variable y must be defined in both branches of If"):

        def f4():
            """Tests the f4 functionality."""
            if True:
                y = 2
            else:
                pass
            return y

        p.parse(f4)
