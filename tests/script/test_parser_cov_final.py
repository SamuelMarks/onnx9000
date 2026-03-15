import pytest
from onnx9000.script.parser import ScriptParser


def test_script_parser_missing_lines():
    p = ScriptParser({})

    def valid_func():
        pass

    class MyClass:
        pass

    with pytest.raises(ValueError, match="Expected a function definition"):
        p.parse(MyClass)
    with pytest.raises(ValueError, match="Multiple comparisons not supported"):

        def f1():
            return 1 < 2 < 3

        p.parse(f1)
    with pytest.raises(ValueError, match="Unsupported comparison"):

        def f2():
            return 1 in [2]

        p.parse(f2)
    with pytest.raises(ValueError, match="Unknown variable: unknown_var"):

        def f3():
            return unknown_var

        p.parse(f3)
    with pytest.raises(
        ValueError, match="Variable y must be defined in both branches of If"
    ):

        def f4():
            x = 1
            if True:
                x = 2
            else:
                y = 3
            return x

        p.parse(f4)
