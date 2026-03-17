"""Module providing core logic and structural definitions."""

from onnx9000.converters.frontend.ast_parser import ScriptCompiler
from onnx9000.converters.frontend.tensor import Tensor


def test_script_compiler_basic() -> None:
    """Tests the test_script_compiler_basic functionality."""

    def simple_func(x):
        """Tests the simple_func functionality."""
        a = x
        return a

    compiler = ScriptCompiler(simple_func)
    builder = compiler.compile(Tensor(name="x"))
    assert builder.name == "simple_func"


def test_script_compiler_return_multiple() -> None:
    """Tests the test_script_compiler_return_multiple functionality."""

    def return_tuple(x, y):
        """Tests the return_tuple functionality."""
        return (x, y)

    def return_list(x, y):
        """Tests the return_list functionality."""
        return [x, y]

    compiler = ScriptCompiler(return_tuple)
    compiler.compile(Tensor(name="x"), Tensor(name="y"))
    compiler2 = ScriptCompiler(return_list)
    compiler2.compile(Tensor(name="x"), Tensor(name="y"))


def test_script_compiler_if() -> None:
    """Tests the test_script_compiler_if functionality."""

    def if_func(x):
        """Tests the if_func functionality."""
        y = x or x
        return y

    compiler = ScriptCompiler(if_func)
    compiler.compile(Tensor(name="x"))


def test_script_compiler_for_while() -> None:
    """Tests the test_script_compiler_for_while functionality."""

    def loops(x):
        """Tests the loops functionality."""
        for _i in [1, 2]:
            pass
        while x:
            pass
        return x

    compiler = ScriptCompiler(loops)
    compiler.compile(Tensor(name="x"))


def test_script_compiler_generic() -> None:
    """Tests the test_script_compiler_generic functionality."""

    class DummyCompiler(ScriptCompiler):
        """Class DummyCompiler implementation."""

        def visit_Pass(self, node):
            """Tests the visit_Pass functionality."""
            return self.generic_visit(node)

    def pass_func() -> None:
        """Tests the pass_func functionality."""
        assert True

    compiler = DummyCompiler(pass_func)
    compiler.compile()


def test_script_compiler_less_args() -> None:
    """Tests the test_script_compiler_less_args functionality."""

    def my_func(x, y):
        """Tests the my_func functionality."""
        a = x
        return a

    compiler = ScriptCompiler(my_func)
    compiler.compile(Tensor(name="x"))
