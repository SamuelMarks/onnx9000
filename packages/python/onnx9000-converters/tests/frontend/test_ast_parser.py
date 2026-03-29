"""Module providing core logic and structural definitions."""

from onnx9000.converters.frontend.ast_parser import ScriptCompiler
from onnx9000.converters.frontend.tensor import Tensor


def test_script_compiler_basic() -> None:
    """Tests the test_script_compiler_basic functionality."""

    def simple_func(x):
        """Test the simple_func functionality."""
        a = x
        return a

    simple_func(1)

    compiler = ScriptCompiler(simple_func)
    builder = compiler.compile(Tensor(name="x"))
    assert builder.name == "simple_func"


def test_script_compiler_return_multiple() -> None:
    """Tests the test_script_compiler_return_multiple functionality."""

    def return_tuple(x, y):
        """Test the return_tuple functionality."""
        return (x, y)

    return_tuple(1, 2)

    def return_list(x, y):
        """Test the return_list functionality."""
        return [x, y]

    return_list(1, 2)

    compiler = ScriptCompiler(return_tuple)
    compiler.compile(Tensor(name="x"), Tensor(name="y"))
    compiler2 = ScriptCompiler(return_list)
    compiler2.compile(Tensor(name="x"), Tensor(name="y"))


def test_script_compiler_if() -> None:
    """Tests the test_script_compiler_if functionality."""

    def if_func(x):
        """Test the if_func functionality."""
        y = x or x
        return y

    if_func(1)

    compiler = ScriptCompiler(if_func)
    compiler.compile(Tensor(name="x"))


def test_script_compiler_for_while() -> None:
    """Tests the test_script_compiler_for_while functionality."""

    def loops(x):
        """Test the loops functionality."""
        for _i in [1, 2]:
            return None
        while x:
            x -= 1
            return None
        return x

    loops(0)
    loops(1)

    compiler = ScriptCompiler(loops)
    compiler.compile(Tensor(name="x"))


def test_script_compiler_generic() -> None:
    """Tests the test_script_compiler_generic functionality."""

    class DummyCompiler(ScriptCompiler):
        """Class DummyCompiler implementation."""

        def visit_Pass(self, node):
            """Test the visit_Pass functionality."""
            return self.generic_visit(node)

    def pass_func() -> None:
        """Test the pass_func functionality."""
        return None

    pass_func()

    compiler = DummyCompiler(pass_func)
    compiler.compile()


def test_script_compiler_less_args() -> None:
    """Tests the test_script_compiler_less_args functionality."""

    def my_func(x, y):
        """Test the my_func functionality."""
        a = x
        return a

    my_func(1, 2)

    compiler = ScriptCompiler(my_func)
    compiler.compile(Tensor(name="x"))
