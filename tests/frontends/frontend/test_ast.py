"""Module providing core logic and structural definitions."""

from onnx9000.frontends.frontend.tracer import script
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_ast_script():
    """Provides semantic functionality and verification."""

    def my_func(x):
        """Provides semantic functionality and verification."""
        if x:
            return x
        else:
            for i in range(10):
                pass
            while True:
                break
        return x

    x = Tensor((), DType.BOOL, "x")
    builder = script(my_func, x)
    assert builder.name == "my_func"
    assert len(builder.nodes) >= 1
    op_types = [n.op_type for n in builder.nodes]
    assert "If" in op_types


def test_ast_parser_missing():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.ast_parser import ScriptCompiler

    def my_func(a, b, c=None):
        """Provides my func functionality and verification."""
        if True:
            pass
        return (a, b), [a]

    t1 = Tensor((10,), DType.FLOAT32, "a")
    t2 = Tensor((10,), DType.FLOAT32, "b")
    parser = ScriptCompiler(my_func)
    parser.compile(t1, t2)


def test_ast_parser_missing_lists():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.ast_parser import ScriptCompiler

    def my_func2(a):
        """Provides my func2 functionality and verification."""
        return [a, a]

    t1 = Tensor((10,), DType.FLOAT32, "a")
    parser2 = ScriptCompiler(my_func2)
    parser2.compile(t1)


def test_ast_parser_missing_tuple():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.ast_parser import ScriptCompiler

    def my_func3(a):
        """Provides my func3 functionality and verification."""
        return (a,)

    t1 = Tensor((10,), DType.FLOAT32, "a")
    parser3 = ScriptCompiler(my_func3)
    parser3.compile(t1)


def test_ast_parser_missing_tensor():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.ast_parser import ScriptCompiler

    def my_func4(a):
        """Provides my func4 functionality and verification."""
        return a

    t1 = Tensor((10,), DType.FLOAT32, "a")
    parser4 = ScriptCompiler(my_func4)
    parser4.compile(t1)


def test_ast_parser_missing_lists_2():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.ast_parser import ScriptCompiler

    def my_func2(a):
        """Provides my func2 functionality and verification."""
        return [a, 5]

    def my_func3(a):
        """Provides my func3 functionality and verification."""
        return a, 5

    t1 = Tensor((10,), DType.FLOAT32, "a")
    parser2 = ScriptCompiler(my_func2)
    parser2.compile(t1)
    parser3 = ScriptCompiler(my_func3)
    parser3.compile(t1)


def test_ast_parser_generic_visit():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.ast_parser import ScriptCompiler
    import ast

    sc = ScriptCompiler(lambda: None)
    sc.generic_visit(ast.Pass())


def test_ast_visit_name():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.ast_parser import ScriptCompiler
    from onnx9000.frontends.frontend.tensor import Tensor

    def simple_func(x):
        """Provides simple func functionality and verification."""
        return x

    comp = ScriptCompiler(simple_func)
    t = Tensor(shape=(), dtype=1, name="x")
    builder = comp.compile(t)
    assert len(builder.outputs) == 1
    assert builder.outputs[0].name == "x"
