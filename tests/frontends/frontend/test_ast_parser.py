"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.frontends.frontend.ast_parser import ScriptCompiler
from onnx9000.frontends.frontend.tensor import Tensor
import ast


def test_script_compiler_basic():
    """Provides semantic functionality and verification."""

    def simple_func(x):
        """Provides simple func functionality and verification."""
        a = x
        return a

    compiler = ScriptCompiler(simple_func)
    builder = compiler.compile(Tensor(name="x"))
    assert builder.name == "simple_func"


def test_script_compiler_return_multiple():
    """Provides semantic functionality and verification."""

    def return_tuple(x, y):
        """Provides return tuple functionality and verification."""
        return x, y

    def return_list(x, y):
        """Provides return list functionality and verification."""
        return [x, y]

    compiler = ScriptCompiler(return_tuple)
    builder = compiler.compile(Tensor(name="x"), Tensor(name="y"))
    compiler2 = ScriptCompiler(return_list)
    builder2 = compiler2.compile(Tensor(name="x"), Tensor(name="y"))


def test_script_compiler_if():
    """Provides semantic functionality and verification."""

    def if_func(x):
        """Provides if func functionality and verification."""
        if x:
            y = x
        else:
            y = x
        return y

    compiler = ScriptCompiler(if_func)
    compiler.compile(Tensor(name="x"))


def test_script_compiler_for_while():
    """Provides semantic functionality and verification."""

    def loops(x):
        """Provides loops functionality and verification."""
        for i in [1, 2]:
            a = i
        while x:
            b = x
        return x

    compiler = ScriptCompiler(loops)
    compiler.compile(Tensor(name="x"))


def test_script_compiler_generic():
    """Provides semantic functionality and verification."""

    class DummyCompiler(ScriptCompiler):
        """Represents the DummyCompiler class."""

        def visit_Pass(self, node):
            """Provides visit Pass functionality and verification."""
            return self.generic_visit(node)

    def pass_func():
        """Provides pass func functionality and verification."""
        pass

    compiler = DummyCompiler(pass_func)
    compiler.compile()


def test_script_compiler_less_args():
    """Provides semantic functionality and verification."""

    def my_func(x, y):
        """Provides my func functionality and verification."""
        a = x
        return a

    compiler = ScriptCompiler(my_func)
    compiler.compile(Tensor(name="x"))
