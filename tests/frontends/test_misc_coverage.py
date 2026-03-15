"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.frontends.frontend.ast_parser import ScriptCompiler
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_ast_parser_miss():
    """Provides semantic functionality and verification."""

    def f1(x, y=None):
        """Provides f1 functionality and verification."""
        return x, y

    def f2(x):
        """Provides f2 functionality and verification."""
        return [x]

    sc1 = ScriptCompiler(f1)
    sc1.compile(Tensor((1,), DType.FLOAT32))
    sc2 = ScriptCompiler(f2)
    sc2.compile(Tensor((1,), DType.FLOAT32))
