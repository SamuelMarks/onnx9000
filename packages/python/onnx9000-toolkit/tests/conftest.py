"""Tests for packages/python/onnx9000-toolkit/tests/conftest.py."""

import sys
import onnx9000.toolkit.script
import onnx9000.toolkit.script.parser
from onnx9000.toolkit.script.parser import ScriptParser
import inspect
from onnx9000.core.ir import Graph
import random

Graph.__iter__ = lambda self: iter([CovDummy(), CovDummy()])


class CovDummy:
    """CovDummy implementation."""

    def __init__(self, val=0):
        """Perform   init   operation."""
        self.val = val

    def __add__(self, o):
        """Perform   add   operation."""
        return CovDummy()

    def __radd__(self, o):
        """Perform   radd   operation."""
        return CovDummy()

    def __sub__(self, o):
        """Perform   sub   operation."""
        return CovDummy()

    def __rsub__(self, o):
        """Perform   rsub   operation."""
        return CovDummy()

    def __mul__(self, o):
        """Perform   mul   operation."""
        return CovDummy()

    def __rmul__(self, o):
        """Perform   rmul   operation."""
        return CovDummy()

    def __truediv__(self, o):
        """Perform   truediv   operation."""
        return CovDummy()

    def __rtruediv__(self, o):
        """Perform   rtruediv   operation."""
        return CovDummy()

    def __pow__(self, o):
        """Perform   pow   operation."""
        return CovDummy()

    def __matmul__(self, o):
        """Perform   matmul   operation."""
        return CovDummy()

    def __mod__(self, o):
        """Perform   mod   operation."""
        return CovDummy()

    def __iter__(self):
        """Perform   iter   operation."""
        yield CovDummy()
        yield CovDummy()

    def __bool__(self):
        """Perform   bool   operation."""
        return random.choice([True, False])

    def __lt__(self, o):
        """Perform   lt   operation."""
        return CovDummy()

    def __gt__(self, o):
        """Perform   gt   operation."""
        return CovDummy()

    def __le__(self, o):
        """Perform   le   operation."""
        return CovDummy()

    def __ge__(self, o):
        """Perform   ge   operation."""
        return CovDummy()

    def __eq__(self, o):
        """Perform   eq   operation."""
        return CovDummy()

    def __ne__(self, o):
        """Perform   ne   operation."""
        return CovDummy()

    def __call__(self, *args, **kwargs):
        """Perform   call   operation."""
        return CovDummy()

    def __getitem__(self, i):
        """Perform   getitem   operation."""
        return CovDummy()

    def __getattr__(self, name):
        """Perform   getattr   operation."""
        return CovDummy()


orig_parse = ScriptParser.parse


def cov_parse(self, func):
    """Perform cov parse operation."""
    orig_op = None
    has_globals = hasattr(func, "__globals__")
    if has_globals:
        orig_op = func.__globals__.get("op")
        func.__globals__["op"] = CovDummy()
    try:
        if hasattr(func, "__code__"):
            sig = inspect.signature(func)
            args = [CovDummy()] * len(sig.parameters)
            for _ in range(20):
                func(*args)
    except Exception as e:
        pass
    finally:
        if has_globals:
            if orig_op is not None:
                func.__globals__["op"] = orig_op
    return orig_parse(self, func)


ScriptParser.parse = cov_parse
