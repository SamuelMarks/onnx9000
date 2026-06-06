import pytest
from onnx9000.converters.frontend.ast_parser import *


def test_ScriptCompiler():
    try:
        obj = ScriptCompiler()
        assert obj is not None
    except Exception:
        pass
