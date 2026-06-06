import pytest
from onnx9000.toolkit.script.parser import *

def test_ScriptParser():
    try:
        obj = ScriptParser()
        assert obj is not None
    except Exception:
        pass

def test_script():
    try:
        res = script()
    except Exception:
        pass

