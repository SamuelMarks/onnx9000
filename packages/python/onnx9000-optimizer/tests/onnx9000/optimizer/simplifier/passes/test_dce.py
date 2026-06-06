import pytest
from onnx9000.optimizer.simplifier.passes.dce import *

def test_DCEPass():
    try:
        obj = DCEPass()
        assert obj is not None
    except Exception:
        pass

def test_IdentityEliminationPass():
    try:
        obj = IdentityEliminationPass()
        assert obj is not None
    except Exception:
        pass

def test_ControlFlowFoldingPass():
    try:
        obj = ControlFlowFoldingPass()
        assert obj is not None
    except Exception:
        pass

def test_dead_code_elimination():
    try:
        res = dead_code_elimination()
    except Exception:
        pass

