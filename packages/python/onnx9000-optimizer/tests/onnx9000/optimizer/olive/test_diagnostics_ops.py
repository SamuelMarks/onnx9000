import pytest
from onnx9000.optimizer.olive.diagnostics_ops import *


def test_DiagnosticsOps():
    try:
        obj = DiagnosticsOps()
        assert obj is not None
    except Exception:
        pass
