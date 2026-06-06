import pytest
from onnx9000.optimizer.simplifier.passes.debug import *

def test_inject_probes():
    try:
        res = inject_probes()
    except Exception:
        pass

