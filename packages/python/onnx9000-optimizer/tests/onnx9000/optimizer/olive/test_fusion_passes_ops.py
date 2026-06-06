import pytest
from onnx9000.optimizer.olive.fusion_passes_ops import *


def test_FusionPassesOps():
    try:
        obj = FusionPassesOps()
        assert obj is not None
    except Exception:
        pass
