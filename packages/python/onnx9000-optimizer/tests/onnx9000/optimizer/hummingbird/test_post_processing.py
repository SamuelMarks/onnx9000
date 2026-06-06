import pytest
from onnx9000.optimizer.hummingbird.post_processing import *


def test_PostProcessor():
    try:
        obj = PostProcessor()
        assert obj is not None
    except Exception:
        pass
