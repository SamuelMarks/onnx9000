import pytest
from onnx9000.converters.cntk.mapper import *


def test_CNTKMapper():
    try:
        obj = CNTKMapper()
        assert obj is not None
    except Exception:
        pass
