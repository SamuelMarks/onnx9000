import pytest
from onnx9000.core.logger import *


def test_get_logger():
    try:
        get_logger()
    except Exception:
        pass
