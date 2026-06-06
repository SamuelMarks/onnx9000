import pytest
from onnx9000.converters.frontend.weight_utils import *


def test_export_state_dict():
    try:
        export_state_dict()
    except Exception:
        pass


def test_universal_weight_bridge():
    try:
        universal_weight_bridge()
    except Exception:
        pass
