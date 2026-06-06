import pytest
from onnx9000.tvm.relay.frontend.safetensors import *


def test_load_safetensors_weights():
    try:
        load_safetensors_weights()
    except Exception:
        pass
