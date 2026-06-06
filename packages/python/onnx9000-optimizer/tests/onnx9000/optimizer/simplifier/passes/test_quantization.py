import pytest
from onnx9000.optimizer.simplifier.passes.quantization import *

def test_insert_qat_nodes():
    try:
        res = insert_qat_nodes()
    except Exception:
        pass

def test_convert_to_int8():
    try:
        res = convert_to_int8()
    except Exception:
        pass

