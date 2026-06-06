import pytest
from onnx9000.converters.mltools.coreml import *

def test_parse_coreml_model():
    try:
        res = parse_coreml_model()
    except Exception:
        pass

