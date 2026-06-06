import pytest
from onnx9000.converters.mltools.sparkml import *

def test_parse_sparkml_pipeline():
    try:
        res = parse_sparkml_pipeline()
    except Exception:
        pass

