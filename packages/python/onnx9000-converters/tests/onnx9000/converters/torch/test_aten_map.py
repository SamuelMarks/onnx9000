import pytest
from onnx9000.converters.torch.aten_map import *


def test_module_load():
    import onnx9000.converters.torch.aten_map

    assert onnx9000.converters.torch.aten_map is not None
