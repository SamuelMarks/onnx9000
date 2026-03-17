"""Mock tests for Pyodide integration."""

import io

from onnx9000.converters.frontend.exporter import export
from onnx9000.converters.frontend.models import ResNet18
from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_pyodide_export_buffer() -> None:
    """Tests the test_pyodide_export_buffer functionality."""
    model = ResNet18()
    x = Tensor((1, 3, 224, 224), DType.FLOAT32, "input")
    buffer = io.BytesIO()
    export(model, x, buffer)
    binary_data = buffer.getvalue()
    assert len(binary_data) > 0
    assert binary_data.startswith(b"\x08")


def test_memory_optimization() -> None:
    """Tests the test_memory_optimization functionality."""
    import gc

    model = ResNet18()
    x = Tensor((1, 3, 224, 224), DType.FLOAT32, "input")
    buffer = io.BytesIO()
    export(model, x, buffer)
    gc.collect()
    assert len(buffer.getvalue()) > 1024
