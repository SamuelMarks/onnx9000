"""Mock tests for Pyodide integration."""

import pytest
from onnx9000.frontends.frontend.exporter import export
from onnx9000.frontends.frontend.models import ResNet18
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType
import io


def test_pyodide_export_buffer():
    """Provides semantic functionality and verification."""
    model = ResNet18()
    x = Tensor((1, 3, 224, 224), DType.FLOAT32, "input")
    buffer = io.BytesIO()
    export(model, x, buffer)
    binary_data = buffer.getvalue()
    assert len(binary_data) > 0
    assert binary_data.startswith(b"\x08")


def test_memory_optimization():
    """Provides semantic functionality and verification."""
    import gc

    model = ResNet18()
    x = Tensor((1, 3, 224, 224), DType.FLOAT32, "input")
    buffer = io.BytesIO()
    export(model, x, buffer)
    gc.collect()
    assert len(buffer.getvalue()) > 1024
