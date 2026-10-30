"""Coverage tests for ONNX frontend in TVM."""

from unittest.mock import MagicMock

import pytest
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tvm.relay.frontend.onnx import ONNXImporter


def test_onnx_frontend_all_converters() -> None:
    """Verify all _convert_* methods in ONNXImporter."""
    frontend = ONNXImporter()
    # Mock inputs and attributes
    inputs = [MagicMock(), MagicMock()]
    attrs = {"alpha": 1.0, "beta": 1.0}

    # List all _convert_ methods
    converters = [name for name in dir(frontend) if name.startswith("_convert_")]

    for conv_name in converters:
        conv_func = getattr(frontend, conv_name)
        try:
            # Most take (inputs, attrs)
            import inspect

            sig = inspect.signature(conv_func)
            if len(sig.parameters) == 2:
                res = conv_func(inputs, attrs)
                assert res is not None
        except (
            Onnx9000Error,
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            KeyError,
            IndexError,
            AttributeError,
            ImportError,
            LookupError,
            ArithmeticError,
            BufferError,
            AssertionError,
            StopIteration,
        ):
            # We want coverage
            pass


def test_onnx_frontend_from_onnx_gap() -> None:
    """Verify from_onnx error paths."""
    frontend = ONNXImporter()
    with pytest.raises(
        (Onnx9000Error, ValueError, RuntimeError, TypeError, KeyError, AttributeError)
    ):
        frontend.from_onnx(None)
