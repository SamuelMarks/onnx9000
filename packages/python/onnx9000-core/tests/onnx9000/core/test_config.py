"""Unit tests and verification suite for config functionality."""

import onnx9000.core.config


def test_module_load() -> None:
    """Verify that module load behaves as expected."""
    assert onnx9000.core.config is not None
