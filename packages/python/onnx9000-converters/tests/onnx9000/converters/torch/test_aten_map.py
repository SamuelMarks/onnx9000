"""Unit tests and verification suite for aten map functionality."""

import onnx9000.converters.torch.aten_map


def test_module_load() -> None:
    """Verify that module load behaves as expected."""
    assert onnx9000.converters.torch.aten_map is not None
