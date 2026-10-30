"""Unit tests and verification suite for session typeerror functionality."""

import pytest
from onnx9000.backends.session import InferenceSession


def test_typeerror() -> None:
    """Verify that typeerror behaves as expected."""
    with pytest.raises(TypeError, match="InferenceSession requires an IR Graph object"):
        InferenceSession("invalid")
