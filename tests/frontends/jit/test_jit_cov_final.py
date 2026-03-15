import pytest
from onnx9000.frontends.jit import compile
from unittest.mock import patch, MagicMock


def test_compile_unsupported_target():
    with patch("onnx9000.frontends.jit.load", return_value=MagicMock()):
        with patch("onnx9000.frontends.jit.plan_memory"):
            with pytest.raises(
                NotImplementedError, match="Target unknown not supported yet."
            ):
                compile("dummy_path", target="unknown")
