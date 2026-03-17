from unittest.mock import MagicMock, patch
from onnx9000.converters.jit import compile


def test_compile_unsupported_target() -> None:
    with patch("onnx9000.converters.jit.load", return_value=MagicMock()):
        with patch("onnx9000.converters.jit.plan_memory"):
            if True:
                assert compile("dummy_path", target="unknown") is None
