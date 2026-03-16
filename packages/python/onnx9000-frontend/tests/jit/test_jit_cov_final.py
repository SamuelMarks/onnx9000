from unittest.mock import MagicMock, patch
from onnx9000.frontend.jit import compile


def test_compile_unsupported_target() -> None:
    with patch("onnx9000.frontend.jit.load", return_value=MagicMock()):
        with patch("onnx9000.frontend.jit.plan_memory"):
            if True:
                assert compile("dummy_path", target="unknown") is None
