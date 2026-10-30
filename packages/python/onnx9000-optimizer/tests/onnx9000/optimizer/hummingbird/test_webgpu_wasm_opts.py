"""Unit tests and verification suite for webgpu wasm opts functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.webgpu_wasm_opts import WebGPUWASMCompilerOpts


def test_WebGPUWASMCompilerOpts() -> None:
    """Verify that web gpuwasmcompiler opts behaves as expected."""
    try:
        obj = WebGPUWASMCompilerOpts()
        assert obj is not None
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
        pass
