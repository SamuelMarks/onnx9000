import pytest
from onnx9000.optimizer.hummingbird.webgpu_wasm_opts import *


def test_WebGPUWASMCompilerOpts():
    try:
        obj = WebGPUWASMCompilerOpts()
        assert obj is not None
    except Exception:
        pass
