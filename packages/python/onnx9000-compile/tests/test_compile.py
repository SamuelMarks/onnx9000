"""Comprehensive unit tests for onnx9000.compile and onnx9000_compile."""

import onnx9000_compile
import pytest
from onnx9000.compile import (
    compile_graph,
    emit_wasm_module,
    float16_to_bytes,
    float32_to_bytes,
    leb128_u,
    run,
)
from onnx9000.core.ir import Graph, Node


def test_run() -> None:
    """Verify default run status check across both namespace imports."""
    assert run() == "[compile] processed"
    assert onnx9000_compile.run() == "[compile] processed"


def test_leb128_u() -> None:
    """Verify unsigned LEB128 encoding for single-byte, multi-byte, and negative inputs."""
    assert leb128_u(0) == b"\x00"
    assert leb128_u(127) == b"\x7f"
    assert leb128_u(128) == b"\x80\x01"
    assert leb128_u(624485) == b"\xe5\x8e\x26"
    with pytest.raises(ValueError, match="non-negative integer"):
        leb128_u(-1)


def test_float_packing() -> None:
    """Verify IEEE-754 float32 and float16 binary packing."""
    assert float32_to_bytes(1.0) == b"\x00\x00\x80?"
    assert float16_to_bytes(1.0) == b"\x00<"


def test_emit_wasm_module_and_compile_graph() -> None:
    """Verify WebAssembly module emission and compile_graph for wasm, c, and invalid targets."""
    raw = emit_wasm_module()
    assert raw == b"\x00asm\x01\x00\x00\x00"

    g = Graph("test_wasm")
    g.add_node(Node("Relu", ["X"], ["Y"]))

    wasm_bytes = compile_graph(g, target="wasm")
    assert isinstance(wasm_bytes, bytes)
    assert wasm_bytes.startswith(b"\x00asm\x01\x00\x00\x00\x00")
    assert b"test_wasm" in wasm_bytes

    c_source = compile_graph(g, target="c")
    assert isinstance(c_source, str)
    assert "void forward_test_wasm(void)" in c_source
    assert "/* op: Relu */" in c_source

    with pytest.raises(ValueError, match="Unsupported compilation target"):
        compile_graph(g, target="invalid_target")
