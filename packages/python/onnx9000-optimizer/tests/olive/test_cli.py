"""Tests for CLI and serialization."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.cli import (
    ModelCache,
    is_package_under_5mb,
    optimize_cli,
    save_onnx,
    save_safetensors,
)
from onnx9000.optimizer.olive.model import OliveModel


def test_cli_cache(tmp_path) -> None:
    """Tests the cli cache functionality."""
    cache = ModelCache(str(tmp_path / ".cache"))
    assert cache.load("test") == {}
    cache.save("test", {"foo": "bar"})
    assert cache.load("test") == {"foo": "bar"}


def test_package_size() -> None:
    """Tests the package size functionality."""
    assert is_package_under_5mb()


def test_save_stubs() -> None:
    """Tests the save stubs functionality."""
    g = Graph("test")
    model = OliveModel(g)
    save_onnx(model, "test.onnx")
    save_safetensors(model, "test.safetensors")


def test_optimize_cli(tmp_path, monkeypatch) -> None:
    """Tests the optimize cli functionality."""
    import onnx9000.core.parser.core

    g = Graph("test")
    monkeypatch.setattr(onnx9000.core.parser.core, "load", lambda x: g)
    out_file = str(tmp_path / "out.onnx")
    optimize_cli("dummy.onnx", out_file, "webgpu")
    optimize_cli("dummy.onnx", out_file, "cpu")
    optimize_cli("dummy.onnx", out_file, "wasm_simd")
