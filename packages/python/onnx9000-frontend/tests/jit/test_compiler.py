"""Module providing core logic and structural definitions."""

import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest
from onnx9000.core import config
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import CompilationError
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.frontend.jit import compiler, hasher, wrapper


@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_hash_graph(mock_loader) -> None:
    """Tests the test_hash_graph functionality."""
    g = Graph("test")
    g.add_tensor(
        Tensor(
            "w",
            (10,),
            DType.FLOAT32,
            is_initializer=True,
            data=np.array([1, 2, 3], dtype=np.float32),
        )
    )
    g.add_tensor(Tensor("in", (10,), DType.FLOAT32))
    g.initializers.append("w")
    g.inputs.append("in")
    h = hasher.hash_graph(g)
    assert isinstance(h, str)
    assert len(h) > 0


@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_wrapper(mock_loader) -> None:
    """Tests the test_wrapper functionality."""
    g = Graph("test")
    g.inputs = ["in1"]

    class MockCppModel:
        """Class MockCppModel implementation."""

        def forward(self, x):
            """Tests the forward functionality."""
            return x * 2

    class MockCppModel2:
        """Class MockCppModel2 implementation."""

        def forward(self, x):
            """Tests the forward functionality."""
            return (x * 2, x * 3)

    cm = wrapper.CompiledModel(MockCppModel(), g)
    res = cm(1)
    assert res == (2,)
    cm2 = wrapper.CompiledModel(MockCppModel2(), g)
    res2 = cm2(1)
    assert res2 == (2, 3)
    with pytest.raises(ValueError):
        cm(1, 2)


@patch("onnx9000.frontend.jit.compiler.config")
@patch("onnx9000.frontend.jit.compiler.sys")
@patch("onnx9000.frontend.jit.compiler.shutil")
@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_get_compiler(mock_loader, mock_shutil, mock_sys, mock_config) -> None:
    """Tests the test_get_compiler functionality."""
    mock_config.ONNX9000_COMPILER = ""
    mock_sys.platform = "win32"
    assert compiler._get_compiler() == "cl.exe"
    mock_sys.platform = "linux"
    mock_shutil.which.side_effect = lambda x: x == "clang++"
    assert compiler._get_compiler() == "clang++"
    mock_shutil.which.side_effect = lambda x: x == "g++"
    assert compiler._get_compiler() == "g++"
    mock_shutil.which.side_effect = lambda x: x == "c++"
    assert compiler._get_compiler() == "c++"
    mock_shutil.which.side_effect = lambda x: False
    with pytest.raises(CompilationError):
        compiler._get_compiler()
    mock_config.ONNX9000_COMPILER = "my_cc"
    assert compiler._get_compiler() == "my_cc"


@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_compile_cpp_cache_hit(mock_loader, tmp_path, monkeypatch) -> None:
    """Tests the test_compile_cpp_cache_hit functionality."""
    monkeypatch.setattr(config, "ONNX9000_CACHE_DIR", tmp_path)
    g = Graph("test")
    h = hasher.hash_graph(g)
    ext = ".pyd" if sys.platform == "win32" else ".so"
    out_path = tmp_path / f"onnx9000_{h}{ext}"
    out_path.write_text("mock")
    res = compiler.compile_cpp(g)
    assert res == out_path


@patch("onnx9000.frontend.jit.compiler.subprocess.run")
@patch("onnx9000.frontend.jit.compiler._get_compiler", return_value="g++")
@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_compile_cpp_compile_success(
    mock_loader, mock_get_cc, mock_run, tmp_path, monkeypatch
) -> None:
    """Tests the test_compile_cpp_compile_success functionality."""
    monkeypatch.setattr(config, "ONNX9000_CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "ONNX9000_DEBUG", False)
    g = Graph("test")
    g.initializers.append("w")
    g.add_tensor(Tensor("w", (10,), DType.FLOAT32, is_initializer=True))
    with patch("onnx9000.frontend.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontend.jit.compiler.Environment") as mock_env:
            mock_env.return_value.get_template.return_value.render.return_value = "template"
            res = compiler.compile_cpp(g)
            ext = ".pyd" if sys.platform == "win32" else ".so"
            assert res.name.endswith(ext)
            assert mock_run.called


@patch("onnx9000.frontend.jit.compiler.subprocess.run")
@patch("onnx9000.frontend.jit.compiler._get_compiler", return_value="g++")
@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_compile_cpp_compile_error(
    mock_loader, mock_get_cc, mock_run, tmp_path, monkeypatch
) -> None:
    """Tests the test_compile_cpp_compile_error functionality."""
    monkeypatch.setattr(config, "ONNX9000_CACHE_DIR", tmp_path)
    g = Graph("test")
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", b"out", b"err")
    with patch("onnx9000.frontend.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontend.jit.compiler.Environment") as mock_env:
            mock_env.return_value.get_template.return_value.render.return_value = "template"
            with pytest.raises(CompilationError):
                compiler.compile_cpp(g)


@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_compile_wasm_cache_hit(mock_loader, tmp_path) -> None:
    """Tests the test_compile_wasm_cache_hit functionality."""
    g = Graph("test")
    h = hasher.hash_graph(g)
    out_dir = tmp_path / "wasm"
    out_dir.mkdir()
    js_path = out_dir / f"onnx9000_{h}.js"
    wasm_path = out_dir / f"onnx9000_{h}.wasm"
    js_path.write_text("mock")
    wasm_path.write_text("mock")
    res = compiler.compile_wasm(g, out_dir)
    assert res == js_path


@patch("onnx9000.frontend.jit.compiler.shutil.which", return_value=None)
@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_compile_wasm_no_compiler(mock_loader, mock_which, tmp_path) -> None:
    """Tests the test_compile_wasm_no_compiler functionality."""
    g = Graph("test")
    with patch("onnx9000.frontend.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontend.jit.compiler.Environment") as mock_env:
            mock_env.return_value.get_template.return_value.render.return_value = "template"
            with pytest.raises(CompilationError, match="Emscripten compiler"):
                compiler.compile_wasm(g, tmp_path)


@patch("onnx9000.frontend.jit.compiler.subprocess.run")
@patch("onnx9000.frontend.jit.compiler.shutil.which", return_value="emcc")
@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_compile_wasm_compile_success(
    mock_loader, mock_which, mock_run, tmp_path, monkeypatch
) -> None:
    """Tests the test_compile_wasm_compile_success functionality."""
    monkeypatch.setattr(config, "ONNX9000_DEBUG", False)
    g = Graph("test")
    with patch("onnx9000.frontend.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontend.jit.compiler.Environment") as mock_env:
            mock_env.return_value.get_template.return_value.render.return_value = "template"
            res = compiler.compile_wasm(g, tmp_path)
            assert res.name.endswith(".js")
            assert mock_run.called


@patch("onnx9000.frontend.jit.compiler.subprocess.run")
@patch("onnx9000.frontend.jit.compiler.shutil.which", return_value="emcc")
@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_compile_wasm_compile_error(mock_loader, mock_which, mock_run, tmp_path) -> None:
    """Tests the test_compile_wasm_compile_error functionality."""
    g = Graph("test")
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", b"out", b"err")
    with patch("onnx9000.frontend.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontend.jit.compiler.Environment") as mock_env:
            mock_env.return_value.get_template.return_value.render.return_value = "template"
            with pytest.raises(CompilationError):
                compiler.compile_wasm(g, tmp_path)


@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_load_module(mock_loader, tmp_path) -> None:
    """Tests the test_load_module functionality."""
    f = tmp_path / "mock_mod.py"
    f.write_text("x = 1")
    mod = compiler.load_module(f)
    assert mod.x == 1
    with pytest.raises(CompilationError):
        compiler.load_module(tmp_path)


@patch("onnx9000.frontend.jit.compiler.subprocess.run")
@patch("onnx9000.frontend.jit.compiler._get_compiler", return_value="g++")
@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_compile_cpp_not_darwin(mock_loader, mock_get_cc, mock_run, tmp_path, monkeypatch) -> None:
    """Tests the test_compile_cpp_not_darwin functionality."""
    monkeypatch.setattr(config, "ONNX9000_CACHE_DIR", tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    g = Graph("test")
    with patch("onnx9000.frontend.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontend.jit.compiler.Environment") as mock_env:
            mock_env.return_value.get_template.return_value.render.return_value = "template"
            res = compiler.compile_cpp(g)
            assert res.name.endswith(".so")


@patch("onnx9000.frontend.jit.compiler.PackageLoader")
def test_hash_graph_nodes(mock_loader) -> None:
    """Tests the test_hash_graph_nodes functionality."""
    g = Graph("test")
    n = Node("Add", ["a"], ["b"], {}, "n1")
    g.add_node(n)
    h = hasher.hash_graph(g)
    assert len(h) > 0
