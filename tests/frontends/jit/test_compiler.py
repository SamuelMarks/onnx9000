"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
import sys
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from onnx9000.core.ir import Graph, Tensor, Node
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import CompilationError
from onnx9000.core import config
from onnx9000.frontends.jit import compiler, hasher, wrapper


def test_hash_graph():
    """Provides semantic functionality and verification."""
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


def test_wrapper():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.inputs = ["in1"]

    class MockCppModel:
        """Represents the MockCppModel class."""

        def forward(self, x):
            """Provides forward functionality and verification."""
            return x * 2

    class MockCppModel2:
        """Represents the MockCppModel2 class."""

        def forward(self, x):
            """Provides forward functionality and verification."""
            return x * 2, x * 3

    cm = wrapper.CompiledModel(MockCppModel(), g)
    res = cm(1)
    assert res == (2,)
    cm2 = wrapper.CompiledModel(MockCppModel2(), g)
    res2 = cm2(1)
    assert res2 == (2, 3)
    with pytest.raises(ValueError):
        cm(1, 2)


@patch("onnx9000.frontends.jit.compiler.config")
@patch("onnx9000.frontends.jit.compiler.sys")
@patch("onnx9000.frontends.jit.compiler.shutil")
def test_get_compiler(mock_shutil, mock_sys, mock_config):
    """Provides semantic functionality and verification."""
    mock_config.ONNX9000_COMPILER = ""
    mock_sys.platform = "win32"
    assert compiler._get_compiler() == "cl.exe"
    mock_sys.platform = "linux"
    mock_shutil.which.side_effect = lambda x: True if x == "clang++" else False
    assert compiler._get_compiler() == "clang++"
    mock_shutil.which.side_effect = lambda x: True if x == "g++" else False
    assert compiler._get_compiler() == "g++"
    mock_shutil.which.side_effect = lambda x: True if x == "c++" else False
    assert compiler._get_compiler() == "c++"
    mock_shutil.which.side_effect = lambda x: False
    with pytest.raises(CompilationError):
        compiler._get_compiler()
    mock_config.ONNX9000_COMPILER = "my_cc"
    assert compiler._get_compiler() == "my_cc"


def test_compile_cpp_cache_hit(tmp_path, monkeypatch):
    """Provides semantic functionality and verification."""
    monkeypatch.setattr(config, "ONNX9000_CACHE_DIR", tmp_path)
    g = Graph("test")
    h = hasher.hash_graph(g)
    ext = ".pyd" if sys.platform == "win32" else ".so"
    out_path = tmp_path / f"onnx9000_{h}{ext}"
    out_path.write_text("mock")
    res = compiler.compile_cpp(g)
    assert res == out_path


@patch("onnx9000.frontends.jit.compiler.subprocess.run")
@patch("onnx9000.frontends.jit.compiler._get_compiler", return_value="g++")
def test_compile_cpp_compile_success(mock_get_cc, mock_run, tmp_path, monkeypatch):
    """Provides semantic functionality and verification."""
    monkeypatch.setattr(config, "ONNX9000_CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "ONNX9000_DEBUG", False)
    g = Graph("test")
    g.initializers.append("w")
    g.add_tensor(Tensor("w", (10,), DType.FLOAT32, is_initializer=True))
    with patch("onnx9000.frontends.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontends.jit.compiler.Environment") as mock_env:
            (
                mock_env.return_value.get_template.return_value.render.return_value
            ) = "template"
            res = compiler.compile_cpp(g)
            ext = ".pyd" if sys.platform == "win32" else ".so"
            assert res.name.endswith(ext)
            assert mock_run.called


@patch("onnx9000.frontends.jit.compiler.subprocess.run")
@patch("onnx9000.frontends.jit.compiler._get_compiler", return_value="g++")
def test_compile_cpp_compile_error(mock_get_cc, mock_run, tmp_path, monkeypatch):
    """Provides semantic functionality and verification."""
    monkeypatch.setattr(config, "ONNX9000_CACHE_DIR", tmp_path)
    g = Graph("test")
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", b"out", b"err")
    with patch("onnx9000.frontends.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontends.jit.compiler.Environment") as mock_env:
            (
                mock_env.return_value.get_template.return_value.render.return_value
            ) = "template"
            with pytest.raises(CompilationError):
                compiler.compile_cpp(g)


def test_compile_wasm_cache_hit(tmp_path):
    """Provides semantic functionality and verification."""
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


@patch("onnx9000.frontends.jit.compiler.shutil.which", return_value=None)
def test_compile_wasm_no_compiler(mock_which, tmp_path):
    """Provides semantic functionality and verification."""
    g = Graph("test")
    with patch("onnx9000.frontends.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontends.jit.compiler.Environment") as mock_env:
            (
                mock_env.return_value.get_template.return_value.render.return_value
            ) = "template"
            with pytest.raises(CompilationError, match="Emscripten compiler"):
                compiler.compile_wasm(g, tmp_path)


@patch("onnx9000.frontends.jit.compiler.subprocess.run")
@patch("onnx9000.frontends.jit.compiler.shutil.which", return_value="emcc")
def test_compile_wasm_compile_success(mock_which, mock_run, tmp_path, monkeypatch):
    """Provides semantic functionality and verification."""
    monkeypatch.setattr(config, "ONNX9000_DEBUG", False)
    g = Graph("test")
    with patch("onnx9000.frontends.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontends.jit.compiler.Environment") as mock_env:
            (
                mock_env.return_value.get_template.return_value.render.return_value
            ) = "template"
            res = compiler.compile_wasm(g, tmp_path)
            assert res.name.endswith(".js")
            assert mock_run.called


@patch("onnx9000.frontends.jit.compiler.subprocess.run")
@patch("onnx9000.frontends.jit.compiler.shutil.which", return_value="emcc")
def test_compile_wasm_compile_error(mock_which, mock_run, tmp_path):
    """Provides semantic functionality and verification."""
    g = Graph("test")
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", b"out", b"err")
    with patch("onnx9000.frontends.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontends.jit.compiler.Environment") as mock_env:
            (
                mock_env.return_value.get_template.return_value.render.return_value
            ) = "template"
            with pytest.raises(CompilationError):
                compiler.compile_wasm(g, tmp_path)


def test_load_module(tmp_path):
    """Provides semantic functionality and verification."""
    f = tmp_path / "mock_mod.py"
    f.write_text("x = 1")
    mod = compiler.load_module(f)
    assert mod.x == 1
    with pytest.raises(CompilationError):
        compiler.load_module(tmp_path)


@patch("onnx9000.frontends.jit.compiler.subprocess.run")
@patch("onnx9000.frontends.jit.compiler._get_compiler", return_value="g++")
def test_compile_cpp_not_darwin(mock_get_cc, mock_run, tmp_path, monkeypatch):
    """Provides semantic functionality and verification."""
    monkeypatch.setattr(config, "ONNX9000_CACHE_DIR", tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    g = Graph("test")
    with patch("onnx9000.frontends.jit.compiler.Generator") as mock_gen:
        mock_gen.return_value.generate.return_value = "code"
        with patch("onnx9000.frontends.jit.compiler.Environment") as mock_env:
            (
                mock_env.return_value.get_template.return_value.render.return_value
            ) = "template"
            res = compiler.compile_cpp(g)
            assert res.name.endswith(".so")


def test_hash_graph_nodes():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    n = Node("Add", ["a"], ["b"], {}, "n1")
    g.add_node(n)
    h = hasher.hash_graph(g)
    assert len(h) > 0
