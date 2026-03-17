import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from onnx9000.converters.jit.compiler import _get_compiler, compile_cpp, compile_wasm, load_module
from onnx9000.core import config
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import CompilationError
from onnx9000.core.ir import Graph, Tensor


def test_get_compiler(monkeypatch) -> None:
    monkeypatch.setattr(config, "ONNX9000_COMPILER", "my_custom_compiler")
    assert _get_compiler() == "my_custom_compiler"
    monkeypatch.setattr(config, "ONNX9000_COMPILER", "")
    with patch("sys.platform", "win32"):
        assert _get_compiler() == "cl.exe"
    with (
        patch("sys.platform", "linux"),
        patch("shutil.which", side_effect=lambda x: x == "clang++"),
    ):
        assert _get_compiler() == "clang++"
    with patch("sys.platform", "linux"):
        with patch("shutil.which", side_effect=lambda x: x == "g++"):
            assert _get_compiler() == "g++"
    with patch("sys.platform", "linux"):
        with patch("shutil.which", side_effect=lambda x: x == "c++"):
            assert _get_compiler() == "c++"
    with patch("sys.platform", "linux"), patch("shutil.which", return_value=False):
        with pytest.raises(CompilationError, match="No C\\+\\+ compiler found"):
            _get_compiler()


@patch("onnx9000.converters.jit.compiler.hash_graph", return_value="test_hash")
@patch("onnx9000.converters.jit.compiler.Generator")
@patch("onnx9000.converters.jit.compiler.Environment")
@patch("onnx9000.converters.jit.compiler.PackageLoader")
@patch("onnx9000.converters.jit.compiler.open")
@patch("subprocess.run")
def test_compile_cpp(
    mock_run, mock_open, mock_loader, mock_env, mock_gen, mock_hash, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(config, "ONNX9000_CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "ONNX9000_DEBUG", False)
    g = Graph("test")
    g.initializers.append("w")
    g.add_tensor(Tensor("w", (1,), DType.FLOAT32, is_initializer=True))
    mock_env.return_value.get_template.return_value.render.return_value = "rendered_template"
    mock_gen.return_value.generate.return_value = "model_code"
    (tmp_path / "onnx9000_test_hash.so").touch()
    with patch("sys.platform", "linux"):
        res = compile_cpp(g)
        assert res == tmp_path / "onnx9000_test_hash.so"
        mock_run.assert_not_called()
    (tmp_path / "onnx9000_test_hash.so").unlink()
    with (
        patch("sys.platform", "darwin"),
        patch("onnx9000.converters.jit.compiler._get_compiler", return_value="clang++"),
    ):
        monkeypatch.setattr(config, "ONNX9000_USE_ACCELERATE", True)
        res = compile_cpp(g)
        assert res == tmp_path / "onnx9000_test_hash.so"
        mock_run.assert_called_once()
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error", output="out")
    with (
        patch("sys.platform", "linux"),
        patch("onnx9000.converters.jit.compiler._get_compiler", return_value="clang++"),
    ):
        with pytest.raises(CompilationError, match="C\\+\\+ Compilation failed"):
            compile_cpp(g)


@patch("onnx9000.converters.jit.compiler.hash_graph", return_value="test_wasm_hash")
@patch("onnx9000.converters.jit.compiler.Generator")
@patch("onnx9000.converters.jit.compiler.Environment")
@patch("onnx9000.converters.jit.compiler.PackageLoader")
@patch("onnx9000.converters.jit.compiler.open")
@patch("subprocess.run")
@patch("shutil.which", return_value=True)
def test_compile_wasm(
    mock_which,
    mock_run,
    mock_open,
    mock_loader,
    mock_env,
    mock_gen,
    mock_hash,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(config, "ONNX9000_DEBUG", False)
    g = Graph("test")
    mock_env.return_value.get_template.return_value.render.return_value = "rendered_template"
    mock_gen.return_value.generate.return_value = "model_code"
    out_dir = tmp_path / "wasm"
    out_dir.mkdir()
    (out_dir / "onnx9000_test_wasm_hash.js").touch()
    (out_dir / "onnx9000_test_wasm_hash.wasm").touch()
    res = compile_wasm(g, out_dir)
    assert res == out_dir / "onnx9000_test_wasm_hash.js"
    mock_run.assert_not_called()
    (out_dir / "onnx9000_test_wasm_hash.js").unlink()
    res = compile_wasm(g, out_dir)
    assert res == out_dir / "onnx9000_test_wasm_hash.js"
    mock_run.assert_called_once()
    mock_which.return_value = False
    with pytest.raises(CompilationError, match="Emscripten compiler"):
        compile_wasm(g, out_dir)
    mock_which.return_value = True
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error", output="out")
    with pytest.raises(CompilationError, match="WASM Compilation failed"):
        compile_wasm(g, out_dir)


@patch("importlib.util.spec_from_file_location")
@patch("importlib.util.module_from_spec")
def test_load_module(mock_mod_from_spec, mock_spec_from_file) -> None:
    mock_spec = MagicMock()
    mock_spec.loader = MagicMock()
    mock_spec_from_file.return_value = mock_spec
    mock_mod = MagicMock()
    mock_mod_from_spec.return_value = mock_mod
    res = load_module(Path("dummy.so"))
    assert res == mock_mod
    mock_spec.loader.exec_module.assert_called_once_with(mock_mod)
    mock_spec_from_file.return_value = None
    with pytest.raises(CompilationError, match="Failed to load module"):
        load_module(Path("dummy.so"))
