"""Tests for packages/python/onnx9000-c-compiler/tests/test_cli.py."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from onnx9000.c_compiler.cli import main
from onnx9000.core.ir import Graph


def test_cli_missing_file(capsys):
    """Test cli missing file."""
    with patch.object(sys, "argv", ["onnx2c", "does_not_exist.onnx"]):
        with patch("onnx9000.c_compiler.cli.os.path.exists", return_value=False):
            with pytest.raises(SystemExit) as e:
                main()
            assert e.value.code == 1


def test_cli_success(capsys):
    """Test cli success."""
    from onnx9000.c_compiler.compiler import C89Compiler

    with patch.object(sys, "argv", ["onnx2c", "test.onnx", "--no-opt"]):
        with patch("onnx9000.c_compiler.cli.os.path.exists", return_value=True):
            m_open = MagicMock()
            m_open.return_value.__enter__.return_value.read.return_value = "test"
            with (
                patch("onnx9000.c_compiler.cli.open", m_open),
                patch("onnx9000.c_compiler.cli.load", return_value=Graph("test_g")),
            ):
                with patch("onnx9000.c_compiler.cli.os.makedirs"):
                    import types

                    mock_module = types.ModuleType("onnx9000.converters.frontend.pyodide_wrapper")
                    mock_module.parse_onnx_to_ir = lambda x: Graph("test_g")
                    with patch.dict(
                        sys.modules, {"onnx9000.converters.frontend.pyodide_wrapper": mock_module}
                    ):
                        with patch.object(C89Compiler, "generate", return_value=("h", "c")):
                            C89Compiler.arena_size = 1024
                            main()
                            assert "Success" in capsys.readouterr().out


def test_generator_extras():
    """Test generator extras."""
    from onnx9000.c_compiler.project_generator import generate_arduino_sketch, generate_cmakelists

    cm = generate_cmakelists("pref_")
    assert "add_library(pref_model STATIC pref_model.c)" in cm
    ino = generate_arduino_sketch("pref_")
    assert "pref_predict" in ino


def test_bundler():
    """Test bundler."""
    import os
    import tempfile

    from onnx9000.c_compiler.bundler import bundle_weights_bin, generate_memory_summary

    with tempfile.TemporaryDirectory() as tmpdir:
        bp = bundle_weights_bin(b"hello", tmpdir, "pref_")
        assert os.path.exists(bp)
        with open(bp, "rb") as f:
            assert f.read() == b"hello"
    summary = generate_memory_summary(1024, 10, 5)
    assert "Peak Arena RAM:  1024 bytes" in summary
    assert "Total Nodes:     10" in summary


def test_cli_extras(capsys):
    """Test cli extras."""
    import struct
    import sys
    from unittest.mock import MagicMock, patch

    from onnx9000.c_compiler.cli import main
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Tensor

    with patch.object(sys, "argv", ["onnx2c", "test.onnx", "--no-math-h", "--no-opt"]):
        with patch("onnx9000.c_compiler.cli.os.path.exists", return_value=True):
            m_open = MagicMock()
            m_open.return_value.__enter__.return_value.read.return_value = "test"
            with (
                patch("onnx9000.c_compiler.cli.open", m_open),
                patch("onnx9000.c_compiler.cli.load", return_value=Graph("test_g")),
            ):
                with patch("onnx9000.c_compiler.cli.os.makedirs"):
                    g = Graph("t")
                    g.tensors["A"] = Tensor(
                        "A", shape=(1,), dtype=DType.FLOAT32, data=struct.pack("<f", 1.0)
                    )
                    import types

                    mock_module = types.ModuleType("onnx9000.converters.frontend.pyodide_wrapper")
                    mock_module.parse_onnx_to_ir = lambda x: g
                    with patch.dict(
                        sys.modules, {"onnx9000.converters.frontend.pyodide_wrapper": mock_module}
                    ):
                        from onnx9000.c_compiler.compiler import C89Compiler

                        with patch.object(C89Compiler, "generate", return_value=("h", "c")):
                            try:
                                main()
                                raise SystemExit
                            except SystemExit:
                                return None


def test_cli_with_opt(capsys):
    """Test cli with opt."""
    import types

    from onnx9000.core.ir import Graph

    mock_module = types.ModuleType("onnx9000.converters.frontend.pyodide_wrapper")
    mock_module.parse_onnx_to_ir = lambda x: Graph("test_g")
    mock_opt_module = types.ModuleType("onnx9000.optimizer.simplifier.api")
    mock_opt_module.simplify = lambda x: x
    mock_opt = types.ModuleType("onnx9000.optimizer.simplifier")
    mock_opt.api = mock_opt_module
    mock_opt_top = types.ModuleType("onnx9000.optimizer")
    mock_opt_top.simplifier = mock_opt

    with patch.dict(
        sys.modules,
        {
            "onnx9000.converters.frontend.pyodide_wrapper": mock_module,
            "onnx9000.optimizer.simplifier.api": mock_opt_module,
            "onnx9000.optimizer.simplifier": mock_opt,
            "onnx9000.optimizer": mock_opt_top,
        },
    ):
        with patch.object(sys, "argv", ["onnx2c", "test.onnx", "--target", "arduino"]):
            with patch("onnx9000.c_compiler.cli.os.path.exists", return_value=True):
                m_open = MagicMock()
                with (
                    patch("onnx9000.c_compiler.cli.open", m_open),
                    patch("onnx9000.c_compiler.cli.load", return_value=Graph("test_g")),
                ):
                    with patch("onnx9000.c_compiler.cli.os.makedirs"):
                        try:
                            with patch(
                                "onnx9000.c_compiler.compiler.C89Compiler.generate",
                                return_value=("h", "c"),
                            ):
                                import onnx9000.c_compiler.cli

                                onnx9000.c_compiler.cli.main()
                                raise SystemExit
                        except SystemExit:
                            return None
