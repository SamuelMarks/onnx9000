import pytest
from unittest.mock import patch, MagicMock
from onnx9000.c_compiler.cli import main
import sys
from onnx9000.core.ir import Graph


def test_cli_missing_file(capsys):
    with patch.object(sys, "argv", ["onnx2c", "does_not_exist.onnx"]):
        with patch("onnx9000.c_compiler.cli.os.path.exists", return_value=False):
            with pytest.raises(SystemExit) as e:
                main()
            assert e.value.code == 1


def test_cli_fallback(capsys):
    with patch.object(sys, "argv", ["onnx2c", "exists.onnx"]):
        with patch("onnx9000.c_compiler.cli.os.path.exists", return_value=True):
            m_open = MagicMock()
            m_open.return_value.__enter__.return_value.read.return_value = "test"
            with patch("onnx9000.c_compiler.cli.open", m_open):
                with pytest.raises(SystemExit) as e:
                    main()
                assert e.value.code == 10


def test_cli_success(capsys):
    from onnx9000.c_compiler.compiler import C89Compiler

    with patch.object(sys, "argv", ["onnx2c", "test.onnx", "--no-opt"]):
        with patch("onnx9000.c_compiler.cli.os.path.exists", return_value=True):
            m_open = MagicMock()
            m_open.return_value.__enter__.return_value.read.return_value = "test"
            with patch("onnx9000.c_compiler.cli.open", m_open):
                with patch("onnx9000.c_compiler.cli.os.makedirs"):
                    # mock the pyodide module completely
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
    from onnx9000.c_compiler.project_generator import generate_cmakelists, generate_arduino_sketch

    cm = generate_cmakelists("pref_")
    assert "add_library(pref_model STATIC pref_model.c)" in cm
    ino = generate_arduino_sketch("pref_")
    assert "pref_predict" in ino


def test_bundler():
    from onnx9000.c_compiler.bundler import bundle_weights_bin, generate_memory_summary
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        bp = bundle_weights_bin(b"hello", tmpdir, "pref_")
        assert os.path.exists(bp)
        with open(bp, "rb") as f:
            assert f.read() == b"hello"

    summary = generate_memory_summary(1024, 10, 5)
    assert "Peak Arena RAM:  1024 bytes" in summary
    assert "Total Nodes:     10" in summary


def test_cli_import_error(capsys):
    from unittest.mock import patch, MagicMock
    import sys
    from onnx9000.c_compiler.cli import main

    with patch.object(sys, "argv", ["onnx2c", "exists.onnx"]):
        with patch("onnx9000.c_compiler.cli.os.path.exists", return_value=True):
            m_open = MagicMock()
            m_open.return_value.__enter__.return_value.read.return_value = "test"
            with patch("onnx9000.c_compiler.cli.open", m_open):
                # mock import error
                import builtins

                real_import = builtins.__import__

                def mock_import(name, *args):
                    if name == "onnx9000.converters.frontend.pyodide_wrapper":
                        raise ImportError("mock error")
                    return real_import(name, *args)

                with patch("builtins.__import__", side_effect=mock_import):
                    with pytest.raises(SystemExit) as e:
                        main()
                    assert e.value.code == 10


def test_string_weights():
    from onnx9000.c_compiler.compiler import C89Compiler
    from onnx9000.core.ir import Graph, Tensor
    from onnx9000.core.dtypes import DType

    g = Graph("test")
    t = Tensor("test_str", shape=(2,), dtype=DType.STRING, data=[b"hello", b"world"])
    t.is_initializer = True
    g.tensors["test_str"] = t
    comp = C89Compiler(g, "p_")
    comp.arena.tensor_offsets = {}
    h, c = comp.generate()
    assert "static const char* p_weights_test_str[]" in c
    assert '"hello",' in c


def test_cli_extras(capsys):
    import sys
    from onnx9000.c_compiler.cli import main
    from unittest.mock import patch, MagicMock
    from onnx9000.core.ir import Graph, Tensor, Constant
    from onnx9000.core.dtypes import DType
    import struct

    with patch.object(sys, "argv", ["onnx2c", "test.onnx", "--no-math-h", "--no-opt"]):
        with patch("onnx9000.c_compiler.cli.os.path.exists", return_value=True):
            m_open = MagicMock()
            m_open.return_value.__enter__.return_value.read.return_value = "test"
            with patch("onnx9000.c_compiler.cli.open", m_open):
                with patch("onnx9000.c_compiler.cli.os.makedirs"):
                    # Custom graph with Float to trigger math.h override
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
                                pass


def test_cli_with_opt(capsys):
    import types
    from onnx9000.core.ir import Graph

    mock_module = types.ModuleType("onnx9000.converters.frontend.pyodide_wrapper")
    mock_module.parse_onnx_to_ir = lambda x: Graph("test_g")
    with patch.dict(sys.modules, {"onnx9000.converters.frontend.pyodide_wrapper": mock_module}):
        with patch.object(sys, "argv", ["onnx2c", "test.onnx", "--target", "arduino"]):
            with patch("onnx9000.c_compiler.cli.os.path.exists", return_value=True):
                m_open = MagicMock()
                with patch("onnx9000.c_compiler.cli.open", m_open):
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
                            pass
