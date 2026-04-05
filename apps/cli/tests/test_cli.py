"""Tests the cli module functionality."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from onnx9000_cli.main import (
    compile_cmd,
    convert_cmd,
    export_cmd,
    inspect_cmd,
    main,
    optimize_cmd,
    quantize_cmd,
    serve_cmd,
    simplify_cmd,
)


def test_cli_commands(capsys) -> None:
    """Tests the cli commands functionality."""
    import argparse

    # Test individual functions
    args = argparse.Namespace(
        model="test.onnx",
        script="test.py",
        src="tf",
        dst="onnx",
        skip_rules="",
        prune_inputs="",
        preserve_nodes="",
        input_shape=[],
        tensor_type=[],
        check_n=0,
        custom_ops=[],
        skip_fusions=False,
        skip_constant_folding=False,
        skip_shape_inference=False,
        skip_fuse_bn=False,
        dry_run=False,
        max_iterations=1,
        log_json=False,
        size_limit_mb=0.0,
        target_opset=None,
        strip_metadata=False,
        sort_value_info=False,
        output="out.onnx",
        format="onnx",
        overwrite=True,
        from_fmt="tf",
        to_fmt="onnx",
        diff_json=False,
        prune=False,
        quantize=False,
        sparsity=0.5,
    )
    inspect_cmd(args)
    assert "Inspecting test.onnx..." in capsys.readouterr().out

    with (
        patch("onnx9000_cli.main.load_onnx") as mock_load,
        patch("onnx9000_cli.main.save_onnx"),
        patch("onnx9000_cli.main.simplify") as mock_simplify,
        patch("importlib.util.spec_from_file_location") as mock_spec,
        patch("importlib.util.module_from_spec") as mock_module_from_spec,
        patch("onnx9000.converters.frontend.tracer.trace") as mock_trace,
        patch("onnx9000.core.exporter.export_graph"),
        patch("onnx9000.c_compiler.compiler.C89Compiler") as mock_compiler_cls,
    ):
        mock_compiler_cls.return_value.generate.return_value = ("header_content", "source_content")
        mock_graph = MagicMock()
        mock_graph.tensors = {}
        mock_graph.nodes = []
        mock_load.return_value = mock_graph
        mock_simplify.return_value = mock_graph

        mock_tracer = MagicMock()
        mock_tracer.to_graph.return_value = mock_graph
        mock_trace_cm = MagicMock()
        mock_trace_cm.__enter__.return_value = mock_tracer
        mock_trace.return_value = mock_trace_cm

        mock_s = MagicMock()
        mock_s.loader = MagicMock()
        mock_spec.return_value = mock_s

        mock_m = MagicMock()
        from onnx9000.converters.frontend.nn.module import Module

        class MockModel(Module):
            """Mock model."""

            def forward(self, x):
                """Forward."""
                return x

        mock_m.MyModel = MockModel
        mock_module_from_spec.return_value = mock_m
        mock_m.__dir__ = lambda self=None: ["MyModel"]

        simplify_cmd(args)
        assert "Simplifying..." in capsys.readouterr().out

        optimize_cmd(args)
        assert "Optimizing test.onnx..." in capsys.readouterr().out

        quantize_cmd(args)
        assert "Quantizing test.onnx..." in capsys.readouterr().out

        export_cmd(args)
        assert "Exporting test.py" in capsys.readouterr().out

        convert_cmd(args)
        assert "Converting from onnx" in capsys.readouterr().out

        serve_cmd(args)
        assert "Serving test.onnx on local server..." in capsys.readouterr().out

        compile_cmd(args)
        assert "Compiling test.onnx..." in capsys.readouterr().out


@patch("sys.argv", ["onnx9000", "inspect", "model.onnx"])
def test_main_valid() -> None:
    """Tests the main valid functionality."""
    with patch("onnx9000_cli.main.inspect_cmd") as mock_cmd:
        main()
        mock_cmd.assert_called_once()


@patch("sys.argv", ["onnx9000"])
def test_main_empty(capsys) -> None:
    """Tests the main empty functionality."""
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    assert "usage:" in capsys.readouterr().out


def test_main_execution() -> None:
    """Tests the main execution functionality."""
    import os
    import subprocess

    main_script = os.path.join(os.path.dirname(__file__), "..", "src", "onnx9000_cli", "main.py")
    result = subprocess.run(
        [sys.executable, main_script, "--help"],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_main_execution_runpy(monkeypatch) -> None:
    """Tests the main execution runpy functionality."""
    import os
    import runpy

    main_script = os.path.join(os.path.dirname(__file__), "..", "src", "onnx9000_cli", "main.py")
    monkeypatch.setattr("sys.argv", ["onnx9000", "--help"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(main_script, run_name="__main__")
    assert e.value.code == 0


def test_module_main_exec(monkeypatch) -> None:
    """Tests the module main exec functionality."""
    import os

    main_script = os.path.join(os.path.dirname(__file__), "..", "src", "onnx9000_cli", "main.py")
    monkeypatch.setattr("sys.argv", ["onnx9000", "--help"])
    with pytest.raises(SystemExit), open(main_script) as f:
        code = compile(f.read(), main_script, "exec")
        exec(code, {"__name__": "__main__"})


def test_run_module(monkeypatch) -> None:
    """Tests the run module functionality."""
    import runpy
    import warnings

    monkeypatch.setattr("sys.argv", ["onnx9000", "--help"])
    with pytest.raises(SystemExit):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message=".*found in sys.modules.*"
            )
            runpy.run_module("onnx9000_cli.main", run_name="__main__")
