"""Tests the cli module functionality."""

import sys
from unittest.mock import patch

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
        overwrite=True,
        diff_json=False,
    )
    inspect_cmd(args)
    assert "Inspecting test.onnx..." in capsys.readouterr().out

    with (
        patch("onnx9000_cli.main.load_onnx") as mock_load,
        patch("onnx9000_cli.main.save_onnx"),
        patch("onnx9000_cli.main.simplify") as mock_simplify,
    ):
        mock_load.return_value = "mock_graph"
        mock_simplify.return_value = "mock_simplified_graph"
        simplify_cmd(args)
    assert "Simplifying..." in capsys.readouterr().out

    optimize_cmd(args)
    assert "Optimizing test.onnx..." in capsys.readouterr().out

    quantize_cmd(args)
    assert "Quantizing test.onnx..." in capsys.readouterr().out

    export_cmd(args)
    assert "Exporting test.py..." in capsys.readouterr().out

    convert_cmd(args)
    assert "Converting from tf to onnx..." in capsys.readouterr().out

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
    import subprocess

    result = subprocess.run(
        [sys.executable, "apps/cli/src/onnx9000_cli/main.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_main_execution_runpy(monkeypatch) -> None:
    """Tests the main execution runpy functionality."""
    import runpy

    monkeypatch.setattr("sys.argv", ["onnx9000", "--help"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path("apps/cli/src/onnx9000_cli/main.py", run_name="__main__")
    assert e.value.code == 0


def test_module_main_exec(monkeypatch) -> None:
    """Tests the module main exec functionality."""
    monkeypatch.setattr("sys.argv", ["onnx9000", "--help"])
    with pytest.raises(SystemExit), open("apps/cli/src/onnx9000_cli/main.py") as f:
        code = compile(f.read(), "apps/cli/src/onnx9000_cli/main.py", "exec")
        exec(code, {"__name__": "__main__"})


def test_run_module(monkeypatch) -> None:
    """Tests the run module functionality."""
    import runpy

    monkeypatch.setattr("sys.argv", ["onnx9000", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module("onnx9000_cli.main", run_name="__main__")
