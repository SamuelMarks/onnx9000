import pytest
from unittest.mock import patch
import sys
from onnx9000_cli.main import (
    main,
    inspect_cmd,
    simplify_cmd,
    optimize_cmd,
    quantize_cmd,
    export_cmd,
    convert_cmd,
    serve_cmd,
    compile_cmd,
)


def test_cli_commands(capsys):
    import argparse

    # Test individual functions
    args = argparse.Namespace(model="test.onnx", script="test.py", src="tf", dst="onnx")
    inspect_cmd(args)
    assert "Inspecting test.onnx..." in capsys.readouterr().out

    simplify_cmd(args)
    assert "Simplifying test.onnx..." in capsys.readouterr().out

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
def test_main_valid():
    with patch("onnx9000_cli.main.inspect_cmd") as mock_cmd:
        main()
        mock_cmd.assert_called_once()


@patch("sys.argv", ["onnx9000"])
def test_main_empty(capsys):
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    assert "usage:" in capsys.readouterr().out


def test_main_execution():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "apps/cli/src/onnx9000_cli/main.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_main_execution_runpy(monkeypatch):
    import runpy

    monkeypatch.setattr("sys.argv", ["onnx9000", "--help"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path("apps/cli/src/onnx9000_cli/main.py", run_name="__main__")
    assert e.value.code == 0


def test_module_main_exec(monkeypatch):
    monkeypatch.setattr("sys.argv", ["onnx9000", "--help"])
    with pytest.raises(SystemExit):
        with open("apps/cli/src/onnx9000_cli/main.py") as f:
            code = compile(f.read(), "apps/cli/src/onnx9000_cli/main.py", "exec")
            exec(code, {"__name__": "__main__"})


def test_run_module(monkeypatch):
    import runpy

    monkeypatch.setattr("sys.argv", ["onnx9000", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module("onnx9000_cli.main", run_name="__main__")
