import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import jit_cmd


def test_jit_cpp_success():
    args = argparse.Namespace(model="test.onnx", target="cpp", output="out.so")
    with patch("onnx9000_cli.main.load_onnx", return_value=MagicMock()):
        with patch("onnx9000.converters.jit.compiler.compile_cpp", return_value="out.so") as mc:
            jit_cmd(args)
            assert mc.call_count == 1


def test_jit_wasm_success():
    args = argparse.Namespace(model="test.onnx", target="wasm", output="out.js")
    with patch("onnx9000_cli.main.load_onnx", return_value=MagicMock()):
        with patch("onnx9000.converters.jit.compiler.compile_wasm", return_value="out.js") as mw:
            jit_cmd(args)
            assert mw.call_count == 1


def test_jit_unknown_target():
    args = argparse.Namespace(model="test.onnx", target="unknown", output="out.js")
    with patch("onnx9000_cli.main.load_onnx", return_value=MagicMock()):
        jit_cmd(args)
