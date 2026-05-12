import argparse
from unittest.mock import mock_open, patch

from onnx9000_cli.main import diffusers_cmd, tensorrt_cmd


def test_tensorrt_cmd():
    args = argparse.Namespace(model="dummy.onnx")
    with patch("builtins.print") as mock_print:
        tensorrt_cmd(args)
    mock_print.assert_any_call("Exporting ONNX model to TensorRT Builder script: dummy.onnx...")
    mock_print.assert_any_call("import tensorrt as trt")


def test_diffusers_cmd_stdout():
    args = argparse.Namespace(model="dummy", prompt="cat", output=None)
    with patch("builtins.print") as mock_print:
        diffusers_cmd(args)
    mock_print.assert_any_call("Initializing Diffusion Pipeline from: dummy...")
    mock_print.assert_any_call("Generated image tensor successfully [1, 3, 512, 512]")


def test_diffusers_cmd_file():
    args = argparse.Namespace(model="dummy", prompt="cat", output="out.png")
    m_open = mock_open()
    with patch("builtins.open", m_open):
        diffusers_cmd(args)
    m_open.assert_called_once_with("out.png", "w")
    m_open().write.assert_called_once_with("Generated tensor mock")


def test_tvm_cmd():
    args = argparse.Namespace(model="dummy.onnx", target="llvm")
    with (
        patch("onnx9000.tvm.build_module.build") as mock_build,
        patch("builtins.print") as mock_print,
    ):
        from onnx9000_cli.main import tvm_cmd

        tvm_cmd(args)
    mock_print.assert_any_call("TVM compiling dummy.onnx for llvm")
    mock_build.assert_called_once()
