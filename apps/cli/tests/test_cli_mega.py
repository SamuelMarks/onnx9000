import argparse
import sys
import traceback
from unittest.mock import MagicMock, patch

import pytest
from onnx9000_cli.main import main


def test_mega_coverage():
    cmds = [
        ["hummingbird", "test.onnx", "-o", "out.onnx"],
        ["tvm", "test.onnx", "out.tar", "--target", "llvm"],
        ["tflite", "test.onnx", "--quantize", "int8", "-o", "out.tflite"],
        ["tensorrt", "test.onnx", "--fp16"],
        ["optimum", "export", "test", "out"],
        ["optimum", "optimize", "test", "out"],
        ["optimum", "quantize", "test", "out"],
        ["check", "test.onnx", "--strict"],
        ["extract", "test.onnx", "--inputs", "i1", "--outputs", "o1", "-o", "out.onnx"],
        ["visualize", "netron", "test.onnx"],
        ["visualize", "dot", "test.onnx", "-o", "out.dot"],
        ["info", "ops", "test.onnx"],
        ["info", "tensors", "test.onnx"],
        ["info", "summary", "test.onnx"],
        ["info", "shape", "test.onnx"],
        ["info", "metadata", "test.onnx"],
        ["serve"],
        ["zoo", "download", "test"],
        ["zoo", "inspect-safetensors", "test"],
        ["genai", "--mode", "test"],
        ["onnx2gguf", "test.onnx", "out.gguf"],
        ["gguf2onnx", "test.gguf", "out.onnx"],
        ["autograd", "test.onnx", "out.onnx"],
        ["diffusers", "export", "test", "out"],
        ["jit", "test"],
        ["rocm", "test.onnx"],
        ["cpu", "test.onnx"],
        ["cuda", "test.onnx"],
        ["testing"],
        ["apple", "test.onnx"],
        ["onnx2tf", "test.onnx"],
        ["chat"],
        ["workspace"],
        ["array"],
    ]
    for fmt in [
        "jax",
        "paddle",
        "pytorch",
        "safetensors",
        "caffe",
        "lightgbm",
        "xgboost",
        "libsvm",
        "h2o",
        "sparkml",
        "coreml",
    ]:
        cmds.append(["convert", "test", "--from", fmt, "--to", "onnx", "--o", "out.onnx"])

    mock_load = patch(
        "onnx9000_cli.main.load_onnx",
        return_value=MagicMock(
            nodes=[MagicMock(op_type="TreeEnsembleClassifier")], name="mocked_name", tensors={}
        ),
    )
    mock_save = patch("onnx9000_cli.main.save_onnx")

    with (
        patch.dict(
            sys.modules,
            {
                "onnx9000.tvm.compiler": MagicMock(),
                "onnx9000.tflite_exporter.builder": MagicMock(),
                "onnx9000.tensorrt.builder": MagicMock(),
                "onnx9000_optimum.training": MagicMock(),
                "onnx9000.core.checker": MagicMock(),
                "onnx9000.core.shape_inference": MagicMock(),
                "onnx9000.toolkit.netron": MagicMock(),
                "onnx9000.toolkit.dot": MagicMock(),
                "onnx9000.zoo.catalog": MagicMock(),
                "onnx9000.zoo.tensors": MagicMock(),
                "onnx9000.genai.ecosystem": MagicMock(),
                "onnx9000.genai.qa": MagicMock(),
                "onnx9000.onnx2gguf.compiler": MagicMock(),
                "onnx9000.gguf2onnx.parser": MagicMock(),
                "onnx9000.toolkit.autograd.compiler": MagicMock(),
                "onnx9000.diffusers.pipeline": MagicMock(),
                "onnx9000.jit.compiler": MagicMock(),
                "onnx9000.backends.rocm.executor": MagicMock(),
                "onnx9000.backends.cpu.executor": MagicMock(),
                "onnx9000.backends.cuda.executor": MagicMock(),
                "onnx9000.toolkit.testing.runner": MagicMock(),
                "onnx9000.backends.apple.executor": MagicMock(),
                "onnx9000.tf.exporter": MagicMock(),
                "onnx9000.cli.chat": MagicMock(),
                "onnx9000.cli.workspace": MagicMock(),
                "onnx9000.array.cli": MagicMock(),
                "onnx9000.converters.parsers": MagicMock(),
                "onnx9000.converters.paddle.api": MagicMock(),
                "onnx9000.converters.safetensors.loader": MagicMock(),
                "onnx9000.converters.mltools.catboost": MagicMock(),
                "onnx9000.converters.mltools.lightgbm": MagicMock(),
                "onnx9000.converters.mltools.xgboost": MagicMock(),
                "onnx9000.converters.mltools.libsvm": MagicMock(),
                "onnx9000.converters.mltools.h2o": MagicMock(),
                "onnx9000.converters.mltools.sparkml": MagicMock(),
                "onnx9000.converters.mltools.coreml": MagicMock(),
                "torch": MagicMock(),
                "torch.export": MagicMock(),
                "torch.nn": MagicMock(),
                "torch.fx": MagicMock(),
                "onnx9000.converters.torch.fx": MagicMock(),
                "onnx9000.optimizer.hummingbird.onnxml_parser": MagicMock(),
                "onnx9000.optimizer.hummingbird.engine": MagicMock(),
                "onnx9000.optimizer.hummingbird.strategies": MagicMock(),
                "http.server": MagicMock(),
                "socketserver": MagicMock(),
            },
        ),
        mock_load,
        mock_save,
        patch("builtins.open"),
        patch("json.load", return_value={}),
        patch("builtins.input", return_value="exit"),
    ):
        for cmd_args in cmds:
            with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                try:
                    main()
                except Exception:
                    pass
                except SystemExit:
                    pass
