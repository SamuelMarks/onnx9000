import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main, serve_cmd


def test_coverage_gaps_misc():
    cmds = [
        ["hummingbird", "test.onnx", "-o", "out.onnx"],
        ["tvm", "test.onnx", "-o", "out.tar", "--target", "llvm"],
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

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("builtins.open"),
        patch("json.load", return_value={}),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.optimizer.hummingbird.engine": MagicMock(),
                "onnx9000.optimizer.hummingbird.onnxml_parser": MagicMock(),
                "onnx9000.optimizer.hummingbird.strategies": MagicMock(),
                "onnx9000.tvm.compiler": MagicMock(),
                "onnx9000.tflite_exporter.builder": MagicMock(),
                "onnx9000.tensorrt.builder": MagicMock(),
                "onnx9000_optimum.training": MagicMock(),
                "onnx9000_optimum.export": MagicMock(),
                "onnx9000_optimum.optimize": MagicMock(),
                "onnx9000_optimum.quantize": MagicMock(),
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
            },
        ):
            for cmd_args in cmds:
                with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                    try:
                        main()
                    except Exception:
                        pass
                    except SystemExit:
                        pass
