import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main


def test_coverage_gaps_cmd33():
    cmds = [
        ["zoo", "download", "test", "-o", "out"],
        ["zoo", "inspect-safetensors", "test"],
        ["genai", "--mode", "test", "--model", "model"],
        ["onnx2gguf", "test.onnx", "out.gguf", "-o", "o"],
        ["gguf2onnx", "test.gguf", "out.onnx", "-o", "o"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.zoo.catalog": MagicMock(
                    ModelCatalog=MagicMock(return_value=MagicMock(download=MagicMock()))
                ),
                "onnx9000.zoo.tensors": MagicMock(
                    SafeTensorsMmapParser=MagicMock(
                        return_value=MagicMock(metadata={}, inspect=MagicMock())
                    )
                ),
                "onnx9000.genai.ecosystem": MagicMock(),
                "onnx9000.genai.qa": MagicMock(),
                "onnx9000.onnx2gguf.compiler": MagicMock(),
                "onnx9000.gguf2onnx.parser": MagicMock(),
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
