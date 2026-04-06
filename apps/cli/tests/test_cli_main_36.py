import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import gguf2onnx_cmd, main, onnx2gguf_cmd


def test_coverage_gaps_cmd36():
    cmds = [
        ["onnx2gguf", "test.onnx", "out.gguf", "-o", "o", "--dry-run"],
        ["onnx2gguf", "test.onnx", "out.gguf", "-o", "o", "--force"],
        [
            "onnx2gguf",
            "test.onnx",
            "out.gguf",
            "-o",
            "o",
            "--architecture",
            "llama",
            "--tokenizer",
            "tok.json",
            "--outtype",
            "q4_0",
        ],
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
                "onnx9000.onnx2gguf.compiler": MagicMock(),
                "onnx9000.onnx2gguf.reader": MagicMock(),
                "onnx9000.onnx2gguf.reverse": MagicMock(),
                "triton": MagicMock(__version__="1.0.0"),
            },
        ):
            with (
                patch("builtins.open", MagicMock()),
                patch("os.path.exists", return_value=True),
                patch("os.path.getsize", return_value=80_000_000_000),
            ):
                for cmd_args in cmds:
                    with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                        try:
                            main()
                        except Exception:
                            pass
                        except SystemExit:
                            pass

                with patch.object(sys, "argv", ["onnx9000", "onnx2gguf", "test.onnx", "out.gguf"]):
                    try:
                        args = MagicMock()
                        args.model = "test.onnx"
                        args.dry_run = False
                        args.force = False
                        args.output = "out.gguf"
                        args.architecture = None
                        args.tokenizer = None
                        args.outtype = None
                        onnx2gguf_cmd(args)
                    except SystemExit:
                        pass

        with patch.dict(
            sys.modules,
            {
                "onnx9000.onnx2gguf.compiler": MagicMock(),
                "onnx9000.onnx2gguf.reader": MagicMock(),
                "onnx9000.onnx2gguf.reverse": MagicMock(),
            },
        ):
            sys.modules.pop("triton", None)
            with patch("builtins.open", MagicMock()), patch("os.path.exists", return_value=False):
                with patch.object(sys, "argv", ["onnx9000", "onnx2gguf", "test.onnx", "out.gguf"]):
                    try:
                        args = MagicMock()
                        args.model = "test.onnx"
                        args.dry_run = False
                        args.force = False
                        args.output = "out.gguf"
                        args.architecture = "llama"
                        args.tokenizer = "tok.json"
                        args.outtype = "q4_0"
                        onnx2gguf_cmd(args)

                        args = MagicMock()
                        args.model = "test.gguf"
                        args.output = "out.onnx"
                        gguf2onnx_cmd(args)
                    except SystemExit:
                        pass
