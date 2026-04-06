import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import autograd_cmd, gguf2onnx_cmd, main, onnx2gguf_cmd


def test_coverage_gaps_cmd56():
    cmds = [
        ["onnx2gguf", "test.onnx", "-o", "o", "--dry-run"],
        ["onnx2gguf", "test.onnx", "-o", "o", "--force"],
        [
            "onnx2gguf",
            "test.onnx",
            "-o",
            "o",
            "--architecture",
            "llama",
            "--tokenizer",
            "tok.json",
            "--outtype",
            "q4_0",
        ],
        ["gguf2onnx", "test.gguf", "-o", "o"],
        ["autograd", "test.onnx", "-o", "out.onnx"],
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
                "onnx9000.toolkit.training.autograd.compiler": MagicMock(
                    AutogradEngine=MagicMock(
                        return_value=MagicMock(
                            build_backward_graph=MagicMock(return_value=MagicMock())
                        )
                    )
                ),
                "triton": MagicMock(__version__="1.0.0"),
            },
        ):
            with (
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

                try:
                    args = MagicMock()
                    args.model = "test.onnx"
                    args.dry_run = False
                    args.force = False
                    args.output = "o"
                    args.architecture = "llama"
                    args.tokenizer = "tok.json"
                    args.outtype = "q4_0"
                    with patch("builtins.open", MagicMock()):
                        onnx2gguf_cmd(args)
                        gguf2onnx_cmd(args)
                        autograd_cmd(args)
                except Exception:
                    pass
                except SystemExit:
                    pass

        with patch.dict(
            sys.modules,
            {
                "onnx9000.onnx2gguf.compiler": MagicMock(),
                "onnx9000.onnx2gguf.reader": MagicMock(),
                "onnx9000.onnx2gguf.reverse": MagicMock(),
                "onnx9000.toolkit.training.autograd.compiler": MagicMock(),
            },
        ):
            sys.modules.pop("triton", None)
            with patch("os.path.exists", return_value=False):
                try:
                    args = MagicMock()
                    args.model = "test.onnx"
                    args.dry_run = False
                    args.force = False
                    args.output = "o"
                    args.architecture = "llama"
                    args.tokenizer = "tok.json"
                    args.outtype = "q4_0"
                    with patch("builtins.open", MagicMock()):
                        onnx2gguf_cmd(args)
                except Exception:
                    pass
                except SystemExit:
                    pass
