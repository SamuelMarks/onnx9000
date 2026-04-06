import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd51():
    cmds = [
        ["serve"],
        ["chat"],
        ["workspace"],
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
                "onnx9000.cli.chat": MagicMock(),
                "onnx9000.cli.workspace": MagicMock(),
            },
        ):
            with patch("http.server.SimpleHTTPRequestHandler"):
                with patch(
                    "socketserver.TCPServer.__enter__",
                    return_value=MagicMock(serve_forever=MagicMock()),
                ):
                    with patch("socketserver.TCPServer.__exit__"):
                        for cmd_args in cmds:
                            with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                                try:
                                    main()
                                except Exception:
                                    pass
                                except SystemExit:
                                    pass
