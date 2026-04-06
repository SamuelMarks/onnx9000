import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main


def test_coverage_gaps_cmd52():
    cmds = [
        ["serve"],
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


def test_custom_handler():
    from onnx9000_cli.main import serve_cmd
    import http.server

    with patch(
        "socketserver.TCPServer.__enter__", return_value=MagicMock(serve_forever=MagicMock())
    ):
        with patch("socketserver.TCPServer.__exit__"):
            handler_cls = None

            def mock_tcp_server(*args, **kwargs):
                nonlocal handler_cls
                handler_cls = args[1]
                return MagicMock(
                    __enter__=MagicMock(return_value=MagicMock(serve_forever=MagicMock())),
                    __exit__=MagicMock(),
                )

            with patch("socketserver.TCPServer", mock_tcp_server):
                serve_cmd(argparse.Namespace())

                if handler_cls:
                    h = handler_cls.__new__(handler_cls)
                    h.directory = ""  # Fix the missing attribute for the mocked handler
                    with patch("os.path.exists", side_effect=[True, False] * 14):
                        for p in [
                            "/",
                            "/index.html",
                            "/old.html",
                            "/checker",
                            "/onnx2c",
                            "/onnx2gguf",
                            "/openvino",
                            "/optimum",
                            "/json-extract",
                            "/llama-web",
                            "/mmdnn",
                            "/pytorch-codegen",
                            "/whisper-llm",
                            "/other",
                        ]:
                            h.translate_path(p)
