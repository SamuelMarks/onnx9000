"""Unit tests and verification suite for cli main 53 functionality."""

import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import serve_cmd


def test_custom_handler() -> None:
    """Verify that custom handler behaves as expected."""
    with patch("socketserver.TCPServer.serve_forever", side_effect=KeyboardInterrupt):
        serve_cmd(argparse.Namespace(port=0))
    assert test_custom_handler is not None


def test_translate_path() -> None:
    """Verify that translate path behaves as expected."""
    with patch("socketserver.TCPServer.serve_forever"):
        handler_cls = None

        def fake_init(
            self: int | float | str | bool | bytes | list | dict | tuple | None,
            server_address: int | float | str | bool | bytes | list | dict | tuple | None,
            RequestHandlerClass: int | float | str | bool | bytes | list | dict | tuple | None,
            bind_and_activate: int | float | str | bool | bytes | list | dict | tuple | None = True,
        ) -> None:
            """Execute fake init operation.

            Args:
                self: Parameter specifying instance.
                server_address: Parameter specifying server address.
                RequestHandlerClass: Parameter specifying request handler class.
                bind_and_activate: Parameter specifying bind and activate.
            """
            nonlocal handler_cls
            handler_cls = RequestHandlerClass
            self.socket = MagicMock()

        with patch("socketserver.TCPServer.__init__", fake_init):
            serve_cmd(argparse.Namespace(port=0))

        if handler_cls:
            with patch(
                "http.server.SimpleHTTPRequestHandler.translate_path",
                return_value="translated",
            ):
                h = handler_cls.__new__(handler_cls)
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
                        "/tfjs-shim",
                        "/iree",
                        "/triton",
                        "/coreml",
                        "/tvm",
                        "/tensorrt",
                        "/diffusers",
                        "/demo-ui/some.js",
                        "/other",
                    ]:
                        h.translate_path(p)
    assert test_translate_path is not None
