import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main, serve_cmd


def test_custom_handler():
    with patch("socketserver.TCPServer.serve_forever", side_effect=KeyboardInterrupt):
        serve_cmd(argparse.Namespace(port=0))


def test_translate_path():
    import http.server
    import socketserver

    with patch("socketserver.TCPServer.serve_forever"):
        handler_cls = None

        def fake_init(self, server_address, RequestHandlerClass, bind_and_activate=True):
            nonlocal handler_cls
            handler_cls = RequestHandlerClass
            self.socket = MagicMock()

        with patch("socketserver.TCPServer.__init__", fake_init):
            serve_cmd(argparse.Namespace(port=0))

        if handler_cls:
            with patch(
                "http.server.SimpleHTTPRequestHandler.translate_path", return_value="translated"
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
                        "/demo-ui/some.js",
                        "/other",
                    ]:
                        h.translate_path(p)
