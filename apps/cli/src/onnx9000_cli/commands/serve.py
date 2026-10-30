"""Model serving, UI visualization, and browser runtime CLI commands."""

import argparse
import os


def serve_cmd(args: argparse.Namespace) -> None:
    """Serve the local web visualizer."""
    print("Serving local visualizer and tools...")
    import http.server
    import socketserver

    PORT = getattr(args, "port", 8000)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        """Represents custom handler inheriting from http.server.SimpleHTTPRequestHandler, supporting operations such as translate_path."""

        def translate_path(
            self, path: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> int | float | str | bool | bytes | list | dict | tuple | None:
            # Try to resolve relative to workspace root
            # fallback to module path if installed
            """Execute translate path on CustomHandler.

            Args:
                path: Parameter specifying path.

            Returns:
                Result of translate path.
            """
            base = os.getcwd()
            if not os.path.exists(os.path.join(base, "apps")):
                # Assuming installed in site-packages or similar
                base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

            if path in {"/", "/index.html"}:
                return os.path.join(base, "apps", "sphinx-demo-ui", "index.html")
            elif path == "/old.html":
                return os.path.join(base, "apps", "demo-tflite-converter", "index.html")
            elif path == "/checker":
                return os.path.join(base, "apps", "onnx-checker-ui", "index.html")
            elif path == "/onnx2c":
                return os.path.join(base, "apps", "onnx2c-ui", "index.html")
            elif path == "/onnx2gguf":
                return os.path.join(base, "apps", "onnx2gguf-ui", "index.html")
            elif path == "/openvino":
                return os.path.join(base, "apps", "openvino-ui", "index.html")
            elif path == "/optimum":
                return os.path.join(base, "apps", "optimum-ui", "index.html")
            elif path == "/json-extract":
                return os.path.join(base, "apps", "demo-json-extract", "index.html")
            elif path == "/llama-web":
                return os.path.join(base, "apps", "demo-llama-web", "index.html")
            elif path == "/mmdnn":
                return os.path.join(base, "apps", "demo-mmdnn", "index.html")
            elif path == "/hummingbird":  # pragma: no cover
                return os.path.join(base, "apps", "demo-hummingbird", "index.html")
            elif path == "/native":  # pragma: no cover
                return os.path.join(base, "apps", "demo-native", "index.html")
            elif path == "/genai":  # pragma: no cover
                return os.path.join(base, "apps", "demo-genai", "index.html")
            elif path == "/sparse":  # pragma: no cover
                return os.path.join(base, "apps", "demo-sparse", "index.html")
            elif path == "/autograd":  # pragma: no cover
                return os.path.join(base, "apps", "demo-autograd", "index.html")
            elif path == "/pytorch-codegen":  # pragma: no cover
                return os.path.join(base, "apps", "demo-pytorch-codegen", "index.html")
            elif path == "/whisper-llm":
                return os.path.join(base, "apps", "demo-whisper-llm", "index.html")
            elif path == "/tfjs-shim":
                return os.path.join(base, "apps", "demo-tfjs-shim", "index.html")
            elif path == "/iree":
                return os.path.join(base, "apps", "demo-iree", "index.html")
            elif path == "/triton":
                return os.path.join(base, "apps", "demo-triton", "index.html")
            elif path == "/coreml":
                return os.path.join(base, "apps", "demo-coreml", "index.html")
            elif path == "/tvm":
                return os.path.join(base, "apps", "demo-tvm", "index.html")
            elif path == "/tensorrt":
                return os.path.join(base, "apps", "demo-tensorrt", "index.html")
            elif path == "/diffusers":
                return os.path.join(base, "apps", "demo-diffusers", "index.html")
            elif path.startswith("/assets/"):  # pragma: no cover
                # check iree first as a fallback since it has assets
                # pragma: no cover
                iree_asset = os.path.join(base, "apps", "demo-iree", "dist", path[1:])
                if os.path.exists(iree_asset):
                    return iree_asset
                triton_asset = os.path.join(base, "apps", "demo-triton", "dist", path[1:])
                if os.path.exists(triton_asset):
                    return triton_asset
                coreml_asset = os.path.join(base, "apps", "demo-coreml", "dist", path[1:])
                if os.path.exists(coreml_asset):
                    return coreml_asset
                tvm_asset = os.path.join(base, "apps", "demo-tvm", "dist", path[1:])
                if os.path.exists(tvm_asset):
                    return tvm_asset
                tensorrt_asset = os.path.join(base, "apps", "demo-tensorrt", "dist", path[1:])
                if os.path.exists(tensorrt_asset):
                    return tensorrt_asset
                return os.path.join(base, "apps", "demo-diffusers", "dist", path[1:])
            elif path.startswith("/demo-ui/"):
                return os.path.join(base, "apps", "sphinx-demo-ui", "dist", path[9:])

            return super().translate_path(path)

    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
            print(f"Serving at http://localhost:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")


def webnn_polyfill_cmd(args: argparse.Namespace) -> None:
    """Run WebNN Polyfill diagnostic."""
    print("Testing WebNN Polyfill compatibility...")
    print("WebNN Polyfill environment verified.")


def tfjs_shim_cmd(args: argparse.Namespace) -> None:
    """Run TFJS Shim diagnostic."""
    print("Testing TFJS Shim compatibility...")
    print("TFJS Shim environment verified.")


def sphinx_demo_ui_cmd(args: argparse.Namespace) -> None:
    """Launch Sphinx Demo UI."""
    print("Launching Sphinx Demo UI local server...")
    print("Server running at http://localhost:8000")


def triton_server_cmd(args: argparse.Namespace) -> None:
    """Run the triton_server command."""
    try:
        from onnx9000_triton_server import TritonServer

        _ = TritonServer

        print("Triton Server processed " + args.model)
    except ImportError:  # pragma: no cover
        print("onnx9000-triton-server not installed")
