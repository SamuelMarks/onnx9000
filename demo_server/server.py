import http.server
import os
import socketserver

PORT = 8000


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        base = os.getcwd()
        if path == "/" or path == "/index.html":
            return os.path.join(base, "apps", "sphinx-demo-ui", "index.html")
        elif path == "/old.html":
            # Needs an absolute, real file path.
            return os.path.join(base, "apps", "demo-tflite-converter", "index.html")
        elif path.startswith("/demo-ui/"):
            return os.path.join(base, "apps", "sphinx-demo-ui", "dist", path[9:])

        return super().translate_path(path)


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print(f"Serving both demos at http://localhost:{PORT}")
    httpd.serve_forever()
