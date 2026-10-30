"""Unit tests for OpenAIServer in `onnx9000.genai.openai_api`."""

from onnx9000.genai.openai_api import OpenAIServer


def test_openai_server() -> None:
    """Verify OpenAIServer route registration and serve startup."""
    srv = OpenAIServer(port=8000)
    srv.add_route("/v1/completions", lambda req: req)
    assert "/v1/completions" in srv.routes
    assert srv.serve() is True
    assert srv.is_running is True
