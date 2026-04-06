import pytest
from onnx9000.genai.openai_api import OpenAIServer


def test_openai_server():
    server = OpenAIServer(8080)
    server.add_route("/v1/chat", lambda x: x)
    assert "/v1/chat" in server.routes
    assert not server.is_running
    assert server.serve(cors=False)
    assert server.is_running
