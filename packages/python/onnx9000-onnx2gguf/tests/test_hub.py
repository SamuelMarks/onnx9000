"""Tests for packages/python/onnx9000-onnx2gguf/tests/test_hub.py."""

from onnx9000.onnx2gguf.hub import fetch_hf_config, generate_readme


def test_generate_readme():
    """Test generate readme."""
    readme = generate_readme("Llama-2-7b", "meta-llama/Llama-2-7b", "Q4_0")
    assert "base_model: meta-llama/Llama-2-7b" in readme
    assert "- **Level:** Q4_0" in readme


def test_fetch_hf_config(monkeypatch):
    """Test fetch hf config."""
    called = []

    def mock_urlopen(req):
        """Perform mock urlopen operation."""
        called.append(req.full_url)
        return MockRes(b'{"architectures": ["LlamaForCausalLM"]}')

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    class MockRes:
        """MockRes implementation."""

        def __init__(self, data):
            """Perform   init   operation."""
            self.data = data

        def read(self):
            """Perform read operation."""
            return self.data

        def __enter__(self):
            """Perform   enter   operation."""
            return self

        def __exit__(self, *args):
            """Perform   exit   operation."""
            return None

    (cfg, tok, url) = fetch_hf_config("dummy/repo", "token")
    assert "architectures" in cfg
    assert url == "https://huggingface.co/dummy/repo"
    import urllib.error

    def mock_err(req):
        """Perform mock err operation."""
        raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", mock_err)
    (cfg2, tok2, url2) = fetch_hf_config("dummy/repo2")
    assert cfg2 == {}
    assert tok2 == ""
