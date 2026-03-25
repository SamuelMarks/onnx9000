from onnx9000.onnx2gguf.hub import fetch_hf_config, generate_readme


def test_generate_readme():
    readme = generate_readme("Llama-2-7b", "meta-llama/Llama-2-7b", "Q4_0")
    assert "base_model: meta-llama/Llama-2-7b" in readme
    assert "- **Level:** Q4_0" in readme


def test_fetch_hf_config(monkeypatch):
    # Mock urllib to test success and failure paths
    called = []

    def mock_urlopen(req):
        called.append(req.full_url)
        return MockRes(b'{"architectures": ["LlamaForCausalLM"]}')

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    class MockRes:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    cfg, tok, url = fetch_hf_config("dummy/repo", "token")
    assert "architectures" in cfg
    assert url == "https://huggingface.co/dummy/repo"

    # Test error
    import urllib.error

    def mock_err(req):
        raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", mock_err)
    cfg2, tok2, url2 = fetch_hf_config("dummy/repo2")
    assert cfg2 == {}
    assert tok2 == ""
