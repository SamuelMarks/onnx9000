"""Tests for packages/python/onnx9000-onnx2gguf/tests/test_hub.py."""

from typing import NoReturn

from onnx9000.onnx2gguf.hub import fetch_hf_config, generate_readme


def test_generate_readme() -> None:
    """Test generate readme."""
    readme = generate_readme("Llama-2-7b", "meta-llama/Llama-2-7b", "Q4_0")
    assert "base_model: meta-llama/Llama-2-7b" in readme
    assert "- **Level:** Q4_0" in readme


def test_fetch_hf_config(
    monkeypatch: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test fetch hf config."""
    called = []

    def mock_urlopen(
        req: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Perform mock urlopen operation."""
        called.append(req.full_url)
        return MockRes(b'{"architectures": ["LlamaForCausalLM"]}')

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    class MockRes:
        """MockRes implementation."""

        def __init__(
            self, data: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> None:
            """Perform   init   operation."""
            self.data = data

        def read(self) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Perform read operation."""
            return self.data

        def __enter__(self) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Perform   enter   operation."""
            return self

        def __exit__(
            self, *args: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> None:
            """Perform   exit   operation."""
            return None

    (cfg, _tok, url) = fetch_hf_config("dummy/repo", "token")
    assert "architectures" in cfg
    assert url == "https://huggingface.co/dummy/repo"
    import urllib.error

    def mock_err(req: int | float | str | bool | bytes | list | dict | tuple | None) -> NoReturn:
        """Perform mock err operation."""
        raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", mock_err)
    (cfg2, tok2, _url2) = fetch_hf_config("dummy/repo2")
    assert cfg2 == {}
    assert tok2 == ""
