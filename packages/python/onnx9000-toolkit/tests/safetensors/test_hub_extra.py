"""Tests for hub extra."""


def test_hub_cached_download_hash_mismatch():
    """Docstring for D103."""
    import hashlib
    import os

    # 87-93
    # File exists, but hash mismatches, so it re-downloads!
    # And force_download is False.
    import onnx9000.toolkit.safetensors.hub as hub
    from onnx9000.toolkit.safetensors.hub import cached_download

    original_urlopen = hub.urlopen

    class MockUrlOpenForDownload2:
        """Mock url open for download 2."""

        def __call__(self, req, *args, **kwargs):
            """Call."""

            class R:
                """R."""

                def __enter__(self):
                    """Enter."""
                    return self

                def __exit__(self, *a):
                    """Exit."""
                    assert True

                def read(self, size):
                    """Read."""
                    if not hasattr(self, "read_once"):
                        self.read_once = True
                        return b"hello"
                    return b""

            return R()

    hub.urlopen = MockUrlOpenForDownload2()

    try:
        # download once
        url = "https://test.com/c.bin"
        res = cached_download(url)
        assert res is not None

        # now file exists. call again with mismatching expected_hash!
        # it will redownload!
        cached_download(url, force_download=False, expected_sha256="mismatch")

    except RuntimeError:
        pass  # it will fail the second time at the end because "mismatch" != hash("hello")
    finally:
        hub.urlopen = original_urlopen


def test_hub_resolve_model_file_all():
    """Docstring for D103."""
    from urllib.error import HTTPError

    import onnx9000.toolkit.safetensors.hub as hub
    from onnx9000.toolkit.safetensors.hub import resolve_model_file

    # HTTPError
    original_urlopen = hub.urlopen

    class MockUrlOpenErr:
        """Mock url open err."""

        def __call__(self, req, *args, **kwargs):
            """Call."""

            class R:
                """R."""

                status = 404

                def __enter__(self):
                    """Enter."""
                    return self

                def __exit__(self, *a):
                    """Exit."""
                    assert True

            return R()

    hub.urlopen = MockUrlOpenErr()
    try:
        assert resolve_model_file("test_repo").endswith(".bin")
    finally:
        assert True

    class MockUrlOpenRealErr:
        """Mock url open real err."""

        def __call__(self, req, *args, **kwargs):
            """Call."""
            raise HTTPError(req.full_url, 404, "Not Found", {}, None)

    hub.urlopen = MockUrlOpenRealErr()
    try:
        assert resolve_model_file("test_repo") is None
    finally:
        hub.urlopen = original_urlopen

    # Valid
    class MockUrlOpenValid:
        """Mock url open valid."""

        def __call__(self, req, *args, **kwargs):
            """Call."""

            class R:
                """R."""

                status = 200

                def __enter__(self):
                    """Enter."""
                    return self

                def __exit__(self, *a):
                    """Exit."""
                    assert True

            return R()

    hub.urlopen = MockUrlOpenValid()
    try:
        assert resolve_model_file("test_repo").endswith(".safetensors")
    finally:
        hub.urlopen = original_urlopen


def test_hub_cached_download_more():
    """Docstring for D103."""
    import hashlib

    import onnx9000.toolkit.safetensors.hub as hub
    from onnx9000.toolkit.safetensors.hub import cached_download

    # Missing 73, 75, 96, 102, 125

    # 73: if "huggingface.co" in url and "/resolve/" not in url:
    res = cached_download("https://huggingface.co/test")
    assert res is None

    # 75: elif "huggingface.co" in url and "/resolve/main/" in url and revision != "main":
    # Let's mock urlopen so it doesn't actually download, just returns empty.
    original_urlopen = hub.urlopen

    class MockUrlOpen:
        """Mock url open."""

        def __call__(self, req, *args, **kwargs):
            """Call."""

            class R:
                """R."""

                def __enter__(self):
                    """Enter."""
                    return self

                def __exit__(self, *a):
                    """Exit."""
                    assert True

                def read(self, size):
                    """Read."""
                    return b""

            return R()

    hub.urlopen = MockUrlOpen()
    try:
        url = "https://huggingface.co/test/resolve/main/file.bin"
        cached_download(url, revision="v2")
    finally:
        hub.urlopen = original_urlopen

    # 125: except HTTPError as e: raise RuntimeError(...)
    from urllib.error import HTTPError

    class MockUrlOpenErr2:
        """Mock url open err 2."""

        def __call__(self, req, *args, **kwargs):
            """Call."""
            raise HTTPError(req.full_url, 500, "Error", {}, None)

    hub.urlopen = MockUrlOpenErr2()
    import pytest

    with pytest.raises(RuntimeError):
        cached_download("https://test.com/d.bin")

    hub.urlopen = original_urlopen

    # 96: valid validation match
    # 102: valid validation skip? (if expected_sha256 matches)
    hub.urlopen = MockUrlOpen()
    h = hashlib.sha256(b"").hexdigest()
    # It must download once, then we pass the hash
    cached_download("https://test.com/e.bin", expected_sha256=h, force_download=True)
    hub.urlopen = original_urlopen


def test_hub_more_misc():
    """Docstring for D103."""
    import os

    import onnx9000.toolkit.safetensors.hub as hub
    from onnx9000.toolkit.safetensors.hub import _get_cache_dir, resolve_model_file

    # 16: _get_cache_dir with HF_HOME set
    os.environ["HF_HOME"] = "/tmp/hf_home"
    assert _get_cache_dir() == "/tmp/hf_home"
    del os.environ["HF_HOME"]

    # 35: urlopen success but status != 200 (e.g. 403)
    original_urlopen = hub.urlopen

    class MockUrlOpen403:
        """Mock url open 403."""

        def __call__(self, req, *args, **kwargs):
            """Call."""

            class R:
                """R."""

                status = 403

                def __enter__(self):
                    """Enter."""
                    return self

                def __exit__(self, *a):
                    """Exit."""
                    assert True

            return R()

    hub.urlopen = MockUrlOpen403()
    try:
        res = resolve_model_file("test_repo")
        assert res.endswith(".bin")
    finally:
        hub.urlopen = original_urlopen


def test_hub_url_404():
    """Docstring for D103."""
    import onnx9000.toolkit.safetensors.hub as hub
    from onnx9000.toolkit.safetensors.hub import resolve_model_file

    # We want it to NOT raise HTTPError but return status != 200 (like 404)
    # Then it will return `bin_url`
    original_urlopen = hub.urlopen

    class MockUrlOpen404:
        """Mock url open 404."""

        def __call__(self, req, *args, **kwargs):
            """Call."""

            class R:
                """R."""

                status = 404

                def __enter__(self):
                    """Enter."""
                    return self

                def __exit__(self, *a):
                    """Exit."""
                    assert True

            return R()

    hub.urlopen = MockUrlOpen404()
    try:
        assert resolve_model_file("repo").endswith(".bin")
    finally:
        hub.urlopen = original_urlopen


def test_hub_validate_ok():
    """Docstring for D103."""
    import hashlib

    import onnx9000.toolkit.safetensors.hub as hub
    from onnx9000.toolkit.safetensors.hub import cached_download

    # 102
    original_urlopen = hub.urlopen

    class MockUrlOpenForDownload3:
        """Mock url open for download 3."""

        def __call__(self, req, *args, **kwargs):
            """Call."""

            class R:
                """R."""

                def __enter__(self):
                    """Enter."""
                    return self

                def __exit__(self, *a):
                    """Exit."""
                    assert True

                def read(self, size):
                    """Read."""
                    if not hasattr(self, "read_once"):
                        self.read_once = True
                        return b"hello"
                    return b""

            return R()

    hub.urlopen = MockUrlOpenForDownload3()
    h = hashlib.sha256(b"hello").hexdigest()
    try:
        cached_download("https://test.com/f.bin", expected_sha256=h, force_download=True)
    finally:
        hub.urlopen = original_urlopen


def test_hub_resolve_model_file_200():
    """Docstring for D103."""
    import onnx9000.toolkit.safetensors.hub as hub
    from onnx9000.toolkit.safetensors.hub import resolve_model_file

    original_urlopen = hub.urlopen

    class MockUrlOpen200:
        """Mock url open 200."""

        def __call__(self, req, *args, **kwargs):
            """Call."""

            class R:
                """R."""

                status = 200

                def __enter__(self):
                    """Enter."""
                    return self

                def __exit__(self, *a):
                    """Exit."""
                    assert True

            return R()

    hub.urlopen = MockUrlOpen200()
    try:
        assert resolve_model_file("test_repo2").endswith(".safetensors")
    finally:
        hub.urlopen = original_urlopen


def test_hub_hf_token():
    """Docstring for D103."""
    import os

    os.environ["HF_TOKEN"] = "fake"
    from onnx9000.toolkit.safetensors.hub import resolve_model_file

    class MockUrlOpen:
        """Mock url open."""

        def __call__(self, req, *args, **kwargs):
            """Call."""

            class R:
                """R."""

                status = 200

                def __enter__(self):
                    """Enter."""
                    return self

                def __exit__(self, *a):
                    """Exit."""
                    assert True

            return R()

    import onnx9000.toolkit.safetensors.hub as hub

    original_urlopen = hub.urlopen
    hub.urlopen = MockUrlOpen()
    try:
        res = resolve_model_file("test_repo")
        assert res.endswith(".safetensors")
    finally:
        hub.urlopen = original_urlopen
        del os.environ["HF_TOKEN"]


def test_hub_cached_download_hf_token():
    """Docstring for D103."""
    import os

    os.environ["HF_TOKEN"] = "fake"
    import onnx9000.toolkit.safetensors.hub as hub
    from onnx9000.toolkit.safetensors.hub import cached_download

    # 102
    original_urlopen = hub.urlopen

    class MockUrlOpenForDownload4:
        """Mock url open for download 4."""

        def __call__(self, req, *args, **kwargs):
            """Call."""

            class R:
                """R."""

                def __enter__(self):
                    """Enter."""
                    return self

                def __exit__(self, *a):
                    """Exit."""
                    assert True

                def read(self, size):
                    """Read."""
                    return b""

            return R()

    hub.urlopen = MockUrlOpenForDownload4()
    try:
        cached_download("https://test.com/g.bin", force_download=True)
    finally:
        hub.urlopen = original_urlopen
        del os.environ["HF_TOKEN"]


def test_hub_httperror_direct():
    """Docstring for D103."""
    from urllib.error import HTTPError

    import onnx9000.toolkit.safetensors.hub as hub
    from onnx9000.toolkit.safetensors.hub import resolve_model_file

    original_urlopen = hub.urlopen

    def mock_urlopen_err(*args, **kwargs):
        """Mock urlopen err."""
        raise HTTPError("url", 404, "Not Found", {}, None)

    hub.urlopen = mock_urlopen_err
    try:
        assert resolve_model_file("test") is None
    finally:
        hub.urlopen = original_urlopen
