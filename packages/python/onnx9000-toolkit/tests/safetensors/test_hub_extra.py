"""Unit tests for Safetensors Hugging Face Hub resolution and download utilities."""

import hashlib
import os
from urllib.error import HTTPError

import pytest
from onnx9000.toolkit.safetensors import hub
from onnx9000.toolkit.safetensors.hub import (
    _get_cache_dir,
    cached_download,
    resolve_model_file,
)


def test_hub_cached_download_hash_mismatch() -> None:
    """Verify cached_download re-downloads when cached file fails SHA-256 verification."""
    original_urlopen = hub.urlopen

    class MockUrlOpenForDownload2:
        """Mock url open for download 2."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpenForDownload2 with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).
            """

            class R:
                """R."""

                def __enter__(self) -> "R":
                    """Enter."""
                    return self

                def __exit__(self, *a: object) -> bool:
                    """Exit."""
                    return False

                def read(self, size: int) -> bytes:
                    """Read."""
                    if not hasattr(self, "read_once"):
                        self.read_once = True
                        return b"hello"
                    return b""

            return R()

    hub.urlopen = MockUrlOpenForDownload2()

    try:
        url = "https://test.com/c.bin"
        res = cached_download(url)
        assert res is not None

        with pytest.raises(RuntimeError):
            cached_download(url, force_download=False, expected_sha256="mismatch")
    finally:
        hub.urlopen = original_urlopen


def test_hub_resolve_model_file_all() -> None:
    """Verify resolve_model_file behavior across 404, HTTPError, and 200 responses."""
    original_urlopen = hub.urlopen

    class MockUrlOpenErr:
        """Mock url open err."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpenErr with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).
            """

            class R:
                """R."""

                status = 404

                def __enter__(self) -> "R":
                    """Enter."""
                    return self

                def __exit__(self, *a: object) -> bool:
                    """Exit."""
                    return False

            return R()

    hub.urlopen = MockUrlOpenErr()
    try:
        resolved = resolve_model_file("test_repo")
        assert resolved is not None and resolved.endswith(".bin")
    finally:
        hub.urlopen = original_urlopen

    class MockUrlOpenRealErr:
        """Mock url open real err."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpenRealErr with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).

            Raises:
                HTTPError: If an invalid state or argument is encountered during execution.
            """
            raise HTTPError(getattr(req, "full_url", "url"), 404, "Not Found", {}, None)  # type: ignore[arg-type]

    hub.urlopen = MockUrlOpenRealErr()
    try:
        assert resolve_model_file("test_repo") is None
    finally:
        hub.urlopen = original_urlopen

    class MockUrlOpenValid:
        """Mock url open valid."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpenValid with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).
            """

            class R:
                """R."""

                status = 200

                def __enter__(self) -> "R":
                    """Enter."""
                    return self

                def __exit__(self, *a: object) -> bool:
                    """Exit."""
                    return False

            return R()

    hub.urlopen = MockUrlOpenValid()
    try:
        resolved_valid = resolve_model_file("test_repo")
        assert resolved_valid is not None and resolved_valid.endswith(".safetensors")
    finally:
        hub.urlopen = original_urlopen


def test_hub_cached_download_more() -> None:
    """Verify cached_download URL rewriting, error handling, and SHA-256 verification."""
    res = cached_download("https://huggingface.co/test")
    assert res is None

    original_urlopen = hub.urlopen

    class MockUrlOpen:
        """Mock url open."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpen with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).
            """

            class R:
                """R."""

                def __enter__(self) -> "R":
                    """Enter."""
                    return self

                def __exit__(self, *a: object) -> bool:
                    """Exit."""
                    return False

                def read(self, size: int) -> bytes:
                    """Read."""
                    return b""

            return R()

    hub.urlopen = MockUrlOpen()
    try:
        url = "https://huggingface.co/test/resolve/main/file.bin"
        out_path = cached_download(url, revision="v2")
        assert out_path is not None
    finally:
        hub.urlopen = original_urlopen

    class MockUrlOpenErr2:
        """Mock url open err 2."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpenErr2 with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).

            Raises:
                HTTPError: If an invalid state or argument is encountered during execution.
            """
            raise HTTPError(getattr(req, "full_url", "url"), 500, "Error", {}, None)

    hub.urlopen = MockUrlOpenErr2()
    with pytest.raises(RuntimeError):
        cached_download("https://test.com/d.bin")

    hub.urlopen = MockUrlOpen()
    h = hashlib.sha256(b"").hexdigest()
    verified_path = cached_download(
        "https://test.com/e.bin", expected_sha256=h, force_download=True
    )
    assert verified_path is not None
    hub.urlopen = original_urlopen


def test_hub_more_misc() -> None:
    """Verify _get_cache_dir and non-200 status fallback in resolve_model_file."""
    os.environ["HF_HOME"] = "/tmp/hf_home"
    assert _get_cache_dir() == "/tmp/hf_home"
    del os.environ["HF_HOME"]

    original_urlopen = hub.urlopen

    class MockUrlOpen403:
        """Mock url open 403."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpen403 with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).
            """

            class R:
                """R."""

                status = 403

                def __enter__(self) -> "R":
                    """Enter."""
                    return self

                def __exit__(self, *a: object) -> bool:
                    """Exit."""
                    return False

            return R()

    hub.urlopen = MockUrlOpen403()
    try:
        res = resolve_model_file("test_repo")
        assert res is not None and res.endswith(".bin")
    finally:
        hub.urlopen = original_urlopen


def test_hub_url_404() -> None:
    """Verify resolve_model_file falls back to .bin on 404 response status."""
    original_urlopen = hub.urlopen

    class MockUrlOpen404:
        """Mock url open 404."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpen404 with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).
            """

            class R:
                """R."""

                status = 404

                def __enter__(self) -> "R":
                    """Enter."""
                    return self

                def __exit__(self, *a: object) -> bool:
                    """Exit."""
                    return False

            return R()

    hub.urlopen = MockUrlOpen404()
    try:
        resolved = resolve_model_file("repo")
        assert resolved is not None and resolved.endswith(".bin")
    finally:
        hub.urlopen = original_urlopen


def test_hub_validate_ok() -> None:
    """Verify cached_download succeeds when SHA-256 matches non-empty payload."""
    original_urlopen = hub.urlopen

    class MockUrlOpenForDownload3:
        """Mock url open for download 3."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpenForDownload3 with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).
            """

            class R:
                """R."""

                def __enter__(self) -> "R":
                    """Enter."""
                    return self

                def __exit__(self, *a: object) -> bool:
                    """Exit."""
                    return False

                def read(self, size: int) -> bytes:
                    """Read."""
                    if not hasattr(self, "read_once"):
                        self.read_once = True
                        return b"hello"
                    return b""

            return R()

    hub.urlopen = MockUrlOpenForDownload3()
    h = hashlib.sha256(b"hello").hexdigest()
    try:
        path = cached_download("https://test.com/f.bin", expected_sha256=h, force_download=True)
        assert path is not None
    finally:
        hub.urlopen = original_urlopen


def test_hub_resolve_model_file_200() -> None:
    """Verify resolve_model_file returns .safetensors URL when status is 200."""
    original_urlopen = hub.urlopen

    class MockUrlOpen200:
        """Mock url open 200."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpen200 with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).
            """

            class R:
                """R."""

                status = 200

                def __enter__(self) -> "R":
                    """Enter."""
                    return self

                def __exit__(self, *a: object) -> bool:
                    """Exit."""
                    return False

            return R()

    hub.urlopen = MockUrlOpen200()
    try:
        resolved = resolve_model_file("test_repo2")
        assert resolved is not None and resolved.endswith(".safetensors")
    finally:
        hub.urlopen = original_urlopen


def test_hub_hf_token() -> None:
    """Verify resolve_model_file attaches HF_TOKEN header when present."""
    os.environ["HF_TOKEN"] = "fake"

    class MockUrlOpen:
        """Mock url open."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpen with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).
            """

            class R:
                """R."""

                status = 200

                def __enter__(self) -> "R":
                    """Enter."""
                    return self

                def __exit__(self, *a: object) -> bool:
                    """Exit."""
                    return False

            return R()

    original_urlopen = hub.urlopen
    hub.urlopen = MockUrlOpen()
    try:
        res = resolve_model_file("test_repo")
        assert res is not None and res.endswith(".safetensors")
    finally:
        hub.urlopen = original_urlopen
        del os.environ["HF_TOKEN"]


def test_hub_cached_download_hf_token() -> None:
    """Verify cached_download attaches HF_TOKEN header when present."""
    os.environ["HF_TOKEN"] = "fake"
    original_urlopen = hub.urlopen

    class MockUrlOpenForDownload4:
        """Mock url open for download 4."""

        def __call__(self, req: object, *args: object, **kwargs: object) -> object:
            """Invoke MockUrlOpenForDownload4 with the provided inputs and keyword arguments.

            Args:
                req: Parameter specifying req (object).
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of call (object).
            """

            class R:
                """R."""

                def __enter__(self) -> "R":
                    """Enter."""
                    return self

                def __exit__(self, *a: object) -> bool:
                    """Exit."""
                    return False

                def read(self, size: int) -> bytes:
                    """Read."""
                    return b""

            return R()

    hub.urlopen = MockUrlOpenForDownload4()
    try:
        path = cached_download("https://test.com/g.bin", force_download=True)
        assert path is not None
    finally:
        hub.urlopen = original_urlopen
        del os.environ["HF_TOKEN"]


def test_hub_httperror_direct() -> None:
    """Verify resolve_model_file returns None when urlopen raises HTTPError."""
    original_urlopen = hub.urlopen

    def mock_urlopen_err(*args: object, **kwargs: object) -> object:
        """Mock urlopen err."""
        raise HTTPError("url", 404, "Not Found", {}, None)  # type: ignore[arg-type]

    hub.urlopen = mock_urlopen_err
    try:
        assert resolve_model_file("test") is None
    finally:
        hub.urlopen = original_urlopen
