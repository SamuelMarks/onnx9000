"""Hugging Face Hub utilities for downloading and caching safetensors models."""

import hashlib
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _get_cache_dir() -> str:
    """Get the default Hugging Face cache directory.

    Returns:
        The path to the Hugging Face cache directory.

    """
    if "HF_HOME" in os.environ:
        return os.environ["HF_HOME"]
    return os.path.expanduser("~/.cache/huggingface/hub")


def resolve_model_file(repo_id: str, revision: str = "main") -> str:
    """Auto-detect if a model repository defaults to .bin (PyTorch) vs .safetensors and prioritize .safetensors.

    Args:
        repo_id: The Hugging Face repository ID.
        revision: The repository revision (branch, tag, or commit hash).

    Returns:
        The URL to the best available weights file, or None if not found.

    """
    base_url = f"https://huggingface.co/{repo_id}/resolve/{revision}"

    safetensors_url = f"{base_url}/model.safetensors"
    req = Request(safetensors_url, method="HEAD")
    if "HF_TOKEN" in os.environ:
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")

    try:
        with urlopen(req) as response:
            if response.status == 200:
                return safetensors_url
    except HTTPError:
        return None

    bin_url = f"{base_url}/pytorch_model.bin"
    return bin_url


def cached_download(
    url: str, force_download: bool = False, expected_sha256: str = None, revision: str = "main"
) -> str:
    """Emulate huggingface_hub cached_download paths if the library is not installed.

    Downloads the file and caches it natively in the Hugging Face cache directory.

    Args:
        url: The URL to download from.
        force_download: Whether to force a re-download if the file is already cached.
        expected_sha256: Optional SHA256 hash to verify the downloaded file.
        revision: The repository revision to use for Hugging Face URLs.

    Returns:
        The path to the cached file.

    Raises:
        RuntimeError: If the download fails or hash validation fails.

    """
    cache_dir = _get_cache_dir()

    # Add revision to URL if it's a huggingface URL and revision is not in it
    if "huggingface.co" in url and "/resolve/" not in url:
        # Assuming URL is like https://huggingface.co/user/repo/blob/main/file
        # which usually isn't what's passed, but just in case
        return None
    elif "huggingface.co" in url and "/resolve/main/" in url and revision != "main":
        url = url.replace("/resolve/main/", f"/resolve/{revision}/")

    # Create a unique filename based on the URL hash to emulate the blob storage
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    blob_dir = os.path.join(cache_dir, "blobs")
    os.makedirs(blob_dir, exist_ok=True)

    cached_path = os.path.join(blob_dir, url_hash)

    # If file exists and not forcing download, return it
    if os.path.exists(cached_path) and not force_download:
        if expected_sha256:
            with open(cached_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                if file_hash != expected_sha256:
                    print(
                        f"Hash mismatch for {cached_path}. Expected {expected_sha256}, got {file_hash}. Re-downloading."
                    )
                    return cached_download(
                        url, force_download=True, expected_sha256=expected_sha256
                    )
        return cached_path

    print(f"Downloading {url} to native cache...")
    req = Request(url)

    if "HF_TOKEN" in os.environ:
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")

    try:
        with urlopen(req) as response:
            hasher = hashlib.sha256()
            with open(cached_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    if expected_sha256:
                        hasher.update(chunk)

            if expected_sha256:
                final_hash = hasher.hexdigest()
                if final_hash != expected_sha256:
                    os.remove(cached_path)
                    raise RuntimeError(
                        f"Hash validation failed for {url}. Expected {expected_sha256}, got {final_hash}."
                    )

    except HTTPError as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

    return cached_path
