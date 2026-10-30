"""Utility functions, deterministic PRNG, and Hugging Face Hub caching for ONNX9000 Diffusers."""

import json
import math
import os
import shutil
import urllib.request
from typing import TypeAlias

from onnx9000.core.exceptions import Onnx9000Error


class PyTorchPCG:
    """A PCG32 pseudo-random number generator matching JavaScript implementations for seed determinism."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the PCG32 generator with a 64-bit integer seed.

        Args:
            seed: Initial integer seed value.
        """
        self.state: int = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.inc: int = 1442695040888963407
        self.next_uint()

    def next_uint(self) -> int:
        """Advance the internal state and return the next 32-bit unsigned integer.

        Returns:
            32-bit unsigned pseudo-random integer in `[0, 2^32 - 1]`.
        """
        oldstate = self.state
        self.state = (oldstate * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
        rot = oldstate >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def next_float(self) -> float:
        """Return a uniform floating-point number in `[0.0, 1.0)`.

        Returns:
            Uniform float in `[0.0, 1.0)`.
        """
        return self.next_uint() / 4294967296.0


def rand(shape: tuple[int, ...], generator: PyTorchPCG | None = None) -> list[float]:
    """Generate a flat list of uniform random numbers in `[0.0, 1.0)` for the given shape.

    Args:
        shape: Dimensions of the output tensor.
        generator: Optional `PyTorchPCG` instance (defaults to seed 42).

    Returns:
        Flat list of uniform floats of length `prod(shape)`.
    """
    gen = generator if generator is not None else PyTorchPCG(42)
    size = 1
    for dim in shape:
        size *= dim
    return [gen.next_float() for _ in range(size)]


def randn(shape: tuple[int, ...], generator: PyTorchPCG | None = None) -> list[float]:
    """Generate a flat list of standard normal random numbers (mean=0, std=1) via Box-Muller.

    Args:
        shape: Dimensions of the output tensor.
        generator: Optional `PyTorchPCG` instance (defaults to seed 42).

    Returns:
        Flat list of standard normal floats of length `prod(shape)`.
    """
    gen = generator if generator is not None else PyTorchPCG(42)
    size = 1
    for dim in shape:
        size *= dim
    out: list[float] = []
    for _ in range((size + 1) // 2):
        u1 = max(gen.next_float(), 1e-7)
        u2 = gen.next_float()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        out.extend([z0, z1])
    return out[:size]


class ProgressBarConfig:
    """Global configuration state for CLI diffusion progress bars."""

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the progress bar configuration.

        Args:
            enabled: Whether progress bar output is enabled.
        """
        self.enabled: bool = enabled


global_progress_bar_config = ProgressBarConfig()


def set_progress_bar_config(enabled: bool = True) -> None:
    """Configure whether the global diffusion progress bar is enabled.

    Args:
        enabled: Boolean flag to enable or disable progress reporting.
    """
    global_progress_bar_config.enabled = enabled


def fetch_hub_file(repo_id: str, filename: str, cache_dir: str = ".cache") -> str:
    """Download a file from Hugging Face Hub with local filesystem caching.

    Args:
        repo_id: Hugging Face model repository identifier (e.g., `"runwayml/stable-diffusion-v1-5"`).
        filename: Relative file path within the repository.
        cache_dir: Local directory path used to cache downloaded files.

    Returns:
        Filesystem path to the cached file.

    Raises:
        Exception: Re-raises any network or I/O error after cleaning up partial downloads.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_repo = repo_id.replace("/", "--")
    safe_file = filename.replace("/", "--")
    out_path = os.path.join(cache_dir, f"{safe_repo}_{safe_file}")

    if os.path.exists(out_path):
        return out_path

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "onnx9000-diffusers/1.0"})
    try:
        with urllib.request.urlopen(req) as response, open(out_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        if os.path.exists(out_path):
            os.remove(out_path)
        raise
    return out_path


JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None


def parse_model_index(repo_id: str, cache_dir: str = ".cache") -> dict[str, JsonValue]:
    """Fetch and parse `model_index.json` for a diffusion repository.

    Args:
        repo_id: Model repository identifier or local identifier.
        cache_dir: Local cache directory.

    Returns:
        Parsed JSON dictionary representing `model_index.json`.
    """
    path = fetch_hub_file(repo_id, "model_index.json", cache_dir)
    with open(path, encoding="utf-8") as f:
        data: dict[str, JsonValue] = json.load(f)
        return data
