import math
import os
import urllib.request
import json
import shutil
from typing import List, Tuple, Dict, Any


class PyTorchPCG:
    """
    A PCG32 pseudo-random number generator that matches JS implementations
    to ensure cross-platform seed determinism.
    """

    def __init__(self, seed: int):
        self.state = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.inc = 1442695040888963407
        self.next_uint()

    def next_uint(self) -> int:
        oldstate = self.state
        self.state = (oldstate * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
        rot = oldstate >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def next_float(self) -> float:
        """Returns a uniform float between 0.0 and 1.0."""
        return self.next_uint() / 4294967296.0


def rand(shape: Tuple[int, ...], generator: PyTorchPCG) -> List[float]:
    """
    Generates a uniform tensor [0, 1) natively matching cross-platform PRNG.

    Args:
        shape (Tuple[int, ...]): The dimensions of the output tensor.
        generator (PyTorchPCG): The PRNG instance.

    Returns:
        List[float]: A flat list representing the tensor data.
    """
    size = 1
    for dim in shape:
        size *= dim
    return [generator.next_float() for _ in range(size)]


def randn(shape: Tuple[int, ...], generator: PyTorchPCG) -> List[float]:
    """
    Generates a standard normal tensor (mean=0, std=1) natively using Box-Muller.

    Args:
        shape (Tuple[int, ...]): The dimensions of the output tensor.
        generator (PyTorchPCG): The PRNG instance.

    Returns:
        List[float]: A flat list representing the standard normal tensor data.
    """
    size = 1
    for dim in shape:
        size *= dim
    out = []
    # Box-Muller transform requires pairs
    for _ in range((size + 1) // 2):
        u1 = max(generator.next_float(), 1e-7)
        u2 = generator.next_float()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        out.extend([z0, z1])
    return out[:size]


class ProgressBarConfig:
    """Global configuration for CLI progress bars."""

    def __init__(self):
        self.enabled = True


global_progress_bar_config = ProgressBarConfig()


def set_progress_bar_config(enabled: bool) -> None:
    """Sets the global progress bar configuration equivalent for CLI environments."""
    global_progress_bar_config.enabled = enabled


def fetch_hub_file(repo_id: str, filename: str, cache_dir: str) -> str:
    """
    Downloads a file from Hugging Face Hub with OS caching.

    Args:
        repo_id (str): The Hugging Face model repository ID.
        filename (str): The file to download.
        cache_dir (str): Local OS cache directory.

    Returns:
        str: Path to the cached downloaded file.
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
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise e
    return out_path


def parse_model_index(repo_id: str, cache_dir: str) -> Dict[str, Any]:
    """
    Provides native configuration parsing for `model_index.json`.

    Args:
        repo_id (str): The model repository ID.
        cache_dir (str): The local cache directory.

    Returns:
        dict: The parsed model index configuration.
    """
    path = fetch_hub_file(repo_id, "model_index.json", cache_dir)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
