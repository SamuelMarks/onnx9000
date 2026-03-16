"""ONNX Backend Test Downloader."""

import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)
ONNX_VERSION = "1.14.0"
ONNX_URL = f"https://github.com/onnx/onnx/archive/refs/tags/v{ONNX_VERSION}.zip"


def download_and_extract_onnx_tests(target_dir: Path) -> Path:
    """
    Downloads the ONNX source code repository ZIP and extracts the standard backend node tests.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / f"onnx-v{ONNX_VERSION}.zip"
    extract_dir = target_dir / f"onnx-{ONNX_VERSION}"
    backend_tests_dir = extract_dir / "onnx" / "backend" / "test" / "data" / "node"
    if backend_tests_dir.exists():
        logger.info(f"ONNX backend tests already downloaded and extracted at {backend_tests_dir}")
        return backend_tests_dir
    if not zip_path.exists():
        logger.info(f"Downloading ONNX v{ONNX_VERSION} tests from {ONNX_URL}...")
        urllib.request.urlretrieve(ONNX_URL, zip_path)
    logger.info(f"Extracting ONNX tests to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    if not backend_tests_dir.exists():
        raise FileNotFoundError(
            f"Failed to locate backend tests in extracted zip at {backend_tests_dir}"
        )
    return backend_tests_dir


def get_node_test_dirs(base_dir: Path) -> list[Path]:
    """Retrieve all standard node test directories."""
    return sorted([d for d in base_dir.iterdir() if d.is_dir() and (d / "model.onnx").exists()])
