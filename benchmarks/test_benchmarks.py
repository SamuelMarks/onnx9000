"""Tests for the benchmark scripts."""

import sys
from unittest.mock import patch

from .memory_profiler import profile_memory
from .safetensors_benchmark import run_benchmark as bench_safetensors
from .tps_benchmark import bench_tps
from .ttft_benchmark import bench_ttft


def test_profile_memory(capsys):
    """Test memory profiling output."""
    profile_memory()
    captured = capsys.readouterr()
    assert "Profiling memory usage..." in captured.out


def test_bench_ttft(capsys):
    """Test TTFT output."""
    bench_ttft()
    captured = capsys.readouterr()
    assert "Benchmarking TTFT..." in captured.out


def test_bench_tps(capsys):
    """Test TPS output."""
    bench_tps()
    captured = capsys.readouterr()
    assert "Benchmarking TPS..." in captured.out


def test_bench_safetensors(capsys):
    """Test safetensors benchmark."""
    # Mock rust safetensors
    import types

    import benchmarks.safetensors_benchmark as sb

    st_mock = types.ModuleType("safetensors")
    st_numpy_mock = types.ModuleType("safetensors.numpy")

    def mock_save(*args):
        pass

    def mock_load(*args):
        return {}

    st_numpy_mock.save_file = mock_save
    st_numpy_mock.load_file = mock_load
    st_mock.numpy = st_numpy_mock
    sys.modules["safetensors"] = st_mock
    sys.modules["safetensors.numpy"] = st_numpy_mock

    # reload the module so it picks up the mock
    import importlib

    importlib.reload(sb)

    sb.run_benchmark()
    captured = capsys.readouterr()
    assert "Generating 100MB dummy tensor..." in captured.out

    # And without
    del sys.modules["safetensors.numpy"]
    del sys.modules["safetensors"]

    with patch.dict(sys.modules, {"safetensors.numpy": None}):
        importlib.reload(sb)
        sb.run_benchmark()

    # test __main__
    with patch.object(sb, "__name__", "__main__"):
        sb.run_benchmark()
