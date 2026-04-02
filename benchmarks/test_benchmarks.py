"""Tests for the benchmark scripts."""

from .memory_profiler import profile_memory
from .tps_benchmark import bench_tps
from .ttft_benchmark import bench_ttft


def test_profile_memory(capsys):
    """Test memory profiling output."""
    profile_memory()
    captured = capsys.readouterr()
    assert "Memory before" in captured.out
    assert "Difference" in captured.out


def test_bench_ttft(capsys):
    """Test TTFT output."""
    bench_ttft()
    captured = capsys.readouterr()
    assert "Time To First Token" in captured.out


def test_bench_tps(capsys):
    """Test TPS output."""
    bench_tps()
    captured = capsys.readouterr()
    assert "Generated" in captured.out
    assert "Throughput" in captured.out
