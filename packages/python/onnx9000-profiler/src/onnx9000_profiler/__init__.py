"""Backward-compatible re-export module for `onnx9000_profiler`."""

from onnx9000.profiler import Profiler, ProfilerResult, profile

__all__ = ["Profiler", "ProfilerResult", "profile"]
