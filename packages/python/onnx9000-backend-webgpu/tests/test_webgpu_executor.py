"""Tests for WebGPU Executor."""

import pytest
from onnx9000.backends.webgpu.executor import WebGPUExecutionProvider
from onnx9000.core.ir import Graph


def test_webgpu_provider_init():
    provider = WebGPUExecutionProvider()
    assert provider.name == "webgpu"
    assert not provider._device_ready


def test_webgpu_provider_initialize():
    provider = WebGPUExecutionProvider()
    provider.initialize()
    assert provider._device_ready


def test_webgpu_provider_execute():
    provider = WebGPUExecutionProvider()
    graph = Graph("test_graph")
    # Execute should auto-initialize and fallback to CPU logic
    results = provider.execute(graph, None, {})
    assert provider._device_ready
    assert isinstance(results, dict)
