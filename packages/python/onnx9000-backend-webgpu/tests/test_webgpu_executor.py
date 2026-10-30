"""Comprehensive unit tests for WebGPU ExecutionProvider and WGSLGenerator."""

from onnx9000.backends.cpu.executor import ExecutionContext
from onnx9000.backends.webgpu import WebGPUExecutionProvider, WGSLGenerator
from onnx9000.core.ir import DynamicDim, Graph, Node


def test_webgpu_provider_init_and_execute() -> None:
    """Verify WebGPUExecutionProvider initialization, manual init, and auto-init during execute."""
    provider = WebGPUExecutionProvider()
    assert provider.name == "webgpu"
    assert not provider._device_ready

    graph = Graph("test_graph")
    ctx = ExecutionContext(options={})
    results = provider.execute(graph, ctx, {})
    assert provider._device_ready
    assert results == {}

    # Calling execute again when already ready skips re-initialization
    results_again = provider.execute(graph, ctx, {})
    assert results_again == {}


def test_wgsl_generator_shaders_and_workgroups() -> None:
    """Verify WGSL shader generation for Relu, Abs, and generic nodes, plus workgroup sizing."""
    gen = WGSLGenerator()

    relu_node = Node("Relu", ["X"], ["Y"], name="relu_0")
    abs_node = Node("Abs", ["Y"], ["Z"], name="")
    abs_node.name = ""
    add_node = Node("Add", ["Z", "X"], ["W"], name="add_0")

    relu_wgsl = gen.generate_shader(relu_node)
    assert "max(0.0, input_a[idx])" in relu_wgsl

    abs_wgsl = gen.generate_shader(abs_node)
    assert "abs(input_a[idx])" in abs_wgsl

    add_wgsl = gen.generate_shader(add_node)
    assert "// op: add" in add_wgsl

    graph = Graph("wgsl_graph")
    graph.add_node(relu_node)
    graph.add_node(abs_node)
    abs_node.name = ""
    graph.add_node(add_node)

    shaders = gen.generate_graph_shaders(graph)
    assert "relu_0" in shaders
    assert "Abs_1" in shaders
    assert "add_0" in shaders

    assert gen.compute_workgroup_size(()) == (1, 1, 1)
    assert gen.compute_workgroup_size((32,)) == (32, 1, 1)
    assert gen.compute_workgroup_size((16, 64)) == (64, 16, 1)
    assert gen.compute_workgroup_size((8, 512, 1024)) == (256, 256, 8)
    assert gen.compute_workgroup_size((DynamicDim("N"), -1, 128)) == (128, 1, 1)
