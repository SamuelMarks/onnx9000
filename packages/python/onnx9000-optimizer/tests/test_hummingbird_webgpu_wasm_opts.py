import pytest
from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.hummingbird.webgpu_wasm_opts import WebGPUWASMCompilerOpts


def test_eliminate_branch_divergence() -> None:
    opts = WebGPUWASMCompilerOpts()
    g = Graph(name="test")

    # Valid graph
    g.nodes.append(Node("MatMul", inputs=["a", "b"], outputs=["c"]))
    opts.eliminate_branch_divergence(g)  # should pass

    # Invalid graph with branches
    g.nodes.append(Node("If", inputs=["cond"], outputs=["out"]))
    with pytest.raises(ValueError):
        opts.eliminate_branch_divergence(g)


def test_prevent_oom() -> None:
    opts = WebGPUWASMCompilerOpts()
    opts.prevent_oom()  # ensure it doesn't crash


def test_ensure_wgsl_compatibility() -> None:
    opts = WebGPUWASMCompilerOpts()
    g = Graph(name="test")
    opts.ensure_wgsl_compatibility(g)


def test_run_dce_and_folding() -> None:
    opts = WebGPUWASMCompilerOpts()
    g = Graph(name="test")
    opts.run_constant_folding(g)
    opts.run_dce(g)
