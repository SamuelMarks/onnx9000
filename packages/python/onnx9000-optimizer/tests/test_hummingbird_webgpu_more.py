import pytest
from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.webgpu_wasm_opts import WebGPUWASMCompilerOpts


def test_webgpu_wasm_opts_stubs():
    g = Graph("g")
    c = WebGPUWASMCompilerOpts()
    c.eliminate_branch_divergence(g)
    c.ensure_constants_fit(g)
    c.auto_chunk_gemm(g)
    c.validate_memory_alignment(g)
    c.pre_transpose_matrices(g)
    c.avoid_fragmented_gather(g)
    c.allocate_scratchpad(g)
    c.enforce_heuristics()
    c.generate_dynamic_axes(g)
    c.verify_texture_limits(g)
    c.fuse_scalar_additions(g)
    c.flatten_subgraphs(g)
    c.strip_metadata(g)
    c.prevent_oom()
    c.add_async_loading_hooks(g)
    c.quantize_gemm_int8(g)
    c.downcast_fp16(g)
    c.ensure_wgsl_compatibility(g)
    c.optimize_topology_for_ort_web(g)
    c.pre_evaluate_static_shapes(g)
    c.run_constant_folding(g)
    c.run_dce(g)


from onnx9000.core.ir import Node


def test_webgpu_branch_divergence():
    g = Graph("g")
    c = WebGPUWASMCompilerOpts()
    g.nodes.append(Node("If", [], []))
    with pytest.raises(ValueError):
        c.eliminate_branch_divergence(g)
