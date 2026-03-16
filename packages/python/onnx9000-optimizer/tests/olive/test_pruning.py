"""Tests for pruning."""

from onnx9000.core.ir import DynamicDim, Graph, Tensor
from onnx9000.optimizer.olive.pruning_ops import Pruner


def test_pruner(tmp_path) -> None:
    g = Graph("test")
    g.metadata = {}
    t = Tensor("t1", "FLOAT32", [DynamicDim("N")], [1.0])
    assert Pruner.global_magnitude_pruning(t) is t
    assert Pruner.block_wise_pruning(t) is t
    assert Pruner.evaluate_norm_bounds(t) == (0.0, 1.0)
    Pruner.modify_constants_in_memory(t)
    assert t.name == "t1_pruned"
    assert Pruner.sparse_block_packing_nm(t) is t
    assert Pruner.emit_sparse_tensor_proto(t) == {"sparse": True}
    p = str(tmp_path / "sparse.bin")
    Pruner.compress_sparse_external(t, p)
    with open(p, "rb") as f:
        assert f.read() == b"SPARSE"
    assert Pruner.detect_implicit_sparsity(t) == 0.5
    assert Pruner.calc_flop_reduction(g, 0.5) == 1000
    assert Pruner.evaluate_dropin_accuracy(g) == 0.99
    Pruner.export_decompression_nodes(g)
    assert g.metadata["decompression_nodes"] is True
    assert Pruner.highlight_dead_channels(t) == [0]
    Pruner.prune_dead_channels(g)
    Pruner.identify_dead_ends(g)
    Pruner.track_dimension_modifications(g)
    Pruner.update_reshape_constants(g)
    Pruner.layer_specific_targets(g, {})
    Pruner.hook_sparse_matmul_webgpu(g)
    Pruner.hook_sparse_conv_wasm(g)
    assert "sparsity" in Pruner.output_sparsity_report(g)
    assert Pruner.calc_zip_size(g) == 500
    assert Pruner.dynamic_random_pruning(t) is t
    assert Pruner.catch_unprunable(g) == ["pos_emb"]
