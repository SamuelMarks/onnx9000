"""Tests the hummingbird post processing more module functionality."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.post_processing import PostProcessor


def test_post_processing_stubs():
    """Tests the post processing stubs functionality."""
    g = Graph("g")
    c = PostProcessor(g, emit_zipmap=True)
    c.apply_argmax_classes("probs", classlabels_ints=[1, 2])
    c.apply_argmax_classes("probs", classlabels_strings=["a", "b"])
    c.apply_zipmap("probs", ["a", "b"])
    c.apply_zipmap("probs", [1, 2])
    c.apply_zipmap("probs", [])
    c.apply_cast("in", "out", 1)
    c.map_hierarchical_probabilities()
    c.combine_multi_output_regression(["a", "b"], "out")
    c.merge_multi_label_classification()
    c.rename_outputs("raw", "target")
    c.apply_top_k("probs", 5)
    c.bypass_activation_for_logits()
    c.apply_calibration_scaling("probs", 2.0)
    c.emit_zipmap = False
    c.apply_zipmap("probs", ["a", "b"])
    c.handle_batch_size_1_drop("probs")
    c.append_confidence_scores("probs")
