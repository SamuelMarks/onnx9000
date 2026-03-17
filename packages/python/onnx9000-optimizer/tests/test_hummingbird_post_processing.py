import pytest
from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.post_processing import PostProcessor


def test_post_processor():
    g = Graph(name="test")
    pp = PostProcessor(g, emit_zipmap=True)

    pp.apply_argmax_classes("raw_scores")
    assert g.nodes[-1].op_type == "ArgMax"

    pp.apply_zipmap("probs", ["a", "b"])
    assert g.nodes[-1].op_type == "ZipMap"
    assert "classlabels_strings" in g.nodes[-1].attrs

    pp.apply_cast("in", "out", 1)
    assert g.nodes[-1].op_type == "Cast"

    pp.combine_multi_output_regression(["in1", "in2"], "out")
    assert g.nodes[-1].op_type == "Concat"

    pp.apply_top_k("probs", 5)
    assert g.nodes[-1].op_type == "TopK"

    pp.apply_calibration_scaling("probs", 2.0)
    assert g.nodes[-1].op_type == "Mul"

    pp.handle_batch_size_1_drop("probs")
    assert g.nodes[-1].op_type == "Squeeze"

    pp.append_confidence_scores("probs")
    assert g.nodes[-1].op_type == "ReduceMax"
