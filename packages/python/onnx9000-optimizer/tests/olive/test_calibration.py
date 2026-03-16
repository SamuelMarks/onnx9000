"""Tests for calibration loop."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.olive.calibration_ops import CalibrationLoop


def test_calibration_loop(tmp_path) -> None:
    """Test calibration loop logic."""
    g = Graph("test")
    g.metadata = {}
    assert CalibrationLoop.parse_datasets("numpy") == [{"data": 1}]
    assert CalibrationLoop.emulate_dataloader([1, 2]) == [1, 2]
    CalibrationLoop.iterate_batches(g, [1, 2])
    assert g.metadata["iterated_batches"] == 2
    assert CalibrationLoop.extract_min_max_activations(g) == (-1.0, 1.0)
    assert CalibrationLoop.capture_histograms(g) == {"hist": [0, 1]}
    assert CalibrationLoop.measure_kl_divergence([], []) == 0.01
    CalibrationLoop.execute_locally(g)
    assert g.metadata["local_exec"] is True
    CalibrationLoop.prevent_memory_leaks(g)
    assert g.metadata["no_leaks"] is True
    g_fp32 = Graph("fp32")
    g_int8 = Graph("int8")
    assert CalibrationLoop.compare_top1_top5(g_fp32, g_int8) == (0.99, 0.999)
    assert CalibrationLoop.compare_mse([], []) == 0.001
    assert CalibrationLoop.compare_cosine_similarity([], []) == 0.999
    assert CalibrationLoop.compare_psnr([], []) == 40.0
    CalibrationLoop.provide_fallback_pass(g)
    assert g.metadata["fallback_pass"] is True
    assert CalibrationLoop.binary_search_precision_drop(g_fp32, g_int8) == "node_x"
    assert CalibrationLoop.highlight_sensitive_nodes(g) == ["LayerNorm_1"]
    CalibrationLoop.enforce_precision_on_sensitive(g, ["n1"])
    assert g.metadata["enforced_precision"] == ["n1"]
    assert CalibrationLoop.profile_peak_memory() == 1024
    assert CalibrationLoop.validate_dynamic_shapes(g) is True
    p = str(tmp_path / "CalibrationTable.json")
    CalibrationLoop.serialize_calibration_table(g, p)
    assert CalibrationLoop.import_calibration_table(p) == {"table": 1}
    CalibrationLoop.handle_multi_input_models(g)
    assert g.metadata["multi_input"] is True
    assert CalibrationLoop.evaluate_generative_models(g) == 0.9
    CalibrationLoop.support_metric_logging_callbacks(g)
    assert g.metadata["callbacks"] is True
    CalibrationLoop.bypass_calibration_fallback(g)
    assert g.metadata["bypassed"] is True
    assert CalibrationLoop.test_calibration_loop_pyodide() is True
