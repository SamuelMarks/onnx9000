"""Tests for target tuning."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.olive.target_tuning import TargetTuner


def test_target_tuner(tmp_path) -> None:
    g = Graph("test")
    g.metadata = {}
    TargetTuner.nchw_to_nhwc_webgpu(g)
    assert g.metadata["nchw_to_nhwc"] is True
    TargetTuner.untranspose_nhwc_accelerate(g)
    TargetTuner.force_fp16_webgpu(g)
    TargetTuner.downcast_fp64_to_fp32_wasm(g)
    TargetTuner.quantize_constants_dp4a(g)
    TargetTuner.remove_dynamic_sequences_wasm(g)
    TargetTuner.translate_mod_to_div_floor(g)
    TargetTuner.chunk_large_matmul(g)
    TargetTuner.inject_ts_tensor_descriptors(g)
    TargetTuner.map_1d_conv_to_2d(g)
    TargetTuner.force_explicit_seq_len(g)
    TargetTuner.pad_constants_for_webgpu(g)
    TargetTuner.optimize_int64_comparisons(g)
    TargetTuner.support_strict_fallback(g)
    TargetTuner.inject_webworker_policies(g)
    TargetTuner.autotune_thread_counts(g)
    assert TargetTuner.evaluate_wgsl_alignment(g) is True
    assert TargetTuner.extract_webgpu_memory_footprint(g) == 1024
    p = str(tmp_path / "weights.bin")
    TargetTuner.export_weights_external_bin(g, p)
    with open(p, "rb") as f:
        assert f.read() == b"WEIGHTS"
    assert TargetTuner.check_webgpu_compatibility(g) is True
    assert TargetTuner.extract_coreml_limitations(g) == ["limitation_1"]
    assert TargetTuner.expose_emscripten_flags() == ["-Os", "-msimd128"]
    assert TargetTuner.estimate_operator_time(10.0) == 0.1
    assert TargetTuner.estimate_operator_time(0.0) == 0.0
    assert "dtype changed" in TargetTuner.warn_dtype_changes()
    assert TargetTuner.benchmark_compiled_models(g) == {"webgpu": 1.0}
    assert TargetTuner.test_mmap_overhead() == 0.5
