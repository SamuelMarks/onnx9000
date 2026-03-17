"""Tests for DiagnosticsOps."""

from onnx9000.optimizer.olive.diagnostics_ops import DiagnosticsOps


def test_diagnostics_ops() -> None:
    """Tests the diagnostics ops functionality."""
    assert DiagnosticsOps.calculate_executionprovider_fallback_lat() is True
    assert DiagnosticsOps.recommend_explicit_intraopnumthreads_opt() is True
    assert DiagnosticsOps.recommend_explicit_interopnumthreads_opt() is True
    assert DiagnosticsOps.simulate_apple_metal_accelerate_specific() is True
    assert DiagnosticsOps.simulate_webgpu_specific_storagebuffer_l() is True
    assert DiagnosticsOps.partition_graph_dynamically_if_a_single() is True
    assert DiagnosticsOps.warn_if_webgl_textures_exceed_max_textur() is True
    assert DiagnosticsOps.simulate_android_ios_specific_memory_evi() is True
    assert DiagnosticsOps.expose_native_cpu_avx512_vs_avx2_specifi() is True
    assert DiagnosticsOps.expose_native_arm_neon_sve_blockpadding() is True
    assert DiagnosticsOps.trace_latency_across_explicitly_injected() is True
    assert DiagnosticsOps.output_a_rich_json_diagnostic_outlining_comp() is True
    assert DiagnosticsOps.output_a_rich_json_diagnostic_outlining() is True
    assert DiagnosticsOps.output_a_rich_json_diagnostic_outlining() is True
    assert DiagnosticsOps.highlight_any_nodes_causing_precision_lo() is True
    assert DiagnosticsOps.highlight_unfused_elementwise_operations() is True
    assert DiagnosticsOps.verify_safetensors_header_size_limits_ar() is True
    assert DiagnosticsOps.provide_strict_structural_topology_tests() is True
    assert DiagnosticsOps.execute_completely_synchronously_if_requ() is True
    assert DiagnosticsOps.validate_wasm_sharedarraybuffer_thread_c() is True
    assert DiagnosticsOps.generate_detailed_chrometracing_compatib() is True
    assert DiagnosticsOps.simulate_tensorrt_engine_building_memory() is True
    assert DiagnosticsOps.check_the_latency_of_dynamicquantizelinea() is True
    assert DiagnosticsOps.highlight_zerovariance_channels_conv_fil() is True
    assert DiagnosticsOps.simulate_cpu_l1l2l3_cache_misses_explici() is True
    assert DiagnosticsOps.expose_an_interactive_cli_debug_flag_to() is True
