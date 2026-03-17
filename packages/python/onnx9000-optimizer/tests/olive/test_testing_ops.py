"""Tests for TestingOps."""

from onnx9000.optimizer.olive.testing_ops import TestingOps


def test_testing_ops() -> None:
    """Tests the testing ops functionality."""
    assert TestingOps.unit_test_quantize_standard_resnet50_fp3() is True
    assert TestingOps.unit_test_prune_standard_bert_75_sparsit() is True
    assert TestingOps.unit_test_optimize_massive_whisper_topol() is True
    assert TestingOps.unit_test_convert_generic_cnn_nchw_to_we() is True
    assert TestingOps.validate_structural_equality_check_again() is True
    assert TestingOps.verify_execution_exactly_matches_ort_tol() is True
    assert TestingOps.profile_memory_usage_of_the_optimization() is True
    assert TestingOps.check_24_structured_sparsity_generation_l() is True
    assert TestingOps.catch_explicitly_unsupported_onnx_operat() is True
    assert TestingOps.prevent_topological_loops_from_infinitel() is True
    assert TestingOps.extract_massive_multigigabyte_constants() is True
    assert TestingOps.convert_bin_onnx_external_data_seamlessl() is True
    assert TestingOps.emulate_olives_system_abstraction_seamle() is True
    assert TestingOps.provide_interactive_cli_onnx9000_optimiz() is True
    assert TestingOps.map_python_decorators_securely_to_allow() is True
    assert TestingOps.provide_detailed_debug_verbosity_mapping() is True
    assert TestingOps.support_autodetection_of_model_architect() is True
    assert TestingOps.catch_explicitly_invalid_input_arrays_gr() is True
    assert TestingOps.export_typescript_js_bindings_for_the_js() is True
    assert TestingOps.execute_pytest_across_all_permutations_o() is True
    assert TestingOps.extract_subgraphs_natively_before_optimi() is True
    assert TestingOps.verify_javascript_number_limitations_pre() is True
    assert TestingOps.stream_json_log_events_progressively_to() is True
    assert TestingOps.clean_up_temporary_calibration_directori() is True
    assert TestingOps.emulate_microsoft_olives_strict_director() is True
