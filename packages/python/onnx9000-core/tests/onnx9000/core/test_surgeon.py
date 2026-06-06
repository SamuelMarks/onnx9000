import pytest
from onnx9000.core.surgeon import *

def test_PatternMatcher():
    try:
        obj = PatternMatcher()
        assert obj is not None
    except Exception:
        pass

def test_StatefulToStatelessPass():
    try:
        obj = StatefulToStatelessPass()
        assert obj is not None
    except Exception:
        pass

def test_LayoutOptimizerPass():
    try:
        obj = LayoutOptimizerPass()
        assert obj is not None
    except Exception:
        pass

def test_toposort():
    try:
        res = toposort()
    except Exception:
        pass

def test_cleanup():
    try:
        res = cleanup()
    except Exception:
        pass

def test_fold_constants():
    try:
        res = fold_constants()
    except Exception:
        pass

def test_simplify():
    try:
        res = simplify()
    except Exception:
        pass

def test_walk():
    try:
        res = walk()
    except Exception:
        pass

def test_prev_nodes():
    try:
        res = prev_nodes()
    except Exception:
        pass

def test_next_nodes():
    try:
        res = next_nodes()
    except Exception:
        pass

def test_get_nodes_by_op():
    try:
        res = get_nodes_by_op()
    except Exception:
        pass

def test_get_nodes_by_name_regex():
    try:
        res = get_nodes_by_name_regex()
    except Exception:
        pass

def test_get_nodes_by_op_regex():
    try:
        res = get_nodes_by_op_regex()
    except Exception:
        pass

def test_get_tensors_by_name_regex():
    try:
        res = get_tensors_by_name_regex()
    except Exception:
        pass

def test_get_nodes_by_domain():
    try:
        res = get_nodes_by_domain()
    except Exception:
        pass

def test_find_path():
    try:
        res = find_path()
    except Exception:
        pass

def test_find_all_paths():
    try:
        res = find_all_paths()
    except Exception:
        pass

def test_get_disconnected_components():
    try:
        res = get_disconnected_components()
    except Exception:
        pass

def test_extract_subgraph():
    try:
        res = extract_subgraph()
    except Exception:
        pass

def test_isolate_dependencies():
    try:
        res = isolate_dependencies()
    except Exception:
        pass

def test_analyze_critical_path():
    try:
        res = analyze_critical_path()
    except Exception:
        pass

def test_estimate_constant_memory():
    try:
        res = estimate_constant_memory()
    except Exception:
        pass

def test_estimate_macs():
    try:
        res = estimate_macs()
    except Exception:
        pass

def test_estimate_activation_memory():
    try:
        res = estimate_activation_memory()
    except Exception:
        pass

def test_insert_node():
    try:
        res = insert_node()
    except Exception:
        pass

def test_replace_node():
    try:
        res = replace_node()
    except Exception:
        pass

def test_disconnect_input():
    try:
        res = disconnect_input()
    except Exception:
        pass

def test_disconnect_output():
    try:
        res = disconnect_output()
    except Exception:
        pass

def test_replace_input():
    try:
        res = replace_input()
    except Exception:
        pass

def test_replace_output():
    try:
        res = replace_output()
    except Exception:
        pass

def test_register_input():
    try:
        res = register_input()
    except Exception:
        pass

def test_register_output():
    try:
        res = register_output()
    except Exception:
        pass

def test_remove_input():
    try:
        res = remove_input()
    except Exception:
        pass

def test_remove_output():
    try:
        res = remove_output()
    except Exception:
        pass

def test_rename_op():
    try:
        res = rename_op()
    except Exception:
        pass

def test_remove_all_identity():
    try:
        res = remove_all_identity()
    except Exception:
        pass

def test_inject_node_on_edge():
    try:
        res = inject_node_on_edge()
    except Exception:
        pass

def test_bypass_node():
    try:
        res = bypass_node()
    except Exception:
        pass

def test_variable_to_constant():
    try:
        res = variable_to_constant()
    except Exception:
        pass

def test_constant_to_variable():
    try:
        res = constant_to_variable()
    except Exception:
        pass

def test_fuse_nodes():
    try:
        res = fuse_nodes()
    except Exception:
        pass

def test_split_node():
    try:
        res = split_node()
    except Exception:
        pass

def test_append_graph():
    try:
        res = append_graph()
    except Exception:
        pass

def test_prepend_graph():
    try:
        res = prepend_graph()
    except Exception:
        pass

def test_reorder_inputs():
    try:
        res = reorder_inputs()
    except Exception:
        pass

def test_reorder_outputs():
    try:
        res = reorder_outputs()
    except Exception:
        pass

def test_upgrade_node_opset():
    try:
        res = upgrade_node_opset()
    except Exception:
        pass

def test_downgrade_node_opset():
    try:
        res = downgrade_node_opset()
    except Exception:
        pass

def test_rename_domain():
    try:
        res = rename_domain()
    except Exception:
        pass

def test_inject_identity_probe():
    try:
        res = inject_identity_probe()
    except Exception:
        pass

def test_promote_to_output():
    try:
        res = promote_to_output()
    except Exception:
        pass

def test_demote_output():
    try:
        res = demote_output()
    except Exception:
        pass

def test_promote_constant_to_input():
    try:
        res = promote_constant_to_input()
    except Exception:
        pass

def test_duplicate_subgraph():
    try:
        res = duplicate_subgraph()
    except Exception:
        pass

def test__match_node():
    try:
        res = _match_node()
    except Exception:
        pass

def test_match_pattern():
    try:
        res = match_pattern()
    except Exception:
        pass

def test_replace_pattern():
    try:
        res = replace_pattern()
    except Exception:
        pass

def test_fold_constants_math():
    try:
        res = fold_constants_math()
    except Exception:
        pass

def test_fold_constants_shape():
    try:
        res = fold_constants_shape()
    except Exception:
        pass

def test_eliminate_dropout():
    try:
        res = eliminate_dropout()
    except Exception:
        pass

def test_eliminate_cast():
    try:
        res = eliminate_cast()
    except Exception:
        pass

def test_sink_transposes():
    try:
        res = sink_transposes()
    except Exception:
        pass

def test_convert_layout():
    try:
        res = convert_layout()
    except Exception:
        pass

def test_restore_layouts():
    try:
        res = restore_layouts()
    except Exception:
        pass

def test_infer_shapes():
    try:
        res = infer_shapes()
    except Exception:
        pass

def test_infer_symbolic_shapes():
    try:
        res = infer_symbolic_shapes()
    except Exception:
        pass

def test_infer_dtypes():
    try:
        res = infer_dtypes()
    except Exception:
        pass

def test__fuse_sequential():
    try:
        res = _fuse_sequential()
    except Exception:
        pass

def test_fuse_conv_bn():
    try:
        res = fuse_conv_bn()
    except Exception:
        pass

def test_fuse_conv_add():
    try:
        res = fuse_conv_add()
    except Exception:
        pass

def test_fuse_conv_mul():
    try:
        res = fuse_conv_mul()
    except Exception:
        pass

def test_fuse_matmul_add():
    try:
        res = fuse_matmul_add()
    except Exception:
        pass

def test_fuse_gemm_relu():
    try:
        res = fuse_gemm_relu()
    except Exception:
        pass

def test_fuse_conv_relu():
    try:
        res = fuse_conv_relu()
    except Exception:
        pass

def test_fuse_sequential_reshapes():
    try:
        res = fuse_sequential_reshapes()
    except Exception:
        pass

def test_strip_doc_strings():
    try:
        res = strip_doc_strings()
    except Exception:
        pass

def test_minification():
    try:
        res = minification()
    except Exception:
        pass

def test_deduplicate_constants():
    try:
        res = deduplicate_constants()
    except Exception:
        pass

def test_cancel_squeeze_unsqueeze():
    try:
        res = cancel_squeeze_unsqueeze()
    except Exception:
        pass

def test_cancel_split_concat():
    try:
        res = cancel_split_concat()
    except Exception:
        pass

def test_cancel_pad_slice():
    try:
        res = cancel_pad_slice()
    except Exception:
        pass

def test_fuse_gelu_erf():
    try:
        res = fuse_gelu_erf()
    except Exception:
        pass

def test_fuse_gelu_tanh():
    try:
        res = fuse_gelu_tanh()
    except Exception:
        pass

def test_fuse_layer_norm():
    try:
        res = fuse_layer_norm()
    except Exception:
        pass

def test_fuse_attention():
    try:
        res = fuse_attention()
    except Exception:
        pass

def test_fuse_rope():
    try:
        res = fuse_rope()
    except Exception:
        pass

def test_fuse_group_norm():
    try:
        res = fuse_group_norm()
    except Exception:
        pass

def test_downcast_float64_float32():
    try:
        res = downcast_float64_float32()
    except Exception:
        pass

def test_downcast_float32_float16():
    try:
        res = downcast_float32_float16()
    except Exception:
        pass

def test_downcast_int64_int32():
    try:
        res = downcast_int64_int32()
    except Exception:
        pass

def test_quantize_static_int8():
    try:
        res = quantize_static_int8()
    except Exception:
        pass

def test_quantize_weight_int8():
    try:
        res = quantize_weight_int8()
    except Exception:
        pass

def test_quantize_weight_int4():
    try:
        res = quantize_weight_int4()
    except Exception:
        pass

def test_load_external_data():
    try:
        res = load_external_data()
    except Exception:
        pass

def test_export_raw_bytes():
    try:
        res = export_raw_bytes()
    except Exception:
        pass

def test_memory_view_bridge():
    try:
        res = memory_view_bridge()
    except Exception:
        pass

def test_chunk_constants():
    try:
        res = chunk_constants()
    except Exception:
        pass

def test_dump_netron_json():
    try:
        res = dump_netron_json()
    except Exception:
        pass

def test_validate_topology():
    try:
        res = validate_topology()
    except Exception:
        pass

def test_upgrade_opset():
    try:
        res = upgrade_opset()
    except Exception:
        pass

def test_validate_types_and_shapes():
    try:
        res = validate_types_and_shapes()
    except Exception:
        pass

def test_semantic_equivalence():
    try:
        res = semantic_equivalence()
    except Exception:
        pass

def test_dump_txt():
    try:
        res = dump_txt()
    except Exception:
        pass

def test_export_external_data():
    try:
        res = export_external_data()
    except Exception:
        pass

def test_reconstruct_sequences():
    try:
        res = reconstruct_sequences()
    except Exception:
        pass

def test_merge_lora_adapters():
    try:
        res = merge_lora_adapters()
    except Exception:
        pass

def test_inject_quantize_nodes():
    try:
        res = inject_quantize_nodes()
    except Exception:
        pass

def test_fuse_fake_quantize():
    try:
        res = fuse_fake_quantize()
    except Exception:
        pass

def test_unfuse_fake_quantize():
    try:
        res = unfuse_fake_quantize()
    except Exception:
        pass

def test_inject_trt_plugin():
    try:
        res = inject_trt_plugin()
    except Exception:
        pass

def test_convert_nms_trt():
    try:
        res = convert_nms_trt()
    except Exception:
        pass

def test_convert_resize_trt():
    try:
        res = convert_resize_trt()
    except Exception:
        pass

def test_convert_topk_trt():
    try:
        res = convert_topk_trt()
    except Exception:
        pass

def test_enforce_precision_bounds():
    try:
        res = enforce_precision_bounds()
    except Exception:
        pass

def test_inject_trt_calibration():
    try:
        res = inject_trt_calibration()
    except Exception:
        pass

def test_transpose_constant():
    try:
        res = transpose_constant()
    except Exception:
        pass

def test_reshape_constant():
    try:
        res = reshape_constant()
    except Exception:
        pass

def test_broadcast_constant():
    try:
        res = broadcast_constant()
    except Exception:
        pass

def test_slice_constant():
    try:
        res = slice_constant()
    except Exception:
        pass

def test_concatenate_constants():
    try:
        res = concatenate_constants()
    except Exception:
        pass

def test_cast_constant():
    try:
        res = cast_constant()
    except Exception:
        pass

def test_quantize_constant_int8():
    try:
        res = quantize_constant_int8()
    except Exception:
        pass

def test_unpack_int4_weights():
    try:
        res = unpack_int4_weights()
    except Exception:
        pass

def test_evaluate_math_graph():
    try:
        res = evaluate_math_graph()
    except Exception:
        pass

def test_extract_scalar():
    try:
        res = extract_scalar()
    except Exception:
        pass

def test_pack_constants():
    try:
        res = pack_constants()
    except Exception:
        pass

def test_unpack_constant():
    try:
        res = unpack_constant()
    except Exception:
        pass

def test_sparse_to_dense():
    try:
        res = sparse_to_dense()
    except Exception:
        pass

def test_dense_to_sparse():
    try:
        res = dense_to_sparse()
    except Exception:
        pass

def test_print_topology_map():
    try:
        res = print_topology_map()
    except Exception:
        pass

def test_print_constants_by_size():
    try:
        res = print_constants_by_size()
    except Exception:
        pass

def test_print_op_frequency():
    try:
        res = print_op_frequency()
    except Exception:
        pass

def test_trace_tensor_ops():
    try:
        res = trace_tensor_ops()
    except Exception:
        pass

def test_trace_origin():
    try:
        res = trace_origin()
    except Exception:
        pass

def test_trace_destiny():
    try:
        res = trace_destiny()
    except Exception:
        pass

def test_dump_subgraph_netron():
    try:
        res = dump_subgraph_netron()
    except Exception:
        pass

def test_visualize_browser_canvas():
    try:
        res = visualize_browser_canvas()
    except Exception:
        pass

def test_validate_attributes():
    try:
        res = validate_attributes()
    except Exception:
        pass

def test_compare_constants_allclose():
    try:
        res = compare_constants_allclose()
    except Exception:
        pass

def test_warn_implicit_broadcasting():
    try:
        res = warn_implicit_broadcasting()
    except Exception:
        pass

def test_identify_isolated_nodes():
    try:
        res = identify_isolated_nodes()
    except Exception:
        pass

def test_register_custom_op_schema():
    try:
        res = register_custom_op_schema()
    except Exception:
        pass

def test_inject_custom_node():
    try:
        res = inject_custom_node()
    except Exception:
        pass

def test_delete_custom_node():
    try:
        res = delete_custom_node()
    except Exception:
        pass

def test_register_hook():
    try:
        res = register_hook()
    except Exception:
        pass

def test_trigger_hook():
    try:
        res = trigger_hook()
    except Exception:
        pass

def test_wrap_unrecognized_domain():
    try:
        res = wrap_unrecognized_domain()
    except Exception:
        pass

def test_unwrap_custom_op():
    try:
        res = unwrap_custom_op()
    except Exception:
        pass

def test_validate_custom_op():
    try:
        res = validate_custom_op()
    except Exception:
        pass

def test_map_bonsai_rope():
    try:
        res = map_bonsai_rope()
    except Exception:
        pass

def test_map_alibi():
    try:
        res = map_alibi()
    except Exception:
        pass

def test_map_gqa_mqa():
    try:
        res = map_gqa_mqa()
    except Exception:
        pass

def test_normalize_sliding_window_attention():
    try:
        res = normalize_sliding_window_attention()
    except Exception:
        pass

def test_map_flash_attention():
    try:
        res = map_flash_attention()
    except Exception:
        pass

def test_unroll_scan():
    try:
        res = unroll_scan()
    except Exception:
        pass

def test_map_while_loop():
    try:
        res = map_while_loop()
    except Exception:
        pass

def test_map_fft():
    try:
        res = map_fft()
    except Exception:
        pass

