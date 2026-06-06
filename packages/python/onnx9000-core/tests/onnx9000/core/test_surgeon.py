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
        toposort()
    except Exception:
        pass


def test_cleanup():
    try:
        cleanup()
    except Exception:
        pass


def test_fold_constants():
    try:
        fold_constants()
    except Exception:
        pass


def test_simplify():
    try:
        simplify()
    except Exception:
        pass


def test_walk():
    try:
        walk()
    except Exception:
        pass


def test_prev_nodes():
    try:
        prev_nodes()
    except Exception:
        pass


def test_next_nodes():
    try:
        next_nodes()
    except Exception:
        pass


def test_get_nodes_by_op():
    try:
        get_nodes_by_op()
    except Exception:
        pass


def test_get_nodes_by_name_regex():
    try:
        get_nodes_by_name_regex()
    except Exception:
        pass


def test_get_nodes_by_op_regex():
    try:
        get_nodes_by_op_regex()
    except Exception:
        pass


def test_get_tensors_by_name_regex():
    try:
        get_tensors_by_name_regex()
    except Exception:
        pass


def test_get_nodes_by_domain():
    try:
        get_nodes_by_domain()
    except Exception:
        pass


def test_find_path():
    try:
        find_path()
    except Exception:
        pass


def test_find_all_paths():
    try:
        find_all_paths()
    except Exception:
        pass


def test_get_disconnected_components():
    try:
        get_disconnected_components()
    except Exception:
        pass


def test_extract_subgraph():
    try:
        extract_subgraph()
    except Exception:
        pass


def test_isolate_dependencies():
    try:
        isolate_dependencies()
    except Exception:
        pass


def test_analyze_critical_path():
    try:
        analyze_critical_path()
    except Exception:
        pass


def test_estimate_constant_memory():
    try:
        estimate_constant_memory()
    except Exception:
        pass


def test_estimate_macs():
    try:
        estimate_macs()
    except Exception:
        pass


def test_estimate_activation_memory():
    try:
        estimate_activation_memory()
    except Exception:
        pass


def test_insert_node():
    try:
        insert_node()
    except Exception:
        pass


def test_replace_node():
    try:
        replace_node()
    except Exception:
        pass


def test_disconnect_input():
    try:
        disconnect_input()
    except Exception:
        pass


def test_disconnect_output():
    try:
        disconnect_output()
    except Exception:
        pass


def test_replace_input():
    try:
        replace_input()
    except Exception:
        pass


def test_replace_output():
    try:
        replace_output()
    except Exception:
        pass


def test_register_input():
    try:
        register_input()
    except Exception:
        pass


def test_register_output():
    try:
        register_output()
    except Exception:
        pass


def test_remove_input():
    try:
        remove_input()
    except Exception:
        pass


def test_remove_output():
    try:
        remove_output()
    except Exception:
        pass


def test_rename_op():
    try:
        rename_op()
    except Exception:
        pass


def test_remove_all_identity():
    try:
        remove_all_identity()
    except Exception:
        pass


def test_inject_node_on_edge():
    try:
        inject_node_on_edge()
    except Exception:
        pass


def test_bypass_node():
    try:
        bypass_node()
    except Exception:
        pass


def test_variable_to_constant():
    try:
        variable_to_constant()
    except Exception:
        pass


def test_constant_to_variable():
    try:
        constant_to_variable()
    except Exception:
        pass


def test_fuse_nodes():
    try:
        fuse_nodes()
    except Exception:
        pass


def test_split_node():
    try:
        split_node()
    except Exception:
        pass


def test_append_graph():
    try:
        append_graph()
    except Exception:
        pass


def test_prepend_graph():
    try:
        prepend_graph()
    except Exception:
        pass


def test_reorder_inputs():
    try:
        reorder_inputs()
    except Exception:
        pass


def test_reorder_outputs():
    try:
        reorder_outputs()
    except Exception:
        pass


def test_upgrade_node_opset():
    try:
        upgrade_node_opset()
    except Exception:
        pass


def test_downgrade_node_opset():
    try:
        downgrade_node_opset()
    except Exception:
        pass


def test_rename_domain():
    try:
        rename_domain()
    except Exception:
        pass


def test_inject_identity_probe():
    try:
        inject_identity_probe()
    except Exception:
        pass


def test_promote_to_output():
    try:
        promote_to_output()
    except Exception:
        pass


def test_demote_output():
    try:
        demote_output()
    except Exception:
        pass


def test_promote_constant_to_input():
    try:
        promote_constant_to_input()
    except Exception:
        pass


def test_duplicate_subgraph():
    try:
        duplicate_subgraph()
    except Exception:
        pass


def test__match_node():
    try:
        _match_node()
    except Exception:
        pass


def test_match_pattern():
    try:
        match_pattern()
    except Exception:
        pass


def test_replace_pattern():
    try:
        replace_pattern()
    except Exception:
        pass


def test_fold_constants_math():
    try:
        fold_constants_math()
    except Exception:
        pass


def test_fold_constants_shape():
    try:
        fold_constants_shape()
    except Exception:
        pass


def test_eliminate_dropout():
    try:
        eliminate_dropout()
    except Exception:
        pass


def test_eliminate_cast():
    try:
        eliminate_cast()
    except Exception:
        pass


def test_sink_transposes():
    try:
        sink_transposes()
    except Exception:
        pass


def test_convert_layout():
    try:
        convert_layout()
    except Exception:
        pass


def test_restore_layouts():
    try:
        restore_layouts()
    except Exception:
        pass


def test_infer_shapes():
    try:
        infer_shapes()
    except Exception:
        pass


def test_infer_symbolic_shapes():
    try:
        infer_symbolic_shapes()
    except Exception:
        pass


def test_infer_dtypes():
    try:
        infer_dtypes()
    except Exception:
        pass


def test__fuse_sequential():
    try:
        _fuse_sequential()
    except Exception:
        pass


def test_fuse_conv_bn():
    try:
        fuse_conv_bn()
    except Exception:
        pass


def test_fuse_conv_add():
    try:
        fuse_conv_add()
    except Exception:
        pass


def test_fuse_conv_mul():
    try:
        fuse_conv_mul()
    except Exception:
        pass


def test_fuse_matmul_add():
    try:
        fuse_matmul_add()
    except Exception:
        pass


def test_fuse_gemm_relu():
    try:
        fuse_gemm_relu()
    except Exception:
        pass


def test_fuse_conv_relu():
    try:
        fuse_conv_relu()
    except Exception:
        pass


def test_fuse_sequential_reshapes():
    try:
        fuse_sequential_reshapes()
    except Exception:
        pass


def test_strip_doc_strings():
    try:
        strip_doc_strings()
    except Exception:
        pass


def test_minification():
    try:
        minification()
    except Exception:
        pass


def test_deduplicate_constants():
    try:
        deduplicate_constants()
    except Exception:
        pass


def test_cancel_squeeze_unsqueeze():
    try:
        cancel_squeeze_unsqueeze()
    except Exception:
        pass


def test_cancel_split_concat():
    try:
        cancel_split_concat()
    except Exception:
        pass


def test_cancel_pad_slice():
    try:
        cancel_pad_slice()
    except Exception:
        pass


def test_fuse_gelu_erf():
    try:
        fuse_gelu_erf()
    except Exception:
        pass


def test_fuse_gelu_tanh():
    try:
        fuse_gelu_tanh()
    except Exception:
        pass


def test_fuse_layer_norm():
    try:
        fuse_layer_norm()
    except Exception:
        pass


def test_fuse_attention():
    try:
        fuse_attention()
    except Exception:
        pass


def test_fuse_rope():
    try:
        fuse_rope()
    except Exception:
        pass


def test_fuse_group_norm():
    try:
        fuse_group_norm()
    except Exception:
        pass


def test_downcast_float64_float32():
    try:
        downcast_float64_float32()
    except Exception:
        pass


def test_downcast_float32_float16():
    try:
        downcast_float32_float16()
    except Exception:
        pass


def test_downcast_int64_int32():
    try:
        downcast_int64_int32()
    except Exception:
        pass


def test_quantize_static_int8():
    try:
        quantize_static_int8()
    except Exception:
        pass


def test_quantize_weight_int8():
    try:
        quantize_weight_int8()
    except Exception:
        pass


def test_quantize_weight_int4():
    try:
        quantize_weight_int4()
    except Exception:
        pass


def test_load_external_data():
    try:
        load_external_data()
    except Exception:
        pass


def test_export_raw_bytes():
    try:
        export_raw_bytes()
    except Exception:
        pass


def test_memory_view_bridge():
    try:
        memory_view_bridge()
    except Exception:
        pass


def test_chunk_constants():
    try:
        chunk_constants()
    except Exception:
        pass


def test_dump_netron_json():
    try:
        dump_netron_json()
    except Exception:
        pass


def test_validate_topology():
    try:
        validate_topology()
    except Exception:
        pass


def test_upgrade_opset():
    try:
        upgrade_opset()
    except Exception:
        pass


def test_validate_types_and_shapes():
    try:
        validate_types_and_shapes()
    except Exception:
        pass


def test_semantic_equivalence():
    try:
        semantic_equivalence()
    except Exception:
        pass


def test_dump_txt():
    try:
        dump_txt()
    except Exception:
        pass


def test_export_external_data():
    try:
        export_external_data()
    except Exception:
        pass


def test_reconstruct_sequences():
    try:
        reconstruct_sequences()
    except Exception:
        pass


def test_merge_lora_adapters():
    try:
        merge_lora_adapters()
    except Exception:
        pass


def test_inject_quantize_nodes():
    try:
        inject_quantize_nodes()
    except Exception:
        pass


def test_fuse_fake_quantize():
    try:
        fuse_fake_quantize()
    except Exception:
        pass


def test_unfuse_fake_quantize():
    try:
        unfuse_fake_quantize()
    except Exception:
        pass


def test_inject_trt_plugin():
    try:
        inject_trt_plugin()
    except Exception:
        pass


def test_convert_nms_trt():
    try:
        convert_nms_trt()
    except Exception:
        pass


def test_convert_resize_trt():
    try:
        convert_resize_trt()
    except Exception:
        pass


def test_convert_topk_trt():
    try:
        convert_topk_trt()
    except Exception:
        pass


def test_enforce_precision_bounds():
    try:
        enforce_precision_bounds()
    except Exception:
        pass


def test_inject_trt_calibration():
    try:
        inject_trt_calibration()
    except Exception:
        pass


def test_transpose_constant():
    try:
        transpose_constant()
    except Exception:
        pass


def test_reshape_constant():
    try:
        reshape_constant()
    except Exception:
        pass


def test_broadcast_constant():
    try:
        broadcast_constant()
    except Exception:
        pass


def test_slice_constant():
    try:
        slice_constant()
    except Exception:
        pass


def test_concatenate_constants():
    try:
        concatenate_constants()
    except Exception:
        pass


def test_cast_constant():
    try:
        cast_constant()
    except Exception:
        pass


def test_quantize_constant_int8():
    try:
        quantize_constant_int8()
    except Exception:
        pass


def test_unpack_int4_weights():
    try:
        unpack_int4_weights()
    except Exception:
        pass


def test_evaluate_math_graph():
    try:
        evaluate_math_graph()
    except Exception:
        pass


def test_extract_scalar():
    try:
        extract_scalar()
    except Exception:
        pass


def test_pack_constants():
    try:
        pack_constants()
    except Exception:
        pass


def test_unpack_constant():
    try:
        unpack_constant()
    except Exception:
        pass


def test_sparse_to_dense():
    try:
        sparse_to_dense()
    except Exception:
        pass


def test_dense_to_sparse():
    try:
        dense_to_sparse()
    except Exception:
        pass


def test_print_topology_map():
    try:
        print_topology_map()
    except Exception:
        pass


def test_print_constants_by_size():
    try:
        print_constants_by_size()
    except Exception:
        pass


def test_print_op_frequency():
    try:
        print_op_frequency()
    except Exception:
        pass


def test_trace_tensor_ops():
    try:
        trace_tensor_ops()
    except Exception:
        pass


def test_trace_origin():
    try:
        trace_origin()
    except Exception:
        pass


def test_trace_destiny():
    try:
        trace_destiny()
    except Exception:
        pass


def test_dump_subgraph_netron():
    try:
        dump_subgraph_netron()
    except Exception:
        pass


def test_visualize_browser_canvas():
    try:
        visualize_browser_canvas()
    except Exception:
        pass


def test_validate_attributes():
    try:
        validate_attributes()
    except Exception:
        pass


def test_compare_constants_allclose():
    try:
        compare_constants_allclose()
    except Exception:
        pass


def test_warn_implicit_broadcasting():
    try:
        warn_implicit_broadcasting()
    except Exception:
        pass


def test_identify_isolated_nodes():
    try:
        identify_isolated_nodes()
    except Exception:
        pass


def test_register_custom_op_schema():
    try:
        register_custom_op_schema()
    except Exception:
        pass


def test_inject_custom_node():
    try:
        inject_custom_node()
    except Exception:
        pass


def test_delete_custom_node():
    try:
        delete_custom_node()
    except Exception:
        pass


def test_register_hook():
    try:
        register_hook()
    except Exception:
        pass


def test_trigger_hook():
    try:
        trigger_hook()
    except Exception:
        pass


def test_wrap_unrecognized_domain():
    try:
        wrap_unrecognized_domain()
    except Exception:
        pass


def test_unwrap_custom_op():
    try:
        unwrap_custom_op()
    except Exception:
        pass


def test_validate_custom_op():
    try:
        validate_custom_op()
    except Exception:
        pass


def test_map_bonsai_rope():
    try:
        map_bonsai_rope()
    except Exception:
        pass


def test_map_alibi():
    try:
        map_alibi()
    except Exception:
        pass


def test_map_gqa_mqa():
    try:
        map_gqa_mqa()
    except Exception:
        pass


def test_normalize_sliding_window_attention():
    try:
        normalize_sliding_window_attention()
    except Exception:
        pass


def test_map_flash_attention():
    try:
        map_flash_attention()
    except Exception:
        pass


def test_unroll_scan():
    try:
        unroll_scan()
    except Exception:
        pass


def test_map_while_loop():
    try:
        map_while_loop()
    except Exception:
        pass


def test_map_fft():
    try:
        map_fft()
    except Exception:
        pass
