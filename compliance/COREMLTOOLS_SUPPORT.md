# Coremltools Support Coverage

Tracking exhaustive coverage of the `coremltools` python package API.


## Detailed API

| Object Name | Type | Signature |
|---|---|---|
| `ClassifierConfig` | Class | `(class_labels, predicted_feature_name, predicted_probabilities_output)` |
| `ComputeUnit` | Class | `(...)` |
| `EnumeratedShapes` | Class | `(shapes, default)` |
| `ImageType` | Class | `(name, shape, scale, bias, color_layout, channel_first, grayscale_use_uint8)` |
| `PassPipeline` | Class | `(pass_names, pipeline_name)` |
| `RangeDim` | Class | `(lower_bound: int, upper_bound: int, default: Optional[int], symbol: Optional[str])` |
| `ReshapeFrequency` | Class | `(...)` |
| `SPECIFICATION_VERSION` | Object | `` |
| `Shape` | Class | `(shape, default)` |
| `SpecializationStrategy` | Class | `(...)` |
| `StateType` | Class | `(wrapped_type: type, name: Optional[str])` |
| `TensorType` | Class | `(name, shape, dtype, default_value)` |
| `colorlayout` | Class | `(...)` |
| `compression_utils` | Object | `` |
| `convert` | Function | `(model, source, inputs, outputs, classifier_config, minimum_deployment_target, convert_to, compute_precision, skip_model_load, compute_units, package_dir, debug, pass_pipeline: Optional[PassPipeline], states)` |
| `converters.ClassifierConfig` | Class | `(class_labels, predicted_feature_name, predicted_probabilities_output)` |
| `converters.ColorLayout` | Class | `(...)` |
| `converters.EnumeratedShapes` | Class | `(shapes, default)` |
| `converters.ImageType` | Class | `(name, shape, scale, bias, color_layout, channel_first, grayscale_use_uint8)` |
| `converters.RangeDim` | Class | `(lower_bound: int, upper_bound: int, default: Optional[int], symbol: Optional[str])` |
| `converters.Shape` | Class | `(shape, default)` |
| `converters.StateType` | Class | `(wrapped_type: type, name: Optional[str])` |
| `converters.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.convert` | Function | `(model, source, inputs, outputs, classifier_config, minimum_deployment_target, convert_to, compute_precision, skip_model_load, compute_units, package_dir, debug, pass_pipeline: Optional[PassPipeline], states)` |
| `converters.libsvm.convert` | Function | `(model, input_names, target_name, probability, input_length)` |
| `converters.mil.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.Builder` | Class | `(...)` |
| `converters.mil.ClassifierConfig` | Class | `(class_labels, predicted_feature_name, predicted_probabilities_output)` |
| `converters.mil.ColorLayout` | Class | `(...)` |
| `converters.mil.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.EnumeratedShapes` | Class | `(shapes, default)` |
| `converters.mil.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.ImageType` | Class | `(name, shape, scale, bias, color_layout, channel_first, grayscale_use_uint8)` |
| `converters.mil.InputSpec` | Class | `(kwargs)` |
| `converters.mil.InputType` | Class | `(name, shape, dtype)` |
| `converters.mil.InternalVar` | Class | `(val, name)` |
| `converters.mil.ListInputType` | Class | `(...)` |
| `converters.mil.ListVar` | Class | `(name, elem_type, init_length, dynamic_length, sym_val, kwargs)` |
| `converters.mil.Operation` | Class | `(kwargs)` |
| `converters.mil.Placeholder` | Class | `(sym_shape, dtype, name, allow_rank0_input)` |
| `converters.mil.Program` | Class | `()` |
| `converters.mil.RangeDim` | Class | `(lower_bound: int, upper_bound: int, default: Optional[int], symbol: Optional[str])` |
| `converters.mil.SPACES` | Object | `` |
| `converters.mil.Shape` | Class | `(shape, default)` |
| `converters.mil.StateType` | Class | `(wrapped_type: type, name: Optional[str])` |
| `converters.mil.Symbol` | Class | `(sym_name)` |
| `converters.mil.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.TupleInputType` | Class | `(...)` |
| `converters.mil.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.backend.backend_helper.NameSanitizer` | Class | `(prefix)` |
| `converters.mil.backend.backend_helper.input_types` | Object | `` |
| `converters.mil.backend.backend_helper.proto` | Object | `` |
| `converters.mil.backend.mil.helper.cast_to_framework_io_dtype` | Function | `(var, is_output)` |
| `converters.mil.backend.mil.helper.create_file_value_tensor` | Function | `(file_name, offset, dim, data_type)` |
| `converters.mil.backend.mil.helper.create_immediate_value` | Function | `(var)` |
| `converters.mil.backend.mil.helper.create_list_scalarvalue` | Function | `(py_list, np_type)` |
| `converters.mil.backend.mil.helper.create_scalar_value` | Function | `(py_scalar)` |
| `converters.mil.backend.mil.helper.create_tensor_value` | Function | `(np_tensor)` |
| `converters.mil.backend.mil.helper.create_tuple_value` | Function | `(py_tuple)` |
| `converters.mil.backend.mil.helper.create_valuetype_list` | Function | `(length, elem_shape, dtype)` |
| `converters.mil.backend.mil.helper.create_valuetype_scalar` | Function | `(data_type)` |
| `converters.mil.backend.mil.helper.create_valuetype_tensor` | Function | `(shape, data_type)` |
| `converters.mil.backend.mil.helper.proto` | Object | `` |
| `converters.mil.backend.mil.helper.set_proto_dim` | Function | `(proto_dim, dim)` |
| `converters.mil.backend.mil.helper.types` | Object | `` |
| `converters.mil.backend.mil.helper.types_to_proto_primitive` | Function | `(valuetype)` |
| `converters.mil.backend.mil.helper.update_listtype` | Function | `(l_type, length, elem_shape, dtype)` |
| `converters.mil.backend.mil.helper.update_tensortype` | Function | `(t_type, shape, data_type)` |
| `converters.mil.backend.mil.load.BlobWriter` | Object | `` |
| `converters.mil.backend.mil.load.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.backend.mil.load.CoreMLProtoExporter` | Class | `(prog: mil.Program, mil_proto: proto.MIL_pb2.Program, predicted_feature_name: str, predicted_probabilities_name: str, classifier_config: mil_input_types.ClassifierConfig, convert_to: str, convert_from: str, specification_version: int)` |
| `converters.mil.backend.mil.load.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.backend.mil.load.MILProtoExporter` | Class | `(prog: Program, weights_dir: str, specification_version: int)` |
| `converters.mil.backend.mil.load.NeuralNetworkImageSize` | Class | `(height, width)` |
| `converters.mil.backend.mil.load.NeuralNetworkImageSizeRange` | Class | `(height_range, width_range)` |
| `converters.mil.backend.mil.load.Operation` | Class | `(kwargs)` |
| `converters.mil.backend.mil.load.Program` | Class | `()` |
| `converters.mil.backend.mil.load.SSAOpRegistry` | Class | `(...)` |
| `converters.mil.backend.mil.load.ScopeInfo` | Class | `(...)` |
| `converters.mil.backend.mil.load.ScopeSource` | Class | `(...)` |
| `converters.mil.backend.mil.load.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.backend.mil.load.any_symbolic` | Function | `(val)` |
| `converters.mil.backend.mil.load.any_variadic` | Function | `(val)` |
| `converters.mil.backend.mil.load.cast_to_framework_io_dtype` | Function | `(var, is_output)` |
| `converters.mil.backend.mil.load.create_file_value_tensor` | Function | `(file_name, offset, dim, data_type)` |
| `converters.mil.backend.mil.load.create_immediate_value` | Function | `(var)` |
| `converters.mil.backend.mil.load.create_list_scalarvalue` | Function | `(py_list, np_type)` |
| `converters.mil.backend.mil.load.create_scalar_value` | Function | `(py_scalar)` |
| `converters.mil.backend.mil.load.create_valuetype_list` | Function | `(length, elem_shape, dtype)` |
| `converters.mil.backend.mil.load.create_valuetype_scalar` | Function | `(data_type)` |
| `converters.mil.backend.mil.load.create_valuetype_tensor` | Function | `(shape, data_type)` |
| `converters.mil.backend.mil.load.flexible_shape_utils` | Object | `` |
| `converters.mil.backend.mil.load.helper` | Object | `` |
| `converters.mil.backend.mil.load.is_symbolic` | Function | `(val)` |
| `converters.mil.backend.mil.load.load` | Function | `(prog: Program, weights_dir: str, resume_on_errors: Optional[bool], specification_version: Optional[int], kwargs) -> proto.Model_pb2.Model` |
| `converters.mil.backend.mil.load.logger` | Object | `` |
| `converters.mil.backend.mil.load.mb` | Class | `(...)` |
| `converters.mil.backend.mil.load.mil` | Object | `` |
| `converters.mil.backend.mil.load.mil_input_types` | Object | `` |
| `converters.mil.backend.mil.load.mil_list` | Class | `(ls)` |
| `converters.mil.backend.mil.load.proto` | Object | `` |
| `converters.mil.backend.mil.load.should_use_weight_file` | Function | `(val: Union[np.ndarray, np.generic], specification_version: Optional[int]) -> bool` |
| `converters.mil.backend.mil.load.types` | Object | `` |
| `converters.mil.backend.mil.load.types_to_proto_primitive` | Function | `(valuetype)` |
| `converters.mil.backend.mil.passes.adjust_io_to_supported_types.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.backend.mil.passes.adjust_io_to_supported_types.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.backend.mil.passes.adjust_io_to_supported_types.adjust_io_to_supported_types` | Class | `(...)` |
| `converters.mil.backend.mil.passes.adjust_io_to_supported_types.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.backend.mil.passes.adjust_io_to_supported_types.logger` | Object | `` |
| `converters.mil.backend.mil.passes.adjust_io_to_supported_types.mb` | Class | `(...)` |
| `converters.mil.backend.mil.passes.adjust_io_to_supported_types.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.backend.mil.passes.adjust_io_to_supported_types.target` | Class | `(...)` |
| `converters.mil.backend.mil.passes.adjust_io_to_supported_types.types` | Object | `` |
| `converters.mil.backend.mil.passes.fuse_activation_silu.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.backend.mil.passes.fuse_activation_silu.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.backend.mil.passes.fuse_activation_silu.fuse_activation_silu` | Class | `(...)` |
| `converters.mil.backend.mil.passes.fuse_activation_silu.mb` | Class | `(...)` |
| `converters.mil.backend.mil.passes.fuse_activation_silu.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.backend.mil.passes.fuse_pow2_sqrt.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.backend.mil.passes.fuse_pow2_sqrt.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.backend.mil.passes.fuse_pow2_sqrt.fuse_pow2_sqrt` | Class | `(...)` |
| `converters.mil.backend.mil.passes.fuse_pow2_sqrt.mb` | Class | `(...)` |
| `converters.mil.backend.mil.passes.fuse_pow2_sqrt.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.backend.mil.passes.insert_image_preprocessing_op.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.backend.mil.passes.insert_image_preprocessing_op.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.backend.mil.passes.insert_image_preprocessing_op.insert_image_preprocessing_ops` | Class | `(...)` |
| `converters.mil.backend.mil.passes.insert_image_preprocessing_op.mb` | Class | `(...)` |
| `converters.mil.backend.mil.passes.insert_image_preprocessing_op.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.backend.mil.passes.insert_image_preprocessing_op.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.backend.mil.passes.sanitize_name_strings.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.backend.mil.passes.sanitize_name_strings.NameSanitizer` | Class | `(prefix)` |
| `converters.mil.backend.mil.passes.sanitize_name_strings.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.backend.mil.passes.sanitize_name_strings.sanitize_name_strings` | Class | `(...)` |
| `converters.mil.backend.mil.passes.test_passes.PASS_REGISTRY` | Object | `` |
| `converters.mil.backend.mil.passes.test_passes.TestAdjustToSupportedTypes` | Class | `(...)` |
| `converters.mil.backend.mil.passes.test_passes.TestImagePreprocessingPass` | Class | `(...)` |
| `converters.mil.backend.mil.passes.test_passes.TestPassFuseActivationSiLU` | Class | `(...)` |
| `converters.mil.backend.mil.passes.test_passes.TestPassFusePow2Sqrt` | Class | `(...)` |
| `converters.mil.backend.mil.passes.test_passes.TestSanitizerPass` | Class | `(...)` |
| `converters.mil.backend.mil.passes.test_passes.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `converters.mil.backend.mil.passes.test_passes.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.backend.mil.passes.test_passes.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.backend.mil.passes.test_passes.mb` | Class | `(...)` |
| `converters.mil.backend.mil.passes.test_passes.mil_list` | Class | `(ls)` |
| `converters.mil.backend.mil.passes.test_passes.target` | Class | `(...)` |
| `converters.mil.backend.mil.passes.test_passes.types` | Object | `` |
| `converters.mil.backend.mil.test_helper.TestNameSanitizer` | Class | `(...)` |
| `converters.mil.backend.mil.test_load.Symbol` | Class | `(sym_name)` |
| `converters.mil.backend.mil.test_load.TestMILDefaultValues` | Class | `(...)` |
| `converters.mil.backend.mil.test_load.TestMILFlexibleShapes` | Class | `(...)` |
| `converters.mil.backend.mil.test_load.TestMILProtoExporter` | Class | `(...)` |
| `converters.mil.backend.mil.test_load.TestMILProtoLoad` | Class | `(...)` |
| `converters.mil.backend.mil.test_load.TestStateModelLoad` | Class | `(...)` |
| `converters.mil.backend.mil.test_load.TestWeightFileSerialization` | Class | `(...)` |
| `converters.mil.backend.mil.test_load.get_new_symbol` | Function | `(name)` |
| `converters.mil.backend.mil.test_load.mb` | Class | `(...)` |
| `converters.mil.backend.mil.test_load.mil` | Object | `` |
| `converters.mil.backend.mil.test_load.proto` | Object | `` |
| `converters.mil.backend.mil.test_load.string_to_nptype` | Function | `(s: str)` |
| `converters.mil.backend.mil.test_load.types` | Object | `` |
| `converters.mil.backend.nn.load.Array` | Class | `(dimensions)` |
| `converters.mil.backend.nn.load.add_enumerated_image_sizes` | Function | `(spec, feature_name, sizes)` |
| `converters.mil.backend.nn.load.add_multiarray_ndshape_enumeration` | Function | `(spec, feature_name, enumerated_shapes)` |
| `converters.mil.backend.nn.load.any_symbolic` | Function | `(val)` |
| `converters.mil.backend.nn.load.any_variadic` | Function | `(val)` |
| `converters.mil.backend.nn.load.convert_ops` | Function | `(const_context, builder, ops, outputs)` |
| `converters.mil.backend.nn.load.flexible_shape_utils` | Object | `` |
| `converters.mil.backend.nn.load.is_symbolic` | Function | `(val)` |
| `converters.mil.backend.nn.load.load` | Function | `(prog, kwargs)` |
| `converters.mil.backend.nn.load.mil_input_types` | Object | `` |
| `converters.mil.backend.nn.load.model` | Object | `` |
| `converters.mil.backend.nn.load.neural_network` | Object | `` |
| `converters.mil.backend.nn.load.set_multiarray_ndshape_range` | Function | `(spec, feature_name, lower_bounds, upper_bounds)` |
| `converters.mil.backend.nn.load.types` | Object | `` |
| `converters.mil.backend.nn.mil_to_nn_mapping_registry.MIL_TO_NN_MAPPING_REGISTRY` | Object | `` |
| `converters.mil.backend.nn.mil_to_nn_mapping_registry.register_mil_to_nn_mapping` | Function | `(func, override)` |
| `converters.mil.backend.nn.op_mapping.MIL_TO_NN_MAPPING_REGISTRY` | Object | `` |
| `converters.mil.backend.nn.op_mapping.SSAOpRegistry` | Class | `(...)` |
| `converters.mil.backend.nn.op_mapping.abs` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.acos` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.add` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.add_const` | Function | `(const_context, builder, name, val)` |
| `converters.mil.backend.nn.op_mapping.add_upsample_nn` | Function | `(const_context, builder, op, scale_factor_h, scale_factor_w)` |
| `converters.mil.backend.nn.op_mapping.any_symbolic` | Function | `(val)` |
| `converters.mil.backend.nn.op_mapping.argsort` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.asin` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.atan` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.atanh` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.avg_pool` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.band_part` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.batch_norm` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.batch_to_space` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.cast` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.ceil` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.clamped_relu` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.clip` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.concat` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.cond` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.const` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.conv` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.conv_helper` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.conv_quantized` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.conv_transpose` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.convert_ops` | Function | `(const_context, builder, ops, outputs)` |
| `converters.mil.backend.nn.op_mapping.cos` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.cosh` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.crop` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.crop_resize` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.cumsum` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.custom_op` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.depth_to_space` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.einsum` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.elu` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.equal` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.erf` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.exp` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.exp2` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.expand_dims` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.fill` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.flatten2d` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.floor` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.floor_div` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.gather` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.gather_along_axis` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.gather_nd` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.gelu` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.greater` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.greater_equal` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.gru` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.identity` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.instance_norm` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.inverse` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.is_symbolic` | Function | `(val)` |
| `converters.mil.backend.nn.op_mapping.is_variadic` | Function | `(val)` |
| `converters.mil.backend.nn.op_mapping.l2_norm` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.l2_pool` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.layer_norm` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.leaky_relu` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.less` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.less_equal` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.linear` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.linear_activation` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.list_gather` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.list_length` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.list_read` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.list_scatter` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.list_write` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.local_response_norm` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.log` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.logger` | Object | `` |
| `converters.mil.backend.nn.op_mapping.logical_and` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.logical_not` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.logical_or` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.logical_xor` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.lstm` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.make_input` | Function | `(const_context, builder, variables)` |
| `converters.mil.backend.nn.op_mapping.make_list` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.matmul` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.max_pool` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.maximum` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.minimum` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.mod` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.mul` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.neural_network` | Object | `` |
| `converters.mil.backend.nn.op_mapping.non_maximum_suppression` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.non_zero` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.not_equal` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.np_val_to_py_type` | Function | `(val)` |
| `converters.mil.backend.nn.op_mapping.one_hot` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.pad` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.pixel_shuffle` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.pow` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.prelu` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.proto` | Object | `` |
| `converters.mil.backend.nn.op_mapping.random_bernoulli` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.random_categorical` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.random_normal` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.random_uniform` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.range_1d` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.real_div` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_argmax` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_argmin` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_l1_norm` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_l2_norm` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_log_sum` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_log_sum_exp` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_max` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_mean` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_min` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_prod` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_sum` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reduce_sum_square` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.register_mil_to_nn_mapping` | Function | `(func, override)` |
| `converters.mil.backend.nn.op_mapping.relu` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.relu6` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reshape` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.resize_bilinear` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.resize_nearest_neighbor` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reverse` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.reverse_sequence` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.rnn` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.round` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.rsqrt` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.scaled_tanh` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.scatter` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.scatter_along_axis` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.scatter_nd` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.select` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.shape` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.sigmoid` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.sigmoid_hard` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.sign` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.silu` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.sin` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.sinh` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.slice_by_index` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.slice_by_size` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.sliding_windows` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.softmax` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.softplus` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.softplus_parametric` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.softsign` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.space_to_batch` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.space_to_depth` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.split` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.sqrt` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.square` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.squeeze` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.stack` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.sub` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.tan` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.tanh` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.threshold` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.thresholded_relu` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.tile` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.topk` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.transpose` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.types` | Object | `` |
| `converters.mil.backend.nn.op_mapping.upsample_bilinear` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.upsample_nearest_neighbor` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.op_mapping.while_loop` | Function | `(const_context, builder, op)` |
| `converters.mil.backend.nn.passes.alert_return_type_cast.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.backend.nn.passes.alert_return_type_cast.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.backend.nn.passes.alert_return_type_cast.alert_return_type_cast` | Class | `(...)` |
| `converters.mil.backend.nn.passes.alert_return_type_cast.logger` | Object | `` |
| `converters.mil.backend.nn.passes.alert_return_type_cast.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.backend.nn.passes.alert_return_type_cast.types` | Object | `` |
| `converters.mil.backend.nn.passes.commingle_loop_vars.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.backend.nn.passes.commingle_loop_vars.commingle_loop_vars` | Class | `(...)` |
| `converters.mil.backend.nn.passes.commingle_loop_vars.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.backend.nn.passes.conv1d_decomposition.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.backend.nn.passes.conv1d_decomposition.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.backend.nn.passes.conv1d_decomposition.Operation` | Class | `(kwargs)` |
| `converters.mil.backend.nn.passes.conv1d_decomposition.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.backend.nn.passes.conv1d_decomposition.decompose_conv1d` | Class | `(...)` |
| `converters.mil.backend.nn.passes.conv1d_decomposition.mb` | Class | `(...)` |
| `converters.mil.backend.nn.passes.conv1d_decomposition.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.backend.nn.passes.handle_return_inputs_as_outputs.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.backend.nn.passes.handle_return_inputs_as_outputs.handle_return_inputs_as_outputs` | Class | `(...)` |
| `converters.mil.backend.nn.passes.handle_return_inputs_as_outputs.mb` | Class | `(...)` |
| `converters.mil.backend.nn.passes.handle_return_inputs_as_outputs.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.backend.nn.passes.handle_return_unused_inputs.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.backend.nn.passes.handle_return_unused_inputs.handle_return_unused_inputs` | Class | `(...)` |
| `converters.mil.backend.nn.passes.handle_return_unused_inputs.mb` | Class | `(...)` |
| `converters.mil.backend.nn.passes.handle_return_unused_inputs.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.backend.nn.passes.handle_unused_inputs.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.backend.nn.passes.handle_unused_inputs.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.backend.nn.passes.handle_unused_inputs.handle_unused_inputs` | Class | `(...)` |
| `converters.mil.backend.nn.passes.handle_unused_inputs.mb` | Class | `(...)` |
| `converters.mil.backend.nn.passes.handle_unused_inputs.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.backend.nn.passes.mlmodel_passes.remove_disconnected_layers` | Function | `(spec)` |
| `converters.mil.backend.nn.passes.mlmodel_passes.remove_redundant_transposes` | Function | `(spec)` |
| `converters.mil.backend.nn.passes.mlmodel_passes.transform_conv_crop` | Function | `(spec)` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.ComputeUnit` | Class | `(...)` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.DEBUG` | Object | `` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.MLModelPassesTest` | Class | `(...)` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.RUN_ALL_TESTS` | Object | `` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.Redundant_Transposees_Test` | Class | `(...)` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.datatypes` | Object | `` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.neural_network` | Object | `` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.print_network_spec` | Function | `(mlmodel_spec, interface_only, style)` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.remove_disconnected_layers` | Function | `(spec)` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.remove_redundant_transposes` | Function | `(spec)` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.suite` | Object | `` |
| `converters.mil.backend.nn.passes.test_mlmodel_passes.transform_conv_crop` | Function | `(spec)` |
| `converters.mil.backend.nn.passes.test_passes.PASS_REGISTRY` | Object | `` |
| `converters.mil.backend.nn.passes.test_passes.TestConv1dDeompositionPasses` | Class | `(...)` |
| `converters.mil.backend.nn.passes.test_passes.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `converters.mil.backend.nn.passes.test_passes.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.backend.nn.passes.test_passes.assert_same_output_names` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.backend.nn.passes.test_passes.backends` | Object | `` |
| `converters.mil.backend.nn.passes.test_passes.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.backend.nn.passes.test_passes.mb` | Class | `(...)` |
| `converters.mil.backend.nn.passes.test_passes.test_commingle_loop_vars` | Function | `()` |
| `converters.mil.backend.nn.passes.test_passes.test_handle_return_inputs_as_outputs` | Function | `()` |
| `converters.mil.backend.nn.passes.test_passes.test_handle_unused_inputs` | Function | `()` |
| `converters.mil.backend.nn.passes.test_passes.testing_reqs` | Object | `` |
| `converters.mil.builder` | Object | `` |
| `converters.mil.conftest.pytest_make_parametrize_id` | Function | `(config, val, argname)` |
| `converters.mil.converter.ConverterRegistry` | Class | `(...)` |
| `converters.mil.converter.MILFrontend` | Class | `(...)` |
| `converters.mil.converter.MILProtoBackend` | Class | `(...)` |
| `converters.mil.converter.NNProtoBackend` | Class | `(...)` |
| `converters.mil.converter.PassPipeline` | Class | `(pass_names, pipeline_name)` |
| `converters.mil.converter.PassPipelineManager` | Class | `(...)` |
| `converters.mil.converter.Program` | Class | `()` |
| `converters.mil.converter.TensorFlow2Frontend` | Class | `(...)` |
| `converters.mil.converter.TensorFlowFrontend` | Class | `(...)` |
| `converters.mil.converter.TorchFrontend` | Class | `(...)` |
| `converters.mil.converter.input_types` | Object | `` |
| `converters.mil.converter.k_num_internal_syms` | Object | `` |
| `converters.mil.converter.k_used_symbols` | Object | `` |
| `converters.mil.converter.mb` | Class | `(...)` |
| `converters.mil.converter.mil_convert` | Function | `(model, convert_from, convert_to, compute_units, kwargs)` |
| `converters.mil.converter.mil_convert_to_proto` | Function | `(model, convert_from, convert_to, converter_registry, main_pipeline, kwargs) -> Tuple[Optional[ct.models.MLModel], Program]` |
| `converters.mil.curr_block` | Function | `()` |
| `converters.mil.debugging_utils.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `converters.mil.debugging_utils.PASS_REGISTRY` | Object | `` |
| `converters.mil.debugging_utils.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.debugging_utils.extract_submodel` | Function | `(model: MLModel, outputs: List[str], inputs: Optional[List[str]], function_name: str) -> MLModel` |
| `converters.mil.debugging_utils.logger` | Object | `` |
| `converters.mil.debugging_utils.mb` | Class | `(...)` |
| `converters.mil.debugging_utils.milproto_to_pymil` | Function | `(model_spec, specification_version, file_weights_dir, kwargs)` |
| `converters.mil.debugging_utils.target` | Class | `(...)` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.arbitrary_cin` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.arbitrary_cout` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.arbitrary_input` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.arbitrary_mean` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.arbitrary_variance` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.arbitrary_weight` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.conv_batchnorm` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.conv_transpose_batchorm` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.mb` | Class | `(...)` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.register_generic_pass` | Function | `(ops_arrangement, var_constraints, transform_pattern, pass_name, namespace)` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.transform_pattern` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_conv_batchnorm_fusion.var_constraints` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.arbitrary_cin` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.arbitrary_cout` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.arbitrary_input` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.arbitrary_perm` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.arbitrary_scalar` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.arbitrary_weight` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.logger` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.mb` | Class | `(...)` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.pattern_to_detect` | Function | `(conv_transpose, transpose, sub)` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.register_generic_pass` | Function | `(ops_arrangement, var_constraints, transform_pattern, pass_name, namespace)` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.transform_pattern` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.transform_transpose_pattern` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.types` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.var_constraints` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_conv_bias_fusion.var_constraints_tranpose` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.arbitrary_cin` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.arbitrary_cout` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.arbitrary_input` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.arbitrary_scalar` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.arbitrary_weight` | Object | `` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.conv_scale_div` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.conv_scale_mul` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.conv_transpose_scale_div` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.conv_transpose_scale_mul` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.mb` | Class | `(...)` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.register_generic_pass` | Function | `(ops_arrangement, var_constraints, transform_pattern, pass_name, namespace)` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.transform_pattern` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_conv_scale_fusion.var_constraints` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.get_new_symbol` | Function | `(name)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.instancenorm_1_constraints` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.instancenorm_2` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.instancenorm_2_constraints` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.instancenorm_3` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.instancenorm_3_constraints` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.instancenorm_4` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.instancenorm_4_constraints` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.instancenorm_or_layernorm` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.layernorm_1_constraints` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.mb` | Class | `(...)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.register_generic_pass` | Function | `(ops_arrangement, var_constraints, transform_pattern, pass_name, namespace)` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.shape` | Object | `` |
| `converters.mil.experimental.passes.generic_layernorm_instancenorm_pattern_fusion.transform_pattern` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_linear_bias_fusion.arbitrary_bias` | Object | `` |
| `converters.mil.experimental.passes.generic_linear_bias_fusion.arbitrary_shape` | Object | `` |
| `converters.mil.experimental.passes.generic_linear_bias_fusion.arbitrary_weight` | Object | `` |
| `converters.mil.experimental.passes.generic_linear_bias_fusion.get_new_symbol` | Function | `(name)` |
| `converters.mil.experimental.passes.generic_linear_bias_fusion.mb` | Class | `(...)` |
| `converters.mil.experimental.passes.generic_linear_bias_fusion.pattern_add` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_linear_bias_fusion.pattern_sub` | Function | `(x)` |
| `converters.mil.experimental.passes.generic_linear_bias_fusion.register_generic_pass` | Function | `(ops_arrangement, var_constraints, transform_pattern, pass_name, namespace)` |
| `converters.mil.experimental.passes.generic_linear_bias_fusion.transform_pattern` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_linear_bias_fusion.var_constraints` | Function | `(pattern)` |
| `converters.mil.experimental.passes.generic_pass_infrastructure.PassContainer` | Class | `(pass_name)` |
| `converters.mil.experimental.passes.generic_pass_infrastructure.Pattern` | Class | `()` |
| `converters.mil.experimental.passes.generic_pass_infrastructure.ScopeInfo` | Class | `(...)` |
| `converters.mil.experimental.passes.generic_pass_infrastructure.ScopeSource` | Class | `(...)` |
| `converters.mil.experimental.passes.generic_pass_infrastructure.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.experimental.passes.generic_pass_infrastructure.fuse_all_blocks` | Function | `(ops_arrangement, var_constraints, transform_pattern, prog)` |
| `converters.mil.experimental.passes.generic_pass_infrastructure.mb` | Class | `(...)` |
| `converters.mil.experimental.passes.generic_pass_infrastructure.pass_registry` | Object | `` |
| `converters.mil.experimental.passes.generic_pass_infrastructure.register_generic_pass` | Function | `(ops_arrangement, var_constraints, transform_pattern, pass_name, namespace)` |
| `converters.mil.frontend.milproto.helper.get_new_symbol` | Function | `(name)` |
| `converters.mil.frontend.milproto.helper.get_proto_dim` | Function | `(dim)` |
| `converters.mil.frontend.milproto.helper.proto_to_types` | Function | `(valuetype)` |
| `converters.mil.frontend.milproto.helper.types` | Object | `` |
| `converters.mil.frontend.milproto.load.BlobReader` | Object | `` |
| `converters.mil.frontend.milproto.load.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.frontend.milproto.load.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.frontend.milproto.load.ListVar` | Class | `(name, elem_type, init_length, dynamic_length, sym_val, kwargs)` |
| `converters.mil.frontend.milproto.load.Placeholder` | Class | `(sym_shape, dtype, name, allow_rank0_input)` |
| `converters.mil.frontend.milproto.load.StateTensorPlaceholder` | Class | `(sym_shape, dtype, name)` |
| `converters.mil.frontend.milproto.load.TranscriptionContext` | Class | `(weights_dir)` |
| `converters.mil.frontend.milproto.load.TupleInputType` | Class | `(...)` |
| `converters.mil.frontend.milproto.load.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.frontend.milproto.load.curr_block` | Function | `()` |
| `converters.mil.frontend.milproto.load.load` | Function | `(model_spec, specification_version, file_weights_dir, kwargs)` |
| `converters.mil.frontend.milproto.load.load_mil_proto` | Function | `(program_spec, specification_version, file_weights_dir)` |
| `converters.mil.frontend.milproto.load.logger` | Object | `` |
| `converters.mil.frontend.milproto.load.mb` | Class | `(...)` |
| `converters.mil.frontend.milproto.load.mil` | Object | `` |
| `converters.mil.frontend.milproto.load.mil_list` | Class | `(ls)` |
| `converters.mil.frontend.milproto.load.optimize_utils` | Object | `` |
| `converters.mil.frontend.milproto.load.proto` | Object | `` |
| `converters.mil.frontend.milproto.load.proto_to_types` | Function | `(valuetype)` |
| `converters.mil.frontend.milproto.load.types` | Object | `` |
| `converters.mil.frontend.milproto.test_load.ComputeUnit` | Class | `(...)` |
| `converters.mil.frontend.milproto.test_load.Program` | Class | `()` |
| `converters.mil.frontend.milproto.test_load.TestE2ENumericalCorrectness` | Class | `(...)` |
| `converters.mil.frontend.milproto.test_load.TestLoadAPIUsage` | Class | `(...)` |
| `converters.mil.frontend.milproto.test_load.TestLoadOperation` | Class | `(...)` |
| `converters.mil.frontend.milproto.test_load.TestScriptedModels` | Class | `(...)` |
| `converters.mil.frontend.milproto.test_load.TestStatefulModelLoad` | Class | `(...)` |
| `converters.mil.frontend.milproto.test_load.TestTensorArray` | Class | `(...)` |
| `converters.mil.frontend.milproto.test_load.compare_backend` | Function | `(mlmodel, input_key_values, expected_outputs, dtype, atol, rtol, also_compare_shapes, state, allow_mismatch_ratio)` |
| `converters.mil.frontend.milproto.test_load.create_tensor_value` | Function | `(np_tensor)` |
| `converters.mil.frontend.milproto.test_load.get_op_names_in_program` | Function | `(prog, func_name, skip_const_ops)` |
| `converters.mil.frontend.milproto.test_load.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.frontend.milproto.test_load.get_pymil_prog_from_mlmodel` | Function | `(mlmodel)` |
| `converters.mil.frontend.milproto.test_load.get_roundtrip_mlmodel` | Function | `(mlmodel)` |
| `converters.mil.frontend.milproto.test_load.mb` | Class | `(...)` |
| `converters.mil.frontend.milproto.test_load.mil_convert` | Function | `(model, convert_from, convert_to, compute_units, kwargs)` |
| `converters.mil.frontend.milproto.test_load.milproto_to_pymil` | Function | `(model_spec, specification_version, file_weights_dir, kwargs)` |
| `converters.mil.frontend.milproto.test_load.proto` | Object | `` |
| `converters.mil.frontend.milproto.test_load.roundtrip_and_compare_mlmodel` | Function | `(mlmodel, input_dict)` |
| `converters.mil.frontend.milproto.test_load.run_compare_tf` | Function | `(graph, feed_dict, output_nodes, inputs_for_conversion, compute_unit, frontend_only, frontend, backend, atol, rtol, freeze_graph, tf_outputs, minimum_deployment_target)` |
| `converters.mil.frontend.milproto.test_load.types` | Object | `` |
| `converters.mil.frontend.tensorflow.TfLSTMBase` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.check_connections` | Function | `(gd)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.connect_dests` | Function | `(g, source, dests)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.connect_edge` | Function | `(g, source, dest)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.connect_edge_at_index` | Function | `(g, source, dest, idx)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.connect_sources` | Function | `(g, sources, dest)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.const_determined_nodes` | Function | `(gd, assume_variable_nodes)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.delete_node` | Function | `(g, node)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.disconnect_control_edge` | Function | `(g, source, dest)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.disconnect_edge` | Function | `(g, source, dest)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.disconnect_vertex_control_ins` | Function | `(g, dest)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.disconnect_vertex_control_outs` | Function | `(g, source)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.disconnect_vertex_ins` | Function | `(g, dest)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.disconnect_vertex_outs` | Function | `(g, source)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.fill_outputs` | Function | `(gd)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.replace_control_dest` | Function | `(g, source, dest, new_dest)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.replace_control_source` | Function | `(g, source, dest, new_source)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.replace_dest` | Function | `(g, source, dest, new_dest)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.replace_node` | Function | `(g, original_node, new_node)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.replace_source` | Function | `(g, source, dest, new_source)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.simple_topsort` | Function | `(inputs)` |
| `converters.mil.frontend.tensorflow.basic_graph_ops.topsort` | Function | `(graph)` |
| `converters.mil.frontend.tensorflow.convert_utils.ListVar` | Class | `(name, elem_type, init_length, dynamic_length, sym_val, kwargs)` |
| `converters.mil.frontend.tensorflow.convert_utils.any_variadic` | Function | `(val)` |
| `converters.mil.frontend.tensorflow.convert_utils.check_output_shapes` | Function | `(x, node)` |
| `converters.mil.frontend.tensorflow.convert_utils.compatible_shapes` | Function | `(tf_shape, inf_shape)` |
| `converters.mil.frontend.tensorflow.convert_utils.connect_global_initializer` | Function | `(graph)` |
| `converters.mil.frontend.tensorflow.convert_utils.convert_graph` | Function | `(context, graph, outputs)` |
| `converters.mil.frontend.tensorflow.convert_utils.is_symbolic` | Function | `(val)` |
| `converters.mil.frontend.tensorflow.convert_utils.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.convert_utils.topsort` | Function | `(graph)` |
| `converters.mil.frontend.tensorflow.convert_utils.types` | Object | `` |
| `converters.mil.frontend.tensorflow.converter.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.frontend.tensorflow.converter.ImageType` | Class | `(name, shape, scale, bias, color_layout, channel_first, grayscale_use_uint8)` |
| `converters.mil.frontend.tensorflow.converter.InputShape` | Class | `(shape, default)` |
| `converters.mil.frontend.tensorflow.converter.InputType` | Class | `(name, shape, dtype)` |
| `converters.mil.frontend.tensorflow.converter.RangeDim` | Class | `(lower_bound: int, upper_bound: int, default: Optional[int], symbol: Optional[str])` |
| `converters.mil.frontend.tensorflow.converter.TFConverter` | Class | `(tfssa, inputs, outputs, opset_version, use_default_fp16_io)` |
| `converters.mil.frontend.tensorflow.converter.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.tensorflow.converter.TranscriptionContext` | Class | `(name)` |
| `converters.mil.frontend.tensorflow.converter.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.frontend.tensorflow.converter.convert_graph` | Function | `(context, graph, outputs)` |
| `converters.mil.frontend.tensorflow.converter.get_new_symbol` | Function | `(name)` |
| `converters.mil.frontend.tensorflow.converter.get_output_names` | Function | `(outputs) -> Optional[List[str]]` |
| `converters.mil.frontend.tensorflow.converter.is_symbolic` | Function | `(val)` |
| `converters.mil.frontend.tensorflow.converter.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.converter.mb` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.converter.mil` | Object | `` |
| `converters.mil.frontend.tensorflow.converter.simple_topsort` | Function | `(inputs)` |
| `converters.mil.frontend.tensorflow.converter.types` | Object | `` |
| `converters.mil.frontend.tensorflow.dialect_ops.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.frontend.tensorflow.dialect_ops.InputSpec` | Class | `(kwargs)` |
| `converters.mil.frontend.tensorflow.dialect_ops.Operation` | Class | `(kwargs)` |
| `converters.mil.frontend.tensorflow.dialect_ops.SSAOpRegistry` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.dialect_ops.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.frontend.tensorflow.dialect_ops.TfLSTMBase` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.dialect_ops.register_op` | Object | `` |
| `converters.mil.frontend.tensorflow.dialect_ops.tf_lstm_block` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.dialect_ops.tf_lstm_block_cell` | Class | `(kwargs)` |
| `converters.mil.frontend.tensorflow.dialect_ops.tf_make_list` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.dialect_ops.types` | Object | `` |
| `converters.mil.frontend.tensorflow.dot_visitor.DotVisitor` | Class | `(annotation)` |
| `converters.mil.frontend.tensorflow.dot_visitor.types` | Object | `` |
| `converters.mil.frontend.tensorflow.load.NetworkEnsemble` | Class | `(instance)` |
| `converters.mil.frontend.tensorflow.load.ParsedTFNode` | Class | `(tfnode)` |
| `converters.mil.frontend.tensorflow.load.SSAFunction` | Class | `(gdict, inputs, outputs, ret)` |
| `converters.mil.frontend.tensorflow.load.TF1Loader` | Class | `(model, debug, kwargs)` |
| `converters.mil.frontend.tensorflow.load.TFConverter` | Class | `(tfssa, inputs, outputs, opset_version, use_default_fp16_io)` |
| `converters.mil.frontend.tensorflow.load.TFLoader` | Class | `(model, debug, kwargs)` |
| `converters.mil.frontend.tensorflow.load.cond_to_where` | Object | `` |
| `converters.mil.frontend.tensorflow.load.constant_propagation` | Object | `` |
| `converters.mil.frontend.tensorflow.load.delete_asserts` | Object | `` |
| `converters.mil.frontend.tensorflow.load.delete_disconnected_nodes` | Object | `` |
| `converters.mil.frontend.tensorflow.load.delete_unnecessary_constant_nodes` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.load.fill_outputs` | Function | `(gd)` |
| `converters.mil.frontend.tensorflow.load.functionalize_loops` | Object | `` |
| `converters.mil.frontend.tensorflow.load.fuse_dilation_conv` | Object | `` |
| `converters.mil.frontend.tensorflow.load.get_output_names` | Function | `(outputs) -> Optional[List[str]]` |
| `converters.mil.frontend.tensorflow.load.insert_get_tuple` | Object | `` |
| `converters.mil.frontend.tensorflow.load.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.load.quantization_pass` | Object | `` |
| `converters.mil.frontend.tensorflow.load.remove_variable_nodes` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.load.tensor_array_resource_removal` | Function | `(gd)` |
| `converters.mil.frontend.tensorflow.naming_utils.escape_fn_name` | Function | `(name)` |
| `converters.mil.frontend.tensorflow.naming_utils.escape_name` | Function | `(name)` |
| `converters.mil.frontend.tensorflow.naming_utils.normalize_names` | Function | `(names)` |
| `converters.mil.frontend.tensorflow.ops.Abs` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Acos` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Add` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.AddN` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.All` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Any` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ArgMax` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ArgMin` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Asin` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Atan` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Atanh` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.AudioSpectrogram` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.AvgPool` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.AvgPool3D` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.BatchToSpaceND` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.BlockLSTM` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.BroadcastTo` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Cast` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Ceil` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ClipByValue` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Complex` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Concat` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ConcatV2` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Const` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Conv2D` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Conv2DBackpropInput` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Conv3D` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Conv3DBackpropInputV2` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Cos` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Cosh` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.CropAndResize` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Cross` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Cumsum` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.DepthToSpace` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.DepthwiseConv2dNative` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ELU` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ERF` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Einsum` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Equal` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.EuclideanNorm` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Exp` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ExpandDims` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ExtractImagePatches` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.FFT` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.FakeQuantWithMinMaxVars` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Fill` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Floor` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.FloorDiv` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.FloorMod` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.FusedBatchNorm` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Gather` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.GatherNd` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.GatherV2` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Greater` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.GreaterEqual` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.IFFT` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.IRFFT` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Identity` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.IdentityN` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Imag` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ImageProjectiveTransformV2` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.InTopK` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.IsFinite` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.LRN` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.LSTMBlockCell` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.LeakyReLU` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Less` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.LessEqual` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Log` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Log1p` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.LogSoftmax` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.LogicalAnd` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.LogicalNot` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.LogicalOr` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.LogicalXor` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.MatMul` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.MatrixBandPart` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.MatrixDiag` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Max` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.MaxPool` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.MaxPool3D` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Maximum` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Mean` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Mfcc` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Min` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Minimum` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.MirrorPad` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Mul` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Multinomial` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Neg` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.NonMaxSuppression` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.NonMaxSuppressionV5` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.NotEqual` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.OneHot` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Pack` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Pad` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.PadV2` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Placeholder` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Pow` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Print` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Prod` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.RFFT` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.RandomStandardNormal` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.RandomUniform` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Range` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Real` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.RealDiv` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Reciprocal` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Relu` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Relu6` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Resampler` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Reshape` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ResizeBilinear` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ResizeNearestNeighbor` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Reverse` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ReverseSequence` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Round` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Rsqrt` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ScatterNd` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Select` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Selu` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Shape` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Sigmoid` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Sign` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Sin` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Sinh` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Size` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Slice` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Softmax` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.SoftmaxCrossEntropyWithLogits` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Softplus` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Softsign` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.SpaceToBatchND` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.SpaceToDepth` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.SparseSoftmaxCrossEntropyWithLogits` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Split` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.SplitV` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Sqrt` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Square` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.SquaredDifference` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Squeeze` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.StopGradient` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.StridedSlice` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Sub` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Sum` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Tan` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Tanh` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.TensorArrayGatherV3` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.TensorArrayReadV3` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.TensorArrayScatterV3` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.TensorArraySizeV3` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.TensorArrayV3` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.TensorArrayWriteV3` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.TensorScatterAdd` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Tile` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.TopK` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Transpose` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Unpack` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.Where` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.While` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.ZerosLike` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.broadcast_shapes` | Function | `(shape_x, shape_y)` |
| `converters.mil.frontend.tensorflow.ops.build_einsum_mil` | Function | `(vars: List[Var], equation: str, name: str) -> Var` |
| `converters.mil.frontend.tensorflow.ops.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.frontend.tensorflow.ops.convert_graph` | Function | `(context, graph, outputs)` |
| `converters.mil.frontend.tensorflow.ops.dynamic_topk` | Function | `(x: Var, k: Var, axis: int, ascending: Optional[bool], name: Optional[str])` |
| `converters.mil.frontend.tensorflow.ops.function_entry` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.get_global` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.get_tuple` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.iff` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.is_current_opset_version_compatible_with` | Function | `(opset_version)` |
| `converters.mil.frontend.tensorflow.ops.is_symbolic` | Function | `(val)` |
| `converters.mil.frontend.tensorflow.ops.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.ops.make_tuple` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.mb` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.ops.promote_input_dtypes` | Function | `(input_vars)` |
| `converters.mil.frontend.tensorflow.ops.register_tf_op` | Function | `(_func, tf_alias, override, strict)` |
| `converters.mil.frontend.tensorflow.ops.set_global` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow.ops.target` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.ops.types` | Object | `` |
| `converters.mil.frontend.tensorflow.parse.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.parse.parse_attr` | Function | `(attr)` |
| `converters.mil.frontend.tensorflow.parse.parse_func` | Function | `(f)` |
| `converters.mil.frontend.tensorflow.parse.parse_list` | Function | `(t)` |
| `converters.mil.frontend.tensorflow.parse.parse_shape` | Function | `(t)` |
| `converters.mil.frontend.tensorflow.parse.parse_string` | Function | `(s)` |
| `converters.mil.frontend.tensorflow.parse.parse_tensor` | Function | `(t)` |
| `converters.mil.frontend.tensorflow.parse.parse_type` | Function | `(t)` |
| `converters.mil.frontend.tensorflow.parse.types` | Object | `` |
| `converters.mil.frontend.tensorflow.parsed_tf_node.ParsedNode` | Class | `()` |
| `converters.mil.frontend.tensorflow.parsed_tf_node.ParsedTFNode` | Class | `(tfnode)` |
| `converters.mil.frontend.tensorflow.parsed_tf_node.types` | Object | `` |
| `converters.mil.frontend.tensorflow.register_tf_op` | Function | `(_func, tf_alias, override, strict)` |
| `converters.mil.frontend.tensorflow.ssa_passes.backfill_make_list_elem_type.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.ssa_passes.backfill_make_list_elem_type.ListVar` | Class | `(name, elem_type, init_length, dynamic_length, sym_val, kwargs)` |
| `converters.mil.frontend.tensorflow.ssa_passes.backfill_make_list_elem_type.backfill_make_list_elem_type` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.ssa_passes.backfill_make_list_elem_type.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.frontend.tensorflow.ssa_passes.backfill_make_list_elem_type.is_symbolic` | Function | `(val)` |
| `converters.mil.frontend.tensorflow.ssa_passes.backfill_make_list_elem_type.mb` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.ssa_passes.backfill_make_list_elem_type.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.frontend.tensorflow.ssa_passes.backfill_make_list_elem_type.types` | Object | `` |
| `converters.mil.frontend.tensorflow.ssa_passes.expand_tf_lstm.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.ssa_passes.expand_tf_lstm.expand_tf_lstm` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.ssa_passes.expand_tf_lstm.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.ssa_passes.expand_tf_lstm.mb` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.ssa_passes.expand_tf_lstm.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.frontend.tensorflow.ssa_passes.test_passes.PASS_REGISTRY` | Object | `` |
| `converters.mil.frontend.tensorflow.ssa_passes.test_passes.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.frontend.tensorflow.ssa_passes.test_passes.assert_same_output_names` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.frontend.tensorflow.ssa_passes.test_passes.mb` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.ssa_passes.test_passes.test_backfill_make_list_elem_type` | Function | `()` |
| `converters.mil.frontend.tensorflow.ssa_passes.test_passes.types` | Object | `` |
| `converters.mil.frontend.tensorflow.ssa_passes.tf_lstm_to_core_lstm.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.ssa_passes.tf_lstm_to_core_lstm.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.frontend.tensorflow.ssa_passes.tf_lstm_to_core_lstm.Operation` | Class | `(kwargs)` |
| `converters.mil.frontend.tensorflow.ssa_passes.tf_lstm_to_core_lstm.SUPPORTED_TF_LSTM_OPS` | Object | `` |
| `converters.mil.frontend.tensorflow.ssa_passes.tf_lstm_to_core_lstm.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.frontend.tensorflow.ssa_passes.tf_lstm_to_core_lstm.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.frontend.tensorflow.ssa_passes.tf_lstm_to_core_lstm.is_symbolic` | Function | `(val)` |
| `converters.mil.frontend.tensorflow.ssa_passes.tf_lstm_to_core_lstm.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.ssa_passes.tf_lstm_to_core_lstm.mb` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.ssa_passes.tf_lstm_to_core_lstm.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.frontend.tensorflow.ssa_passes.tf_lstm_to_core_lstm.tf_lstm_to_core_lstm` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_composite_ops.TensorFlowBaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_composite_ops.TestCompositeOp` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_composite_ops.backends` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_composite_ops.compute_units` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_composite_ops.make_tf_graph` | Function | `(input_types)` |
| `converters.mil.frontend.tensorflow.test.test_composite_ops.mb` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_composite_ops.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.frontend.tensorflow.test.test_composite_ops.register_tf_op` | Function | `(_func, tf_alias, override, strict)` |
| `converters.mil.frontend.tensorflow.test.test_composite_ops.tf` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.InputSpec` | Class | `(kwargs)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.Operation` | Class | `(kwargs)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.TensorFlowBaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.TestCustomMatMul` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.TestCustomTopK` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.backends` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.compute_units` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.is_symbolic` | Function | `(val)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.make_tf_graph` | Function | `(input_types)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.mb` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.register_op` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.register_tf_op` | Function | `(_func, tf_alias, override, strict)` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.tf` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_custom_ops.types` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_graphs.TensorFlowBaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_graphs.TestTFGraphs` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_graphs.backends` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_graphs.compute_units` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_graphs.make_tf_graph` | Function | `(input_types)` |
| `converters.mil.frontend.tensorflow.test.test_graphs.tf` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_load.EnumeratedShapes` | Class | `(shapes, default)` |
| `converters.mil.frontend.tensorflow.test.test_load.ImageType` | Class | `(name, shape, scale, bias, color_layout, channel_first, grayscale_use_uint8)` |
| `converters.mil.frontend.tensorflow.test.test_load.MSG_TF1_NOT_FOUND` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_load.RangeDim` | Class | `(lower_bound: int, upper_bound: int, default: Optional[int], symbol: Optional[str])` |
| `converters.mil.frontend.tensorflow.test.test_load.TFConverter` | Class | `(tfssa, inputs, outputs, opset_version, use_default_fp16_io)` |
| `converters.mil.frontend.tensorflow.test.test_load.TensorFlowBaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_load.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.tensorflow.test.test_load.TestTf1ModelFormats` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_load.TestTfModelInputsOutputs` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_load.backends` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_load.converter` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_load.frontend` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_load.ft` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_load.get_tf_keras_io_names` | Function | `(model)` |
| `converters.mil.frontend.tensorflow.test.test_load.make_tf_graph` | Function | `(input_types)` |
| `converters.mil.frontend.tensorflow.test.test_load.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.frontend.tensorflow.test.test_load.tf` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_ops.MSG_TF1_NOT_FOUND` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_ops.Operation` | Class | `(kwargs)` |
| `converters.mil.frontend.tensorflow.test.test_ops.PREBUILT_TF1_WHEEL_VERSION` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_ops.Program` | Class | `()` |
| `converters.mil.frontend.tensorflow.test.test_ops.RangeDim` | Class | `(lower_bound: int, upper_bound: int, default: Optional[int], symbol: Optional[str])` |
| `converters.mil.frontend.tensorflow.test.test_ops.TensorFlowBaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestActivation` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestAddN` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestAddOrdering` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestArgSort` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestAudioSpectrogram` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestBatchNormalization` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestBatchToSpaceND` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestBroadcastTo` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestCast` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestClipByValue` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestComplex` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestConcat` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestCond` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestContribLSTMBlockCell` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestContribResampler` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestConv` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestConv3d` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestConvTranspose` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestCross` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestCumSum` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestDebugging` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestDepthToSpace` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestDepthwiseConv` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestDuplicateOutputs` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestDynamicTile` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestEinsum` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestElementWiseBinary` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestElementWiseUnary` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestExpandDims` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestFakeQuant` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestFft` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestFill` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestGather` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestGelu` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestIdentity` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestIdentityN` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestIfft` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestImag` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestImageResizing` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestIrfft` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestIsFinite` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestL2Normalization` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestLinear` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestLocalResponseNormalization` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestLogSoftMax` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestMatrixBandPart` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestMatrixDiag` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestMfcc` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestNonMaximumSuppression` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestNormalization` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestOneHot` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestPack` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestPad` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestPadV2` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestPlaceholderAsOutput` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestPool1d` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestPool2d` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestPool3d` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestPrint` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestRandom` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestRange` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestReal` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestReduction` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestReshape` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestReverse` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestReverseSequence` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestRfft` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestScatter` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestSelect` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestSeparableConv` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestShape` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestSize` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestSliceByIndex` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestSliceBySize` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestSoftmaxCrossEntropyWithLogits` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestSpaceToBatchND` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestSpaceToDepth` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestSplit` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestSqueeze` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestStack` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestTensorArray` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestTensorScatterAdd` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestTile` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestTopK` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestTranspose` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestUnstack` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestVariable` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestWhere` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestWhileLoop` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.TestZerosLike` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.Testlog1p` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_ops.backends` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_ops.compute_units` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_ops.einsum_equations` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_ops.freeze_g` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_ops.gen_input_shapes_einsum` | Function | `(equation: str, dynamic: bool, backend: Tuple[str, str])` |
| `converters.mil.frontend.tensorflow.test.test_ops.get_tf_node_names` | Function | `(tf_nodes, mode)` |
| `converters.mil.frontend.tensorflow.test.test_ops.layer_counts` | Function | `(spec, layer_type)` |
| `converters.mil.frontend.tensorflow.test.test_ops.load_tf_pb` | Function | `(pb_file)` |
| `converters.mil.frontend.tensorflow.test.test_ops.make_tf_graph` | Function | `(input_types)` |
| `converters.mil.frontend.tensorflow.test.test_ops.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.frontend.tensorflow.test.test_ops.tf` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_ops.types` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_parse.TestParse` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_parse.mil_types` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_parse.parse` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_parsed_tf_node.ParsedTFNode` | Class | `(tfnode)` |
| `converters.mil.frontend.tensorflow.test.test_parsed_tf_node.TestParsedTFNode` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.MSG_TF1_NOT_FOUND` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.TestInputOutputConversionAPI` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.TestTensorFlow1ConverterExamples` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.TestTf1Inputs` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.TestiOS16DefaultIODtype` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.assert_cast_ops_count` | Function | `(mlmodel, expected_count)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.assert_input_dtype` | Function | `(mlmodel, expected_type_str, expected_name, index)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.assert_ops_in_mil_program` | Function | `(mlmodel, expected_op_list)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.assert_output_dtype` | Function | `(mlmodel, expected_type_str, expected_name, index)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.assert_prog_input_type` | Function | `(prog, expected_dtype_str, expected_name, index)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.assert_prog_output_type` | Function | `(prog, expected_dtype_str, expected_name, index)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.assert_spec_input_image_type` | Function | `(spec, expected_feature_type)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.assert_spec_output_image_type` | Function | `(spec, expected_feature_type)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.backends` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.compute_units` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.float32_input_model_add_op` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.float32_input_model_relu_ops` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.float32_two_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.float32_two_output_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.float64_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.int32_float32_two_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.int32_float32_two_output_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.int32_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.int32_two_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.int32_two_output_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.int64_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.int8_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.linear_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.proto` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.rank3_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.rank4_grayscale_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.rank4_grayscale_input_model_with_channel_first_output` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.rank4_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.rank4_input_model_with_channel_first_output` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.tf` | Object | `` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.uint8_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow.test.test_tf_conversion_api.verify_prediction` | Function | `(mlmodel, multiarray_type)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.TensorFlow2BaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.TensorFlowBaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.compare_backend` | Function | `(mlmodel, input_key_values, expected_outputs, dtype, atol, rtol, also_compare_shapes, state, allow_mismatch_ratio)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.coremltoolsutils` | Object | `` |
| `converters.mil.frontend.tensorflow.test.testing_utils.ct` | Object | `` |
| `converters.mil.frontend.tensorflow.test.testing_utils.ct_convert` | Function | `(program, source, inputs, outputs, classifier_config, minimum_deployment_target, convert_to, compute_precision, skip_model_load, converter, kwargs)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.get_tf_keras_io_names` | Function | `(model)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.get_tf_node_names` | Function | `(tf_nodes, mode)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.layer_counts` | Function | `(spec, layer_type)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.load_tf_pb` | Function | `(pb_file)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.make_tf2_graph` | Function | `(input_types)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.make_tf_graph` | Function | `(input_types)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.run_compare_tf` | Function | `(graph, feed_dict, output_nodes, inputs_for_conversion, compute_unit, frontend_only, frontend, backend, atol, rtol, freeze_graph, tf_outputs, minimum_deployment_target)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.tf` | Object | `` |
| `converters.mil.frontend.tensorflow.test.testing_utils.tf_graph_to_mlmodel` | Function | `(graph, feed_dict, output_nodes, frontend, backend, compute_unit, inputs_for_conversion, minimum_deployment_target)` |
| `converters.mil.frontend.tensorflow.test.testing_utils.validate_minimum_deployment_target` | Function | `(minimum_deployment_target: ct.target, backend: Tuple[str, str])` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.cond_to_where.CondToWhere` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.cond_to_where.FindAllUpstreamTerminals` | Class | `(fn, control_dependencies)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.cond_to_where.compute_max_rank` | Function | `(graph)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.cond_to_where.cond_to_where` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.cond_to_where.delete_node` | Function | `(g, node)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.cond_to_where.disconnect_edge` | Function | `(g, source, dest)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.cond_to_where.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.constant_propagation.const_determined_nodes` | Function | `(gd, assume_variable_nodes)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.constant_propagation.constant_propagation` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.constant_propagation.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.constant_propagation.numpy_val_to_builtin_val` | Function | `(npval)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.constant_propagation.types` | Object | `` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_asserts.delete_asserts` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_asserts.delete_node` | Function | `(g, node)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_asserts.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_constant.check_connections` | Function | `(gd)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_constant.convert_constant_nodes_to_const_ops` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_constant.delete_node` | Function | `(g, node)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_constant.delete_nodes_with_only_constant_descendents` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_constant.delete_unnecessary_constant_nodes` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_constant.disconnect_edge` | Function | `(g, source, dest)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_constant.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_disconnected_nodes.delete_disconnected_nodes` | Function | `(gd)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.delete_unnecessary_constant_nodes` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.FindAllReachableNodes` | Class | `(fn)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.FindImmediateDownstreamNodes` | Class | `(fn)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.FindImmediateUpstreamNodes` | Class | `(fn)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.FindSubgraph` | Class | `(terminal_nodes)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.FunctionalizeLoops` | Class | `()` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.ParsedTFNode` | Class | `(tfnode)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.SSAFunction` | Class | `(gdict, inputs, outputs, ret)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.connect_dests` | Function | `(g, source, dests)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.connect_edge` | Function | `(g, source, dest)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.connect_sources` | Function | `(g, sources, dest)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.delete_node` | Function | `(g, node)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.disconnect_edge` | Function | `(g, source, dest)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.functionalize_loops` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.replace_dest` | Function | `(g, source, dest, new_dest)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.functionalize_loops.replace_source` | Function | `(g, source, dest, new_source)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.fuse_dilation_conv.delete_node` | Function | `(g, node)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.fuse_dilation_conv.fuse_dilated_conv` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.fuse_dilation_conv.fuse_dilation_conv` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.fuse_dilation_conv.replace_source` | Function | `(g, source, dest, new_source)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.insert_get_tuple.ParsedTFNode` | Class | `(tfnode)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.insert_get_tuple.insert_get_tuple` | Function | `(gddict)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.quantization_pass.delete_fakequant_node_and_repair_graph` | Function | `(g, node)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.quantization_pass.delete_node` | Function | `(g, node)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.quantization_pass.quantization_pass` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.quantization_pass.quantization_pass_impl` | Function | `(fn)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.remove_variable_nodes` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.tensor_array_resource_removal` | Function | `(gd)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.tensor_array_transform.tensor_array_resource_removal` | Function | `(gd)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.variable_node_transform.delete_node` | Function | `(g, node)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.variable_node_transform.disconnect_vertex_ins` | Function | `(g, dest)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.variable_node_transform.remove_variable_node_impl` | Function | `(fn, tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.variable_node_transform.remove_variable_nodes` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.visitors.FindAllDownstreamTerminals` | Class | `(fn)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.visitors.FindAllReachableNodes` | Class | `(fn)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.visitors.FindAllUpstreamTerminals` | Class | `(fn, control_dependencies)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.visitors.FindImmediateDownstreamNodes` | Class | `(fn)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.visitors.FindImmediateUpstreamNodes` | Class | `(fn)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.visitors.FindSubgraph` | Class | `(terminal_nodes)` |
| `converters.mil.frontend.tensorflow.tf_graph_pass.visitors.ParsedTFNode` | Class | `(tfnode)` |
| `converters.mil.frontend.tensorflow.tf_lstm_block` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.tf_lstm_block_cell` | Class | `(kwargs)` |
| `converters.mil.frontend.tensorflow.tf_make_list` | Class | `(...)` |
| `converters.mil.frontend.tensorflow.tf_op_registry.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.tf_op_registry.register_tf_op` | Function | `(_func, tf_alias, override, strict)` |
| `converters.mil.frontend.tensorflow.tfssa.DotVisitor` | Class | `(annotation)` |
| `converters.mil.frontend.tensorflow.tfssa.NetworkEnsemble` | Class | `(instance)` |
| `converters.mil.frontend.tensorflow.tfssa.ParsedNode` | Class | `()` |
| `converters.mil.frontend.tensorflow.tfssa.SSAFunction` | Class | `(gdict, inputs, outputs, ret)` |
| `converters.mil.frontend.tensorflow.tfssa.check_connections` | Function | `(gd)` |
| `converters.mil.frontend.tensorflow.tfssa.const_determined_nodes` | Function | `(gd, assume_variable_nodes)` |
| `converters.mil.frontend.tensorflow.tfssa.escape_fn_name` | Function | `(name)` |
| `converters.mil.frontend.tensorflow.tfssa.logger` | Object | `` |
| `converters.mil.frontend.tensorflow.tfssa.types` | Object | `` |
| `converters.mil.frontend.tensorflow2.converter.TF2Converter` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.converter.TFConverter` | Class | `(tfssa, inputs, outputs, opset_version, use_default_fp16_io)` |
| `converters.mil.frontend.tensorflow2.converter.simple_topsort` | Function | `(inputs)` |
| `converters.mil.frontend.tensorflow2.load.NetworkEnsemble` | Class | `(instance)` |
| `converters.mil.frontend.tensorflow2.load.ParsedTFNode` | Class | `(tfnode)` |
| `converters.mil.frontend.tensorflow2.load.SSAFunction` | Class | `(gdict, inputs, outputs, ret)` |
| `converters.mil.frontend.tensorflow2.load.TF2Converter` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.load.TF2Loader` | Class | `(model, debug, kwargs)` |
| `converters.mil.frontend.tensorflow2.load.TFLoader` | Class | `(model, debug, kwargs)` |
| `converters.mil.frontend.tensorflow2.load.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.tensorflow2.load.constant_propagation` | Object | `` |
| `converters.mil.frontend.tensorflow2.load.delete_disconnected_nodes` | Object | `` |
| `converters.mil.frontend.tensorflow2.load.delete_unnecessary_constant_nodes` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow2.load.fill_outputs` | Function | `(gd)` |
| `converters.mil.frontend.tensorflow2.load.flatten_sub_graph_namespaces` | Function | `(tf_ssa)` |
| `converters.mil.frontend.tensorflow2.load.fuse_dilation_conv` | Object | `` |
| `converters.mil.frontend.tensorflow2.load.insert_get_tuple` | Object | `` |
| `converters.mil.frontend.tensorflow2.load.logger` | Object | `` |
| `converters.mil.frontend.tensorflow2.load.remove_variable_nodes` | Function | `(tfssa)` |
| `converters.mil.frontend.tensorflow2.load.rewrite_control_flow_functions` | Object | `` |
| `converters.mil.frontend.tensorflow2.load.tensor_array_resource_removal` | Function | `(gd)` |
| `converters.mil.frontend.tensorflow2.ops.FusedBatchNormV3` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow2.ops.StatelessIf` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow2.ops.StatelessWhile` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow2.ops.TensorListFromTensor` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow2.ops.TensorListGather` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow2.ops.TensorListGetItem` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow2.ops.TensorListLength` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow2.ops.TensorListReserve` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow2.ops.TensorListScatterIntoExistingList` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow2.ops.TensorListSetItem` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow2.ops.TensorListStack` | Function | `(context, node)` |
| `converters.mil.frontend.tensorflow2.ops.any_symbolic` | Function | `(val)` |
| `converters.mil.frontend.tensorflow2.ops.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.frontend.tensorflow2.ops.convert_graph` | Function | `(context, graph, outputs)` |
| `converters.mil.frontend.tensorflow2.ops.mb` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.ops.ops` | Object | `` |
| `converters.mil.frontend.tensorflow2.ops.register_tf_op` | Function | `(_func, tf_alias, override, strict)` |
| `converters.mil.frontend.tensorflow2.register_tf_op` | Function | `(_func, tf_alias, override, strict)` |
| `converters.mil.frontend.tensorflow2.ssa_passes.remove_vacuous_cond.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.ssa_passes.remove_vacuous_cond.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.frontend.tensorflow2.ssa_passes.remove_vacuous_cond.logger` | Object | `` |
| `converters.mil.frontend.tensorflow2.ssa_passes.remove_vacuous_cond.mb` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.ssa_passes.remove_vacuous_cond.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.frontend.tensorflow2.ssa_passes.remove_vacuous_cond.remove_vacuous_cond` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.ssa_passes.test_v2_passes.PASS_REGISTRY` | Object | `` |
| `converters.mil.frontend.tensorflow2.ssa_passes.test_v2_passes.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.frontend.tensorflow2.ssa_passes.test_v2_passes.assert_same_output_names` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.frontend.tensorflow2.ssa_passes.test_v2_passes.mb` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.ssa_passes.test_v2_passes.test_remove_vacuous_cond` | Function | `()` |
| `converters.mil.frontend.tensorflow2.ssa_passes.test_v2_passes.types` | Object | `` |
| `converters.mil.frontend.tensorflow2.ssa_passes.test_v2_passes.validate_model` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.TestTF2FlexibleInput` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.TestTensorFlow2ConverterExamples` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.TestTf2InputOutputConversionAPI` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.TestTf2iOS16DefaultIODtype` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.backends` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.float32_input_model_add_op` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.float32_input_model_relu_ops` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.float32_two_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.float32_two_output_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.float64_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.int32_float32_two_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.int32_float32_two_output_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.int32_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.int32_two_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.int32_two_output_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.int64_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.int8_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.linear_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.rank3_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.rank4_grayscale_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.rank4_grayscale_input_model_with_channel_first_output` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.rank4_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.rank4_input_model_with_channel_first_output` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.types` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_tf2_conversion_api.uint8_input_model` | Function | `()` |
| `converters.mil.frontend.tensorflow2.test.test_v2_load.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_load.TestTf2ModelFormats` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_load.backends` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_v2_load.converter` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_v2_load.frontend` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_v2_load.get_tf_keras_io_names` | Function | `(model)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_load.tf` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.TensorFlow2BaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.TensorFlowBaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.TestActivationSiLU` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.TestControlFlowFromAutoGraph` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.TestElementWiseBinaryTF2` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.TestImageResample` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.TestImageTransform` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.TestNormalizationTF2` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.TestPartitionedCall` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.TestResizeNearestNeighbor` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.TestTensorList` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.backends` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.compute_units` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.make_tf_graph` | Function | `(input_types)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.testing_reqs` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops.tf` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TensorFlow2BaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TensorFlowBaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestActivation` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestBatchNormalization` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestBinary` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestConcatenate` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestConvTranspose` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestConvolution` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestCropping` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestDense` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestEmbedding` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestFlatten` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestGelu` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestGlobalPooling` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestInstanceNormalization` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestLambda` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestMasking` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestNormalization` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestPadding` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestPermute` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestPooling` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestRecurrent` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestRepeatVector` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestReshape` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestSkips` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.TestUpSampling` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.backends` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.compute_units` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.is_symbolic_dim_in_prog` | Function | `(prog)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.testing_reqs` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.test_v2_ops_tf_keras.tf` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.RangeDim` | Class | `(lower_bound: int, upper_bound: int, default: Optional[int], symbol: Optional[str])` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.TensorFlow2BaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.TensorFlowBaseTest` | Class | `(...)` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.compare_backend` | Function | `(mlmodel, input_key_values, expected_outputs, dtype, atol, rtol, also_compare_shapes, state, allow_mismatch_ratio)` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.coremltoolsutils` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.ct_convert` | Function | `(program, source, inputs, outputs, classifier_config, minimum_deployment_target, convert_to, compute_precision, skip_model_load, converter, kwargs)` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.get_tf_node_names` | Function | `(tf_nodes, mode)` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.make_tf2_graph` | Function | `(input_types)` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.run_compare_tf2` | Function | `(model, input_dict, output_names, inputs_for_conversion, compute_unit, frontend_only, frontend, backend, debug, atol, rtol, minimum_deployment_target)` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.run_compare_tf_keras` | Function | `(model, input_values, inputs_for_conversion, compute_unit, frontend_only, frontend, backend, atol, rtol)` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.tf` | Object | `` |
| `converters.mil.frontend.tensorflow2.test.testing_utils.validate_minimum_deployment_target` | Function | `(minimum_deployment_target: ct.target, backend: Tuple[str, str])` |
| `converters.mil.frontend.tensorflow2.tf_graph_pass.flatten_sub_graph_namespaces` | Function | `(tf_ssa)` |
| `converters.mil.frontend.tensorflow2.tf_graph_pass.rewrite_control_flow_functions.ParsedTFNode` | Class | `(tfnode)` |
| `converters.mil.frontend.tensorflow2.tf_graph_pass.rewrite_control_flow_functions.connect_edge` | Function | `(g, source, dest)` |
| `converters.mil.frontend.tensorflow2.tf_graph_pass.rewrite_control_flow_functions.connect_edge_at_index` | Function | `(g, source, dest, idx)` |
| `converters.mil.frontend.tensorflow2.tf_graph_pass.rewrite_control_flow_functions.delete_node` | Function | `(g, node)` |
| `converters.mil.frontend.tensorflow2.tf_graph_pass.rewrite_control_flow_functions.disconnect_edge` | Function | `(g, source, dest)` |
| `converters.mil.frontend.tensorflow2.tf_graph_pass.rewrite_control_flow_functions.flatten_sub_graph_namespaces` | Function | `(tf_ssa)` |
| `converters.mil.frontend.tensorflow2.tf_graph_pass.rewrite_control_flow_functions.logger` | Object | `` |
| `converters.mil.frontend.tensorflow2.tf_graph_pass.rewrite_control_flow_functions.replace_dest` | Function | `(g, source, dest, new_dest)` |
| `converters.mil.frontend.tensorflow2.tf_graph_pass.rewrite_control_flow_functions.replace_node` | Function | `(g, original_node, new_node)` |
| `converters.mil.frontend.tensorflow2.tf_graph_pass.rewrite_control_flow_functions.rewrite_control_flow_functions` | Function | `(tf_ssa)` |
| `converters.mil.frontend.torch.converter.ColorLayout` | Class | `(...)` |
| `converters.mil.frontend.torch.converter.CompressionInfo` | Class | `(...)` |
| `converters.mil.frontend.torch.converter.CompressionType` | Class | `(...)` |
| `converters.mil.frontend.torch.converter.EnumeratedShapes` | Class | `(shapes, default)` |
| `converters.mil.frontend.torch.converter.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.frontend.torch.converter.ImageType` | Class | `(name, shape, scale, bias, color_layout, channel_first, grayscale_use_uint8)` |
| `converters.mil.frontend.torch.converter.InputType` | Class | `(name, shape, dtype)` |
| `converters.mil.frontend.torch.converter.InternalTorchIRGraph` | Class | `(params: Dict[str, np.ndarray], inputs: Dict[str, TensorType], outputs: List[str], nodes: Optional[List[InternalTorchIRNode]], buffers: Optional[Dict[str, torch.Tensor]], input_name_to_source_buffer_name: Optional[Dict[str, str]], output_name_to_target_buffer_name: Optional[Dict[str, str]])` |
| `converters.mil.frontend.torch.converter.InternalTorchIRNode` | Class | `(kind: str, inputs: List[str], outputs: List[str], kwinputs: Optional[Dict[str, str]], name: Optional[str], parent: Optional[Union[InternalTorchIRGraph, InternalTorchIRBlock]], attr: Optional[Dict[str, Any]], blocks: Optional[List[InternalTorchIRBlock]], model_hierarchy: Optional[str], meta: Optional[Dict])` |
| `converters.mil.frontend.torch.converter.NUM_TO_NUMPY_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.converter.Placeholder` | Class | `(sym_shape, dtype, name, allow_rank0_input)` |
| `converters.mil.frontend.torch.converter.Program` | Class | `()` |
| `converters.mil.frontend.torch.converter.QuantizationContext` | Class | `(context: TranscriptionContext)` |
| `converters.mil.frontend.torch.converter.ScopeInfo` | Class | `(...)` |
| `converters.mil.frontend.torch.converter.ScopeSource` | Class | `(...)` |
| `converters.mil.frontend.torch.converter.StateType` | Class | `(wrapped_type: type, name: Optional[str])` |
| `converters.mil.frontend.torch.converter.TORCH_DTYPE_TO_MIL_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.converter.TORCH_DTYPE_TO_NUM` | Object | `` |
| `converters.mil.frontend.torch.converter.TORCH_EXPORT_BASED_FRONTENDS` | Object | `` |
| `converters.mil.frontend.torch.converter.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.torch.converter.TorchConverter` | Class | `(loaded_model: Union[RecursiveScriptModule, ExportedProgram], inputs: Optional[List[TensorType]], outputs: Optional[List[TensorType]], cut_at_symbols: Optional[List[str]], opset_version: Optional[int], use_default_fp16_io: bool, states: Optional[List[StateType]])` |
| `converters.mil.frontend.torch.converter.TorchFrontend` | Class | `(...)` |
| `converters.mil.frontend.torch.converter.TranscriptionContext` | Class | `(name: Optional[str], frontend: TorchFrontend)` |
| `converters.mil.frontend.torch.converter.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.frontend.torch.converter.WRAPPED_SCALAR_INPUT_SUFFIX` | Object | `` |
| `converters.mil.frontend.torch.converter.any_symbolic` | Function | `(val)` |
| `converters.mil.frontend.torch.converter.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.frontend.torch.converter.convert_nodes` | Function | `(context: TranscriptionContext, graph: InternalTorchIRGraph, early_exit: Optional[bool]) -> None` |
| `converters.mil.frontend.torch.converter.flatten_graph_input_values` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.converter.flatten_graph_output_values` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.converter.frontend_utils` | Object | `` |
| `converters.mil.frontend.torch.converter.generate_tensor_assignment_ops` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.converter.is_current_opset_version_compatible_with` | Function | `(opset_version)` |
| `converters.mil.frontend.torch.converter.is_float` | Function | `(t)` |
| `converters.mil.frontend.torch.converter.is_symbolic` | Function | `(val)` |
| `converters.mil.frontend.torch.converter.logger` | Object | `` |
| `converters.mil.frontend.torch.converter.mb` | Class | `(...)` |
| `converters.mil.frontend.torch.converter.mil` | Object | `` |
| `converters.mil.frontend.torch.converter.optimize_utils` | Object | `` |
| `converters.mil.frontend.torch.converter.populate_native_const_model_hierarchy` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.converter.prune_weights` | Class | `(...)` |
| `converters.mil.frontend.torch.converter.remove_getattr_nodes` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.converter.transform_inplace_ops` | Function | `(graph: InternalTorchIRGraph, name_remap_dict: Optional[Dict[str, str]]) -> None` |
| `converters.mil.frontend.torch.converter.types` | Object | `` |
| `converters.mil.frontend.torch.dialect_ops.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.frontend.torch.dialect_ops.InputSpec` | Class | `(kwargs)` |
| `converters.mil.frontend.torch.dialect_ops.Operation` | Class | `(kwargs)` |
| `converters.mil.frontend.torch.dialect_ops.SSAOpRegistry` | Class | `(...)` |
| `converters.mil.frontend.torch.dialect_ops.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.frontend.torch.dialect_ops.get_new_symbol` | Function | `(name)` |
| `converters.mil.frontend.torch.dialect_ops.get_param_val` | Function | `(param)` |
| `converters.mil.frontend.torch.dialect_ops.is_compatible_symbolic_vector` | Function | `(val_a, val_b)` |
| `converters.mil.frontend.torch.dialect_ops.register_op` | Object | `` |
| `converters.mil.frontend.torch.dialect_ops.solve_slice_by_index_shape` | Function | `(x_shape, begin, end, stride, begin_mask, end_mask, squeeze_mask)` |
| `converters.mil.frontend.torch.dialect_ops.torch_tensor_assign` | Class | `(...)` |
| `converters.mil.frontend.torch.dialect_ops.torch_upsample_bilinear` | Class | `(...)` |
| `converters.mil.frontend.torch.dialect_ops.torch_upsample_nearest_neighbor` | Class | `(...)` |
| `converters.mil.frontend.torch.dialect_ops.types` | Object | `` |
| `converters.mil.frontend.torch.dim_order_ops.NUM_TO_DTYPE_STRING` | Object | `` |
| `converters.mil.frontend.torch.dim_order_ops.NUM_TO_NUMPY_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.dim_order_ops.NUM_TO_TORCH_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.dim_order_ops.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.frontend.torch.dim_order_ops.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.frontend.torch.dim_order_ops.dtype_to_32bit` | Function | `(dtype)` |
| `converters.mil.frontend.torch.dim_order_ops.mb` | Class | `(...)` |
| `converters.mil.frontend.torch.dim_order_ops.register_torch_op` | Function | `(_func, torch_alias, override)` |
| `converters.mil.frontend.torch.exir_utils.RangeDim` | Class | `(lower_bound: int, upper_bound: int, default: Optional[int], symbol: Optional[str])` |
| `converters.mil.frontend.torch.exir_utils.TORCH_DTYPE_TO_MIL_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.exir_utils.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.torch.exir_utils.WRAPPED_SCALAR_INPUT_SUFFIX` | Object | `` |
| `converters.mil.frontend.torch.exir_utils.extract_io_from_exir_program` | Function | `(exported_program) -> Tuple[List[TensorType], List[str], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, str], Dict[str, str]]` |
| `converters.mil.frontend.torch.exir_utils.logger` | Object | `` |
| `converters.mil.frontend.torch.exir_utils.types` | Object | `` |
| `converters.mil.frontend.torch.internal_graph.InternalTorchIRBlock` | Class | `(parent: Optional[InternalTorchIRNode], nodes: Optional[List[InternalTorchIRNode]], inputs: Optional[List[str]], outputs: Optional[List[str]])` |
| `converters.mil.frontend.torch.internal_graph.InternalTorchIRGraph` | Class | `(params: Dict[str, np.ndarray], inputs: Dict[str, TensorType], outputs: List[str], nodes: Optional[List[InternalTorchIRNode]], buffers: Optional[Dict[str, torch.Tensor]], input_name_to_source_buffer_name: Optional[Dict[str, str]], output_name_to_target_buffer_name: Optional[Dict[str, str]])` |
| `converters.mil.frontend.torch.internal_graph.InternalTorchIRNode` | Class | `(kind: str, inputs: List[str], outputs: List[str], kwinputs: Optional[Dict[str, str]], name: Optional[str], parent: Optional[Union[InternalTorchIRGraph, InternalTorchIRBlock]], attr: Optional[Dict[str, Any]], blocks: Optional[List[InternalTorchIRBlock]], model_hierarchy: Optional[str], meta: Optional[Dict])` |
| `converters.mil.frontend.torch.internal_graph.TORCH_DTYPE_TO_NUM` | Object | `` |
| `converters.mil.frontend.torch.internal_graph.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.torch.internal_graph.extract_io_from_exir_program` | Function | `(exported_program) -> Tuple[List[TensorType], List[str], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, str], Dict[str, str]]` |
| `converters.mil.frontend.torch.internal_graph.logger` | Object | `` |
| `converters.mil.frontend.torch.internal_graph.sanitize_op_kind` | Function | `(op_kind: str) -> str` |
| `converters.mil.frontend.torch.is_torch_fx_node_supported` | Function | `(torch_fx_node: torch.fx.Node) -> bool` |
| `converters.mil.frontend.torch.load.Program` | Class | `()` |
| `converters.mil.frontend.torch.load.StateType` | Class | `(wrapped_type: type, name: Optional[str])` |
| `converters.mil.frontend.torch.load.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.torch.load.TorchConverter` | Class | `(loaded_model: Union[RecursiveScriptModule, ExportedProgram], inputs: Optional[List[TensorType]], outputs: Optional[List[TensorType]], cut_at_symbols: Optional[List[str]], opset_version: Optional[int], use_default_fp16_io: bool, states: Optional[List[StateType]])` |
| `converters.mil.frontend.torch.load.TorchFrontend` | Class | `(...)` |
| `converters.mil.frontend.torch.load.is_torch_model` | Function | `(model_spec: Union[str, RecursiveScriptModule]) -> bool` |
| `converters.mil.frontend.torch.load.load` | Function | `(spec: Union[RecursiveScriptModule, ExportedProgram, str], inputs: List[TensorType], specification_version: int, debug: bool, outputs: Optional[List[TensorType]], cut_at_symbols: Optional[List[str]], use_default_fp16_io: bool, states: Optional[List[StateType]], kwargs) -> Program` |
| `converters.mil.frontend.torch.load.logger` | Object | `` |
| `converters.mil.frontend.torch.ops.InternalTorchIRGraph` | Class | `(params: Dict[str, np.ndarray], inputs: Dict[str, TensorType], outputs: List[str], nodes: Optional[List[InternalTorchIRNode]], buffers: Optional[Dict[str, torch.Tensor]], input_name_to_source_buffer_name: Optional[Dict[str, str]], output_name_to_target_buffer_name: Optional[Dict[str, str]])` |
| `converters.mil.frontend.torch.ops.InternalTorchIRNode` | Class | `(kind: str, inputs: List[str], outputs: List[str], kwinputs: Optional[Dict[str, str]], name: Optional[str], parent: Optional[Union[InternalTorchIRGraph, InternalTorchIRBlock]], attr: Optional[Dict[str, Any]], blocks: Optional[List[InternalTorchIRBlock]], model_hierarchy: Optional[str], meta: Optional[Dict])` |
| `converters.mil.frontend.torch.ops.ListVar` | Class | `(name, elem_type, init_length, dynamic_length, sym_val, kwargs)` |
| `converters.mil.frontend.torch.ops.MAX_SIZE_CONSTANT_FOLDING` | Object | `` |
| `converters.mil.frontend.torch.ops.MIL_DTYPE_TO_TORCH_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.ops.NUMPY_DTYPE_TO_TORCH_NUM` | Object | `` |
| `converters.mil.frontend.torch.ops.NUM_TO_DTYPE_STRING` | Object | `` |
| `converters.mil.frontend.torch.ops.NUM_TO_NUMPY_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.ops.NUM_TO_TORCH_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.ops.PYTORCH_DEFAULT_VALUE` | Object | `` |
| `converters.mil.frontend.torch.ops.SSAOpRegistry` | Class | `(...)` |
| `converters.mil.frontend.torch.ops.ScopeInfo` | Class | `(...)` |
| `converters.mil.frontend.torch.ops.ScopeSource` | Class | `(...)` |
| `converters.mil.frontend.torch.ops.Symbol` | Class | `(sym_name)` |
| `converters.mil.frontend.torch.ops.TORCH_DTYPE_TO_NUM` | Object | `` |
| `converters.mil.frontend.torch.ops.TORCH_EXPORT_BASED_FRONTENDS` | Object | `` |
| `converters.mil.frontend.torch.ops.TORCH_STRING_ARGS` | Object | `` |
| `converters.mil.frontend.torch.ops.TorchFrontend` | Class | `(...)` |
| `converters.mil.frontend.torch.ops.TranscriptionContext` | Class | `(weights_dir)` |
| `converters.mil.frontend.torch.ops.VALUE_CLOSE_TO_INFINITY` | Object | `` |
| `converters.mil.frontend.torch.ops.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.frontend.torch.ops.acos` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.acosh` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.adaptive_avg_pool1d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.adaptive_avg_pool2d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.adaptive_max_pool1d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.adaptive_max_pool2d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.add` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.addmm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.affine_grid_generator` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.amax` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.amin` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.any_symbolic` | Function | `(val)` |
| `converters.mil.frontend.torch.ops.append` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.arange` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.arange_start_step` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.argmax` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.argsort` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.asin` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.asinh` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.atan` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.atan2` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.atanh` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.atleast_1d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.avg_pool1d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.avg_pool2d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.avg_pool3d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.baddbmm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.batch_norm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.bitwise_and` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.bitwise_not` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.broadcast_tensors` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.frontend.torch.ops.cat` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.ceil` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.clamp` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.clamp_max` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.clamp_min` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.col2im` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.complex` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.constant` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.constantchunk` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.convert_block` | Function | `(context, block, inputs)` |
| `converters.mil.frontend.torch.ops.convert_nodes` | Function | `(context: TranscriptionContext, graph: InternalTorchIRGraph, early_exit: Optional[bool]) -> None` |
| `converters.mil.frontend.torch.ops.convert_single_node` | Function | `(context: TranscriptionContext, node: InternalTorchIRNode) -> None` |
| `converters.mil.frontend.torch.ops.copy` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.cos` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.cosh` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.cosine_similarity` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.cross` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.cumprod` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.cumsum` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.diagonal` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.dim` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.div` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.dot` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.dtype` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.dtype_to_32bit` | Function | `(dtype)` |
| `converters.mil.frontend.torch.ops.einsum` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.elu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.embedding` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.eq` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.erf` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.exp` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.exp2` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.expand` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.expand_as` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.expm1` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.eye` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.fft_fft` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.fft_fftn` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.fft_ifft` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.fft_ifftn` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.fft_irfft` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.fft_irfftn` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.fft_rfft` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.fft_rfftn` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.fill` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.flatten` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.flip` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.fliplr` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.floor` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.floor_divide` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.frac` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.frobenius_norm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.full` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.full_like` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.gather` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.ge` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.gelu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.getitem` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.glu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.grid_sampler` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.group_norm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.gru` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.gt` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.hann_window` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.hardsigmoid` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.hardswish` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.hardtanh` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.hstack` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.im2col` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.imag` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.implicittensortonum` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.index` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.index_put` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.index_select` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.instance_norm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.is_bool` | Function | `(t)` |
| `converters.mil.frontend.torch.ops.is_current_opset_version_compatible_with` | Function | `(opset_version)` |
| `converters.mil.frontend.torch.ops.is_floating_point` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.is_symbolic` | Function | `(val)` |
| `converters.mil.frontend.torch.ops.isnan` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.item` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.layer_norm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.le` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.leaky_relu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.linalg_inv` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.linalg_matrix_norm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.linalg_norm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.linalg_vecdot` | Function | `(context: TranscriptionContext, node: InternalTorchIRNode)` |
| `converters.mil.frontend.torch.ops.linalg_vector_norm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.linear` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.linspace` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.listconstruct` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.log` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.log10` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.log1p` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.log2` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.log_softmax` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.logger` | Object | `` |
| `converters.mil.frontend.torch.ops.logical_and` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.logical_not` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.logical_or` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.logical_xor` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.loop` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.lstm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.lt` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.masked_fill` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.matmul` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.max` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.max_pool1d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.maximum` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.mb` | Class | `(...)` |
| `converters.mil.frontend.torch.ops.mean` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.meshgrid` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.min` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.minimum` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.mish` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.mse_loss` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.mul` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.multinomial` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.mv` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.nan_to_num` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.narrow` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.native_dropout` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.native_group_norm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.ne` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.neg` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.new_full` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.new_zeros` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.nll_loss` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.nonzero` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.nonzero_numpy` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.noop` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.norm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.frontend.torch.ops.numel` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.numtotensor` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.one_hot` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.ones` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.ones_like` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.outer` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.pad` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.permute_copy` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.pixel_shuffle` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.pixel_unshuffle` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.pow` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.prelu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.promote_input_dtypes` | Function | `(input_vars)` |
| `converters.mil.frontend.torch.ops.rand` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.rand_like` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.randint` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.randn` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.randn_like` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.real` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.reciprocal` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.reflection_pad2d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.register_torch_op` | Function | `(_func, torch_alias, override)` |
| `converters.mil.frontend.torch.ops.relu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.relu6` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.remainder` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.repeat` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.repeat_interleave` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.replication_pad2d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.reshape_as` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.rms_norm` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.rnn_relu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.rnn_tanh` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.roll` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.rrelu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.rsqrt` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.scalar_tensor` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.scaled_dot_product_attention` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.scatter` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.scatter_add` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.searchsorted` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.select` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.select_scatter` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.selu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.sigmoid` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.sign` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.silu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.sin` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.sinh` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.size` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.slice` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.slice_scatter` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.softmax` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.softplus` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.softsign` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.solve_slice_by_index_shape` | Function | `(x_shape, begin, end, stride, begin_mask, end_mask, squeeze_mask)` |
| `converters.mil.frontend.torch.ops.sort` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.split` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.sqrt` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.square` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.squeeze` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.stack` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.std` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.std_correction` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.stft` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.sub` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.tan` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.tanh` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.target` | Class | `(...)` |
| `converters.mil.frontend.torch.ops.tensor` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.threshold` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.tile` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.to` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.topk` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.torchvision_nms` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.trace` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.transpose` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.tril` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.triu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.true_divide` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.tupleconstruct` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.tupleindex` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.tupleunpack` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.type_as` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.types` | Object | `` |
| `converters.mil.frontend.torch.ops.unbind` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.unflatten` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.unsqueeze` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.upsample_bilinear2d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.upsample_linear1d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.upsample_nearest1d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.upsample_nearest2d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.var` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.var_correction` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.view` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.view_as_real` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.where` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.where_scalarself` | Function | `(context: TranscriptionContext, node: InternalTorchIRNode)` |
| `converters.mil.frontend.torch.ops.zeros` | Function | `(context, node)` |
| `converters.mil.frontend.torch.ops.zeros_like` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.MSG_TORCHAO_NOT_FOUND` | Object | `` |
| `converters.mil.frontend.torch.quantization_ops.NUM_TO_NUMPY_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.quantization_ops.NUM_TO_TORCH_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.quantization_ops.TORCH_DTYPE_TO_NUM` | Object | `` |
| `converters.mil.frontend.torch.quantization_ops.TORCH_EXPORT_BASED_FRONTENDS` | Object | `` |
| `converters.mil.frontend.torch.quantization_ops.TORCH_QTYPE_TO_NP_TYPE` | Object | `` |
| `converters.mil.frontend.torch.quantization_ops.TORCH_QTYPE_TO_STR` | Object | `` |
| `converters.mil.frontend.torch.quantization_ops.TorchFrontend` | Class | `(...)` |
| `converters.mil.frontend.torch.quantization_ops.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.frontend.torch.quantization_ops.choose_qparams_per_token_asymmetric` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.dequantize` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.dequantize_affine` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.logger` | Object | `` |
| `converters.mil.frontend.torch.quantization_ops.mb` | Class | `(...)` |
| `converters.mil.frontend.torch.quantization_ops.promote_input_dtypes` | Function | `(input_vars)` |
| `converters.mil.frontend.torch.quantization_ops.quant_noop` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantize_per_channel` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantize_per_tensor` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantized_add` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantized_add_relu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantized_conv2d` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantized_conv2d_relu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantized_embedding` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantized_embedding_4bit` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantized_linear` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantized_linear_relu` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantized_matmul` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.quantized_mul` | Function | `(context, node)` |
| `converters.mil.frontend.torch.quantization_ops.register_torch_op` | Function | `(_func, torch_alias, override)` |
| `converters.mil.frontend.torch.quantization_ops.types` | Object | `` |
| `converters.mil.frontend.torch.register_torch_op` | Function | `(_func, torch_alias, override)` |
| `converters.mil.frontend.torch.ssa_passes.torch_tensor_assign_to_core.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.frontend.torch.ssa_passes.torch_tensor_assign_to_core.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.frontend.torch.ssa_passes.torch_tensor_assign_to_core.mb` | Class | `(...)` |
| `converters.mil.frontend.torch.ssa_passes.torch_tensor_assign_to_core.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.frontend.torch.ssa_passes.torch_tensor_assign_to_core.torch_tensor_assign_to_core` | Class | `(...)` |
| `converters.mil.frontend.torch.ssa_passes.torch_upsample_to_core_upsample.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.frontend.torch.ssa_passes.torch_upsample_to_core_upsample.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.frontend.torch.ssa_passes.torch_upsample_to_core_upsample.logger` | Object | `` |
| `converters.mil.frontend.torch.ssa_passes.torch_upsample_to_core_upsample.mb` | Class | `(...)` |
| `converters.mil.frontend.torch.ssa_passes.torch_upsample_to_core_upsample.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.frontend.torch.ssa_passes.torch_upsample_to_core_upsample.target_ops` | Object | `` |
| `converters.mil.frontend.torch.ssa_passes.torch_upsample_to_core_upsample.torch_upsample_to_core_upsample` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_custom_ops.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.frontend.torch.test.test_custom_ops.InputSpec` | Class | `(kwargs)` |
| `converters.mil.frontend.torch.test.test_custom_ops.Operation` | Class | `(kwargs)` |
| `converters.mil.frontend.torch.test.test_custom_ops.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.frontend.torch.test.test_custom_ops.TestCompositeOp` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_custom_ops.TestCustomOp` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_custom_ops.TorchBaseTest` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_custom_ops.convert_to_mlmodel` | Function | `(model_spec, tensor_inputs, backend, converter_input_type, compute_unit, minimum_deployment_target, converter)` |
| `converters.mil.frontend.torch.test.test_custom_ops.cosine_similarity` | Function | `(context, node)` |
| `converters.mil.frontend.torch.test.test_custom_ops.cosine_similarity_main` | Function | `(context, node)` |
| `converters.mil.frontend.torch.test.test_custom_ops.custom_cosine_similarity` | Object | `` |
| `converters.mil.frontend.torch.test.test_custom_ops.default_cosine_similarity` | Object | `` |
| `converters.mil.frontend.torch.test.test_custom_ops.mb` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_custom_ops.register_op` | Object | `` |
| `converters.mil.frontend.torch.test.test_custom_ops.register_torch_op` | Function | `(_func, torch_alias, override)` |
| `converters.mil.frontend.torch.test.test_custom_ops.types` | Object | `` |
| `converters.mil.frontend.torch.test.test_examples.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.frontend.torch.test.test_examples.TestModelScripting` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_examples.backends` | Object | `` |
| `converters.mil.frontend.torch.test.test_internal_graph.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.frontend.torch.test.test_internal_graph.InternalTorchIRNode` | Class | `(kind: str, inputs: List[str], outputs: List[str], kwinputs: Optional[Dict[str, str]], name: Optional[str], parent: Optional[Union[InternalTorchIRGraph, InternalTorchIRBlock]], attr: Optional[Dict[str, Any]], blocks: Optional[List[InternalTorchIRBlock]], model_hierarchy: Optional[str], meta: Optional[Dict])` |
| `converters.mil.frontend.torch.test.test_internal_graph.TestTorchOps` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_internal_graph.TranscriptionContext` | Class | `(name: Optional[str], frontend: TorchFrontend)` |
| `converters.mil.frontend.torch.test.test_internal_graph.get_new_symbol` | Function | `(name)` |
| `converters.mil.frontend.torch.test.test_internal_graph.mb` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_internal_graph.ops` | Object | `` |
| `converters.mil.frontend.torch.test.test_internal_graph.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.frontend.torch.test.test_internal_graph.torch` | Object | `` |
| `converters.mil.frontend.torch.test.test_internal_graph.types` | Object | `` |
| `converters.mil.frontend.torch.test.test_internal_graph.utils` | Object | `` |
| `converters.mil.frontend.torch.test.test_passes.InternalTorchIRBlock` | Class | `(parent: Optional[InternalTorchIRNode], nodes: Optional[List[InternalTorchIRNode]], inputs: Optional[List[str]], outputs: Optional[List[str]])` |
| `converters.mil.frontend.torch.test.test_passes.InternalTorchIRGraph` | Class | `(params: Dict[str, np.ndarray], inputs: Dict[str, TensorType], outputs: List[str], nodes: Optional[List[InternalTorchIRNode]], buffers: Optional[Dict[str, torch.Tensor]], input_name_to_source_buffer_name: Optional[Dict[str, str]], output_name_to_target_buffer_name: Optional[Dict[str, str]])` |
| `converters.mil.frontend.torch.test.test_passes.InternalTorchIRNode` | Class | `(kind: str, inputs: List[str], outputs: List[str], kwinputs: Optional[Dict[str, str]], name: Optional[str], parent: Optional[Union[InternalTorchIRGraph, InternalTorchIRBlock]], attr: Optional[Dict[str, Any]], blocks: Optional[List[InternalTorchIRBlock]], model_hierarchy: Optional[str], meta: Optional[Dict])` |
| `converters.mil.frontend.torch.test.test_passes.TestTorchPasses` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_passes.flatten_graph_input_values` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.test.test_passes.flatten_graph_output_values` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.test.test_passes.transform_inplace_ops` | Function | `(graph: InternalTorchIRGraph, name_remap_dict: Optional[Dict[str, str]]) -> None` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.MSG_TORCHAO_NOT_FOUND` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TestFxNodeSupport` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TestGrayscaleImagePredictions` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TestInputOutputConversionAPI` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TestPyTorchConverterExamples` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TestQuantizationConversionAPI` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TestTorchInputs` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TestTorchOpsRegistry` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TestTorchScriptValidation` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TestTorchao` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TestUtilsImport` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TestiOS16DefaultIODtype` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.TorchOpsRegistry` | Class | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.any_symbolic` | Function | `(val)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.assert_cast_ops_count` | Function | `(mlmodel, expected_count)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.assert_input_dtype` | Function | `(mlmodel, expected_type_str, expected_name, index)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.assert_ops_in_mil_program` | Function | `(mlmodel, expected_op_list)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.assert_output_dtype` | Function | `(mlmodel, expected_type_str, expected_name, index)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.assert_prog_input_type` | Function | `(prog, expected_dtype_str, expected_name, index)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.assert_prog_output_type` | Function | `(prog, expected_dtype_str, expected_name, index)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.assert_spec_input_image_type` | Function | `(spec, expected_feature_type)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.assert_spec_output_image_type` | Function | `(spec, expected_feature_type)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.backends` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.float32_input_model_add_op` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.float32_input_model_relu_ops` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.float32_two_input_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.float32_two_output_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.float64_input_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.int32_float32_two_output_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.int32_input_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.int64_input_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.linear_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.proto` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.rank3_input_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.rank4_grayscale_input_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.rank4_input_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.register_torch_op` | Function | `(_func, torch_alias, override)` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.torch_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_conversion_api.verify_prediction` | Function | `(mlmodel, multiarray_type)` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.ScopeSource` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.TestExecuTorchExamples` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.TestTorchExportConversionAPI` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.TorchBaseTest` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.TorchFrontend` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.WRAPPED_SCALAR_INPUT_SUFFIX` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.assert_spec_input_image_type` | Function | `(spec, expected_feature_type)` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.backends` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.compute_units` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.export_torch_model_to_frontend` | Function | `(model, input_data, frontend, use_scripting, torch_export_dynamic_shapes)` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.frontends` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.proto` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.testing_reqs` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_export_conversion_api.verify_prediction` | Function | `(mlmodel, multiarray_type)` |
| `converters.mil.frontend.torch.test.test_torch_export_quantization.CoreMLQuantizer` | Class | `(config: _Optional[_LinearQuantizerConfig])` |
| `converters.mil.frontend.torch.test.test_torch_export_quantization.LinearQuantizerConfig` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_export_quantization.QuantizationScheme` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_export_quantization.TestTorchExportQuantization` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_export_quantization.TorchBaseTest` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_export_quantization.TorchFrontend` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_export_quantization.frontends` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_export_quantization.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.frontend.torch.test.test_torch_export_quantization.types` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.COMMON_SHAPES` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.COMMON_SHAPES_ALL` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.ModuleWrapper` | Class | `(function, kwargs)` |
| `converters.mil.frontend.torch.test.test_torch_ops.NUMPY_DTYPE_TO_TORCH_NUM` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.NUM_TO_TORCH_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.Operation` | Class | `(kwargs)` |
| `converters.mil.frontend.torch.test.test_torch_ops.Program` | Class | `()` |
| `converters.mil.frontend.torch.test.test_torch_ops.RangeDim` | Class | `(lower_bound: int, upper_bound: int, default: Optional[int], symbol: Optional[str])` |
| `converters.mil.frontend.torch.test.test_torch_ops.Shape` | Class | `(shape, default)` |
| `converters.mil.frontend.torch.test.test_torch_ops.StripCellAndHidden` | Class | `(flagReturnTuple_)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TORCH_EXPORT_BASED_FRONTENDS` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestAMaxAMin` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestActivation` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestAdaptiveAvgPool` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestAdaptiveMaxPool` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestAddmm` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestAffineGrid` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestArange` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestArgSort` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestArgmax` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestArgmin` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestAtLeastND` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestAtan2` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestAvgPool` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestBaddbmm` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestBatchNorm` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestBitWiseLogical` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestBitwiseAnd` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestBitwiseNot` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestBoolOps` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestBroadcastTensors` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestComplex` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestConcat` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestConv` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestConvTranspose` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestCopy` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestCosineSimilarity` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestCross` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestCumSum` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestCumprod` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestDim` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestDot` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestDuplicateOutputTensors` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestDynamicConv` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestEinsum` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestElementWiseUnary` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestEmbedding` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestEmpty` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestExpand` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestExpandDims` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestEye` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestFft` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestFill` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestFlatten` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestFlip` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestFliplr` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestFold` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestFrac` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestFull` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestGRU` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestGather` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestGlu` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestGridSample` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestGroupNorm` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestHannWindow` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestHardswish` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestHstack` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestImag` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestIndex` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestIndexPut` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestIndexSelect` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestInstanceNorm` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLSTM` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLSTMWithPackedSequence` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLayerNorm` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLinAlgMatrixNorms` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLinAlgNorms` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLinAlgVectorNorms` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLinear` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLinspace` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLog10` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLog2` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLogicalAnd` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLogicalNot` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLogicalOr` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLogicalXor` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestLoss` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestMaskedFill` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestMatMul` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestMaxPool` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestMaximumMinimum` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestMeshgrid` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestMultinomial` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestMv` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestNLLLoss` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestNanToNum` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestNarrow` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestNewFull` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestNewZeros` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestNms` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestNonZero` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestNormalize` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestNorms` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestNumel` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestOneHot` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestOnes` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestOnesLike` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestOuter` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestPad` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestPixelShuffle` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestPixelUnshuffle` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestRNN` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestRand` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestRandLike` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestRandint` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestRandn` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestRandnLike` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestReal` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestReduction` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestRemainder` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestRepeat` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestRepeatInterleave` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestReshape` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestReshapeAs` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestRoll` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestSTFT` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestScalarTensor` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestScaledDotProductAttention` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestScatter` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestScriptedModels` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestSearchsorted` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestSelect` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestSelectScatter` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestSlice` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestSliceScatter` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestSort` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestSpectrogram` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestSplit` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestSqueeze` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestStack` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestStackedBLSTM` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestSum` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTensorAssign` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTensorSize` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTile` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTo` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTopk` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTorchTensor` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTrace` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTransformer` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTranspose` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTransposeCopy` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTril` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTriu` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTupleIndex` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTupleUnpack` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestTypeAs` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestUnbind` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestUnflatten` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestUnfold` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestUnique` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestUpsample` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestVarStd` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestViewAsReal` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestWeightNorm` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestWhere` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestZeros` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TestaLinAlgVectorDot` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TorchBaseTest` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.TorchFrontend` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_ops.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.frontend.torch.test.test_torch_ops.backends` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.compute_units` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.contains_op` | Function | `(torch, op_string)` |
| `converters.mil.frontend.torch.test.test_torch_ops.einsum_equations` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.export_torch_model_to_frontend` | Function | `(model, input_data, frontend, use_scripting, torch_export_dynamic_shapes)` |
| `converters.mil.frontend.torch.test.test_torch_ops.frontends` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.gen_input_shapes_einsum` | Function | `(equation: str, dynamic: bool, backend: Tuple[str, str])` |
| `converters.mil.frontend.torch.test.test_torch_ops.generate_input_data` | Function | `(input_size, rand_range, dtype, torch_device) -> Union[torch.Tensor, List[torch.Tensor]]` |
| `converters.mil.frontend.torch.test.test_torch_ops.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.frontend.torch.test.test_torch_ops.hardcoded_einsum_equations` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.frontend.torch.test.test_torch_ops.testing_reqs` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.torch` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.types` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_ops.version_lt` | Function | `(module, target_version)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.MSG_TORCHAO_NOT_FOUND` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.MSG_TORCH_VISION_NOT_FOUND` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.TestPyTorchQuantizationOps` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.TestPytorchCarryCompressionInfo` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.TestPytorchQuantizedOps` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.TestTorchvisionQuantizedModels` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.TorchBaseTest` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.TorchFrontend` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.TorchQuantizationBaseTest` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.compute_units` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.create_quantize_friendly_weight` | Function | `(weight: np.ndarray, nbits: int, signed: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.create_sparse_weight` | Function | `(weight, target_sparsity)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.create_unique_weight` | Function | `(weight, nbits, vector_size, vector_axis)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.cto` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.frontends` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.get_test_model_and_data` | Function | `(multi_layer: bool, quantize_config: Optional[OpCompressorConfig], use_linear: bool)` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.testing_reqs` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_quantization_ops.types` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.TestStateConversionAPI` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.TorchFrontend` | Class | `(...)` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.any_symbolic` | Function | `(val)` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.assert_output_dtype` | Function | `(mlmodel, expected_type_str, expected_name, index)` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.assert_prog_output_type` | Function | `(prog, expected_dtype_str, expected_name, index)` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.assert_spec_input_image_type` | Function | `(spec, expected_feature_type)` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.assert_spec_output_image_type` | Function | `(spec, expected_feature_type)` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.compute_units` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.export_torch_model_to_frontend` | Function | `(model, input_data, frontend, use_scripting, torch_export_dynamic_shapes)` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.float16_buffer_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.float32_buffer_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.float32_buffer_model_two_inputs_two_states` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.float32_buffer_model_with_two_inputs` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.float32_buffer_not_returned_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.float32_buffer_not_returned_model_2` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.float32_buffer_sequantial_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.float32_non_persistent_buffer_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.float32_two_buffers_model` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.frontends` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.proto` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.rank4_grayscale_input_model_with_buffer` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.rank4_input_model_with_buffer` | Function | `()` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.torch` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.types` | Object | `` |
| `converters.mil.frontend.torch.test.test_torch_stateful_model.verify_prediction` | Function | `(mlmodel, multiarray_type)` |
| `converters.mil.frontend.torch.test.testing_utils.ModuleWrapper` | Class | `(function, kwargs)` |
| `converters.mil.frontend.torch.test.testing_utils.RangeDim` | Class | `(lower_bound: int, upper_bound: int, default: Optional[int], symbol: Optional[str])` |
| `converters.mil.frontend.torch.test.testing_utils.TORCH_DTYPE_TO_MIL_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.test.testing_utils.TORCH_EXPORT_BASED_FRONTENDS` | Object | `` |
| `converters.mil.frontend.torch.test.testing_utils.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.frontend.torch.test.testing_utils.TorchBaseTest` | Class | `(...)` |
| `converters.mil.frontend.torch.test.testing_utils.TorchFrontend` | Class | `(...)` |
| `converters.mil.frontend.torch.test.testing_utils.contains_op` | Function | `(torch, op_string)` |
| `converters.mil.frontend.torch.test.testing_utils.convert_and_compare` | Function | `(input_data, model_spec, expected_results, atol, rtol, backend, converter_input_type, compute_unit, minimum_deployment_target, converter)` |
| `converters.mil.frontend.torch.test.testing_utils.convert_to_coreml_inputs` | Function | `(input_description, inputs)` |
| `converters.mil.frontend.torch.test.testing_utils.convert_to_mlmodel` | Function | `(model_spec, tensor_inputs, backend, converter_input_type, compute_unit, minimum_deployment_target, converter)` |
| `converters.mil.frontend.torch.test.testing_utils.coremltoolsutils` | Object | `` |
| `converters.mil.frontend.torch.test.testing_utils.ct_convert` | Function | `(program, source, inputs, outputs, classifier_config, minimum_deployment_target, convert_to, compute_precision, skip_model_load, converter, kwargs)` |
| `converters.mil.frontend.torch.test.testing_utils.debug_save_mlmodels` | Object | `` |
| `converters.mil.frontend.torch.test.testing_utils.export_torch_model_to_frontend` | Function | `(model, input_data, frontend, use_scripting, torch_export_dynamic_shapes)` |
| `converters.mil.frontend.torch.test.testing_utils.flatten_and_detach_torch_results` | Function | `(torch_results)` |
| `converters.mil.frontend.torch.test.testing_utils.frontend` | Object | `` |
| `converters.mil.frontend.torch.test.testing_utils.frontends` | Object | `` |
| `converters.mil.frontend.torch.test.testing_utils.generate_input_data` | Function | `(input_size, rand_range, dtype, torch_device) -> Union[torch.Tensor, List[torch.Tensor]]` |
| `converters.mil.frontend.torch.test.testing_utils.logger` | Object | `` |
| `converters.mil.frontend.torch.test.testing_utils.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.frontend.torch.test.testing_utils.validate_minimum_deployment_target` | Function | `(minimum_deployment_target: ct.target, backend: Tuple[str, str])` |
| `converters.mil.frontend.torch.torch_op_registry.TorchOpsRegistry` | Class | `()` |
| `converters.mil.frontend.torch.torch_op_registry.is_torch_fx_node_supported` | Function | `(torch_fx_node: torch.fx.Node) -> bool` |
| `converters.mil.frontend.torch.torch_op_registry.logger` | Object | `` |
| `converters.mil.frontend.torch.torch_op_registry.register_torch_op` | Function | `(_func, torch_alias, override)` |
| `converters.mil.frontend.torch.torch_op_registry.sanitize_op_kind` | Function | `(op_kind: str) -> str` |
| `converters.mil.frontend.torch.torch_op_registry.unify_inplace_and_functional` | Function | `(op_kind: str) -> str` |
| `converters.mil.frontend.torch.torch_tensor_assign` | Class | `(...)` |
| `converters.mil.frontend.torch.torch_upsample_bilinear` | Class | `(...)` |
| `converters.mil.frontend.torch.torch_upsample_nearest_neighbor` | Class | `(...)` |
| `converters.mil.frontend.torch.torchir_passes.InternalTorchIRGraph` | Class | `(params: Dict[str, np.ndarray], inputs: Dict[str, TensorType], outputs: List[str], nodes: Optional[List[InternalTorchIRNode]], buffers: Optional[Dict[str, torch.Tensor]], input_name_to_source_buffer_name: Optional[Dict[str, str]], output_name_to_target_buffer_name: Optional[Dict[str, str]])` |
| `converters.mil.frontend.torch.torchir_passes.InternalTorchIRNode` | Class | `(kind: str, inputs: List[str], outputs: List[str], kwinputs: Optional[Dict[str, str]], name: Optional[str], parent: Optional[Union[InternalTorchIRGraph, InternalTorchIRBlock]], attr: Optional[Dict[str, Any]], blocks: Optional[List[InternalTorchIRBlock]], model_hierarchy: Optional[str], meta: Optional[Dict])` |
| `converters.mil.frontend.torch.torchir_passes.flatten_graph_input_values` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.torchir_passes.flatten_graph_output_values` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.torchir_passes.generate_tensor_assignment_ops` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.torchir_passes.logger` | Object | `` |
| `converters.mil.frontend.torch.torchir_passes.populate_native_const_model_hierarchy` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.torchir_passes.remove_getattr_nodes` | Function | `(graph: InternalTorchIRGraph) -> None` |
| `converters.mil.frontend.torch.torchir_passes.transform_inplace_ops` | Function | `(graph: InternalTorchIRGraph, name_remap_dict: Optional[Dict[str, str]]) -> None` |
| `converters.mil.frontend.torch.torchscript_utils.version_lt` | Function | `(module, target_version)` |
| `converters.mil.frontend.torch.utils.MIL_DTYPE_TO_TORCH_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.utils.NUMPY_DTYPE_TO_TORCH_NUM` | Object | `` |
| `converters.mil.frontend.torch.utils.NUM_TO_DTYPE_STRING` | Object | `` |
| `converters.mil.frontend.torch.utils.NUM_TO_NUMPY_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.utils.NUM_TO_TORCH_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.utils.TORCH_DTYPE_TO_MIL_DTYPE` | Object | `` |
| `converters.mil.frontend.torch.utils.TORCH_DTYPE_TO_NUM` | Object | `` |
| `converters.mil.frontend.torch.utils.TORCH_EXPORT_BASED_FRONTENDS` | Object | `` |
| `converters.mil.frontend.torch.utils.TORCH_QTYPE_TO_NP_TYPE` | Object | `` |
| `converters.mil.frontend.torch.utils.TORCH_QTYPE_TO_STR` | Object | `` |
| `converters.mil.frontend.torch.utils.TorchFrontend` | Class | `(...)` |
| `converters.mil.frontend.torch.utils.dtype_to_32bit` | Function | `(dtype)` |
| `converters.mil.frontend.torch.utils.sanitize_op_kind` | Function | `(op_kind: str) -> str` |
| `converters.mil.frontend.torch.utils.types` | Object | `` |
| `converters.mil.frontend.torch.utils.unify_inplace_and_functional` | Function | `(op_kind: str) -> str` |
| `converters.mil.get_existing_symbol` | Function | `(name)` |
| `converters.mil.get_new_symbol` | Function | `(name)` |
| `converters.mil.get_new_variadic_symbol` | Function | `()` |
| `converters.mil.input_types.ClassifierConfig` | Class | `(class_labels, predicted_feature_name, predicted_probabilities_output)` |
| `converters.mil.input_types.ColorLayout` | Class | `(...)` |
| `converters.mil.input_types.EnumeratedShapes` | Class | `(shapes, default)` |
| `converters.mil.input_types.ImageType` | Class | `(name, shape, scale, bias, color_layout, channel_first, grayscale_use_uint8)` |
| `converters.mil.input_types.InputType` | Class | `(name, shape, dtype)` |
| `converters.mil.input_types.RangeDim` | Class | `(lower_bound: int, upper_bound: int, default: Optional[int], symbol: Optional[str])` |
| `converters.mil.input_types.Shape` | Class | `(shape, default)` |
| `converters.mil.input_types.StateType` | Class | `(wrapped_type: type, name: Optional[str])` |
| `converters.mil.input_types.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.input_types.is_symbolic` | Function | `(val)` |
| `converters.mil.input_types.types` | Object | `` |
| `converters.mil.mil.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.Builder` | Class | `(...)` |
| `converters.mil.mil.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.InternalVar` | Class | `(val, name)` |
| `converters.mil.mil.ListInputType` | Class | `(...)` |
| `converters.mil.mil.ListVar` | Class | `(name, elem_type, init_length, dynamic_length, sym_val, kwargs)` |
| `converters.mil.mil.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.Placeholder` | Class | `(sym_shape, dtype, name, allow_rank0_input)` |
| `converters.mil.mil.Program` | Class | `()` |
| `converters.mil.mil.PyFunctionInputType` | Class | `(...)` |
| `converters.mil.mil.SPACES` | Object | `` |
| `converters.mil.mil.Symbol` | Class | `(sym_name)` |
| `converters.mil.mil.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.TupleInputType` | Class | `(...)` |
| `converters.mil.mil.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.block.BLOCK_STACK` | Object | `` |
| `converters.mil.mil.block.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.block.CacheDoublyLinkedList` | Class | `(array: Optional[List[Operation]])` |
| `converters.mil.mil.block.ComplexVar` | Class | `(name, sym_type, sym_val, op, op_output_idx, real: Optional[Var], imag: Optional[Var])` |
| `converters.mil.mil.block.DEBUG` | Object | `` |
| `converters.mil.mil.block.DotVisitor` | Class | `(annotation)` |
| `converters.mil.mil.block.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.block.InternalVar` | Class | `(val, name)` |
| `converters.mil.mil.block.InvalidBlockStateError` | Class | `(...)` |
| `converters.mil.mil.block.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.block.SCOPE_STACK` | Object | `` |
| `converters.mil.mil.block.SPACES` | Object | `` |
| `converters.mil.mil.block.ScopeSource` | Class | `(...)` |
| `converters.mil.mil.block.VALID_OPS_TO_COPY_SCOPE_INFO` | Object | `` |
| `converters.mil.mil.block.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.block.add_graph_pass_scope` | Function | `(src_scopes: Dict[ScopeSource, List[str]], graph_pass_scopes: Dict[ScopeSource, List[str]]) -> Dict[ScopeSource, List[str]]` |
| `converters.mil.mil.block.curr_block` | Function | `()` |
| `converters.mil.mil.block.curr_opset_version` | Function | `()` |
| `converters.mil.mil.block.is_current_opset_version_compatible_with` | Function | `(opset_version)` |
| `converters.mil.mil.block.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.block.k_used_symbols` | Object | `` |
| `converters.mil.mil.block.logger` | Object | `` |
| `converters.mil.mil.block.types` | Object | `` |
| `converters.mil.mil.builder.AvailableTarget` | Class | `(...)` |
| `converters.mil.mil.builder.BeforeOpContextManager` | Class | `(before_op: mil.Operation)` |
| `converters.mil.mil.builder.Builder` | Class | `(...)` |
| `converters.mil.mil.builder.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.builder.InternalInputType` | Class | `(...)` |
| `converters.mil.mil.builder.InternalVar` | Class | `(val, name)` |
| `converters.mil.mil.builder.ListOrTensorOrDictInputType` | Class | `(...)` |
| `converters.mil.mil.builder.Placeholder` | Class | `(sym_shape, dtype, name, allow_rank0_input)` |
| `converters.mil.mil.builder.SCOPE_STACK` | Object | `` |
| `converters.mil.mil.builder.ScopeContextManager` | Class | `(scopes: List[ScopeInfo])` |
| `converters.mil.mil.builder.ScopeInfo` | Class | `(...)` |
| `converters.mil.mil.builder.ScopeSource` | Class | `(...)` |
| `converters.mil.mil.builder.StateTensorPlaceholder` | Class | `(sym_shape, dtype, name)` |
| `converters.mil.mil.builder.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.builder.TupleInputType` | Class | `(...)` |
| `converters.mil.mil.builder.VALID_OPS_TO_COPY_SCOPE_INFO` | Object | `` |
| `converters.mil.mil.builder.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.builder.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.builder.curr_block` | Function | `()` |
| `converters.mil.mil.builder.is_python_value` | Function | `(val)` |
| `converters.mil.mil.builder.logger` | Object | `` |
| `converters.mil.mil.builder.mil` | Object | `` |
| `converters.mil.mil.curr_block` | Function | `()` |
| `converters.mil.mil.get_existing_symbol` | Function | `(name)` |
| `converters.mil.mil.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.get_new_variadic_symbol` | Function | `()` |
| `converters.mil.mil.input_type.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.input_type.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.input_type.InternalInputType` | Class | `(...)` |
| `converters.mil.mil.input_type.InternalVar` | Class | `(val, name)` |
| `converters.mil.mil.input_type.ListInputType` | Class | `(...)` |
| `converters.mil.mil.input_type.ListOrTensorOrDictInputType` | Class | `(...)` |
| `converters.mil.mil.input_type.PyFunctionInputType` | Class | `(...)` |
| `converters.mil.mil.input_type.StateInputType` | Class | `(...)` |
| `converters.mil.mil.input_type.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.input_type.TupleInputType` | Class | `(...)` |
| `converters.mil.mil.input_type.get_type_info` | Function | `(t)` |
| `converters.mil.mil.input_type.types` | Object | `` |
| `converters.mil.mil.mil_list` | Class | `(ls)` |
| `converters.mil.mil.operation.ALL` | Object | `` |
| `converters.mil.mil.operation.ComplexVar` | Class | `(name, sym_type, sym_val, op, op_output_idx, real: Optional[Var], imag: Optional[Var])` |
| `converters.mil.mil.operation.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.operation.InternalVar` | Class | `(val, name)` |
| `converters.mil.mil.operation.ListVar` | Class | `(name, elem_type, init_length, dynamic_length, sym_val, kwargs)` |
| `converters.mil.mil.operation.NONE` | Object | `` |
| `converters.mil.mil.operation.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.operation.SPACES` | Object | `` |
| `converters.mil.mil.operation.SYMBOL` | Object | `` |
| `converters.mil.mil.operation.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.operation.TupleInputType` | Class | `(...)` |
| `converters.mil.mil.operation.VALUE` | Object | `` |
| `converters.mil.mil.operation.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.operation.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.operation.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.mil.operation.is_internal_input` | Function | `(arg_name)` |
| `converters.mil.mil.operation.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.operation.mil_list` | Class | `(ls)` |
| `converters.mil.mil.operation.precondition` | Function | `(allow)` |
| `converters.mil.mil.operation.types` | Object | `` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.ComplexVar` | Class | `(name, sym_type, sym_val, op, op_output_idx, real: Optional[Var], imag: Optional[Var])` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.SSAOpRegistry` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_abs` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_fft` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_fftn` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_ifft` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_ifftn` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_imag` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_irfft` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_irfftn` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_real` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_rfft` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_rfftn` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_shape` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.complex_stft` | Class | `(...)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.fft_canonicalize_length_dim` | Function | `(input_data: Var, length: Optional[Var], dim: Optional[Var], c2r: bool) -> Tuple[int, int]` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.fft_canonicalize_shapes_dims` | Function | `(input_data: Var, shapes: Optional[Var], dims: Optional[Var], c2r: bool) -> Tuple[Tuple[int], Tuple[int]]` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.infer_complex_dtype` | Function | `(real_dtype, imag_dtype)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.infer_fp_dtype_from_complex` | Function | `(complex_dtype)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.operation` | Object | `` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.complex_dialect_ops.types` | Object | `` |
| `converters.mil.mil.ops.defs.coreml_dialect.coreml_update_state` | Class | `(...)` |
| `converters.mil.mil.ops.defs.coreml_dialect.ops.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.coreml_dialect.ops.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.coreml_dialect.ops.StateInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.coreml_dialect.ops.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.coreml_dialect.ops.coreml_update_state` | Class | `(...)` |
| `converters.mil.mil.ops.defs.coreml_dialect.ops.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.coreml_dialect.ops.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.abs` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.acos` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.activation.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.activation.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.activation.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.activation.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.activation.activation_with_alpha` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.activation_with_alpha_and_beta` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.clamped_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.elementwise_unary` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.elu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.gelu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.leaky_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.linear_activation` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS15.activation.prelu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.activation.relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.relu6` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.scaled_tanh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.sigmoid` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.sigmoid_hard` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.silu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.softmax` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.softplus` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.softplus_parametric` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.softsign` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.thresholded_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.activation.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.add` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.affine` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.argsort` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.asin` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.atan` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.atanh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.avg_pool` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.band_part` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.batch_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.cast` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.ceil` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.clamped_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.classify.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.classify.ListInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.classify.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.classify.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.classify.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.classify.classify` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.classify.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.classify.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.clip` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.concat` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.cond` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.const` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.Const` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.InternalInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.ListInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.NONE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.PyFunctionInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.SYMBOL` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.TupleInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.cond` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.const` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.get_existing_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.infer_type_with_broadcast` | Function | `(typea, typeb, primitive_type)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.is_compatible_type` | Function | `(type1, type2)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.is_subtype` | Function | `(type1, type2)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.list_gather` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.list_length` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.list_read` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.list_scatter` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.list_write` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.logger` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.make_list` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.mil_list` | Class | `(ls)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.numpy_val_to_builtin_val` | Function | `(npval)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.promoted_primitive_type` | Function | `(type1, type2)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.select` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.types_list` | Function | `(arg, init_length, dynamic_length)` |
| `converters.mil.mil.ops.defs.iOS15.control_flow.while_loop` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.conv.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.conv.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.conv.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.conv.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.conv.conv` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.conv.conv_quantized` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.conv.conv_transpose` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.conv.curr_opset_version` | Function | `()` |
| `converters.mil.mil.ops.defs.iOS15.conv.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.conv.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.conv.spatial_dimensions_out_shape` | Function | `(pad_type, input_shape, kernel_shape, strides, dilations, custom_pad, ceil_mode)` |
| `converters.mil.mil.ops.defs.iOS15.conv.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.conv_quantized` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.conv_transpose` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.cos` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.cosh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.crop` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.crop_resize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.cumsum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.depth_to_space` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.einsum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.add` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.elementwise_binary` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.elementwise_binary_logical` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.equal` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.floor_div` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.greater` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.greater_equal` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.infer_type_with_broadcast` | Function | `(typea, typeb, primitive_type)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.less` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.less_equal` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.logical_and` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.logical_or` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.logical_xor` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.maximum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.minimum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.mod` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.mul` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.not_equal` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.pow` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.promoted_primitive_type` | Function | `(type1, type2)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.real_div` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.sub` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_binary.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.SYMBOL` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.abs` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.acos` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.asin` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.atan` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.atanh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.cast` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.ceil` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.clip` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.cos` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.cosh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.elementwise_unary` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.elementwise_unary_with_int` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.erf` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.exp` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.exp2` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.floor` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.inverse` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.log` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.logical_not` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.round` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.rsqrt` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.sign` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.sin` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.sinh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.sqrt` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.square` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.string_to_builtin` | Function | `(s)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.string_to_nptype` | Function | `(s: str)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.tan` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.tanh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.threshold` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.elementwise_unary.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.elu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.equal` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.erf` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.exp` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.exp2` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.expand_dims` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.fill` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.flatten2d` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.floor` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.floor_div` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.gather` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.gather_along_axis` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.gather_nd` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.gelu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.greater` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.greater_equal` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.gru` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.identity` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.affine` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.crop` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.crop_resize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.resample` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.resize_bilinear` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.resize_nearest_neighbor` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.upsample_bilinear` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.image_resizing.upsample_nearest_neighbor` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.instance_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.inverse` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.l2_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.l2_pool` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.layer_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.leaky_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.less` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.less_equal` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.linear.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.linear.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.linear.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.linear.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.linear.TupleInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.linear.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.linear.broadcast_shapes` | Function | `(shape_x, shape_y)` |
| `converters.mil.mil.ops.defs.iOS15.linear.einsum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.linear.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.linear.linear` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.linear.matmul` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.linear.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.mil.ops.defs.iOS15.linear.parse_einsum_equation` | Function | `(equation: str) -> Tuple[List[str]]` |
| `converters.mil.mil.ops.defs.iOS15.linear.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS15.linear.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.linear.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.linear_activation` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.list_gather` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.list_length` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.list_read` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.list_scatter` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.list_write` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.local_response_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.log` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.logical_and` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.logical_not` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.logical_or` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.logical_xor` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.lstm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.make_list` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.matmul` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.max_pool` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.maximum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.minimum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.mod` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.mul` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.non_maximum_suppression` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.non_zero` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.normalization.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.batch_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.instance_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.l2_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.layer_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.local_response_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS15.normalization.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.normalization.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.not_equal` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.one_hot` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.pad` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.pixel_shuffle` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.pool.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.pool.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.pool.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.pool.Pooling` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.pool.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.pool.avg_pool` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.pool.curr_opset_version` | Function | `()` |
| `converters.mil.mil.ops.defs.iOS15.pool.l2_pool` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.pool.max_pool` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.pool.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.pool.spatial_dimensions_out_shape` | Function | `(pad_type, input_shape, kernel_shape, strides, dilations, custom_pad, ceil_mode)` |
| `converters.mil.mil.ops.defs.iOS15.pool.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.pow` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.prelu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.random.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.random.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.random.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.random.RandomDistribution` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.random.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.random.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.random.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.defs.iOS15.random.get_new_variadic_symbol` | Function | `()` |
| `converters.mil.mil.ops.defs.iOS15.random.random_bernoulli` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.random.random_categorical` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.random.random_normal` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.random.random_uniform` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.random.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.random.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.random_bernoulli` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.random_categorical` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.random_normal` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.random_uniform` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.range_1d` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.real_div` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.recurrent.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.recurrent.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.recurrent.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.recurrent.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.recurrent.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.ops.defs.iOS15.recurrent.gru` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.recurrent.lstm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.recurrent.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.recurrent.rnn` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.recurrent.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.reduce_argmax` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduce_argmin` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduce_l1_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduce_l2_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduce_log_sum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduce_log_sum_exp` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduce_max` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.reduce_mean` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduce_min` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduce_prod` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduce_sum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduce_sum_square` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.ReductionAxes` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.ReductionAxis` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.reduction.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_arg` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_argmax` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_argmin` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_l1_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_l2_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_log_sum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_log_sum_exp` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_max` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_mean` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_min` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_prod` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_sum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.reduce_sum_square` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reduction.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.reduction.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.relu6` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.resample` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reshape` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.resize_bilinear` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.resize_nearest_neighbor` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reverse` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.reverse_sequence` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.rnn` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.round` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.rsqrt` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.scaled_tanh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.scatter` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_along_axis` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.SYMBOL` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.compute_gather` | Function | `(params, indices, axis, batch_dims)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.gather` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.gather_along_axis` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.gather_nd` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.is_compatible_symbolic_vector` | Function | `(val_a, val_b)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.scatter` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.scatter_along_axis` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.scatter_nd` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.scatter_gather.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.scatter_nd` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.select` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.shape` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.sigmoid` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.sigmoid_hard` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.sign` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.silu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.sin` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.sinh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.slice_by_index` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.slice_by_size` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.sliding_windows` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.softmax` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.softplus` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.softplus_parametric` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.softsign` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.space_to_batch` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.space_to_depth` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.split` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.sqrt` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.square` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.squeeze` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.stack` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.sub` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tan` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tanh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.target` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.ListOrTensorOrDictInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.MAX_SIZE_CONSTANT_FOLDING` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.NONE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.SYMBOL` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.TupleInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.argsort` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.band_part` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.concat` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.cumsum` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.fill` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.flatten2d` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.get_new_variadic_symbol` | Function | `()` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.identity` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.is_compatible_symbolic_vector` | Function | `(val_a, val_b)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.non_maximum_suppression` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.non_zero` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.one_hot` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.pad` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.range_1d` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.shape` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.split` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.stack` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.tile` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.topk` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_operation.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.SYMBOL` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.any_variadic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.batch_to_space` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.depth_to_space` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.expand_dims` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.get_new_variadic_symbol` | Function | `()` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.get_param_val` | Function | `(param)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.get_squeeze_axes` | Function | `(squeeze_mask, rank)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.isscalar` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.logger` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.pixel_shuffle` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.reshape` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.reshape_with_symbol` | Function | `(v, shape)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.reverse` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.reverse_sequence` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.slice_by_index` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.slice_by_size` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.sliding_windows` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.solve_slice_by_index_shape` | Function | `(x_shape, begin, end, stride, begin_mask, end_mask, squeeze_mask)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.solve_slice_by_index_slice` | Function | `(x_shape, begin, end, stride, begin_mask, end_mask, squeeze_mask)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.space_to_batch` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.space_to_depth` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.squeeze` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.transpose` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tensor_transformation.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS15.threshold` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.thresholded_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.tile` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.topk` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.transpose` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.upsample_bilinear` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.upsample_nearest_neighbor` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS15.while_loop` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_affine_dequantize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_cast` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_lut_to_dense` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_ops.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_ops.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_ops.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_ops.constexpr_affine_dequantize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_ops.constexpr_cast` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_ops.constexpr_lut_to_dense` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_ops.constexpr_sparse_to_dense` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_ops.optimize_utils` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_ops.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_ops.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.constexpr_sparse_to_dense` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.crop_resize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.fill_like` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.gather` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.gather_nd` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.image_resizing.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.image_resizing.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.image_resizing.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.image_resizing.crop_resize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.image_resizing.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.image_resizing.resample` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.image_resizing.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.image_resizing.upsample_bilinear` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.pixel_unshuffle` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.resample` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.reshape_like` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.SYMBOL` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.compute_gather` | Function | `(params, indices, axis, batch_dims)` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.gather` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.gather_along_axis` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.gather_nd` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.scatter_gather.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.target` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_operation.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_operation.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_operation.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_operation.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_operation.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.tensor_operation.fill_like` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_operation.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_operation.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.tensor_operation.topk` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_operation.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.tensor_transformation.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_transformation.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_transformation.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_transformation.TupleInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_transformation.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_transformation.pixel_unshuffle` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_transformation.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.tensor_transformation.reshape_like` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.tensor_transformation.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS16.topk` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS16.upsample_bilinear` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.activation.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.activation.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.activation.clamped_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.activation.elu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.activation.leaky_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.activation.linear_activation` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.activation.prelu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.activation.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.activation.scaled_tanh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.activation.sigmoid_hard` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.activation.softplus_parametric` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.activation.thresholded_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.activation.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.batch_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.cast` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.clamped_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.clip` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.conv.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.conv.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.conv.conv` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.conv.conv_transpose` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.conv.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.conv.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.conv_transpose` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.crop_resize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.dequantize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.elementwise_unary.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.elementwise_unary.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.elementwise_unary.cast` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.elementwise_unary.clip` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.elementwise_unary.inverse` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.elementwise_unary.log` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.elementwise_unary.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.elementwise_unary.rsqrt` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.elementwise_unary.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.elu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.expand_dims` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.gather` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.gather_along_axis` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.gather_nd` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.gru` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.image_resizing.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.image_resizing.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.image_resizing.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.image_resizing.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.image_resizing.crop_resize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.image_resizing.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.defs.iOS17.image_resizing.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.image_resizing.resample` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.image_resizing.resize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.image_resizing.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.instance_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.inverse` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.l2_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.layer_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.leaky_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.linear.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.linear.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.linear.linear` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.linear.matmul` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.linear.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.linear.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.linear_activation` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.local_response_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.log` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.lstm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.matmul` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.non_maximum_suppression` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.normalization.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.normalization.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.normalization.batch_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.normalization.instance_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.normalization.l2_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.normalization.layer_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.normalization.local_response_norm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.normalization.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.normalization.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.prelu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.quantization_ops.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.quantization_ops.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.quantization_ops.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.quantization_ops.VALUE` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.quantization_ops.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.ops.defs.iOS17.quantization_ops.dequantize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.quantization_ops.optimize_utils` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.quantization_ops.precondition` | Function | `(allow)` |
| `converters.mil.mil.ops.defs.iOS17.quantization_ops.quantize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.quantization_ops.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.quantization_ops.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.quantize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.recurrent.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.recurrent.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.recurrent.gru` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.recurrent.lstm` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.recurrent.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.recurrent.rnn` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.recurrent.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.reduce_argmax` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.reduce_argmin` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.reduction.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.reduction.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.reduction.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.reduction.reduce_arg` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.reduction.reduce_argmax` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.reduction.reduce_argmin` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.reduction.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.reduction.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.resample` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.reshape` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.reshape_like` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.resize` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.reverse` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.reverse_sequence` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.rnn` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.rsqrt` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.scaled_tanh` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.scatter` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.scatter_along_axis` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.scatter_gather.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.scatter_gather.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.scatter_gather.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.scatter_gather.gather` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.scatter_gather.gather_along_axis` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.scatter_gather.gather_nd` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.scatter_gather.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.scatter_gather.scatter` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.scatter_gather.scatter_along_axis` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.scatter_gather.scatter_nd` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.scatter_gather.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.scatter_nd` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.sigmoid_hard` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.sliding_windows` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.softplus_parametric` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.squeeze` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.target` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_operation.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_operation.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_operation.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_operation.non_maximum_suppression` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_operation.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.tensor_operation.topk` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_operation.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.TupleInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.expand_dims` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.reshape` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.reshape_like` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.reverse` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.reverse_sequence` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.slice_by_index` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.slice_by_size` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.sliding_windows` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.squeeze` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.transpose` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.tensor_transformation.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS17.thresholded_relu` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.topk` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS17.transpose` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.SSAOpRegistry` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.compression.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.compression.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.compression.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.compression.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.ops.defs.iOS18.compression.constexpr_blockwise_shift_scale` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.compression.constexpr_cast` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.compression.constexpr_lut_to_dense` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.compression.constexpr_lut_to_sparse` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.compression.constexpr_sparse_blockwise_shift_scale` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.compression.constexpr_sparse_to_dense` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.compression.optimize_utils` | Object | `` |
| `converters.mil.mil.ops.defs.iOS18.compression.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS18.compression.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS18.constexpr_blockwise_shift_scale` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.constexpr_lut_to_dense` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.constexpr_lut_to_sparse` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.constexpr_sparse_blockwise_shift_scale` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.constexpr_sparse_to_dense` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.gru` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.read_state` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.recurrent.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.recurrent.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.recurrent.gru` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.recurrent.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS18.recurrent.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS18.scaled_dot_product_attention` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.slice_update` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.states.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.states.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.states.StateInputType` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.states.read_state` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.states.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS18.states.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS18.target` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.tensor_transformation.DefaultInputs` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.tensor_transformation.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.tensor_transformation.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.tensor_transformation.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.tensor_transformation.get_param_val` | Function | `(param)` |
| `converters.mil.mil.ops.defs.iOS18.tensor_transformation.is_compatible_symbolic_vector` | Function | `(val_a, val_b)` |
| `converters.mil.mil.ops.defs.iOS18.tensor_transformation.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS18.tensor_transformation.slice_update` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.tensor_transformation.solve_slice_by_index_shape` | Function | `(x_shape, begin, end, stride, begin_mask, end_mask, squeeze_mask)` |
| `converters.mil.mil.ops.defs.iOS18.tensor_transformation.solve_slice_by_index_slice` | Function | `(x_shape, begin, end, stride, begin_mask, end_mask, squeeze_mask)` |
| `converters.mil.mil.ops.defs.iOS18.tensor_transformation.types` | Object | `` |
| `converters.mil.mil.ops.defs.iOS18.transformers.InputSpec` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.transformers.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.transformers.TensorInputType` | Class | `(type_domain, kwargs)` |
| `converters.mil.mil.ops.defs.iOS18.transformers.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS18.transformers.broadcast_shapes` | Function | `(shape_x, shape_y)` |
| `converters.mil.mil.ops.defs.iOS18.transformers.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.defs.iOS18.transformers.register_op` | Object | `` |
| `converters.mil.mil.ops.defs.iOS18.transformers.scaled_dot_product_attention` | Class | `(...)` |
| `converters.mil.mil.ops.defs.iOS18.transformers.types` | Object | `` |
| `converters.mil.mil.ops.helper.AvailableTarget` | Class | `(...)` |
| `converters.mil.mil.ops.helper.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.ops.registry.Builder` | Class | `(...)` |
| `converters.mil.mil.ops.registry.SSAOpRegistry` | Class | `(...)` |
| `converters.mil.mil.ops.registry.curr_opset_version` | Function | `()` |
| `converters.mil.mil.ops.registry.logger` | Object | `` |
| `converters.mil.mil.ops.registry.target` | Class | `(...)` |
| `converters.mil.mil.ops.tests.coreml_dialect.test_coreml_dialect.TestCoreMLUpdateState` | Class | `(...)` |
| `converters.mil.mil.ops.tests.coreml_dialect.test_coreml_dialect.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.ops.tests.coreml_dialect.test_coreml_dialect.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.coreml_dialect.test_coreml_dialect.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.backends_internal` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.clean_up_backends` | Function | `(backends: List[BackendConfig], minimum_opset_version: ct.target, force_include_iOS15_test: bool) -> List[BackendConfig]` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestClampedReLU` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestELU` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestGeLU` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestInputWeightDifferentDtypesErrorOut` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestLeakyReLU` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestLinearActivation` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestPReLU` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestReLU` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestReLU6` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestScaledTanh` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestSiLU` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestSigmoid` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestSigmoidHard` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestSoftmax` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestSoftplus` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestSoftplusParametric` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestSoftsign` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.TestThresholdedReLU` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.mark_api_breaking` | Function | `(breaking_opset_version: ct.target)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.ssa_fn` | Function | `(func)` |
| `converters.mil.mil.ops.tests.iOS14.test_activation.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.TestCond` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.TestConst` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.TestList` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.TestSelect` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.TestWhileLoop` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.UNK_SYM` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.construct_inputs_from_placeholders` | Function | `(input_placeholders: Dict[str, Placeholder], upper_bound: int) -> [List[TensorType]]` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.ssa_fn` | Function | `(func)` |
| `converters.mil.mil.ops.tests.iOS14.test_control_flow.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.TestConv` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.TestConvTranspose` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.TestInvalidConvConfig` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_conv.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.TestElementwiseBinary` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.TestEqual` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.TestGreater` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.TestGreaterEqual` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.TestLess` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.TestLessEqual` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.TestNotEqual` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.ssa_fn` | Function | `(func)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_binary.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_unary.TestElementwiseUnary` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_unary.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_unary.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_unary.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_unary.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_unary.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_unary.ssa_fn` | Function | `(func)` |
| `converters.mil.mil.ops.tests.iOS14.test_elementwise_unary.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.TestCrop` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.TestCropResize` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.TestResizeBilinear` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.TestResizeNearestNeighbor` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.TestUpsampleBilinear` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.TestUpsampleNearestNeighbor` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.mark_api_breaking` | Function | `(breaking_opset_version: ct.target)` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_image_resizing.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.TestEinsum` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.TestLinear` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.TestMatMul` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.ssa_fn` | Function | `(func)` |
| `converters.mil.mil.ops.tests.iOS14.test_linear.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.MSG_TF2_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.TestNormalizationBatchNorm` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.TestNormalizationInstanceNorm` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.TestNormalizationL2Norm` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.TestNormalizationLayerNorm` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.TestNormalizationLocalResponseNorm` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.UNK_SYM` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.construct_inputs_from_placeholders` | Function | `(input_placeholders: Dict[str, Placeholder], upper_bound: int) -> [List[TensorType]]` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_normalization.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_pool.TestAvgPool` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_pool.TestL2Pool` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_pool.TestMaxPool` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_pool.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_pool.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_pool.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_pool.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_pool.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_pool.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_random.TestRandomBernoulli` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_random.TestRandomCategorical` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_random.TestRandomNormal` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_random.TestRandomUniform` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_random.UNK_SYM` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_random.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_random.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_random.get_core_ml_prediction` | Function | `(build, input_placeholders, input_values, backend, compute_unit)` |
| `converters.mil.mil.ops.tests.iOS14.test_random.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_random.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_random.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.TestGRU` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.TestLSTM` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.TestRNN` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.construct_inputs_from_placeholders` | Function | `(input_placeholders: Dict[str, Placeholder], upper_bound: int) -> [List[TensorType]]` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.new_backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_recurrent.ssa_fn` | Function | `(func)` |
| `converters.mil.mil.ops.tests.iOS14.test_reduction.TestReduction` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_reduction.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_reduction.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_reduction.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_reduction.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_reduction.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.mil.ops.tests.iOS14.test_reduction.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_reduction.ssa_fn` | Function | `(func)` |
| `converters.mil.mil.ops.tests.iOS14.test_reduction.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.MSG_TF2_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.TestGather` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.TestGatherAlongAxis` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.TestGatherNd` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.TestScatter` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.TestScatterAlongAxis` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.TestScatterNd` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.mark_api_breaking` | Function | `(breaking_opset_version: ct.target)` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_scatter_gather.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.MSG_TF2_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestArgSort` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestBandPart` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestConcat` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestCumSum` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestDynamicTile` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestFill` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestFlatten2d` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestIdentity` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestNonMaximumSuppression` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestNonZero` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestOneHot` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestPad` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestRange1d` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestShape` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestTile` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.TestTopK` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.UNK_SYM` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.UNK_VARIADIC` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.construct_inputs_from_placeholders` | Function | `(input_placeholders: Dict[str, Placeholder], upper_bound: int) -> [List[TensorType]]` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.mark_api_breaking` | Function | `(breaking_opset_version: ct.target)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.ssa_fn` | Function | `(func)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_operation.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestBatchToSpace` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestConcat` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestDepthToSpace` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestExpandDims` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestPixelShuffle` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestReshape` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestReverse` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestReverseSequence` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestSliceByIndex` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestSliceBySize` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestSlidingWindows` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestSpaceToBatch` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestSpaceToDepth` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestSplit` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestSqueeze` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestStack` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.TestTranspose` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.UNK_SYM` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.UNK_VARIADIC` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.construct_inputs_from_placeholders` | Function | `(input_placeholders: Dict[str, Placeholder], upper_bound: int) -> [List[TensorType]]` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.ssa_fn` | Function | `(func)` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.testing_reqs` | Object | `` |
| `converters.mil.mil.ops.tests.iOS14.test_tensor_transformation.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS15.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS15.backends_internal` | Object | `` |
| `converters.mil.mil.ops.tests.iOS15.clean_up_backends` | Function | `(backends: List[BackendConfig], minimum_opset_version: ct.target, force_include_iOS15_test: bool) -> List[BackendConfig]` |
| `converters.mil.mil.ops.tests.iOS15.test_elementwise_unary.MockInputVar` | Class | `(val, sym_type)` |
| `converters.mil.mil.ops.tests.iOS15.test_elementwise_unary.TestCast` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS15.test_elementwise_unary.elementwise_unary` | Object | `` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.TestAffine` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.TestCropResize` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.TestResample` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.TestResizeBilinear` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.TestUpsampleNearestNeighborFractionalScales` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.mark_api_breaking` | Function | `(breaking_opset_version: ct.target)` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS15.test_image_resizing.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS15.test_tensor_transformation.TestReshape` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS15.test_tensor_transformation.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS15.test_tensor_transformation.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS15.test_tensor_transformation.mark_api_breaking` | Function | `(breaking_opset_version: ct.target)` |
| `converters.mil.mil.ops.tests.iOS15.test_tensor_transformation.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS15.test_tensor_transformation.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS15.test_tensor_transformation.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.backends_internal` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.clean_up_backends` | Function | `(backends: List[BackendConfig], minimum_opset_version: ct.target, force_include_iOS15_test: bool) -> List[BackendConfig]` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.TestConstexprAffineDequantize` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.TestConstexprCast` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.TestConstexprLutToDense` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.TestConstexprSparseToDense` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.constexpr_ops` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.mark_api_breaking` | Function | `(breaking_opset_version: ct.target)` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.ssa_fn` | Function | `(func)` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.testing_reqs` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_constexpr_ops.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_conv.TestConvolution` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_conv.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_conv.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.ops.tests.iOS16.test_conv.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_conv.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_image_resizing.TestCropResize` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_image_resizing.TestResample` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_image_resizing.TestUpsampleBilinear` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_image_resizing.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_image_resizing.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_image_resizing.mark_api_breaking` | Function | `(breaking_opset_version: ct.target)` |
| `converters.mil.mil.ops.tests.iOS16.test_image_resizing.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_image_resizing.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS16.test_image_resizing.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_scatter_gather.TestGather` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_scatter_gather.TestGatherAlongAxis` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_scatter_gather.TestGatherNd` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_scatter_gather.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_scatter_gather.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_scatter_gather.mark_api_breaking` | Function | `(breaking_opset_version: ct.target)` |
| `converters.mil.mil.ops.tests.iOS16.test_scatter_gather.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_scatter_gather.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS16.test_scatter_gather.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_operation.TestFillLike` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_operation.TestTopK` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_operation.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_operation.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_operation.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_operation.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_operation.testing_reqs` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_operation.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_transformation.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_transformation.TestPixelUnshuffle` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_transformation.TestReshapeLike` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_transformation.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_transformation.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_transformation.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_transformation.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_transformation.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_transformation.testing_reqs` | Object | `` |
| `converters.mil.mil.ops.tests.iOS16.test_tensor_transformation.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.backends_internal` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.clean_up_backends` | Function | `(backends: List[BackendConfig], minimum_opset_version: ct.target, force_include_iOS15_test: bool) -> List[BackendConfig]` |
| `converters.mil.mil.ops.tests.iOS17.test_activation.TestInputWeightDifferentDtypes` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_activation.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_activation.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_activation.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_activation.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS17.test_activation.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_conv.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_conv.TestConv` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_conv.TestConvTranspose` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_conv.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_conv.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_elementwise_unary.PassPipeline` | Class | `(pass_names, pipeline_name)` |
| `converters.mil.mil.ops.tests.iOS17.test_elementwise_unary.TestElementwiseUnary` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_elementwise_unary.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.ops.tests.iOS17.test_elementwise_unary.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_elementwise_unary.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_elementwise_unary.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS17.test_elementwise_unary.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.ops.tests.iOS17.test_elementwise_unary.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_elementwise_unary.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.ops.tests.iOS17.test_elementwise_unary.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS17.test_elementwise_unary.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_image_resizing.TestCropResize` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_image_resizing.TestResample` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_image_resizing.TestResize` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_image_resizing.UNK_SYM` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_image_resizing.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_image_resizing.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_image_resizing.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_image_resizing.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS17.test_image_resizing.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_linear.TestLinear` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_linear.TestMatMul` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_linear.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_linear.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.mil.ops.tests.iOS17.test_linear.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_linear.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_linear.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.mil.ops.tests.iOS17.test_linear.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.ops.tests.iOS17.test_linear.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS17.test_linear.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_normalization.MSG_TF2_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_normalization.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_normalization.TestNormalizationBatchNorm` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_normalization.TestNormalizationInstanceNorm` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_normalization.TestNormalizationL2Norm` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_normalization.TestNormalizationLayerNorm` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_normalization.TestNormalizationLocalResponseNorm` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_normalization.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_normalization.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.BackendConfig` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.TestDequantize` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.TestQuantizationBase` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.TestQuantize` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS17.test_quantization.ssa_fn` | Function | `(func)` |
| `converters.mil.mil.ops.tests.iOS17.test_recurrent.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_recurrent.TestGRU` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_recurrent.TestLSTM` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_recurrent.TestRNN` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_recurrent.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_recurrent.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_reduction.TestReduction` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_reduction.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_reduction.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_reduction.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_reduction.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS17.test_reduction.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_scatter_gather.TestGather` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_scatter_gather.TestGatherAlongAxis` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_scatter_gather.TestGatherNd` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_scatter_gather.TestScatter` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_scatter_gather.TestScatterAlongAxis` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_scatter_gather.TestScatterNd` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_scatter_gather.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_scatter_gather.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_scatter_gather.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_scatter_gather.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS17.test_scatter_gather.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_operation.TestTopK` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_operation.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_operation.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_operation.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_operation.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_operation.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.TestExpandDims` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.TestReshape` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.TestReshapeLike` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.TestReverse` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.TestReverseSequence` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.TestSliceByIndex` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.TestSliceBySize` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.TestSlidingWindows` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.TestSqueeze` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.TestTranspose` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS17.test_tensor_transformation.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.backends_internal` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.clean_up_backends` | Function | `(backends: List[BackendConfig], minimum_opset_version: ct.target, force_include_iOS15_test: bool) -> List[BackendConfig]` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.TestConstexprBlockwiseDequantize` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.TestConstexprLut` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.TestConstexprLutToSparse` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.TestConstexprSparseBlockwiseShiftScale` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.TestConstexprSparseToDense` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.TestJointCompressionOps` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.constexpr_blockwise_shift_scale` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.constexpr_lut_to_dense` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.constexpr_lut_to_sparse` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.constexpr_sparse_blockwise_shift_scale` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.constexpr_sparse_to_dense` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.promote_input_dtypes` | Function | `(input_vars)` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_compression.utils` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_recurrent.TestGRU` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_recurrent.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_recurrent.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_recurrent.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_recurrent.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_states.TestCoreMLUpdateState` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_states.TestReadState` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_states.TestStatefulModel` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_states.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_states.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_states.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_states.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.mil.ops.tests.iOS18.test_states.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS18.test_states.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_tensor_transformation.TestSliceUpdate` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_tensor_transformation.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_tensor_transformation.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_tensor_transformation.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_tensor_transformation.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS18.test_tensor_transformation.target` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_tensor_transformation.types` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_transformers.TestScaledDotProductAttention` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_transformers.backends` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_transformers.compute_units` | Object | `` |
| `converters.mil.mil.ops.tests.iOS18.test_transformers.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.ops.tests.iOS18.test_transformers.mb` | Class | `(...)` |
| `converters.mil.mil.ops.tests.iOS18.test_transformers.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.iOS18.test_transformers.types` | Object | `` |
| `converters.mil.mil.ops.tests.test_utils.TestAggregatePadding` | Class | `(...)` |
| `converters.mil.mil.ops.tests.test_utils.TestDilation` | Class | `(...)` |
| `converters.mil.mil.ops.tests.test_utils.TestOutputShape` | Class | `(...)` |
| `converters.mil.mil.ops.tests.test_utils.aggregated_pad` | Function | `(pad_type, kernel_shape, input_shape, strides, dilations, custom_pad)` |
| `converters.mil.mil.ops.tests.test_utils.effective_kernel` | Function | `(kernel_shape, dilations)` |
| `converters.mil.mil.ops.tests.test_utils.spatial_dimensions_out_shape` | Function | `(pad_type, input_shape, kernel_shape, strides, dilations, custom_pad, ceil_mode)` |
| `converters.mil.mil.ops.tests.testing_utils.BackendConfig` | Class | `(...)` |
| `converters.mil.mil.ops.tests.testing_utils.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.ops.tests.testing_utils.PassPipeline` | Class | `(pass_names, pipeline_name)` |
| `converters.mil.mil.ops.tests.testing_utils.Placeholder` | Class | `(sym_shape, dtype, name, allow_rank0_input)` |
| `converters.mil.mil.ops.tests.testing_utils.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.mil.ops.tests.testing_utils.UNK_SYM` | Object | `` |
| `converters.mil.mil.ops.tests.testing_utils.UNK_VARIADIC` | Object | `` |
| `converters.mil.mil.ops.tests.testing_utils.compare_backend` | Function | `(mlmodel, input_key_values, expected_outputs, dtype, atol, rtol, also_compare_shapes, state, allow_mismatch_ratio)` |
| `converters.mil.mil.ops.tests.testing_utils.construct_inputs_from_placeholders` | Function | `(input_placeholders: Dict[str, Placeholder], upper_bound: int) -> [List[TensorType]]` |
| `converters.mil.mil.ops.tests.testing_utils.ct_convert` | Function | `(program, source, inputs, outputs, classifier_config, minimum_deployment_target, convert_to, compute_precision, skip_model_load, converter, kwargs)` |
| `converters.mil.mil.ops.tests.testing_utils.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.ops.tests.testing_utils.logger` | Object | `` |
| `converters.mil.mil.ops.tests.testing_utils.mark_api_breaking` | Function | `(breaking_opset_version: ct.target)` |
| `converters.mil.mil.ops.tests.testing_utils.mil` | Object | `` |
| `converters.mil.mil.ops.tests.testing_utils.run_compare_builder` | Function | `(build, input_placeholders, input_values, expected_output_types, expected_outputs, compute_unit, frontend_only, backend: Optional[BackendConfig], atol, rtol, inputs, also_compare_shapes, converter, pass_pipeline: Optional[PassPipeline], pred_iters: Optional[int])` |
| `converters.mil.mil.ops.tests.testing_utils.validate_minimum_deployment_target` | Function | `(minimum_deployment_target: ct.target, backend: Tuple[str, str])` |
| `converters.mil.mil.passes.adjust_io_to_supported_types` | Object | `` |
| `converters.mil.mil.passes.alert_return_type_cast` | Object | `` |
| `converters.mil.mil.passes.backfill_make_list_elem_type` | Object | `` |
| `converters.mil.mil.passes.cleanup` | Object | `` |
| `converters.mil.mil.passes.commingle_loop_vars` | Object | `` |
| `converters.mil.mil.passes.conv1d_decomposition` | Object | `` |
| `converters.mil.mil.passes.defs.cleanup.const_deduplication.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.const_deduplication.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.cleanup.const_deduplication.ListVar` | Class | `(name, elem_type, init_length, dynamic_length, sym_val, kwargs)` |
| `converters.mil.mil.passes.defs.cleanup.const_deduplication.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.cleanup.const_deduplication.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.passes.defs.cleanup.const_deduplication.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.cleanup.const_deduplication.const_deduplication` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.const_deduplication.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.cleanup.const_deduplication.types` | Object | `` |
| `converters.mil.mil.passes.defs.cleanup.const_elimination.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.const_elimination.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.cleanup.const_elimination.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.cleanup.const_elimination.const_elimination` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.const_elimination.logger` | Object | `` |
| `converters.mil.mil.passes.defs.cleanup.const_elimination.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.const_elimination.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.cleanup.dead_code_elimination.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.dead_code_elimination.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.cleanup.dead_code_elimination.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.cleanup.dead_code_elimination.dead_code_elimination` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.dead_code_elimination.logger` | Object | `` |
| `converters.mil.mil.passes.defs.cleanup.dead_code_elimination.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.cleanup.dedup_op_and_var_names.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.dedup_op_and_var_names.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.passes.defs.cleanup.dedup_op_and_var_names.dedup_op_and_var_names` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.dedup_op_and_var_names.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.cleanup.expand_dynamic_linear.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.expand_dynamic_linear.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.cleanup.expand_dynamic_linear.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.cleanup.expand_dynamic_linear.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.cleanup.expand_dynamic_linear.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.passes.defs.cleanup.expand_dynamic_linear.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.cleanup.expand_dynamic_linear.expand_dynamic_linear` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.expand_dynamic_linear.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.expand_dynamic_linear.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.cleanup.fuse_reduce_mean.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.fuse_reduce_mean.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.cleanup.fuse_reduce_mean.fuse_reduce_mean` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.fuse_reduce_mean.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.passes.defs.cleanup.fuse_reduce_mean.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.fuse_reduce_mean.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.cleanup.loop_invariant_elimination.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.loop_invariant_elimination.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.cleanup.loop_invariant_elimination.loop_invariant_elimination` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.loop_invariant_elimination.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.loop_invariant_elimination.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.cleanup.noop_elimination.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.noop_elimination.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.cleanup.noop_elimination.noop_elimination` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.noop_elimination.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.cleanup.remove_redundant_ops.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.remove_redundant_ops.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.cleanup.remove_redundant_ops.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.cleanup.remove_redundant_ops.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.passes.defs.cleanup.remove_redundant_ops.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.cleanup.remove_redundant_ops.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.cleanup.remove_redundant_ops.remove_redundant_ops` | Class | `()` |
| `converters.mil.mil.passes.defs.cleanup.remove_symbolic_reshape.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.remove_symbolic_reshape.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.cleanup.remove_symbolic_reshape.any_variadic` | Function | `(val)` |
| `converters.mil.mil.passes.defs.cleanup.remove_symbolic_reshape.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.cleanup.remove_symbolic_reshape.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.passes.defs.cleanup.remove_symbolic_reshape.logger` | Object | `` |
| `converters.mil.mil.passes.defs.cleanup.remove_symbolic_reshape.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.remove_symbolic_reshape.num_symbolic` | Function | `(val)` |
| `converters.mil.mil.passes.defs.cleanup.remove_symbolic_reshape.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.cleanup.remove_symbolic_reshape.remove_symbolic_reshape` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.topological_reorder.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.topological_reorder.CacheDoublyLinkedList` | Class | `(array: Optional[List[Operation]])` |
| `converters.mil.mil.passes.defs.cleanup.topological_reorder.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.cleanup.topological_reorder.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.cleanup.topological_reorder.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.cleanup.topological_reorder.topological_reorder` | Class | `(...)` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.ComplexVar` | Class | `(name, sym_type, sym_val, op, op_output_idx, real: Optional[Var], imag: Optional[Var])` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.LowerComplex` | Class | `(...)` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.fft_canonicalize_length_dim` | Function | `(input_data: Var, length: Optional[Var], dim: Optional[Var], c2r: bool) -> Tuple[int, int]` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.fft_canonicalize_shapes_dims` | Function | `(input_data: Var, shapes: Optional[Var], dims: Optional[Var], c2r: bool) -> Tuple[Tuple[int], Tuple[int]]` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.lower_complex_dialect_ops` | Class | `(...)` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.lower_complex_dialect_ops.types` | Object | `` |
| `converters.mil.mil.passes.defs.optimize_activation.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation.FusePreluPattern1` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation.FusePreluPattern2` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.optimize_activation.fuse_all_blocks` | Function | `(ops_arrangement, var_constraints, transform_pattern, prog)` |
| `converters.mil.mil.passes.defs.optimize_activation.fuse_gelu_exact` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation.fuse_gelu_tanh_approximation` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation.fuse_leaky_relu` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation.fuse_prelu` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.passes.defs.optimize_activation.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation.prelu_to_lrelu` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.optimize_activation_quantization.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation_quantization.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.optimize_activation_quantization.OpLinearQuantizerConfig` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation_quantization.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.optimize_activation_quantization.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.optimize_activation_quantization.insert_suffix_quantize_dequantize_pair` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation_quantization.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_activation_quantization.optimize_utils` | Object | `` |
| `converters.mil.mil.passes.defs.optimize_activation_quantization.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.optimize_activation_quantization.types` | Object | `` |
| `converters.mil.mil.passes.defs.optimize_conv.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_conv.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.optimize_conv.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.optimize_conv.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.optimize_conv.add_conv_transpose_output_shape` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_conv.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.passes.defs.optimize_conv.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.optimize_conv.compose_conv1d` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_conv.fuse_conv_batchnorm` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_conv.fuse_conv_bias` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_conv.fuse_conv_scale` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_conv.fuse_dilated_conv` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_conv.fuse_pad_conv` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_conv.logger` | Object | `` |
| `converters.mil.mil.passes.defs.optimize_conv.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_conv.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.optimize_conv.types` | Object | `` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.broadcast_shapes` | Function | `(shape_x, shape_y)` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.divide_to_multiply` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.fuse_elementwise_to_batchnorm` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.rank0_expand_dims_swap` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.optimize_elementwise_binary.select_optimization` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_linear.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_linear.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.optimize_linear.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.optimize_linear.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.optimize_linear.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.passes.defs.optimize_linear.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.optimize_linear.fuse_linear_bias` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_linear.fuse_matmul_weight_bias` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_linear.fuse_transpose_matmul` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_linear.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_linear.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.optimize_normalization.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_normalization.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.optimize_normalization.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.optimize_normalization.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.optimize_normalization.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.passes.defs.optimize_normalization.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.optimize_normalization.fuse_layernorm_or_instancenorm` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_normalization.logger` | Object | `` |
| `converters.mil.mil.passes.defs.optimize_normalization.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_normalization.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.optimize_quantization.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_quantization.AvailableTarget` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_quantization.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.optimize_quantization.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.optimize_quantization.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.passes.defs.optimize_quantization.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.optimize_quantization.canonicalize_quantized_lut_pattern` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_quantization.dequantize_quantize_pair_elimination` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_quantization.dequantize_to_constexpr` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_quantization.distributive_quantized_binary_op_scale_normalization` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_quantization.int_op_canonicalization` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_quantization.is_current_opset_version_compatible_with` | Function | `(opset_version)` |
| `converters.mil.mil.passes.defs.optimize_quantization.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_quantization.merge_affine_dequantize_with_consecutive_ops` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_quantization.nullify_redundant_quantization_zero_point` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_quantization.optimize_utils` | Object | `` |
| `converters.mil.mil.passes.defs.optimize_quantization.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.optimize_quantization.reorder_lut_per_channel_scale` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_quantization.types` | Object | `` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.CastOptimizationNode` | Class | `(op_type, match_criterion)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.RangeTuple` | Object | `` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.TransformAxisUpdateOps` | Class | `(op, transpose_axes, var_to_hypothetical_value_dict)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.builtin_to_range` | Function | `(builtin_type: type) -> RangeTuple` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.builtin_to_resolution` | Function | `(builtin_type: type)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.cast_optimization` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.logger` | Object | `` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.merge_consecutive_paddings` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.merge_consecutive_relus` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.merge_consecutive_reshapes` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.merge_consecutive_transposes` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.reduce_transposes` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.optimize_repeat_ops.string_to_builtin` | Function | `(s)` |
| `converters.mil.mil.passes.defs.optimize_state.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_state.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.optimize_state.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.optimize_state.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.optimize_state.ScopeInfo` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_state.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.optimize_state.canonicalize_inplace_pattern` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_state.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_state.prefer_state_in_downstream` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_state.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.AvailableTarget` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.concat_to_pixel_shuffle` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.detect_concat_interleave` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.expand_high_rank_reshape_and_transpose` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.fuse_onehot_matmul_to_gather` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.fuse_squeeze_expand_dims` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.fuse_stack_split` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.guard_negative_gather_indices` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.is_current_opset_version_compatible_with` | Function | `(opset_version)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.replace_stack_reshape` | Class | `(...)` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.types` | Object | `` |
| `converters.mil.mil.passes.defs.optimize_tensor_operation.use_reflection_padding` | Class | `(...)` |
| `converters.mil.mil.passes.defs.preprocess.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.preprocess.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.preprocess.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.passes.defs.preprocess.NameSanitizer` | Class | `(prefix)` |
| `converters.mil.mil.passes.defs.preprocess.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.preprocess.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.preprocess.image_input_preprocess` | Class | `(...)` |
| `converters.mil.mil.passes.defs.preprocess.input_types` | Object | `` |
| `converters.mil.mil.passes.defs.preprocess.logger` | Object | `` |
| `converters.mil.mil.passes.defs.preprocess.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.preprocess.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.preprocess.sanitize_input_output_names` | Class | `(...)` |
| `converters.mil.mil.passes.defs.preprocess.types` | Object | `` |
| `converters.mil.mil.passes.defs.preprocess.update_output_dtypes` | Class | `(...)` |
| `converters.mil.mil.passes.defs.quantization.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.quantization.AbstractQuantizationPass` | Class | `(op_selector)` |
| `converters.mil.mil.passes.defs.quantization.AvailableTarget` | Class | `(...)` |
| `converters.mil.mil.passes.defs.quantization.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.defs.quantization.CastTypeQuantization` | Class | `(op_selector)` |
| `converters.mil.mil.passes.defs.quantization.ComputePrecision` | Class | `(...)` |
| `converters.mil.mil.passes.defs.quantization.FP16ComputePrecision` | Class | `(op_selector)` |
| `converters.mil.mil.passes.defs.quantization.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.passes.defs.quantization.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.quantization.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.quantization.SSAOpRegistry` | Class | `(...)` |
| `converters.mil.mil.passes.defs.quantization.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.passes.defs.quantization.add_fp16_cast` | Class | `(...)` |
| `converters.mil.mil.passes.defs.quantization.add_int16_cast` | Class | `(op_selector)` |
| `converters.mil.mil.passes.defs.quantization.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.quantization.input_types` | Object | `` |
| `converters.mil.mil.passes.defs.quantization.is_current_opset_version_compatible_with` | Function | `(opset_version)` |
| `converters.mil.mil.passes.defs.quantization.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.passes.defs.quantization.logger` | Object | `` |
| `converters.mil.mil.passes.defs.quantization.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.quantization.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.quantization.string_to_builtin` | Function | `(s)` |
| `converters.mil.mil.passes.defs.quantization.types` | Object | `` |
| `converters.mil.mil.passes.defs.randomize.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.randomize.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.defs.randomize.WeightRandomizer` | Class | `(...)` |
| `converters.mil.mil.passes.defs.randomize.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.defs.randomize.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.randomize.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.symbol_transform.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.symbol_transform.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.passes.defs.symbol_transform.Placeholder` | Class | `(sym_shape, dtype, name, allow_rank0_input)` |
| `converters.mil.mil.passes.defs.symbol_transform.Program` | Class | `()` |
| `converters.mil.mil.passes.defs.symbol_transform.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.passes.defs.symbol_transform.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.passes.defs.symbol_transform.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.passes.defs.symbol_transform.logger` | Object | `` |
| `converters.mil.mil.passes.defs.symbol_transform.materialize_symbolic_shape_program` | Class | `()` |
| `converters.mil.mil.passes.defs.symbol_transform.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.symbol_transform.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.symbol_transform.types` | Object | `` |
| `converters.mil.mil.passes.defs.transformer.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.defs.transformer.logger` | Object | `` |
| `converters.mil.mil.passes.defs.transformer.mb` | Class | `(...)` |
| `converters.mil.mil.passes.defs.transformer.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.defs.transformer.scaled_dot_product_attention_sliced_q` | Class | `()` |
| `converters.mil.mil.passes.defs.transformer.target` | Class | `(...)` |
| `converters.mil.mil.passes.defs.transformer.types` | Object | `` |
| `converters.mil.mil.passes.expand_tf_lstm` | Object | `` |
| `converters.mil.mil.passes.fuse_activation_silu` | Object | `` |
| `converters.mil.mil.passes.fuse_pow2_sqrt` | Object | `` |
| `converters.mil.mil.passes.graph_pass.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.graph_pass.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.graph_pass.PassOption` | Class | `(option_name: Text, option_val: Union[Text, Callable[[Operation], bool]])` |
| `converters.mil.mil.passes.graph_pass.Program` | Class | `()` |
| `converters.mil.mil.passes.graph_pass.ScopeInfo` | Class | `(...)` |
| `converters.mil.mil.passes.graph_pass.ScopeSource` | Class | `(...)` |
| `converters.mil.mil.passes.graph_pass.mb` | Class | `(...)` |
| `converters.mil.mil.passes.handle_return_inputs_as_outputs` | Object | `` |
| `converters.mil.mil.passes.handle_return_unused_inputs` | Object | `` |
| `converters.mil.mil.passes.handle_unused_inputs` | Object | `` |
| `converters.mil.mil.passes.helper.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.helper.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.mil.passes.helper.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.passes.helper.block_context_manager` | Function | `(_func: Optional[Callable])` |
| `converters.mil.mil.passes.helper.classproperty` | Class | `(...)` |
| `converters.mil.mil.passes.insert_image_preprocessing_op` | Object | `` |
| `converters.mil.mil.passes.lower_complex_dialect_ops` | Object | `` |
| `converters.mil.mil.passes.mlmodel_passes` | Object | `` |
| `converters.mil.mil.passes.optimize_activation` | Object | `` |
| `converters.mil.mil.passes.optimize_activation_quantization` | Object | `` |
| `converters.mil.mil.passes.optimize_conv` | Object | `` |
| `converters.mil.mil.passes.optimize_elementwise_binary` | Object | `` |
| `converters.mil.mil.passes.optimize_linear` | Object | `` |
| `converters.mil.mil.passes.optimize_normalization` | Object | `` |
| `converters.mil.mil.passes.optimize_quantization` | Object | `` |
| `converters.mil.mil.passes.optimize_repeat_ops` | Object | `` |
| `converters.mil.mil.passes.optimize_state` | Object | `` |
| `converters.mil.mil.passes.optimize_tensor_operation` | Object | `` |
| `converters.mil.mil.passes.pass_pipeline.OpPalettizerConfig` | Class | `(...)` |
| `converters.mil.mil.passes.pass_pipeline.OpThresholdPrunerConfig` | Class | `(...)` |
| `converters.mil.mil.passes.pass_pipeline.OptimizationConfig` | Class | `(...)` |
| `converters.mil.mil.passes.pass_pipeline.PASS_REGISTRY` | Object | `` |
| `converters.mil.mil.passes.pass_pipeline.PassOption` | Class | `(option_name: Text, option_val: Union[Text, Callable[[Operation], bool]])` |
| `converters.mil.mil.passes.pass_pipeline.PassPipeline` | Class | `(pass_names, pipeline_name)` |
| `converters.mil.mil.passes.pass_pipeline.PassPipelineManager` | Class | `(...)` |
| `converters.mil.mil.passes.pass_pipeline.Program` | Class | `()` |
| `converters.mil.mil.passes.pass_pipeline.logger` | Object | `` |
| `converters.mil.mil.passes.pass_registry.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.mil.passes.pass_registry.PASS_REGISTRY` | Object | `` |
| `converters.mil.mil.passes.pass_registry.PassRegistry` | Class | `()` |
| `converters.mil.mil.passes.pass_registry.logger` | Object | `` |
| `converters.mil.mil.passes.pass_registry.register_pass` | Function | `(namespace: Text, override: bool, name: Optional[Text])` |
| `converters.mil.mil.passes.preprocess` | Object | `` |
| `converters.mil.mil.passes.remove_vacuous_cond` | Object | `` |
| `converters.mil.mil.passes.sanitize_name_strings` | Object | `` |
| `converters.mil.mil.passes.symbol_transform` | Object | `` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.CONSTEXPR_FUNCS` | Object | `` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.CONSTEXPR_OPS` | Object | `` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.PASS_REGISTRY` | Object | `` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.Symbol` | Class | `(sym_name)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.TestConstDeduplication` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.TestConstElimination` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.TestDeadCodeElimination` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.TestDedupOpAndVarNames` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.TestExpandDynamicLinear` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.TestLoopInvariantElimination` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.TestNoopElimination` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.TestReduceMeanFusion` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.TestRemoveRedundantOps` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.TestRemoveSymbolicReshape` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.TestTopologicalReorder` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.assert_op_count_match` | Function | `(program, expect, op, verbose)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.assert_same_output_names` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.get_op_names_in_program` | Function | `(prog, func_name, skip_const_ops)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.get_op_types_in_block` | Function | `(block: Block, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.mb` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.mil` | Object | `` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.remove_redundant_ops` | Class | `()` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.topological_reorder` | Object | `` |
| `converters.mil.mil.passes.tests.test_cleanup_passes.types` | Object | `` |
| `converters.mil.mil.passes.tests.test_lower_complex_dialect_ops.ComputeUnit` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_lower_complex_dialect_ops.ScopeInfo` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_lower_complex_dialect_ops.ScopeSource` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_lower_complex_dialect_ops.TestLowerComplexDialectOps` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_lower_complex_dialect_ops.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `converters.mil.mil.passes.tests.test_lower_complex_dialect_ops.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.mil.passes.tests.test_lower_complex_dialect_ops.ct_convert` | Function | `(program, source, inputs, outputs, classifier_config, minimum_deployment_target, convert_to, compute_precision, skip_model_load, converter, kwargs)` |
| `converters.mil.mil.passes.tests.test_lower_complex_dialect_ops.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.passes.tests.test_lower_complex_dialect_ops.mb` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_optimize_linear_passes.PASS_REGISTRY` | Object | `` |
| `converters.mil.mil.passes.tests.test_optimize_linear_passes.TestFuseLinearBias` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_optimize_linear_passes.TestFuseMatmulWeightBias` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_optimize_linear_passes.TestFuseTransposeMatmul` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_optimize_linear_passes.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `converters.mil.mil.passes.tests.test_optimize_linear_passes.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.mil.passes.tests.test_optimize_linear_passes.assert_op_count_match` | Function | `(program, expect, op, verbose)` |
| `converters.mil.mil.passes.tests.test_optimize_linear_passes.assert_same_output_names` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.mil.passes.tests.test_optimize_linear_passes.backends` | Object | `` |
| `converters.mil.mil.passes.tests.test_optimize_linear_passes.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.passes.tests.test_optimize_linear_passes.mb` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_pass_pipeline.PassPipeline` | Class | `(pass_names, pipeline_name)` |
| `converters.mil.mil.passes.tests.test_pass_pipeline.PassPipelineManager` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_pass_pipeline.TestPassPipeline` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_pass_pipeline.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.mil.passes.tests.test_pass_pipeline.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.passes.tests.test_pass_pipeline.mb` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.AvailableTarget` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.CONSTEXPR_FUNCS` | Object | `` |
| `converters.mil.mil.passes.tests.test_passes.CONSTEXPR_OPS` | Object | `` |
| `converters.mil.mil.passes.tests.test_passes.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.passes.tests.test_passes.PASS_REGISTRY` | Object | `` |
| `converters.mil.mil.passes.tests.test_passes.ScopeInfo` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.ScopeSource` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestAddConvTransposeOutputShape` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestCastOptimizationAcrossBlocks` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestCastOptimizationCastFusion` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestCastOptimizationComplexPatterns` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestCastOptimizationReduendantCastRemoval` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestChildOrdering` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestConcatInterleave` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestConcatToPixelShuffle` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestConv1dChannellastCompositionPasses` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestConv1dCompositionPasses` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestConvBatchNormFusion` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestConvBiasFusion` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestConvScaleFusion` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestDivideToMultiply` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestExpandHighRankReshapeAndTranspose` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestFuseDilatedConv` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestFuseElementwiseToBatchNorm` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestFuseLayerNormOrInstanceNorm` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestFuseLinearBias` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestFuseMatmulWeightBias` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestFuseOnehotMatmulToGather` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestFusePadConv` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestFuseSqueezeExpandDims` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestGeluFusion` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestGraphPassScopePreservation` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestGuardNegativeGatherIndices` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestImageInputPreprocess` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestLeakyReluFusion` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestMergeConsecutivePaddings` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestMergeConsecutiveRelus` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestMergeConsecutiveReshapes` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestMergeConsecutiveTransposes` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestPreluFusion` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestPreluToLrelu` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestRandomizeWeights` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestRank0ExpandDimsSwap` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestReplaceStackReshape` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestSanitizeInputOutputNames` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestScaledDotProductAttentionSlicedQ` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestSelectOptimization` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestSkipConstexprOps` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestStackSplitFusion` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestUpdateOutputDtypes` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.TestUseReflectionPadding` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `converters.mil.mil.passes.tests.test_passes.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.mil.passes.tests.test_passes.assert_op_count_match` | Function | `(program, expect, op, verbose)` |
| `converters.mil.mil.passes.tests.test_passes.assert_same_output_names` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.mil.passes.tests.test_passes.backends` | Object | `` |
| `converters.mil.mil.passes.tests.test_passes.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.mil.passes.tests.test_passes.cast_optimization` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.cto` | Object | `` |
| `converters.mil.mil.passes.tests.test_passes.get_op_types_in_block` | Function | `(block: Block, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.passes.tests.test_passes.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.passes.tests.test_passes.mb` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_passes.mil` | Object | `` |
| `converters.mil.mil.passes.tests.test_passes.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.passes.tests.test_passes.register_generic_pass` | Function | `(ops_arrangement, var_constraints, transform_pattern, pass_name, namespace)` |
| `converters.mil.mil.passes.tests.test_passes.types` | Object | `` |
| `converters.mil.mil.passes.tests.test_quantization_passes.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.mil.passes.tests.test_quantization_passes.QuantizationBaseTest` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.TestDequantizeQuantizePairElimination` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.TestDequantizeToConstexpr` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.TestDistributiveQuantizedBinaryOpScaleNormalization` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.TestFP16CastTransform` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.TestInt32CastToInt16` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.TestIntOpCanonicalization` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.TestNullifyRedundantQuantizationZeroPoint` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.TestReorderLutPerChannelScale` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.TestReorderQuantizedLut` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.TestTensorwiseAffineDequantizeConstElimination` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.TestTransformFunctionSignatures` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.add_fp16_cast` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `converters.mil.mil.passes.tests.test_quantization_passes.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.mb` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_quantization_passes.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.passes.tests.test_quantization_passes.quantization` | Object | `` |
| `converters.mil.mil.passes.tests.test_quantization_passes.types` | Object | `` |
| `converters.mil.mil.passes.tests.test_reduce_transposes_pass.PASS_REGISTRY` | Object | `` |
| `converters.mil.mil.passes.tests.test_reduce_transposes_pass.TestTransposePassUtilityMethods` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_reduce_transposes_pass.TransformAxisUpdateOps` | Class | `(op, transpose_axes, var_to_hypothetical_value_dict)` |
| `converters.mil.mil.passes.tests.test_reduce_transposes_pass.TransposeOptimizationPass` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_reduce_transposes_pass.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `converters.mil.mil.passes.tests.test_reduce_transposes_pass.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.mil.passes.tests.test_reduce_transposes_pass.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.passes.tests.test_reduce_transposes_pass.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.passes.tests.test_reduce_transposes_pass.mb` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_state_passes.AvailableTarget` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_state_passes.TestCanonicalizeInplacePattern` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_state_passes.TestPreferStateInDownstream` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_state_passes.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `converters.mil.mil.passes.tests.test_state_passes.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.passes.tests.test_state_passes.mb` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_state_passes.types` | Object | `` |
| `converters.mil.mil.passes.tests.test_symbol_transform.PASS_REGISTRY` | Object | `` |
| `converters.mil.mil.passes.tests.test_symbol_transform.TestMaterializeSymbolicShapeProgram` | Class | `(...)` |
| `converters.mil.mil.passes.tests.test_symbol_transform.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `converters.mil.mil.passes.tests.test_symbol_transform.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.passes.tests.test_symbol_transform.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.passes.tests.test_symbol_transform.mb` | Class | `(...)` |
| `converters.mil.mil.passes.tf_lstm_to_core_lstm` | Object | `` |
| `converters.mil.mil.passes.torch_tensor_assign_to_core` | Object | `` |
| `converters.mil.mil.passes.torch_upsample_to_core_upsample` | Object | `` |
| `converters.mil.mil.passes.transformer` | Object | `` |
| `converters.mil.mil.precondition` | Function | `(allow)` |
| `converters.mil.mil.program.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.program.InternalInputType` | Class | `(...)` |
| `converters.mil.mil.program.ListVar` | Class | `(name, elem_type, init_length, dynamic_length, sym_val, kwargs)` |
| `converters.mil.mil.program.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.program.Placeholder` | Class | `(sym_shape, dtype, name, allow_rank0_input)` |
| `converters.mil.mil.program.Program` | Class | `()` |
| `converters.mil.mil.program.ScopeSource` | Class | `(...)` |
| `converters.mil.mil.program.StateTensorPlaceholder` | Class | `(sym_shape, dtype, name)` |
| `converters.mil.mil.program.Symbol` | Class | `(sym_name)` |
| `converters.mil.mil.program.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.program.get_existing_symbol` | Function | `(name)` |
| `converters.mil.mil.program.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.program.get_new_variadic_symbol` | Function | `()` |
| `converters.mil.mil.program.k_num_internal_syms` | Object | `` |
| `converters.mil.mil.program.k_used_symbols` | Object | `` |
| `converters.mil.mil.program.logger` | Object | `` |
| `converters.mil.mil.program.types` | Object | `` |
| `converters.mil.mil.register_op` | Object | `` |
| `converters.mil.mil.scope.SCOPE_STACK` | Object | `` |
| `converters.mil.mil.scope.ScopeContextManager` | Class | `(scopes: List[ScopeInfo])` |
| `converters.mil.mil.scope.ScopeInfo` | Class | `(...)` |
| `converters.mil.mil.scope.ScopeSource` | Class | `(...)` |
| `converters.mil.mil.scope.ScopeStack` | Class | `()` |
| `converters.mil.mil.scope.VALID_OPS_TO_COPY_SCOPE_INFO` | Object | `` |
| `converters.mil.mil.scope.add_graph_pass_scope` | Function | `(src_scopes: Dict[ScopeSource, List[str]], graph_pass_scopes: Dict[ScopeSource, List[str]]) -> Dict[ScopeSource, List[str]]` |
| `converters.mil.mil.tests.test_block.CONSTEXPR_FUNCS` | Object | `` |
| `converters.mil.mil.tests.test_block.CacheDoublyLinkedList` | Class | `(array: Optional[List[Operation]])` |
| `converters.mil.mil.tests.test_block.TestCacheDoublyLinkedList` | Class | `(...)` |
| `converters.mil.mil.tests.test_block.assert_same_output_names` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.mil.tests.test_block.assert_same_output_shapes` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.mil.tests.test_block.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.tests.test_block.mb` | Class | `(...)` |
| `converters.mil.mil.tests.test_block.test_add_op` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_duplicate_outputs_add_consuming_block_once` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_duplicate_outputs_substituion` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_empty_block` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_op_removal_and_insertion` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_remove_duplicate_ops` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_remove_duplicate_ops_not_affect_others` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_remove_op` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_remove_op2` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_remove_ops_fail_for_block_output` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_replace_nonreplaceable_vars` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_replace_nonreplaceable_vars_force` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_simple_substituion` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_simple_transpose_squash` | Function | `()` |
| `converters.mil.mil.tests.test_block.test_substitute_nested_op` | Function | `()` |
| `converters.mil.mil.tests.test_debug.TestExtractSubModel` | Class | `(...)` |
| `converters.mil.mil.tests.test_debug.compute_ground_truth_answer` | Function | `(input)` |
| `converters.mil.mil.tests.test_debug.extract_submodel` | Function | `(model: MLModel, outputs: List[str], inputs: Optional[List[str]], function_name: str) -> MLModel` |
| `converters.mil.mil.tests.test_debug.get_new_symbol` | Function | `(name)` |
| `converters.mil.mil.tests.test_debug.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.mil.tests.test_debug.get_simple_program` | Function | `(opset_version)` |
| `converters.mil.mil.tests.test_debug.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.tests.test_debug.mb` | Class | `(...)` |
| `converters.mil.mil.tests.test_debug.types` | Object | `` |
| `converters.mil.mil.tests.test_programs.CONSTEXPR_FUNCS` | Object | `` |
| `converters.mil.mil.tests.test_programs.ComplexVar` | Class | `(name, sym_type, sym_val, op, op_output_idx, real: Optional[Var], imag: Optional[Var])` |
| `converters.mil.mil.tests.test_programs.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.mil.tests.test_programs.Program` | Class | `()` |
| `converters.mil.mil.tests.test_programs.ScopeInfo` | Class | `(...)` |
| `converters.mil.mil.tests.test_programs.ScopeSource` | Class | `(...)` |
| `converters.mil.mil.tests.test_programs.TestBeforeOp` | Class | `(...)` |
| `converters.mil.mil.tests.test_programs.TestMILBasic` | Class | `(...)` |
| `converters.mil.mil.tests.test_programs.TestMILBuilderAPI` | Class | `(...)` |
| `converters.mil.mil.tests.test_programs.TestMILProgramVersionHandling` | Class | `(...)` |
| `converters.mil.mil.tests.test_programs.TestScope` | Class | `(...)` |
| `converters.mil.mil.tests.test_programs.add_graph_pass_scope` | Function | `(src_scopes: Dict[ScopeSource, List[str]], graph_pass_scopes: Dict[ScopeSource, List[str]]) -> Dict[ScopeSource, List[str]]` |
| `converters.mil.mil.tests.test_programs.get_simple_nested_block_program` | Function | `(opset_version)` |
| `converters.mil.mil.tests.test_programs.get_simple_pixel_unshuffle_program` | Function | `(opset_version)` |
| `converters.mil.mil.tests.test_programs.get_simple_topk_pixel_unshuffle_program` | Function | `(opset_version)` |
| `converters.mil.mil.tests.test_programs.get_simple_topk_program` | Function | `(opset_version)` |
| `converters.mil.mil.tests.test_programs.logger` | Object | `` |
| `converters.mil.mil.tests.test_programs.mb` | Class | `(...)` |
| `converters.mil.mil.tests.test_programs.mil` | Object | `` |
| `converters.mil.mil.tests.test_programs.test_conv_example` | Function | `()` |
| `converters.mil.mil.tests.test_programs.test_reserved_node_names` | Function | `()` |
| `converters.mil.mil.tests.test_programs.test_single_layer_example` | Function | `()` |
| `converters.mil.mil.tests.test_programs.test_while_example` | Function | `()` |
| `converters.mil.mil.tests.test_programs.types` | Object | `` |
| `converters.mil.mil.tests.test_types.ImageType` | Class | `(name, shape, scale, bias, color_layout, channel_first, grayscale_use_uint8)` |
| `converters.mil.mil.tests.test_types.StateType` | Class | `(wrapped_type: type, name: Optional[str])` |
| `converters.mil.mil.tests.test_types.TensorType` | Class | `(name, shape, dtype, default_value)` |
| `converters.mil.mil.tests.test_types.TestInputTypes` | Class | `(...)` |
| `converters.mil.mil.tests.test_types.TestTypeMapping` | Class | `(...)` |
| `converters.mil.mil.tests.test_types.TestTypes` | Class | `(...)` |
| `converters.mil.mil.tests.test_types.optimize_utils` | Object | `` |
| `converters.mil.mil.tests.test_types.type_mapping` | Object | `` |
| `converters.mil.mil.tests.test_types.types` | Object | `` |
| `converters.mil.mil.types.BUILTIN_TO_PROTO_TYPES` | Object | `` |
| `converters.mil.mil.types.IMMEDIATE_VALUE_TYPES_IN_BYTES` | Object | `` |
| `converters.mil.mil.types.PROTO_TO_BUILTIN_TYPE` | Object | `` |
| `converters.mil.mil.types.SUB_BYTE_DTYPE_METADATA_KEY` | Object | `` |
| `converters.mil.mil.types.annotate.annotate` | Function | `(return_type, kwargs)` |
| `converters.mil.mil.types.annotate.annotated_class_list` | Object | `` |
| `converters.mil.mil.types.annotate.annotated_function_list` | Object | `` |
| `converters.mil.mil.types.annotate.apply_delayed_types` | Function | `(type_map, fnlist)` |
| `converters.mil.mil.types.annotate.class_annotate` | Function | `()` |
| `converters.mil.mil.types.annotate.delay_type` | Object | `` |
| `converters.mil.mil.types.annotate.delay_type_cls` | Class | `(...)` |
| `converters.mil.mil.types.apply_delayed_types` | Function | `(type_map, fnlist)` |
| `converters.mil.mil.types.bool` | Class | `(v)` |
| `converters.mil.mil.types.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.mil.types.class_annotate` | Function | `()` |
| `converters.mil.mil.types.complex` | Object | `` |
| `converters.mil.mil.types.complex128` | Object | `` |
| `converters.mil.mil.types.complex64` | Object | `` |
| `converters.mil.mil.types.delay_type` | Object | `` |
| `converters.mil.mil.types.dict` | Function | `(keytype, valuetype)` |
| `converters.mil.mil.types.double` | Object | `` |
| `converters.mil.mil.types.empty_dict` | Class | `(...)` |
| `converters.mil.mil.types.empty_list` | Class | `(...)` |
| `converters.mil.mil.types.float` | Object | `` |
| `converters.mil.mil.types.fp16` | Object | `` |
| `converters.mil.mil.types.fp32` | Object | `` |
| `converters.mil.mil.types.fp64` | Object | `` |
| `converters.mil.mil.types.get_nbits_int_builtin_type` | Function | `(nbits: int, signed: True) -> type` |
| `converters.mil.mil.types.get_type_info.FunctionType` | Class | `(inputs, output, python_function)` |
| `converters.mil.mil.types.get_type_info.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.get_type_info.get_python_method_type` | Function | `(py_function)` |
| `converters.mil.mil.types.get_type_info.get_type_info` | Function | `(t)` |
| `converters.mil.mil.types.get_type_info.void` | Class | `(...)` |
| `converters.mil.mil.types.global_methods.global_invremap` | Object | `` |
| `converters.mil.mil.types.global_methods.global_remap` | Object | `` |
| `converters.mil.mil.types.global_remap` | Object | `` |
| `converters.mil.mil.types.globals_pseudo_type` | Class | `(...)` |
| `converters.mil.mil.types.int16` | Object | `` |
| `converters.mil.mil.types.int32` | Object | `` |
| `converters.mil.mil.types.int4` | Object | `` |
| `converters.mil.mil.types.int64` | Object | `` |
| `converters.mil.mil.types.int8` | Object | `` |
| `converters.mil.mil.types.is_bool` | Function | `(t)` |
| `converters.mil.mil.types.is_builtin` | Function | `(t)` |
| `converters.mil.mil.types.is_compatible_type` | Function | `(type1, type2)` |
| `converters.mil.mil.types.is_complex` | Function | `(t)` |
| `converters.mil.mil.types.is_dict` | Function | `(t)` |
| `converters.mil.mil.types.is_float` | Function | `(t)` |
| `converters.mil.mil.types.is_int` | Function | `(t)` |
| `converters.mil.mil.types.is_list` | Function | `(t)` |
| `converters.mil.mil.types.is_primitive` | Function | `(btype)` |
| `converters.mil.mil.types.is_scalar` | Function | `(btype)` |
| `converters.mil.mil.types.is_signed_int` | Function | `(t)` |
| `converters.mil.mil.types.is_state` | Function | `(t)` |
| `converters.mil.mil.types.is_str` | Function | `(t)` |
| `converters.mil.mil.types.is_sub_byte` | Function | `(t)` |
| `converters.mil.mil.types.is_subtype` | Function | `(type1, type2)` |
| `converters.mil.mil.types.is_tensor` | Function | `(tensor_type)` |
| `converters.mil.mil.types.is_tensor_and_is_compatible` | Function | `(tensor_type1, tensor_type2, allow_promotion)` |
| `converters.mil.mil.types.is_tuple` | Function | `(t)` |
| `converters.mil.mil.types.is_unsigned_int` | Function | `(t)` |
| `converters.mil.mil.types.list` | Function | `(arg, init_length, dynamic_length)` |
| `converters.mil.mil.types.np_dtype_to_py_type` | Function | `(np_dtype)` |
| `converters.mil.mil.types.np_int4_dtype` | Object | `` |
| `converters.mil.mil.types.np_uint1_dtype` | Object | `` |
| `converters.mil.mil.types.np_uint2_dtype` | Object | `` |
| `converters.mil.mil.types.np_uint3_dtype` | Object | `` |
| `converters.mil.mil.types.np_uint4_dtype` | Object | `` |
| `converters.mil.mil.types.np_uint6_dtype` | Object | `` |
| `converters.mil.mil.types.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.mil.types.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.types.numpy_val_to_builtin_val` | Function | `(npval)` |
| `converters.mil.mil.types.promote_dtypes` | Function | `(dtypes)` |
| `converters.mil.mil.types.promote_types` | Function | `(dtype1, dtype2)` |
| `converters.mil.mil.types.state` | Function | `(state_type)` |
| `converters.mil.mil.types.str` | Class | `(v)` |
| `converters.mil.mil.types.string_to_builtin` | Function | `(s)` |
| `converters.mil.mil.types.string_to_nptype` | Function | `(s: str)` |
| `converters.mil.mil.types.symbolic.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.types.symbolic.any_variadic` | Function | `(val)` |
| `converters.mil.mil.types.symbolic.is_compatible_symbolic_vector` | Function | `(val_a, val_b)` |
| `converters.mil.mil.types.symbolic.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.types.symbolic.is_variadic` | Function | `(val)` |
| `converters.mil.mil.types.symbolic.isscalar` | Function | `(val)` |
| `converters.mil.mil.types.symbolic.k_num_internal_syms` | Object | `` |
| `converters.mil.mil.types.symbolic.k_used_symbols` | Object | `` |
| `converters.mil.mil.types.symbolic.num_symbolic` | Function | `(val)` |
| `converters.mil.mil.types.tensor` | Function | `(primitive, shape)` |
| `converters.mil.mil.types.tensor_has_complete_shape` | Function | `(tensor_type)` |
| `converters.mil.mil.types.tuple` | Function | `(args)` |
| `converters.mil.mil.types.type_bool.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_bool.annotate` | Function | `(return_type, kwargs)` |
| `converters.mil.mil.types.type_bool.bool` | Class | `(v)` |
| `converters.mil.mil.types.type_bool.class_annotate` | Function | `()` |
| `converters.mil.mil.types.type_bool.delay_type` | Object | `` |
| `converters.mil.mil.types.type_bool.is_bool` | Function | `(t)` |
| `converters.mil.mil.types.type_complex.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_complex.annotate` | Function | `(return_type, kwargs)` |
| `converters.mil.mil.types.type_complex.bool` | Class | `(v)` |
| `converters.mil.mil.types.type_complex.class_annotate` | Function | `()` |
| `converters.mil.mil.types.type_complex.complex` | Object | `` |
| `converters.mil.mil.types.type_complex.complex128` | Object | `` |
| `converters.mil.mil.types.type_complex.complex64` | Object | `` |
| `converters.mil.mil.types.type_complex.delay_type` | Object | `` |
| `converters.mil.mil.types.type_complex.is_complex` | Function | `(t)` |
| `converters.mil.mil.types.type_complex.logger` | Object | `` |
| `converters.mil.mil.types.type_complex.make_complex` | Function | `(width)` |
| `converters.mil.mil.types.type_dict.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_dict.annotate` | Function | `(return_type, kwargs)` |
| `converters.mil.mil.types.type_dict.dict` | Function | `(keytype, valuetype)` |
| `converters.mil.mil.types.type_dict.empty_dict` | Class | `(...)` |
| `converters.mil.mil.types.type_dict.get_type_info` | Function | `(t)` |
| `converters.mil.mil.types.type_dict.is_dict` | Function | `(t)` |
| `converters.mil.mil.types.type_dict.memoize` | Function | `(f)` |
| `converters.mil.mil.types.type_dict.type_bool` | Object | `` |
| `converters.mil.mil.types.type_dict.type_int` | Object | `` |
| `converters.mil.mil.types.type_dict.void` | Class | `(...)` |
| `converters.mil.mil.types.type_double.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_double.annotate` | Function | `(return_type, kwargs)` |
| `converters.mil.mil.types.type_double.bool` | Class | `(v)` |
| `converters.mil.mil.types.type_double.class_annotate` | Function | `()` |
| `converters.mil.mil.types.type_double.delay_type` | Object | `` |
| `converters.mil.mil.types.type_double.double` | Object | `` |
| `converters.mil.mil.types.type_double.float` | Object | `` |
| `converters.mil.mil.types.type_double.fp16` | Object | `` |
| `converters.mil.mil.types.type_double.fp32` | Object | `` |
| `converters.mil.mil.types.type_double.fp64` | Object | `` |
| `converters.mil.mil.types.type_double.is_float` | Function | `(t)` |
| `converters.mil.mil.types.type_double.logger` | Object | `` |
| `converters.mil.mil.types.type_double.make_float` | Function | `(width)` |
| `converters.mil.mil.types.type_globals_pseudo_type.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_globals_pseudo_type.globals_pseudo_type` | Class | `(...)` |
| `converters.mil.mil.types.type_int.SUB_BYTE_DTYPE_METADATA_KEY` | Object | `` |
| `converters.mil.mil.types.type_int.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_int.annotate` | Function | `(return_type, kwargs)` |
| `converters.mil.mil.types.type_int.bool` | Class | `(v)` |
| `converters.mil.mil.types.type_int.class_annotate` | Function | `()` |
| `converters.mil.mil.types.type_int.delay_type` | Object | `` |
| `converters.mil.mil.types.type_int.int16` | Object | `` |
| `converters.mil.mil.types.type_int.int32` | Object | `` |
| `converters.mil.mil.types.type_int.int4` | Object | `` |
| `converters.mil.mil.types.type_int.int64` | Object | `` |
| `converters.mil.mil.types.type_int.int8` | Object | `` |
| `converters.mil.mil.types.type_int.is_int` | Function | `(t)` |
| `converters.mil.mil.types.type_int.is_signed_int` | Function | `(t)` |
| `converters.mil.mil.types.type_int.is_sub_byte` | Function | `(t)` |
| `converters.mil.mil.types.type_int.is_unsigned_int` | Function | `(t)` |
| `converters.mil.mil.types.type_int.logger` | Object | `` |
| `converters.mil.mil.types.type_int.make_int` | Function | `(width_, unsigned_)` |
| `converters.mil.mil.types.type_int.np_int4_dtype` | Object | `` |
| `converters.mil.mil.types.type_int.np_uint1_dtype` | Object | `` |
| `converters.mil.mil.types.type_int.np_uint2_dtype` | Object | `` |
| `converters.mil.mil.types.type_int.np_uint3_dtype` | Object | `` |
| `converters.mil.mil.types.type_int.np_uint4_dtype` | Object | `` |
| `converters.mil.mil.types.type_int.np_uint6_dtype` | Object | `` |
| `converters.mil.mil.types.type_int.uint` | Object | `` |
| `converters.mil.mil.types.type_int.uint1` | Object | `` |
| `converters.mil.mil.types.type_int.uint16` | Object | `` |
| `converters.mil.mil.types.type_int.uint2` | Object | `` |
| `converters.mil.mil.types.type_int.uint3` | Object | `` |
| `converters.mil.mil.types.type_int.uint32` | Object | `` |
| `converters.mil.mil.types.type_int.uint4` | Object | `` |
| `converters.mil.mil.types.type_int.uint6` | Object | `` |
| `converters.mil.mil.types.type_int.uint64` | Object | `` |
| `converters.mil.mil.types.type_int.uint8` | Object | `` |
| `converters.mil.mil.types.type_list.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_list.annotate` | Function | `(return_type, kwargs)` |
| `converters.mil.mil.types.type_list.empty_list` | Class | `(...)` |
| `converters.mil.mil.types.type_list.get_type_info` | Function | `(t)` |
| `converters.mil.mil.types.type_list.is_list` | Function | `(t)` |
| `converters.mil.mil.types.type_list.list` | Function | `(arg, init_length, dynamic_length)` |
| `converters.mil.mil.types.type_list.memoize` | Function | `(f)` |
| `converters.mil.mil.types.type_list.type_int` | Object | `` |
| `converters.mil.mil.types.type_list.void` | Class | `(...)` |
| `converters.mil.mil.types.type_mapping.BUILTIN_TO_PROTO_TYPES` | Object | `` |
| `converters.mil.mil.types.type_mapping.PROTO_TO_BUILTIN_TYPE` | Object | `` |
| `converters.mil.mil.types.type_mapping.RangeTuple` | Object | `` |
| `converters.mil.mil.types.type_mapping.SUB_BYTE_DTYPE_METADATA_KEY` | Object | `` |
| `converters.mil.mil.types.type_mapping.builtin_to_range` | Function | `(builtin_type: type) -> RangeTuple` |
| `converters.mil.mil.types.type_mapping.builtin_to_resolution` | Function | `(builtin_type: type)` |
| `converters.mil.mil.types.type_mapping.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.mil.types.type_mapping.get_nbits_int_builtin_type` | Function | `(nbits: int, signed: True) -> type` |
| `converters.mil.mil.types.type_mapping.get_type_info` | Function | `(t)` |
| `converters.mil.mil.types.type_mapping.infer_complex_dtype` | Function | `(real_dtype, imag_dtype)` |
| `converters.mil.mil.types.type_mapping.infer_fp_dtype_from_complex` | Function | `(complex_dtype)` |
| `converters.mil.mil.types.type_mapping.is_bool` | Function | `(t)` |
| `converters.mil.mil.types.type_mapping.is_builtin` | Function | `(t)` |
| `converters.mil.mil.types.type_mapping.is_complex` | Function | `(t)` |
| `converters.mil.mil.types.type_mapping.is_dict` | Function | `(t)` |
| `converters.mil.mil.types.type_mapping.is_float` | Function | `(t)` |
| `converters.mil.mil.types.type_mapping.is_int` | Function | `(t)` |
| `converters.mil.mil.types.type_mapping.is_list` | Function | `(t)` |
| `converters.mil.mil.types.type_mapping.is_primitive` | Function | `(btype)` |
| `converters.mil.mil.types.type_mapping.is_scalar` | Function | `(btype)` |
| `converters.mil.mil.types.type_mapping.is_str` | Function | `(t)` |
| `converters.mil.mil.types.type_mapping.is_sub_byte` | Function | `(t)` |
| `converters.mil.mil.types.type_mapping.is_subtype` | Function | `(type1, type2)` |
| `converters.mil.mil.types.type_mapping.is_subtype_tensor` | Function | `(type1, type2)` |
| `converters.mil.mil.types.type_mapping.is_tensor` | Function | `(tensor_type)` |
| `converters.mil.mil.types.type_mapping.is_tuple` | Function | `(t)` |
| `converters.mil.mil.types.type_mapping.np_dtype_to_py_type` | Function | `(np_dtype)` |
| `converters.mil.mil.types.type_mapping.np_int4_dtype` | Object | `` |
| `converters.mil.mil.types.type_mapping.np_uint1_dtype` | Object | `` |
| `converters.mil.mil.types.type_mapping.np_uint2_dtype` | Object | `` |
| `converters.mil.mil.types.type_mapping.np_uint3_dtype` | Object | `` |
| `converters.mil.mil.types.type_mapping.np_uint4_dtype` | Object | `` |
| `converters.mil.mil.types.type_mapping.np_uint6_dtype` | Object | `` |
| `converters.mil.mil.types.type_mapping.np_val_to_py_type` | Function | `(val)` |
| `converters.mil.mil.types.type_mapping.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.mil.types.type_mapping.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.types.type_mapping.numpy_val_to_builtin_val` | Function | `(npval)` |
| `converters.mil.mil.types.type_mapping.promote_dtypes` | Function | `(dtypes)` |
| `converters.mil.mil.types.type_mapping.promote_types` | Function | `(dtype1, dtype2)` |
| `converters.mil.mil.types.type_mapping.string_to_builtin` | Function | `(s)` |
| `converters.mil.mil.types.type_mapping.string_to_nptype` | Function | `(s: str)` |
| `converters.mil.mil.types.type_mapping.type_to_builtin_type` | Function | `(type)` |
| `converters.mil.mil.types.type_mapping.types` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_bool` | Class | `(v)` |
| `converters.mil.mil.types.type_mapping.types_complex128` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_complex64` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_fp16` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_fp32` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_fp64` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_int16` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_int32` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_int4` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_int64` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_int8` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_str` | Class | `(v)` |
| `converters.mil.mil.types.type_mapping.types_uint1` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_uint16` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_uint2` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_uint3` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_uint32` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_uint4` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_uint6` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_uint64` | Object | `` |
| `converters.mil.mil.types.type_mapping.types_uint8` | Object | `` |
| `converters.mil.mil.types.type_mapping.unknown` | Class | `(val)` |
| `converters.mil.mil.types.type_spec.FunctionType` | Class | `(inputs, output, python_function)` |
| `converters.mil.mil.types.type_spec.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_state.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_state.get_type_info` | Function | `(t)` |
| `converters.mil.mil.types.type_state.is_state` | Function | `(t)` |
| `converters.mil.mil.types.type_state.memoize` | Function | `(f)` |
| `converters.mil.mil.types.type_state.state` | Function | `(state_type)` |
| `converters.mil.mil.types.type_str.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_str.annotate` | Function | `(return_type, kwargs)` |
| `converters.mil.mil.types.type_str.class_annotate` | Function | `()` |
| `converters.mil.mil.types.type_str.delay_type` | Object | `` |
| `converters.mil.mil.types.type_str.str` | Class | `(v)` |
| `converters.mil.mil.types.type_tensor.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_tensor.builtin_to_string` | Function | `(builtin_type)` |
| `converters.mil.mil.types.type_tensor.canonical_shape` | Function | `(shape)` |
| `converters.mil.mil.types.type_tensor.get_type_info` | Function | `(t)` |
| `converters.mil.mil.types.type_tensor.is_compatible_type` | Function | `(type1, type2)` |
| `converters.mil.mil.types.type_tensor.is_subtype` | Function | `(type1, type2)` |
| `converters.mil.mil.types.type_tensor.is_symbolic` | Function | `(val)` |
| `converters.mil.mil.types.type_tensor.is_tensor` | Function | `(tensor_type)` |
| `converters.mil.mil.types.type_tensor.is_tensor_and_is_compatible` | Function | `(tensor_type1, tensor_type2, allow_promotion)` |
| `converters.mil.mil.types.type_tensor.logger` | Object | `` |
| `converters.mil.mil.types.type_tensor.memoize` | Function | `(f)` |
| `converters.mil.mil.types.type_tensor.nptype_from_builtin` | Function | `(btype)` |
| `converters.mil.mil.types.type_tensor.numpy_type_to_builtin_type` | Function | `(nptype) -> type` |
| `converters.mil.mil.types.type_tensor.promote_types` | Function | `(dtype1, dtype2)` |
| `converters.mil.mil.types.type_tensor.tensor` | Function | `(primitive, shape)` |
| `converters.mil.mil.types.type_tensor.tensor_has_complete_shape` | Function | `(tensor_type)` |
| `converters.mil.mil.types.type_to_builtin_type` | Function | `(type)` |
| `converters.mil.mil.types.type_tuple.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_tuple.annotate` | Function | `(return_type, kwargs)` |
| `converters.mil.mil.types.type_tuple.empty_list` | Class | `(...)` |
| `converters.mil.mil.types.type_tuple.get_type_info` | Function | `(t)` |
| `converters.mil.mil.types.type_tuple.memoize` | Function | `(f)` |
| `converters.mil.mil.types.type_tuple.tuple` | Function | `(args)` |
| `converters.mil.mil.types.type_tuple.type_int` | Object | `` |
| `converters.mil.mil.types.type_tuple.type_unknown` | Object | `` |
| `converters.mil.mil.types.type_unknown.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_unknown.unknown` | Class | `(val)` |
| `converters.mil.mil.types.type_void.Type` | Class | `(name, tparam, python_class)` |
| `converters.mil.mil.types.type_void.void` | Class | `(...)` |
| `converters.mil.mil.types.uint` | Object | `` |
| `converters.mil.mil.types.uint1` | Object | `` |
| `converters.mil.mil.types.uint16` | Object | `` |
| `converters.mil.mil.types.uint2` | Object | `` |
| `converters.mil.mil.types.uint3` | Object | `` |
| `converters.mil.mil.types.uint32` | Object | `` |
| `converters.mil.mil.types.uint4` | Object | `` |
| `converters.mil.mil.types.uint6` | Object | `` |
| `converters.mil.mil.types.uint64` | Object | `` |
| `converters.mil.mil.types.uint8` | Object | `` |
| `converters.mil.mil.types.unknown` | Class | `(val)` |
| `converters.mil.mil.types.void` | Class | `(...)` |
| `converters.mil.mil.utils.CacheDoublyLinkedList` | Class | `(array: Optional[List[Operation]])` |
| `converters.mil.mil.utils.OpNode` | Class | `(op: Operation)` |
| `converters.mil.mil.utils.Operation` | Class | `(kwargs)` |
| `converters.mil.mil.var.ComplexVar` | Class | `(name, sym_type, sym_val, op, op_output_idx, real: Optional[Var], imag: Optional[Var])` |
| `converters.mil.mil.var.InternalVar` | Class | `(val, name)` |
| `converters.mil.mil.var.ListVar` | Class | `(name, elem_type, init_length, dynamic_length, sym_val, kwargs)` |
| `converters.mil.mil.var.ScopeSource` | Class | `(...)` |
| `converters.mil.mil.var.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil.var.any_symbolic` | Function | `(val)` |
| `converters.mil.mil.var.types` | Object | `` |
| `converters.mil.mil.visitors.dot_visitor.DotVisitor` | Class | `(annotation)` |
| `converters.mil.mil.visitors.dot_visitor.Var` | Class | `(name, sym_type, sym_val, op, op_output_idx)` |
| `converters.mil.mil_list` | Class | `(ls)` |
| `converters.mil.register_op` | Object | `` |
| `converters.mil.register_tf_op` | Function | `(_func, tf_alias, override, strict)` |
| `converters.mil.register_torch_op` | Function | `(_func, torch_alias, override)` |
| `converters.mil.test_inputs_outputs_shape.MSG_TF2_NOT_FOUND` | Object | `` |
| `converters.mil.test_inputs_outputs_shape.MSG_TORCH_NOT_FOUND` | Object | `` |
| `converters.mil.test_inputs_outputs_shape.TestConvModule` | Class | `(in_channels, out_channels, kernel_size)` |
| `converters.mil.test_inputs_outputs_shape.TestFlexibleInputShapesTF` | Class | `(...)` |
| `converters.mil.test_inputs_outputs_shape.TestFlexibleInputShapesTorch` | Class | `(...)` |
| `converters.mil.test_inputs_outputs_shape.TestOutputShapes` | Class | `(...)` |
| `converters.mil.test_inputs_outputs_shape.TestSimpleModule` | Class | `(...)` |
| `converters.mil.test_inputs_outputs_shape.backends` | Object | `` |
| `converters.mil.test_inputs_outputs_shape.compute_units` | Object | `` |
| `converters.mil.test_inputs_outputs_shape.mb` | Class | `(...)` |
| `converters.mil.testing_reqs.BackendConfig` | Class | `(...)` |
| `converters.mil.testing_reqs.backends` | Object | `` |
| `converters.mil.testing_reqs.backends_internal` | Object | `` |
| `converters.mil.testing_reqs.clean_up_backends` | Function | `(backends: List[BackendConfig], minimum_opset_version: ct.target, force_include_iOS15_test: bool) -> List[BackendConfig]` |
| `converters.mil.testing_reqs.compute_units` | Object | `` |
| `converters.mil.testing_reqs.cur_str_val` | Object | `` |
| `converters.mil.testing_reqs.macos_compatible_with_deployment_target` | Function | `(minimum_deployment_target)` |
| `converters.mil.testing_reqs.precisions` | Object | `` |
| `converters.mil.testing_reqs.targets` | Object | `` |
| `converters.mil.testing_reqs.tf` | Object | `` |
| `converters.mil.testing_reqs.torch` | Object | `` |
| `converters.mil.testing_utils.AbstractGraphPass` | Class | `(...)` |
| `converters.mil.testing_utils.Block` | Class | `(block_inputs, outer_op, name)` |
| `converters.mil.testing_utils.Function` | Class | `(inputs, opset_version)` |
| `converters.mil.testing_utils.IOS_TO_MINIMUM_MACOS_VERSION` | Object | `` |
| `converters.mil.testing_utils.PASS_REGISTRY` | Object | `` |
| `converters.mil.testing_utils.Program` | Class | `()` |
| `converters.mil.testing_utils.ScopeSource` | Class | `(...)` |
| `converters.mil.testing_utils.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `converters.mil.testing_utils.assert_cast_ops_count` | Function | `(mlmodel, expected_count)` |
| `converters.mil.testing_utils.assert_input_dtype` | Function | `(mlmodel, expected_type_str, expected_name, index)` |
| `converters.mil.testing_utils.assert_model_is_valid` | Function | `(program, inputs, backend, verbose, expected_output_shapes, minimum_deployment_target: ct.target)` |
| `converters.mil.testing_utils.assert_numerical_value` | Function | `(mil_var, expected_value)` |
| `converters.mil.testing_utils.assert_op_count_match` | Function | `(program, expect, op, verbose)` |
| `converters.mil.testing_utils.assert_ops_in_mil_program` | Function | `(mlmodel, expected_op_list)` |
| `converters.mil.testing_utils.assert_output_dtype` | Function | `(mlmodel, expected_type_str, expected_name, index)` |
| `converters.mil.testing_utils.assert_prog_input_type` | Function | `(prog, expected_dtype_str, expected_name, index)` |
| `converters.mil.testing_utils.assert_prog_output_type` | Function | `(prog, expected_dtype_str, expected_name, index)` |
| `converters.mil.testing_utils.assert_same_input_names` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.testing_utils.assert_same_input_types` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.testing_utils.assert_same_output_names` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.testing_utils.assert_same_output_shapes` | Function | `(prog1, prog2, func_name)` |
| `converters.mil.testing_utils.assert_same_output_types` | Function | `(prog1: Program, prog2: Program, func_name: str)` |
| `converters.mil.testing_utils.assert_spec_input_image_type` | Function | `(spec, expected_feature_type)` |
| `converters.mil.testing_utils.assert_spec_input_type` | Function | `(spec, expected_feature_type, expected_name, index)` |
| `converters.mil.testing_utils.assert_spec_output_image_type` | Function | `(spec, expected_feature_type)` |
| `converters.mil.testing_utils.assert_spec_output_type` | Function | `(spec, expected_feature_type, expected_name, index)` |
| `converters.mil.testing_utils.compare_backend` | Function | `(mlmodel, input_key_values, expected_outputs, dtype, atol, rtol, also_compare_shapes, state, allow_mismatch_ratio)` |
| `converters.mil.testing_utils.compare_shapes` | Function | `(mlmodel, input_key_values, expected_outputs, pred)` |
| `converters.mil.testing_utils.compute_snr_and_psnr` | Function | `(x, y)` |
| `converters.mil.testing_utils.coremltoolsutils` | Object | `` |
| `converters.mil.testing_utils.ct_convert` | Function | `(program, source, inputs, outputs, classifier_config, minimum_deployment_target, convert_to, compute_precision, skip_model_load, converter, kwargs)` |
| `converters.mil.testing_utils.debug_save_mlmodel_config_file_name` | Object | `` |
| `converters.mil.testing_utils.debug_save_mlmodels` | Object | `` |
| `converters.mil.testing_utils.einsum_equations` | Object | `` |
| `converters.mil.testing_utils.gen_activation_stats_for_program` | Function | `(prog)` |
| `converters.mil.testing_utils.gen_input_shapes_einsum` | Function | `(equation: str, dynamic: bool, backend: Tuple[str, str])` |
| `converters.mil.testing_utils.get_core_ml_prediction` | Function | `(build, input_placeholders, input_values, backend, compute_unit)` |
| `converters.mil.testing_utils.get_op_names_in_program` | Function | `(prog, func_name, skip_const_ops)` |
| `converters.mil.testing_utils.get_op_types_in_block` | Function | `(block: Block, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.testing_utils.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `converters.mil.testing_utils.hardcoded_einsum_equations` | Object | `` |
| `converters.mil.testing_utils.lines` | Object | `` |
| `converters.mil.testing_utils.macos_compatible_with_deployment_target` | Function | `(minimum_deployment_target)` |
| `converters.mil.testing_utils.mil` | Object | `` |
| `converters.mil.testing_utils.proto` | Object | `` |
| `converters.mil.testing_utils.random_gen` | Function | `(shape, rand_min, rand_max, eps_from_int, allow_duplicate, dtype)` |
| `converters.mil.testing_utils.random_gen_input_feature_type` | Function | `(input_desc)` |
| `converters.mil.testing_utils.run_core_ml_predict` | Function | `(mlmodel, input_key_values, state)` |
| `converters.mil.testing_utils.ssa_fn` | Function | `(func)` |
| `converters.mil.testing_utils.str_to_proto_feature_type` | Function | `(dtype: str) -> proto.FeatureTypes_pb2.ArrayFeatureType` |
| `converters.mil.testing_utils.to_tuple` | Function | `(v)` |
| `converters.mil.testing_utils.validate_minimum_deployment_target` | Function | `(minimum_deployment_target: ct.target, backend: Tuple[str, str])` |
| `converters.mil.testing_utils.verify_prediction` | Function | `(mlmodel, multiarray_type)` |
| `converters.sklearn.convert` | Function | `(sk_obj, input_features, output_feature_names)` |
| `converters.xgboost.convert` | Function | `(model, feature_names, target, force_32bit_float, mode, class_labels, n_classes)` |
| `models.CompiledMLModel` | Class | `(path: str, compute_units: _ComputeUnit, function_name: _Optional[str], optimization_hints: _Optional[dict], asset: _Optional[_MLModelAsset])` |
| `models.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `models.array_feature_extractor.SPECIFICATION_VERSION` | Object | `` |
| `models.array_feature_extractor.create_array_feature_extractor` | Function | `(input_features, output_name, extract_indices, output_type)` |
| `models.array_feature_extractor.datatypes` | Object | `` |
| `models.array_feature_extractor.proto` | Object | `` |
| `models.array_feature_extractor.set_transform_interface_params` | Function | `(spec, input_features, output_features, are_optional, training_features, array_datatype)` |
| `models.compute_device.MLCPUComputeDevice` | Class | `(proxy)` |
| `models.compute_device.MLComputeDevice` | Class | `(...)` |
| `models.compute_device.MLGPUComputeDevice` | Class | `(proxy)` |
| `models.compute_device.MLNeuralEngineComputeDevice` | Class | `(proxy)` |
| `models.compute_plan.MLComputePlan` | Class | `(proxy)` |
| `models.compute_plan.MLComputePlanCost` | Class | `(weight: float)` |
| `models.compute_plan.MLComputePlanDeviceUsage` | Class | `(preferred_compute_device: _MLComputeDevice, supported_compute_devices: _List[_MLComputeDevice])` |
| `models.compute_plan.MLModelStructure` | Class | `(neuralnetwork: _Optional[MLModelStructureNeuralNetwork], program: _Optional[MLModelStructureProgram], pipeline: _Optional[MLModelStructurePipeline])` |
| `models.compute_plan.MLModelStructureNeuralNetwork` | Class | `(layers: _List[MLModelStructureNeuralNetworkLayer])` |
| `models.compute_plan.MLModelStructureNeuralNetworkLayer` | Class | `(name: str, type: str, input_names: _List[str], output_names: _List[str], __proxy__: _Any)` |
| `models.compute_plan.MLModelStructurePipeline` | Class | `(sub_models: _Tuple[str, MLModelStructure])` |
| `models.compute_plan.MLModelStructureProgram` | Class | `(functions: _Dict[str, MLModelStructureProgramFunction])` |
| `models.compute_plan.MLModelStructureProgramArgument` | Class | `(bindings: _List[MLModelStructureProgramBinding])` |
| `models.compute_plan.MLModelStructureProgramBinding` | Class | `(name: _Optional[str], value: _Optional[MLModelStructureProgramValue])` |
| `models.compute_plan.MLModelStructureProgramBlock` | Class | `(inputs: _List[MLModelStructureProgramNamedValueType], operations: _List[MLModelStructureProgramOperation], output_names: _List[str])` |
| `models.compute_plan.MLModelStructureProgramFunction` | Class | `(inputs: _List[MLModelStructureProgramNamedValueType], block: MLModelStructureProgramBlock)` |
| `models.compute_plan.MLModelStructureProgramNamedValueType` | Class | `(name: str, type: MLModelStructureProgramValueType)` |
| `models.compute_plan.MLModelStructureProgramOperation` | Class | `(inputs: _Dict[str, MLModelStructureProgramArgument], operator_name: str, outputs: _List[MLModelStructureProgramNamedValueType], blocks: _List[MLModelStructureProgramBlock], __proxy__: _Any)` |
| `models.compute_plan.MLModelStructureProgramValue` | Class | `()` |
| `models.compute_plan.MLModelStructureProgramValueType` | Class | `()` |
| `models.datatypes.Array` | Class | `(dimensions)` |
| `models.datatypes.Dictionary` | Class | `(key_type)` |
| `models.datatypes.Double` | Class | `()` |
| `models.datatypes.Int64` | Class | `()` |
| `models.datatypes.String` | Class | `()` |
| `models.datatypes.proto` | Object | `` |
| `models.feature_vectorizer.SPECIFICATION_VERSION` | Object | `` |
| `models.feature_vectorizer.create_feature_vectorizer` | Function | `(input_features, output_feature_name, known_size_map)` |
| `models.feature_vectorizer.datatypes` | Object | `` |
| `models.feature_vectorizer.is_valid_feature_list` | Function | `(features)` |
| `models.feature_vectorizer.process_or_validate_features` | Function | `(features, num_dimensions, feature_type_map)` |
| `models.feature_vectorizer.proto` | Object | `` |
| `models.feature_vectorizer.set_transform_interface_params` | Function | `(spec, input_features, output_features, are_optional, training_features, array_datatype)` |
| `models.ml_program.compression_utils.affine_quantize_weights` | Function | `(mlmodel, mode, op_selector, dtype)` |
| `models.ml_program.compression_utils.decompress_weights` | Function | `(mlmodel)` |
| `models.ml_program.compression_utils.palettize_weights` | Function | `(mlmodel, nbits, mode, op_selector, lut_function)` |
| `models.ml_program.compression_utils.sparsify_weights` | Function | `(mlmodel, mode, threshold, target_percentile, op_selector)` |
| `models.ml_program.experimental.async_wrapper.ComputeUnit` | Class | `(...)` |
| `models.ml_program.experimental.async_wrapper.Device` | Class | `(name: str, type: DeviceType, identifier: str, udid: str, os_version: str, os_build_number: str, developer_mode_state: str, state: DeviceState, session: _Optional[_Type[_AppSession]])` |
| `models.ml_program.experimental.async_wrapper.LocalMLModelAsyncWrapper` | Class | `(spec_or_path: Union[proto.Model_pb2.Model, str], weights_dir: str, compute_units: ComputeUnit, function_name: Optional[str], optimization_hints: Optional[Dict[str, Any]])` |
| `models.ml_program.experimental.async_wrapper.MLComputePlan` | Class | `(proxy)` |
| `models.ml_program.experimental.async_wrapper.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `models.ml_program.experimental.async_wrapper.MLModelAsyncWrapper` | Class | `(spec_or_path: Union[proto.Model_pb2.Model, str], weights_dir: str, compute_units: ComputeUnit, function_name: Optional[str], optimization_hints: Optional[Dict[str, Any]])` |
| `models.ml_program.experimental.async_wrapper.MLModelStructure` | Class | `(neuralnetwork: _Optional[MLModelStructureNeuralNetwork], program: _Optional[MLModelStructureProgram], pipeline: _Optional[MLModelStructurePipeline])` |
| `models.ml_program.experimental.async_wrapper.MLState` | Class | `(proxy)` |
| `models.ml_program.experimental.async_wrapper.RemoteMLModelAsyncWrapper` | Class | `(spec_or_path: Union[proto.Model_pb2.Model, str], weights_dir: str, device: Device, compute_units: ComputeUnit, function_name: Optional[str], optimization_hints: Optional[Dict[str, Any]])` |
| `models.ml_program.experimental.async_wrapper.compile_model` | Function | `(model: _Union[_proto.Model_pb2.Model, str], destination_path: _Optional[str]) -> str` |
| `models.ml_program.experimental.async_wrapper.proto` | Object | `` |
| `models.ml_program.experimental.compute_plan_utils.ComputePlan` | Class | `(infos: _Dict[ModelStructurePath, ComputePlan.OperationOrLayerInfo])` |
| `models.ml_program.experimental.compute_plan_utils.ComputeUnit` | Class | `(...)` |
| `models.ml_program.experimental.compute_plan_utils.Device` | Class | `(name: str, type: DeviceType, identifier: str, udid: str, os_version: str, os_build_number: str, developer_mode_state: str, state: DeviceState, session: _Optional[_Type[_AppSession]])` |
| `models.ml_program.experimental.compute_plan_utils.MLCPUComputeDevice` | Class | `(proxy)` |
| `models.ml_program.experimental.compute_plan_utils.MLComputeDevice` | Class | `(...)` |
| `models.ml_program.experimental.compute_plan_utils.MLComputePlan` | Class | `(proxy)` |
| `models.ml_program.experimental.compute_plan_utils.MLComputePlanCost` | Class | `(weight: float)` |
| `models.ml_program.experimental.compute_plan_utils.MLComputePlanDeviceUsage` | Class | `(preferred_compute_device: _MLComputeDevice, supported_compute_devices: _List[_MLComputeDevice])` |
| `models.ml_program.experimental.compute_plan_utils.MLGPUComputeDevice` | Class | `(proxy)` |
| `models.ml_program.experimental.compute_plan_utils.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `models.ml_program.experimental.compute_plan_utils.MLModelStructure` | Class | `(neuralnetwork: _Optional[MLModelStructureNeuralNetwork], program: _Optional[MLModelStructureProgram], pipeline: _Optional[MLModelStructurePipeline])` |
| `models.ml_program.experimental.compute_plan_utils.MLModelStructureNeuralNetworkLayer` | Class | `(name: str, type: str, input_names: _List[str], output_names: _List[str], __proxy__: _Any)` |
| `models.ml_program.experimental.compute_plan_utils.MLModelStructureProgramOperation` | Class | `(inputs: _Dict[str, MLModelStructureProgramArgument], operator_name: str, outputs: _List[MLModelStructureProgramNamedValueType], blocks: _List[MLModelStructureProgramBlock], __proxy__: _Any)` |
| `models.ml_program.experimental.compute_plan_utils.MLNeuralEngineComputeDevice` | Class | `(proxy)` |
| `models.ml_program.experimental.compute_plan_utils.apply_compute_plan` | Function | `(model: MLModel, compute_plan: MLComputePlan, backend_assignment_fn: Optional[Callable[[proto.MIL_pb2.Operation, Optional[MLComputePlanDeviceUsage]], Optional[List[str]]]]) -> MLModel` |
| `models.ml_program.experimental.compute_plan_utils.load_compute_plan_from_path_on_device` | Function | `(path: str, compute_units: ComputeUnit, device: Optional[Device]) -> MLComputePlan` |
| `models.ml_program.experimental.compute_plan_utils.map_model_spec_to_path` | Function | `(model_spec: _proto.Model_pb2.Model, components: _List[ModelStructurePath.Component]) -> _List[_Tuple[_proto.MIL_pb2.Operation, ModelStructurePath]]` |
| `models.ml_program.experimental.compute_plan_utils.map_model_structure_to_path` | Function | `(model_structure: MLModelStructure, components: _List[ModelStructurePath.Component]) -> _List[_Tuple[ModelStructureLayerOrOperation, ModelStructurePath]]` |
| `models.ml_program.experimental.compute_plan_utils.proto` | Object | `` |
| `models.ml_program.experimental.compute_plan_utils.set_intended_backends` | Function | `(model: MLModel, backend_assignment_fn: Callable[[proto.MIL_pb2.Operation], Optional[List[str]]]) -> MLModel` |
| `models.ml_program.experimental.compute_plan_utils.set_intended_backends_attr` | Function | `(op: proto.MIL_pb2.Operation, backend_names: Optional[List[str]])` |
| `models.ml_program.experimental.debugging_utils.ComputeUnit` | Class | `(...)` |
| `models.ml_program.experimental.debugging_utils.Device` | Class | `(name: str, type: DeviceType, identifier: str, udid: str, os_version: str, os_build_number: str, developer_mode_state: str, state: DeviceState, session: _Optional[_Type[_AppSession]])` |
| `models.ml_program.experimental.debugging_utils.K` | Object | `` |
| `models.ml_program.experimental.debugging_utils.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `models.ml_program.experimental.debugging_utils.MLModelAsyncWrapper` | Class | `(spec_or_path: Union[proto.Model_pb2.Model, str], weights_dir: str, compute_units: ComputeUnit, function_name: Optional[str], optimization_hints: Optional[Dict[str, Any]])` |
| `models.ml_program.experimental.debugging_utils.MLModelComparator` | Class | `(reference_model: MLModel, target_model: MLModel, function_name: Optional[str], optimization_hints: Optional[Dict[str, Any]], num_predict_intermediate_outputs: int, reference_device: Optional[Device], target_device: Optional[Device])` |
| `models.ml_program.experimental.debugging_utils.MLModelInspector` | Class | `(model: MLModel, compute_units: ComputeUnit, function_name: Optional[str], optimization_hints: Optional[Dict[str, Any]], device: Optional[Device])` |
| `models.ml_program.experimental.debugging_utils.MLModelValidator` | Class | `(model: MLModel, function_name: Optional[str], compute_units: ComputeUnit, optimization_hints: Optional[Dict[str, Any]], num_predict_intermediate_outputs: int, device: Optional[Device])` |
| `models.ml_program.experimental.debugging_utils.V` | Object | `` |
| `models.ml_program.experimental.debugging_utils.compute_snr_and_psnr` | Function | `(x: np.array, y: np.array) -> Tuple[float, float]` |
| `models.ml_program.experimental.debugging_utils.proto` | Object | `` |
| `models.ml_program.experimental.debugging_utils.skip_op_by_type` | Function | `(op: proto.MIL_pb2.Operation, op_types: Iterable[str]) -> bool` |
| `models.ml_program.experimental.model_structure_path.MLModelStructure` | Class | `(neuralnetwork: _Optional[MLModelStructureNeuralNetwork], program: _Optional[MLModelStructureProgram], pipeline: _Optional[MLModelStructurePipeline])` |
| `models.ml_program.experimental.model_structure_path.MLModelStructureNeuralNetwork` | Class | `(layers: _List[MLModelStructureNeuralNetworkLayer])` |
| `models.ml_program.experimental.model_structure_path.MLModelStructureNeuralNetworkLayer` | Class | `(name: str, type: str, input_names: _List[str], output_names: _List[str], __proxy__: _Any)` |
| `models.ml_program.experimental.model_structure_path.MLModelStructureProgram` | Class | `(functions: _Dict[str, MLModelStructureProgramFunction])` |
| `models.ml_program.experimental.model_structure_path.MLModelStructureProgramBlock` | Class | `(inputs: _List[MLModelStructureProgramNamedValueType], operations: _List[MLModelStructureProgramOperation], output_names: _List[str])` |
| `models.ml_program.experimental.model_structure_path.MLModelStructureProgramOperation` | Class | `(inputs: _Dict[str, MLModelStructureProgramArgument], operator_name: str, outputs: _List[MLModelStructureProgramNamedValueType], blocks: _List[MLModelStructureProgramBlock], __proxy__: _Any)` |
| `models.ml_program.experimental.model_structure_path.ModelStructureLayerOrOperation` | Object | `` |
| `models.ml_program.experimental.model_structure_path.ModelStructurePath` | Class | `(components: _Tuple[Component, ...])` |
| `models.ml_program.experimental.model_structure_path.map_model_spec_to_path` | Function | `(model_spec: _proto.Model_pb2.Model, components: _List[ModelStructurePath.Component]) -> _List[_Tuple[_proto.MIL_pb2.Operation, ModelStructurePath]]` |
| `models.ml_program.experimental.model_structure_path.map_model_structure_to_path` | Function | `(model_structure: MLModelStructure, components: _List[ModelStructurePath.Component]) -> _List[_Tuple[ModelStructureLayerOrOperation, ModelStructurePath]]` |
| `models.ml_program.experimental.perf_utils.ComputeUnit` | Class | `(...)` |
| `models.ml_program.experimental.perf_utils.Device` | Class | `(name: str, type: DeviceType, identifier: str, udid: str, os_version: str, os_build_number: str, developer_mode_state: str, state: DeviceState, session: _Optional[_Type[_AppSession]])` |
| `models.ml_program.experimental.perf_utils.MLComputePlanDeviceUsage` | Class | `(preferred_compute_device: _MLComputeDevice, supported_compute_devices: _List[_MLComputeDevice])` |
| `models.ml_program.experimental.perf_utils.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `models.ml_program.experimental.perf_utils.MLModelAsyncWrapper` | Class | `(spec_or_path: Union[proto.Model_pb2.Model, str], weights_dir: str, compute_units: ComputeUnit, function_name: Optional[str], optimization_hints: Optional[Dict[str, Any]])` |
| `models.ml_program.experimental.perf_utils.MLModelBenchmarker` | Class | `(model: MLModel, device: Optional[Device])` |
| `models.ml_program.experimental.perf_utils.ModelStructurePath` | Class | `(components: _Tuple[Component, ...])` |
| `models.ml_program.experimental.perf_utils.map_model_spec_to_path` | Function | `(model_spec: _proto.Model_pb2.Model, components: _List[ModelStructurePath.Component]) -> _List[_Tuple[_proto.MIL_pb2.Operation, ModelStructurePath]]` |
| `models.ml_program.experimental.perf_utils.map_model_structure_to_path` | Function | `(model_structure: MLModelStructure, components: _List[ModelStructurePath.Component]) -> _List[_Tuple[ModelStructureLayerOrOperation, ModelStructurePath]]` |
| `models.ml_program.experimental.perf_utils.proto` | Object | `` |
| `models.ml_program.experimental.remote_device.AppSigningCredentials` | Class | `(development_team: _Optional[str], provisioning_profile_uuid: _Optional[str], bundle_identifier: _Optional[str])` |
| `models.ml_program.experimental.remote_device.ComputePlan` | Class | `(infos: _Dict[ModelStructurePath, ComputePlan.OperationOrLayerInfo])` |
| `models.ml_program.experimental.remote_device.Device` | Class | `(name: str, type: DeviceType, identifier: str, udid: str, os_version: str, os_build_number: str, developer_mode_state: str, state: DeviceState, session: _Optional[_Type[_AppSession]])` |
| `models.ml_program.experimental.remote_device.DeviceState` | Class | `(...)` |
| `models.ml_program.experimental.remote_device.DeviceType` | Class | `(...)` |
| `models.ml_program.experimental.remote_device.ModelStructurePath` | Class | `(components: _Tuple[Component, ...])` |
| `models.ml_program.experimental.torch.debugging_utils.Device` | Class | `(name: str, type: DeviceType, identifier: str, udid: str, os_version: str, os_build_number: str, developer_mode_state: str, state: DeviceState, session: _Optional[_Type[_AppSession]])` |
| `models.ml_program.experimental.torch.debugging_utils.FrameInfo` | Class | `(filename: str, lineno: int)` |
| `models.ml_program.experimental.torch.debugging_utils.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `models.ml_program.experimental.torch.debugging_utils.MLModelInspector` | Class | `(model: MLModel, compute_units: ComputeUnit, function_name: Optional[str], optimization_hints: Optional[Dict[str, Any]], device: Optional[Device])` |
| `models.ml_program.experimental.torch.debugging_utils.ScopeSource` | Class | `(...)` |
| `models.ml_program.experimental.torch.debugging_utils.TorchExportMLModelComparator` | Class | `(model: ExportedProgram, num_predict_intermediate_outputs: int, target_device: Optional[Device], converter_kwargs)` |
| `models.ml_program.experimental.torch.debugging_utils.TorchNode` | Object | `` |
| `models.ml_program.experimental.torch.debugging_utils.TorchNodeToMILOperationMapping` | Class | `(node_to_operations_map: Dict[TorchNode, List[proto.MIL_pb2.Operation]], operation_output_name_to_node_map: Dict[str, TorchNode])` |
| `models.ml_program.experimental.torch.debugging_utils.TorchScriptMLModelComparator` | Class | `(model: torch.nn.Module, example_inputs: Tuple[torch.tensor], num_predict_intermediate_outputs: int, target_device: Optional[Device], converter_kwargs)` |
| `models.ml_program.experimental.torch.debugging_utils.TorchScriptModuleAnnotator` | Class | `(name_prefix: str)` |
| `models.ml_program.experimental.torch.debugging_utils.TorchScriptModuleInfo` | Class | `(name: str, call_sequence: int, hierarchy: Tuple[str], input_names: Tuple[str], output_names: Tuple[str], submodules: Tuple[Key], code: str)` |
| `models.ml_program.experimental.torch.debugging_utils.TorchScriptModuleMappingInfo` | Class | `(source: TorchScriptModuleInfo, source_to_target_ops_mapping: OrderedDict[TorchScriptNodeInfo:(List[proto.MIL_pb2.Operation])], deps: Dict[TorchScriptModuleInfo.Key, Iterable[TorchScriptModuleMappingInfo]], outputs: List[proto.MIL_pb2.Operation], submodules: List[TorchScriptModuleMappingInfo])` |
| `models.ml_program.experimental.torch.debugging_utils.TorchScriptNodeInfo` | Class | `(source_range: str, modules: Tuple[TorchScriptModuleInfo.Key], desc: str, kind: str, input_names: Tuple[str], output_names: Tuple[str])` |
| `models.ml_program.experimental.torch.debugging_utils.convert_and_retrieve_op_mapping` | Function | `(model: Union[ExportedProgram, ScriptModule], converter_kwargs) -> Tuple[MLModel, TorchNodeToMILOperationMapping]` |
| `models.ml_program.experimental.torch.debugging_utils.get_stack_frame_infos` | Function | `(node: TorchNode) -> Optional[List[FrameInfo]]` |
| `models.ml_program.experimental.torch.debugging_utils.inline_and_annotate_module` | Function | `(model: ScriptModule, name_prefix: str) -> TorchScriptModuleAnnotator` |
| `models.ml_program.experimental.torch.debugging_utils.proto` | Object | `` |
| `models.ml_program.experimental.torch.perf_utils.Device` | Class | `(name: str, type: DeviceType, identifier: str, udid: str, os_version: str, os_build_number: str, developer_mode_state: str, state: DeviceState, session: _Optional[_Type[_AppSession]])` |
| `models.ml_program.experimental.torch.perf_utils.MLModelBenchmarker` | Class | `(model: MLModel, device: Optional[Device])` |
| `models.ml_program.experimental.torch.perf_utils.TorchMLModelBenchmarker` | Class | `(model: Union[ExportedProgram, torch.jit.ScriptModule], device: Optional[Device], converter_kwargs)` |
| `models.ml_program.experimental.torch.perf_utils.TorchNode` | Object | `` |
| `models.ml_program.experimental.torch.perf_utils.TorchScriptNodeInfo` | Class | `(source_range: str, modules: Tuple[TorchScriptModuleInfo.Key], desc: str, kind: str, input_names: Tuple[str], output_names: Tuple[str])` |
| `models.ml_program.experimental.torch.perf_utils.proto` | Object | `` |
| `models.model.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `models.model.MLModelAsset` | Class | `(proxy)` |
| `models.model.MLState` | Class | `(proxy)` |
| `models.model.logger` | Object | `` |
| `models.nearest_neighbors.KNearestNeighborsClassifierBuilder` | Class | `(input_name, output_name, number_of_dimensions, default_class_label, kwargs)` |
| `models.nearest_neighbors.builder.KNearestNeighborsClassifierBuilder` | Class | `(input_name, output_name, number_of_dimensions, default_class_label, kwargs)` |
| `models.nearest_neighbors.builder.datatypes` | Object | `` |
| `models.nearest_neighbors.builder.proto` | Object | `` |
| `models.neural_network.AdamParams` | Class | `(lr, batch, beta1, beta2, eps)` |
| `models.neural_network.NeuralNetworkBuilder` | Class | `(input_features, output_features, mode, spec, nn_spec, disable_rank5_shape_mapping, training_features, use_float_arraytype)` |
| `models.neural_network.SgdParams` | Class | `(lr, batch, momentum)` |
| `models.neural_network.builder.AdamParams` | Class | `(lr, batch, beta1, beta2, eps)` |
| `models.neural_network.builder.NeuralNetworkBuilder` | Class | `(input_features, output_features, mode, spec, nn_spec, disable_rank5_shape_mapping, training_features, use_float_arraytype)` |
| `models.neural_network.builder.SgdParams` | Class | `(lr, batch, momentum)` |
| `models.neural_network.builder.datatypes` | Object | `` |
| `models.neural_network.builder.set_training_features` | Function | `(spec, training_features)` |
| `models.neural_network.builder.set_transform_interface_params` | Function | `(spec, input_features, output_features, are_optional, training_features, array_datatype)` |
| `models.neural_network.flexible_shape_utils.NeuralNetworkImageSize` | Class | `(height, width)` |
| `models.neural_network.flexible_shape_utils.NeuralNetworkImageSizeRange` | Class | `(height_range, width_range)` |
| `models.neural_network.flexible_shape_utils.NeuralNetworkMultiArrayShape` | Class | `(channel, height, width)` |
| `models.neural_network.flexible_shape_utils.NeuralNetworkMultiArrayShapeRange` | Class | `(input_ranges)` |
| `models.neural_network.flexible_shape_utils.Shape` | Class | `(shape_value)` |
| `models.neural_network.flexible_shape_utils.ShapeRange` | Class | `(lowerBound, upperBound)` |
| `models.neural_network.flexible_shape_utils.Size` | Class | `(size_value)` |
| `models.neural_network.flexible_shape_utils.add_enumerated_image_sizes` | Function | `(spec, feature_name, sizes)` |
| `models.neural_network.flexible_shape_utils.add_enumerated_multiarray_shapes` | Function | `(spec, feature_name, shapes)` |
| `models.neural_network.flexible_shape_utils.add_multiarray_ndshape_enumeration` | Function | `(spec, feature_name, enumerated_shapes)` |
| `models.neural_network.flexible_shape_utils.set_multiarray_ndshape_range` | Function | `(spec, feature_name, lower_bounds, upper_bounds)` |
| `models.neural_network.flexible_shape_utils.update_image_size_range` | Function | `(spec, feature_name, size_range)` |
| `models.neural_network.flexible_shape_utils.update_multiarray_shape_range` | Function | `(spec, feature_name, shape_range)` |
| `models.neural_network.printer.print_network_spec` | Function | `(mlmodel_spec, interface_only, style)` |
| `models.neural_network.quantization_utils.AdvancedQuantizedLayerSelector` | Class | `(skip_layer_types, minimum_conv_kernel_channels, minimum_conv_weight_count)` |
| `models.neural_network.quantization_utils.MatrixMultiplyLayerSelector` | Class | `(minimum_weight_count, minimum_input_channels, minimum_output_channels, maximum_input_channels, maximum_output_channels, include_layers_with_names)` |
| `models.neural_network.quantization_utils.ModelMetrics` | Class | `(spec)` |
| `models.neural_network.quantization_utils.NoiseMetrics` | Class | `()` |
| `models.neural_network.quantization_utils.OutputMetric` | Class | `(name, type)` |
| `models.neural_network.quantization_utils.QuantizedLayerSelector` | Class | `()` |
| `models.neural_network.quantization_utils.TopKMetrics` | Class | `(topk)` |
| `models.neural_network.quantization_utils.activate_int8_int8_matrix_multiplications` | Function | `(spec, selector)` |
| `models.neural_network.quantization_utils.compare_models` | Function | `(full_precision_model, quantized_model, sample_data)` |
| `models.neural_network.quantization_utils.quantize_weights` | Function | `(full_precision_model, nbits, quantization_mode, sample_data, kwargs)` |
| `models.neural_network.spec_inspection_utils.proto` | Object | `` |
| `models.neural_network.update_optimizer_utils.AdamParams` | Class | `(lr, batch, beta1, beta2, eps)` |
| `models.neural_network.update_optimizer_utils.Batch` | Class | `(value, allowed_set)` |
| `models.neural_network.update_optimizer_utils.RangeParam` | Class | `(value, min, max)` |
| `models.neural_network.update_optimizer_utils.SgdParams` | Class | `(lr, batch, momentum)` |
| `models.neural_network.utils.NeuralNetworkBuilder` | Class | `(input_features, output_features, mode, spec, nn_spec, disable_rank5_shape_mapping, training_features, use_float_arraytype)` |
| `models.neural_network.utils.make_image_input` | Function | `(model, input_name, is_bgr, red_bias, blue_bias, green_bias, gray_bias, scale, image_format)` |
| `models.neural_network.utils.make_nn_classifier` | Function | `(model, class_labels, predicted_feature_name, predicted_probabilities_output)` |
| `models.pipeline.Pipeline` | Class | `(input_features, output_features, training_features)` |
| `models.pipeline.PipelineClassifier` | Class | `(input_features, class_labels, output_features, training_features)` |
| `models.pipeline.PipelineRegressor` | Class | `(input_features, output_features, training_features)` |
| `models.pipeline.set_classifier_interface_params` | Function | `(spec, features, class_labels, model_accessor_for_class_labels, output_features, training_features)` |
| `models.pipeline.set_regressor_interface_params` | Function | `(spec, features, output_features, training_features)` |
| `models.pipeline.set_training_features` | Function | `(spec, training_features)` |
| `models.pipeline.set_transform_interface_params` | Function | `(spec, input_features, output_features, are_optional, training_features, array_datatype)` |
| `models.tree_ensemble.TreeEnsembleBase` | Class | `()` |
| `models.tree_ensemble.TreeEnsembleClassifier` | Class | `(features, class_labels, output_features)` |
| `models.tree_ensemble.TreeEnsembleRegressor` | Class | `(features, target)` |
| `models.tree_ensemble.set_classifier_interface_params` | Function | `(spec, features, class_labels, model_accessor_for_class_labels, output_features, training_features)` |
| `models.tree_ensemble.set_regressor_interface_params` | Function | `(spec, features, output_features, training_features)` |
| `models.utils.MultiFunctionDescriptor` | Class | `(model_path: _Optional[str])` |
| `models.utils.bisect_model` | Function | `(model: _Union[str, _ct.models.MLModel], output_dir: str, merge_chunks_to_pipeline: _Optional[bool], check_output_correctness: _Optional[bool])` |
| `models.utils.change_input_output_tensor_type` | Function | `(ml_model: _ct.models.MLModel, from_type: _proto.FeatureTypes_pb2.ArrayFeatureType, to_type: _proto.FeatureTypes_pb2.ArrayFeatureType, function_names: _Optional[_List[str]], input_names: _Optional[_List[str]], output_names: _Optional[_List[str]]) -> _ct.models.model.MLModel` |
| `models.utils.compile_model` | Function | `(model: _Union[_proto.Model_pb2.Model, str], destination_path: _Optional[str]) -> str` |
| `models.utils.convert_double_to_float_multiarray_type` | Function | `(spec)` |
| `models.utils.evaluate_classifier` | Function | `(model, data, target, verbose)` |
| `models.utils.evaluate_classifier_with_probabilities` | Function | `(model, data, probabilities, verbose)` |
| `models.utils.evaluate_regressor` | Function | `(model, data, target, verbose)` |
| `models.utils.evaluate_transformer` | Function | `(model, input_data, reference_output, verbose)` |
| `models.utils.load_spec` | Function | `(model_path: str) -> _proto.Model_pb2` |
| `models.utils.make_pipeline` | Function | `(models: _ct.models.MLModel, compute_units: _Union[None, _ct.ComputeUnit]) -> _ct.models.MLModel` |
| `models.utils.materialize_dynamic_shape_mlmodel` | Function | `(dynamic_shape_mlmodel: _ct.models.MLModel, function_name_to_materialization_map: _Dict[str, _Dict[str, _Tuple[int]]], destination_path: str, source_function_name: str) -> None` |
| `models.utils.randomize_weights` | Function | `(mlmodel: _ct.models.MLModel)` |
| `models.utils.rename_feature` | Function | `(spec, current_name, new_name, rename_inputs, rename_outputs)` |
| `models.utils.save_multifunction` | Function | `(desc: MultiFunctionDescriptor, destination_path: str)` |
| `models.utils.save_spec` | Function | `(spec, filename, auto_set_specification_version, weights_dir)` |
| `optimize.coreml.CoreMLOpMetaData` | Class | `(...)` |
| `optimize.coreml.CoreMLWeightMetaData` | Class | `(...)` |
| `optimize.coreml.OpLinearQuantizerConfig` | Class | `(...)` |
| `optimize.coreml.OpMagnitudePrunerConfig` | Class | `(...)` |
| `optimize.coreml.OpPalettizerConfig` | Class | `(...)` |
| `optimize.coreml.OpThresholdPrunerConfig` | Class | `(...)` |
| `optimize.coreml.OptimizationConfig` | Class | `(...)` |
| `optimize.coreml.decompress_weights` | Function | `(mlmodel: _model.MLModel)` |
| `optimize.coreml.experimental.OpActivationLinearQuantizerConfig` | Class | `(...)` |
| `optimize.coreml.experimental.linear_quantize_activations` | Function | `(mlmodel: ct.models.MLModel, config: _OptimizationConfig, sample_data: List[Dict[Optional[str], np.ndarray]], calibration_op_group_size: int)` |
| `optimize.coreml.experimental.test_post_training_quantization.TestActivationQuantization` | Class | `(...)` |
| `optimize.coreml.experimental.test_post_training_quantization.TestCompressionPasses` | Class | `(...)` |
| `optimize.coreml.experimental.test_post_training_quantization.TestGetActivationStats` | Class | `(...)` |
| `optimize.coreml.experimental.test_post_training_quantization.cto` | Object | `` |
| `optimize.coreml.get_weights_metadata` | Function | `(mlmodel: _model.MLModel, weight_threshold: int)` |
| `optimize.coreml.linear_quantize_activations` | Function | `(mlmodel: _model.MLModel, config: _OptimizationConfig, sample_data: List[Dict[Optional[str], np.ndarray]], calibration_op_group_size: int)` |
| `optimize.coreml.linear_quantize_weights` | Function | `(mlmodel: _model.MLModel, config: _OptimizationConfig, joint_compression: bool)` |
| `optimize.coreml.palettize_weights` | Function | `(mlmodel: _model.MLModel, config: _OptimizationConfig, joint_compression: bool)` |
| `optimize.coreml.prune_weights` | Function | `(mlmodel: _model.MLModel, config: _OptimizationConfig, joint_compression: bool)` |
| `optimize.torch.base_model_optimizer.BaseDataCalibratedModelOptimizer` | Class | `(model: _torch.nn.Module, config: _Optional[_OptimizationConfig])` |
| `optimize.torch.base_model_optimizer.BaseModelOptimizer` | Class | `(model: _torch.nn.Module, config: _Optional[_OptimizationConfig])` |
| `optimize.torch.base_model_optimizer.BasePostTrainingModelOptimizer` | Class | `(model: _torch.nn.Module, config: _Optional[_OptimizationConfig])` |
| `optimize.torch.base_model_optimizer.BaseTrainingTimeModelOptimizer` | Class | `(model: _torch.nn.Module, config: _Optional[_OptimizationConfig])` |
| `optimize.torch.layerwise_compression.DefaultInputCacher` | Class | `(...)` |
| `optimize.torch.layerwise_compression.FirstLayerInputCacher` | Class | `(model: _nn.Module, layers: str)` |
| `optimize.torch.layerwise_compression.GPTFirstLayerInputCacher` | Class | `(model: _nn.Module, layers: _Union[str, _List])` |
| `optimize.torch.layerwise_compression.GPTQ` | Class | `(layer: _nn.Module, config: ModuleGPTQConfig)` |
| `optimize.torch.layerwise_compression.LayerwiseCompressionAlgorithm` | Class | `(...)` |
| `optimize.torch.layerwise_compression.LayerwiseCompressionAlgorithmConfig` | Class | `(...)` |
| `optimize.torch.layerwise_compression.LayerwiseCompressor` | Class | `(model: _nn.Module, config: LayerwiseCompressorConfig)` |
| `optimize.torch.layerwise_compression.LayerwiseCompressorConfig` | Class | `(...)` |
| `optimize.torch.layerwise_compression.ModuleGPTQConfig` | Class | `(...)` |
| `optimize.torch.layerwise_compression.ModuleSparseGPTConfig` | Class | `(...)` |
| `optimize.torch.layerwise_compression.SparseGPT` | Class | `(layer: _nn.Module, config: ModuleSparseGPTConfig)` |
| `optimize.torch.layerwise_compression.algorithms.GPTQ` | Class | `(layer: _nn.Module, config: ModuleGPTQConfig)` |
| `optimize.torch.layerwise_compression.algorithms.LayerwiseCompressionAlgorithm` | Class | `(...)` |
| `optimize.torch.layerwise_compression.algorithms.LayerwiseCompressionAlgorithmConfig` | Class | `(...)` |
| `optimize.torch.layerwise_compression.algorithms.ModuleGPTQConfig` | Class | `(...)` |
| `optimize.torch.layerwise_compression.algorithms.ModuleSparseGPTConfig` | Class | `(...)` |
| `optimize.torch.layerwise_compression.algorithms.OBSCompressionAlgorithm` | Class | `(layer: _nn.Module, config: LayerwiseCompressionAlgorithmConfig)` |
| `optimize.torch.layerwise_compression.algorithms.QuantizationGranularity` | Class | `(...)` |
| `optimize.torch.layerwise_compression.algorithms.SparseGPT` | Class | `(layer: _nn.Module, config: ModuleSparseGPTConfig)` |
| `optimize.torch.layerwise_compression.input_cacher.DefaultInputCacher` | Class | `(...)` |
| `optimize.torch.layerwise_compression.input_cacher.FirstLayerInputCacher` | Class | `(model: _nn.Module, layers: str)` |
| `optimize.torch.layerwise_compression.input_cacher.GPTFirstLayerInputCacher` | Class | `(model: _nn.Module, layers: _Union[str, _List])` |
| `optimize.torch.layerwise_compression.input_cacher.StopExecution` | Class | `(...)` |
| `optimize.torch.layerwise_compression.layerwise_compressor.LayerwiseCompressor` | Class | `(model: _nn.Module, config: LayerwiseCompressorConfig)` |
| `optimize.torch.layerwise_compression.layerwise_compressor.LayerwiseCompressorConfig` | Class | `(...)` |
| `optimize.torch.optimization_config.ModuleOptimizationConfig` | Class | `(...)` |
| `optimize.torch.optimization_config.OptimizationConfig` | Class | `(...)` |
| `optimize.torch.optimization_config.PalettizationGranularity` | Class | `(...)` |
| `optimize.torch.optimization_config.QuantizationGranularity` | Class | `(...)` |
| `optimize.torch.palettization.DKMPalettizer` | Class | `(model: _nn.Module, config: _Optional[_DKMPalettizerConfig])` |
| `optimize.torch.palettization.DKMPalettizerConfig` | Class | `(...)` |
| `optimize.torch.palettization.FakePalettize` | Class | `(observer: _ObserverBase, n_bits: int, cluster_dim: int, enable_per_channel_scale: bool, group_size: _Optional[int], quant_min: int, quant_max: int, lut_dtype: str, advanced_options: dict, observer_kwargs)` |
| `optimize.torch.palettization.ModuleDKMPalettizerConfig` | Class | `(...)` |
| `optimize.torch.palettization.ModulePostTrainingPalettizerConfig` | Class | `(...)` |
| `optimize.torch.palettization.ModuleSKMPalettizerConfig` | Class | `(...)` |
| `optimize.torch.palettization.PostTrainingPalettizer` | Class | `(model: _torch.nn.Module, config: PostTrainingPalettizerConfig)` |
| `optimize.torch.palettization.PostTrainingPalettizerConfig` | Class | `(...)` |
| `optimize.torch.palettization.SKMPalettizer` | Class | `(model: _torch.nn.Module, config: _Optional[SKMPalettizerConfig])` |
| `optimize.torch.palettization.SKMPalettizerConfig` | Class | `(...)` |
| `optimize.torch.palettization.fake_palettize.DEFAULT_PALETTIZATION_ADVANCED_OPTIONS` | Object | `` |
| `optimize.torch.palettization.fake_palettize.FakePalettize` | Class | `(observer: _ObserverBase, n_bits: int, cluster_dim: int, enable_per_channel_scale: bool, group_size: _Optional[int], quant_min: int, quant_max: int, lut_dtype: str, advanced_options: dict, observer_kwargs)` |
| `optimize.torch.palettization.fake_palettize.dist_batch_cdist_square` | Class | `(...)` |
| `optimize.torch.palettization.palettization_config.DEFAULT_PALETTIZATION_ADVANCED_OPTIONS` | Object | `` |
| `optimize.torch.palettization.palettization_config.DEFAULT_PALETTIZATION_OPTIONS` | Object | `` |
| `optimize.torch.palettization.palettization_config.DEFAULT_PALETTIZATION_SCHEME` | Object | `` |
| `optimize.torch.palettization.palettization_config.DKMPalettizerConfig` | Class | `(...)` |
| `optimize.torch.palettization.palettization_config.ModuleDKMPalettizerConfig` | Class | `(...)` |
| `optimize.torch.palettization.palettization_config.PalettizationGranularity` | Class | `(...)` |
| `optimize.torch.palettization.palettization_config.SUPPORTED_PYTORCH_QAT_MODULES` | Object | `` |
| `optimize.torch.palettization.palettizer.DKMPalettizer` | Class | `(model: _nn.Module, config: _Optional[_DKMPalettizerConfig])` |
| `optimize.torch.palettization.palettizer.Palettizer` | Class | `(...)` |
| `optimize.torch.palettization.post_training_palettization.ModulePostTrainingPalettizerConfig` | Class | `(...)` |
| `optimize.torch.palettization.post_training_palettization.PalettizationGranularity` | Class | `(...)` |
| `optimize.torch.palettization.post_training_palettization.PostTrainingPalettizer` | Class | `(model: _torch.nn.Module, config: PostTrainingPalettizerConfig)` |
| `optimize.torch.palettization.post_training_palettization.PostTrainingPalettizerConfig` | Class | `(...)` |
| `optimize.torch.palettization.sensitive_k_means.ModuleSKMPalettizerConfig` | Class | `(...)` |
| `optimize.torch.palettization.sensitive_k_means.PalettizationGranularity` | Class | `(...)` |
| `optimize.torch.palettization.sensitive_k_means.SKMPalettizer` | Class | `(model: _torch.nn.Module, config: _Optional[SKMPalettizerConfig])` |
| `optimize.torch.palettization.sensitive_k_means.SKMPalettizerConfig` | Class | `(...)` |
| `optimize.torch.pruning.ConstantSparsityScheduler` | Class | `(...)` |
| `optimize.torch.pruning.MagnitudePruner` | Class | `(model: _torch.nn.Module, config: _Optional[MagnitudePrunerConfig])` |
| `optimize.torch.pruning.MagnitudePrunerConfig` | Class | `(...)` |
| `optimize.torch.pruning.ModuleMagnitudePrunerConfig` | Class | `(...)` |
| `optimize.torch.pruning.PolynomialDecayScheduler` | Class | `(...)` |
| `optimize.torch.pruning.magnitude_pruner.MagnitudePruner` | Class | `(model: _torch.nn.Module, config: _Optional[MagnitudePrunerConfig])` |
| `optimize.torch.pruning.magnitude_pruner.MagnitudePrunerConfig` | Class | `(...)` |
| `optimize.torch.pruning.magnitude_pruner.ModuleMagnitudePrunerConfig` | Class | `(...)` |
| `optimize.torch.pruning.pruning_scheduler.ConstantSparsityScheduler` | Class | `(...)` |
| `optimize.torch.pruning.pruning_scheduler.PolynomialDecayScheduler` | Class | `(...)` |
| `optimize.torch.pruning.pruning_scheduler.PruningScheduler` | Class | `(...)` |
| `optimize.torch.quantization.LinearQuantizer` | Class | `(model: _torch.nn.Module, config: _Optional[_LinearQuantizerConfig])` |
| `optimize.torch.quantization.LinearQuantizerConfig` | Class | `(...)` |
| `optimize.torch.quantization.ModuleLinearQuantizerConfig` | Class | `(...)` |
| `optimize.torch.quantization.ModulePostTrainingQuantizerConfig` | Class | `(...)` |
| `optimize.torch.quantization.ObserverType` | Class | `(...)` |
| `optimize.torch.quantization.PostTrainingQuantizer` | Class | `(model: _torch.nn.Module, config: PostTrainingQuantizerConfig)` |
| `optimize.torch.quantization.PostTrainingQuantizerConfig` | Class | `(...)` |
| `optimize.torch.quantization.QuantizationScheme` | Class | `(...)` |
| `optimize.torch.quantization.modules.conv_transpose.ConvTranspose1d` | Class | `(in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t, padding: _size_1_t, output_padding: _size_1_t, groups: int, bias: bool, dilation: _size_1_t, padding_mode: str, qconfig, device, dtype)` |
| `optimize.torch.quantization.modules.conv_transpose.ConvTranspose2d` | Class | `(in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t, padding: _size_2_t, output_padding: _size_2_t, groups: int, bias: bool, dilation: _size_2_t, padding_mode: str, qconfig, device, dtype)` |
| `optimize.torch.quantization.modules.conv_transpose.ConvTranspose3d` | Class | `(in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t, padding: _size_3_t, output_padding: _size_3_t, groups: int, bias: bool, dilation: _size_3_t, padding_mode: str, qconfig, device, dtype)` |
| `optimize.torch.quantization.modules.conv_transpose.MOD` | Object | `` |
| `optimize.torch.quantization.modules.conv_transpose_fused.ConvTransposeBn1d` | Class | `(in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t, padding: _size_1_t, output_padding: _size_1_t, dilation: _size_1_t, groups: int, bias: bool, padding_mode: str, eps, momentum, freeze_bn, qconfig)` |
| `optimize.torch.quantization.modules.conv_transpose_fused.ConvTransposeBn2d` | Class | `(in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t, padding: _size_2_t, output_padding: _size_2_t, dilation: _size_2_t, groups: int, bias: bool, padding_mode: str, eps, momentum, freeze_bn, qconfig)` |
| `optimize.torch.quantization.modules.conv_transpose_fused.ConvTransposeBn3d` | Class | `(in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t, padding: _size_3_t, output_padding: _size_3_t, dilation: _size_3_t, groups: int, bias: bool, padding_mode: str, eps, momentum, freeze_bn, qconfig)` |
| `optimize.torch.quantization.modules.conv_transpose_fused.MOD` | Object | `` |
| `optimize.torch.quantization.modules.conv_transpose_fused.iConvTransposeBn1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.conv_transpose_fused.iConvTransposeBn2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.conv_transpose_fused.iConvTransposeBn3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvAct1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvAct2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvAct3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvBnAct1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvBnAct2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvBnAct3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvTransposeAct1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvTransposeAct2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvTransposeAct3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvTransposeBn1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvTransposeBn2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvTransposeBn3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvTransposeBnAct1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvTransposeBnAct2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.ConvTransposeBnAct3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.fused_modules.LinearAct` | Class | `(linear: _nn.Linear, act: _nn.Module)` |
| `optimize.torch.quantization.modules.learnable_fake_quantize.LearnableFakeQuantize` | Class | `(observer: _ObserverBase, dtype: _torch.dtype, qscheme: _torch.qscheme, reduce_range: bool, observer_kwargs)` |
| `optimize.torch.quantization.modules.learnable_fake_quantize.grad_scale` | Function | `(x: _torch.Tensor, scale: _torch.Tensor)` |
| `optimize.torch.quantization.modules.learnable_fake_quantize.round_pass` | Function | `(x: _torch.Tensor)` |
| `optimize.torch.quantization.modules.observers.CustomObserverBase` | Class | `(dtype: _torch.dtype, qscheme: _torch.qscheme, reduce_range: bool, quant_min: _Optional[int], quant_max: _Optional[int], ch_axis: int, factory_kwargs: _Any, is_dynamic: bool)` |
| `optimize.torch.quantization.modules.observers.EMAMinMaxObserver` | Class | `(dtype: _torch.dtype, qscheme: _torch.qscheme, reduce_range: bool, quant_min: _Optional[int], quant_max: _Optional[int], ch_axis: int, ema_ratio: float, factory_kwargs: _Any, is_dynamic: bool)` |
| `optimize.torch.quantization.modules.observers.NoopObserver` | Class | `(dtype: _torch.dtype, custom_op_name: str, factory_kwargs: _Dict[str, _Any])` |
| `optimize.torch.quantization.modules.qat_modules.ConvAct1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.ConvAct2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.ConvAct3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.ConvBnAct1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.ConvBnAct2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.ConvBnAct3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.ConvTransposeAct1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.ConvTransposeAct2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.ConvTransposeAct3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.ConvTransposeBnAct1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.ConvTransposeBnAct2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.ConvTransposeBnAct3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.qat_modules.LinearAct` | Class | `(linear: _nnqat.Linear, act: _nn.Module, qconfig: _aoquant.QConfig)` |
| `optimize.torch.quantization.modules.qat_modules.freeze_bn_stats` | Function | `(mod)` |
| `optimize.torch.quantization.modules.qat_modules.update_bn_stats` | Function | `(mod)` |
| `optimize.torch.quantization.modules.quantized_modules.QuantizedConvAct1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.quantized_modules.QuantizedConvAct2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.quantized_modules.QuantizedConvAct3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.quantized_modules.QuantizedConvTransposeAct1d` | Class | `(...)` |
| `optimize.torch.quantization.modules.quantized_modules.QuantizedConvTransposeAct2d` | Class | `(...)` |
| `optimize.torch.quantization.modules.quantized_modules.QuantizedConvTransposeAct3d` | Class | `(...)` |
| `optimize.torch.quantization.modules.quantized_modules.QuantizedLinearAct` | Class | `(linear: _reference.Linear, act: _nn.Module)` |
| `optimize.torch.quantization.post_training_quantization.ModulePostTrainingQuantizerConfig` | Class | `(...)` |
| `optimize.torch.quantization.post_training_quantization.PostTrainingQuantizer` | Class | `(model: _torch.nn.Module, config: PostTrainingQuantizerConfig)` |
| `optimize.torch.quantization.post_training_quantization.PostTrainingQuantizerConfig` | Class | `(...)` |
| `optimize.torch.quantization.post_training_quantization.QuantizationGranularity` | Class | `(...)` |
| `optimize.torch.quantization.post_training_quantization.optimize_utils` | Object | `` |
| `optimize.torch.quantization.quantization_config.LinearQuantizerConfig` | Class | `(...)` |
| `optimize.torch.quantization.quantization_config.ModuleLinearQuantizerConfig` | Class | `(...)` |
| `optimize.torch.quantization.quantization_config.ObserverType` | Class | `(...)` |
| `optimize.torch.quantization.quantization_config.QuantizationScheme` | Class | `(...)` |
| `optimize.torch.quantization.quantizer.LinearQuantizer` | Class | `(model: _torch.nn.Module, config: _Optional[_LinearQuantizerConfig])` |
| `optimize.torch.quantization.quantizer.Quantizer` | Class | `(...)` |
| `precision` | Class | `(...)` |
| `proto.ArrayFeatureExtractor_pb2.ArrayFeatureExtractor` | Object | `` |
| `proto.ArrayFeatureExtractor_pb2.DESCRIPTOR` | Object | `` |
| `proto.AudioFeaturePrint_pb2.AudioFeaturePrint` | Object | `` |
| `proto.AudioFeaturePrint_pb2.DESCRIPTOR` | Object | `` |
| `proto.BayesianProbitRegressor_pb2.BayesianProbitRegressor` | Object | `` |
| `proto.BayesianProbitRegressor_pb2.DESCRIPTOR` | Object | `` |
| `proto.CategoricalMapping_pb2.ArrayFeatureType` | Object | `` |
| `proto.CategoricalMapping_pb2.CategoricalMapping` | Object | `` |
| `proto.CategoricalMapping_pb2.DESCRIPTOR` | Object | `` |
| `proto.CategoricalMapping_pb2.DataStructures__pb2` | Object | `` |
| `proto.CategoricalMapping_pb2.DictionaryFeatureType` | Object | `` |
| `proto.CategoricalMapping_pb2.DoubleFeatureType` | Object | `` |
| `proto.CategoricalMapping_pb2.DoubleRange` | Object | `` |
| `proto.CategoricalMapping_pb2.DoubleVector` | Object | `` |
| `proto.CategoricalMapping_pb2.FeatureType` | Object | `` |
| `proto.CategoricalMapping_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.CategoricalMapping_pb2.FloatVector` | Object | `` |
| `proto.CategoricalMapping_pb2.ImageFeatureType` | Object | `` |
| `proto.CategoricalMapping_pb2.Int64FeatureType` | Object | `` |
| `proto.CategoricalMapping_pb2.Int64Range` | Object | `` |
| `proto.CategoricalMapping_pb2.Int64Set` | Object | `` |
| `proto.CategoricalMapping_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.CategoricalMapping_pb2.Int64ToStringMap` | Object | `` |
| `proto.CategoricalMapping_pb2.Int64Vector` | Object | `` |
| `proto.CategoricalMapping_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.CategoricalMapping_pb2.SequenceFeatureType` | Object | `` |
| `proto.CategoricalMapping_pb2.SizeRange` | Object | `` |
| `proto.CategoricalMapping_pb2.StateFeatureType` | Object | `` |
| `proto.CategoricalMapping_pb2.StringFeatureType` | Object | `` |
| `proto.CategoricalMapping_pb2.StringToDoubleMap` | Object | `` |
| `proto.CategoricalMapping_pb2.StringToInt64Map` | Object | `` |
| `proto.CategoricalMapping_pb2.StringVector` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.ArrayFeatureType` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.ClassConfidenceThresholding` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.DESCRIPTOR` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.DataStructures__pb2` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.DictionaryFeatureType` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.DoubleFeatureType` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.DoubleRange` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.DoubleVector` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.FeatureType` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.FloatVector` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.ImageFeatureType` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.Int64FeatureType` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.Int64Range` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.Int64Set` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.Int64ToStringMap` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.Int64Vector` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.SequenceFeatureType` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.SizeRange` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.StateFeatureType` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.StringFeatureType` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.StringToDoubleMap` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.StringToInt64Map` | Object | `` |
| `proto.ClassConfidenceThresholding_pb2.StringVector` | Object | `` |
| `proto.CustomModel_pb2.CustomModel` | Object | `` |
| `proto.CustomModel_pb2.DESCRIPTOR` | Object | `` |
| `proto.DataStructures_pb2.ArrayFeatureType` | Object | `` |
| `proto.DataStructures_pb2.DESCRIPTOR` | Object | `` |
| `proto.DataStructures_pb2.DictionaryFeatureType` | Object | `` |
| `proto.DataStructures_pb2.DoubleFeatureType` | Object | `` |
| `proto.DataStructures_pb2.DoubleRange` | Object | `` |
| `proto.DataStructures_pb2.DoubleVector` | Object | `` |
| `proto.DataStructures_pb2.FeatureType` | Object | `` |
| `proto.DataStructures_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.DataStructures_pb2.FloatVector` | Object | `` |
| `proto.DataStructures_pb2.ImageFeatureType` | Object | `` |
| `proto.DataStructures_pb2.Int64FeatureType` | Object | `` |
| `proto.DataStructures_pb2.Int64Range` | Object | `` |
| `proto.DataStructures_pb2.Int64Set` | Object | `` |
| `proto.DataStructures_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.DataStructures_pb2.Int64ToStringMap` | Object | `` |
| `proto.DataStructures_pb2.Int64Vector` | Object | `` |
| `proto.DataStructures_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.DataStructures_pb2.SequenceFeatureType` | Object | `` |
| `proto.DataStructures_pb2.SizeRange` | Object | `` |
| `proto.DataStructures_pb2.StateFeatureType` | Object | `` |
| `proto.DataStructures_pb2.StringFeatureType` | Object | `` |
| `proto.DataStructures_pb2.StringToDoubleMap` | Object | `` |
| `proto.DataStructures_pb2.StringToInt64Map` | Object | `` |
| `proto.DataStructures_pb2.StringVector` | Object | `` |
| `proto.DictVectorizer_pb2.ArrayFeatureType` | Object | `` |
| `proto.DictVectorizer_pb2.DESCRIPTOR` | Object | `` |
| `proto.DictVectorizer_pb2.DataStructures__pb2` | Object | `` |
| `proto.DictVectorizer_pb2.DictVectorizer` | Object | `` |
| `proto.DictVectorizer_pb2.DictionaryFeatureType` | Object | `` |
| `proto.DictVectorizer_pb2.DoubleFeatureType` | Object | `` |
| `proto.DictVectorizer_pb2.DoubleRange` | Object | `` |
| `proto.DictVectorizer_pb2.DoubleVector` | Object | `` |
| `proto.DictVectorizer_pb2.FeatureType` | Object | `` |
| `proto.DictVectorizer_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.DictVectorizer_pb2.FloatVector` | Object | `` |
| `proto.DictVectorizer_pb2.ImageFeatureType` | Object | `` |
| `proto.DictVectorizer_pb2.Int64FeatureType` | Object | `` |
| `proto.DictVectorizer_pb2.Int64Range` | Object | `` |
| `proto.DictVectorizer_pb2.Int64Set` | Object | `` |
| `proto.DictVectorizer_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.DictVectorizer_pb2.Int64ToStringMap` | Object | `` |
| `proto.DictVectorizer_pb2.Int64Vector` | Object | `` |
| `proto.DictVectorizer_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.DictVectorizer_pb2.SequenceFeatureType` | Object | `` |
| `proto.DictVectorizer_pb2.SizeRange` | Object | `` |
| `proto.DictVectorizer_pb2.StateFeatureType` | Object | `` |
| `proto.DictVectorizer_pb2.StringFeatureType` | Object | `` |
| `proto.DictVectorizer_pb2.StringToDoubleMap` | Object | `` |
| `proto.DictVectorizer_pb2.StringToInt64Map` | Object | `` |
| `proto.DictVectorizer_pb2.StringVector` | Object | `` |
| `proto.FeatureTypes_pb2.ArrayFeatureType` | Object | `` |
| `proto.FeatureTypes_pb2.DESCRIPTOR` | Object | `` |
| `proto.FeatureTypes_pb2.DictionaryFeatureType` | Object | `` |
| `proto.FeatureTypes_pb2.DoubleFeatureType` | Object | `` |
| `proto.FeatureTypes_pb2.FeatureType` | Object | `` |
| `proto.FeatureTypes_pb2.ImageFeatureType` | Object | `` |
| `proto.FeatureTypes_pb2.Int64FeatureType` | Object | `` |
| `proto.FeatureTypes_pb2.SequenceFeatureType` | Object | `` |
| `proto.FeatureTypes_pb2.SizeRange` | Object | `` |
| `proto.FeatureTypes_pb2.StateFeatureType` | Object | `` |
| `proto.FeatureTypes_pb2.StringFeatureType` | Object | `` |
| `proto.FeatureVectorizer_pb2.DESCRIPTOR` | Object | `` |
| `proto.FeatureVectorizer_pb2.FeatureVectorizer` | Object | `` |
| `proto.GLMClassifier_pb2.ArrayFeatureType` | Object | `` |
| `proto.GLMClassifier_pb2.DESCRIPTOR` | Object | `` |
| `proto.GLMClassifier_pb2.DataStructures__pb2` | Object | `` |
| `proto.GLMClassifier_pb2.DictionaryFeatureType` | Object | `` |
| `proto.GLMClassifier_pb2.DoubleFeatureType` | Object | `` |
| `proto.GLMClassifier_pb2.DoubleRange` | Object | `` |
| `proto.GLMClassifier_pb2.DoubleVector` | Object | `` |
| `proto.GLMClassifier_pb2.FeatureType` | Object | `` |
| `proto.GLMClassifier_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.GLMClassifier_pb2.FloatVector` | Object | `` |
| `proto.GLMClassifier_pb2.GLMClassifier` | Object | `` |
| `proto.GLMClassifier_pb2.ImageFeatureType` | Object | `` |
| `proto.GLMClassifier_pb2.Int64FeatureType` | Object | `` |
| `proto.GLMClassifier_pb2.Int64Range` | Object | `` |
| `proto.GLMClassifier_pb2.Int64Set` | Object | `` |
| `proto.GLMClassifier_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.GLMClassifier_pb2.Int64ToStringMap` | Object | `` |
| `proto.GLMClassifier_pb2.Int64Vector` | Object | `` |
| `proto.GLMClassifier_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.GLMClassifier_pb2.SequenceFeatureType` | Object | `` |
| `proto.GLMClassifier_pb2.SizeRange` | Object | `` |
| `proto.GLMClassifier_pb2.StateFeatureType` | Object | `` |
| `proto.GLMClassifier_pb2.StringFeatureType` | Object | `` |
| `proto.GLMClassifier_pb2.StringToDoubleMap` | Object | `` |
| `proto.GLMClassifier_pb2.StringToInt64Map` | Object | `` |
| `proto.GLMClassifier_pb2.StringVector` | Object | `` |
| `proto.GLMRegressor_pb2.DESCRIPTOR` | Object | `` |
| `proto.GLMRegressor_pb2.GLMRegressor` | Object | `` |
| `proto.Gazetteer_pb2.ArrayFeatureType` | Object | `` |
| `proto.Gazetteer_pb2.DESCRIPTOR` | Object | `` |
| `proto.Gazetteer_pb2.DataStructures__pb2` | Object | `` |
| `proto.Gazetteer_pb2.DictionaryFeatureType` | Object | `` |
| `proto.Gazetteer_pb2.DoubleFeatureType` | Object | `` |
| `proto.Gazetteer_pb2.DoubleRange` | Object | `` |
| `proto.Gazetteer_pb2.DoubleVector` | Object | `` |
| `proto.Gazetteer_pb2.FeatureType` | Object | `` |
| `proto.Gazetteer_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.Gazetteer_pb2.FloatVector` | Object | `` |
| `proto.Gazetteer_pb2.Gazetteer` | Object | `` |
| `proto.Gazetteer_pb2.ImageFeatureType` | Object | `` |
| `proto.Gazetteer_pb2.Int64FeatureType` | Object | `` |
| `proto.Gazetteer_pb2.Int64Range` | Object | `` |
| `proto.Gazetteer_pb2.Int64Set` | Object | `` |
| `proto.Gazetteer_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.Gazetteer_pb2.Int64ToStringMap` | Object | `` |
| `proto.Gazetteer_pb2.Int64Vector` | Object | `` |
| `proto.Gazetteer_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.Gazetteer_pb2.SequenceFeatureType` | Object | `` |
| `proto.Gazetteer_pb2.SizeRange` | Object | `` |
| `proto.Gazetteer_pb2.StateFeatureType` | Object | `` |
| `proto.Gazetteer_pb2.StringFeatureType` | Object | `` |
| `proto.Gazetteer_pb2.StringToDoubleMap` | Object | `` |
| `proto.Gazetteer_pb2.StringToInt64Map` | Object | `` |
| `proto.Gazetteer_pb2.StringVector` | Object | `` |
| `proto.Identity_pb2.DESCRIPTOR` | Object | `` |
| `proto.Identity_pb2.Identity` | Object | `` |
| `proto.Imputer_pb2.ArrayFeatureType` | Object | `` |
| `proto.Imputer_pb2.DESCRIPTOR` | Object | `` |
| `proto.Imputer_pb2.DataStructures__pb2` | Object | `` |
| `proto.Imputer_pb2.DictionaryFeatureType` | Object | `` |
| `proto.Imputer_pb2.DoubleFeatureType` | Object | `` |
| `proto.Imputer_pb2.DoubleRange` | Object | `` |
| `proto.Imputer_pb2.DoubleVector` | Object | `` |
| `proto.Imputer_pb2.FeatureType` | Object | `` |
| `proto.Imputer_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.Imputer_pb2.FloatVector` | Object | `` |
| `proto.Imputer_pb2.ImageFeatureType` | Object | `` |
| `proto.Imputer_pb2.Imputer` | Object | `` |
| `proto.Imputer_pb2.Int64FeatureType` | Object | `` |
| `proto.Imputer_pb2.Int64Range` | Object | `` |
| `proto.Imputer_pb2.Int64Set` | Object | `` |
| `proto.Imputer_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.Imputer_pb2.Int64ToStringMap` | Object | `` |
| `proto.Imputer_pb2.Int64Vector` | Object | `` |
| `proto.Imputer_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.Imputer_pb2.SequenceFeatureType` | Object | `` |
| `proto.Imputer_pb2.SizeRange` | Object | `` |
| `proto.Imputer_pb2.StateFeatureType` | Object | `` |
| `proto.Imputer_pb2.StringFeatureType` | Object | `` |
| `proto.Imputer_pb2.StringToDoubleMap` | Object | `` |
| `proto.Imputer_pb2.StringToInt64Map` | Object | `` |
| `proto.Imputer_pb2.StringVector` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.ArrayFeatureType` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.DESCRIPTOR` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.DataStructures__pb2` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.DictionaryFeatureType` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.DoubleFeatureType` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.DoubleRange` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.DoubleVector` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.FeatureType` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.FloatVector` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.ImageFeatureType` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.Int64FeatureType` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.Int64Range` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.Int64Set` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.Int64ToStringMap` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.Int64Vector` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.ItemSimilarityRecommender` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.SequenceFeatureType` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.SizeRange` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.StateFeatureType` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.StringFeatureType` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.StringToDoubleMap` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.StringToInt64Map` | Object | `` |
| `proto.ItemSimilarityRecommender_pb2.StringVector` | Object | `` |
| `proto.LinkedModel_pb2.ArrayFeatureType` | Object | `` |
| `proto.LinkedModel_pb2.BoolParameter` | Object | `` |
| `proto.LinkedModel_pb2.DESCRIPTOR` | Object | `` |
| `proto.LinkedModel_pb2.DataStructures__pb2` | Object | `` |
| `proto.LinkedModel_pb2.DictionaryFeatureType` | Object | `` |
| `proto.LinkedModel_pb2.DoubleFeatureType` | Object | `` |
| `proto.LinkedModel_pb2.DoubleParameter` | Object | `` |
| `proto.LinkedModel_pb2.DoubleRange` | Object | `` |
| `proto.LinkedModel_pb2.DoubleVector` | Object | `` |
| `proto.LinkedModel_pb2.FeatureType` | Object | `` |
| `proto.LinkedModel_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.LinkedModel_pb2.FloatVector` | Object | `` |
| `proto.LinkedModel_pb2.ImageFeatureType` | Object | `` |
| `proto.LinkedModel_pb2.Int64FeatureType` | Object | `` |
| `proto.LinkedModel_pb2.Int64Parameter` | Object | `` |
| `proto.LinkedModel_pb2.Int64Range` | Object | `` |
| `proto.LinkedModel_pb2.Int64Set` | Object | `` |
| `proto.LinkedModel_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.LinkedModel_pb2.Int64ToStringMap` | Object | `` |
| `proto.LinkedModel_pb2.Int64Vector` | Object | `` |
| `proto.LinkedModel_pb2.LinkedModel` | Object | `` |
| `proto.LinkedModel_pb2.LinkedModelFile` | Object | `` |
| `proto.LinkedModel_pb2.Parameters__pb2` | Object | `` |
| `proto.LinkedModel_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.LinkedModel_pb2.SequenceFeatureType` | Object | `` |
| `proto.LinkedModel_pb2.SizeRange` | Object | `` |
| `proto.LinkedModel_pb2.StateFeatureType` | Object | `` |
| `proto.LinkedModel_pb2.StringFeatureType` | Object | `` |
| `proto.LinkedModel_pb2.StringParameter` | Object | `` |
| `proto.LinkedModel_pb2.StringToDoubleMap` | Object | `` |
| `proto.LinkedModel_pb2.StringToInt64Map` | Object | `` |
| `proto.LinkedModel_pb2.StringVector` | Object | `` |
| `proto.MIL_pb2.Argument` | Object | `` |
| `proto.MIL_pb2.BFLOAT16` | Object | `` |
| `proto.MIL_pb2.BOOL` | Object | `` |
| `proto.MIL_pb2.Block` | Object | `` |
| `proto.MIL_pb2.DESCRIPTOR` | Object | `` |
| `proto.MIL_pb2.DataType` | Object | `` |
| `proto.MIL_pb2.DictionaryType` | Object | `` |
| `proto.MIL_pb2.DictionaryValue` | Object | `` |
| `proto.MIL_pb2.Dimension` | Object | `` |
| `proto.MIL_pb2.FLOAT16` | Object | `` |
| `proto.MIL_pb2.FLOAT32` | Object | `` |
| `proto.MIL_pb2.FLOAT64` | Object | `` |
| `proto.MIL_pb2.FLOAT8E4M3FN` | Object | `` |
| `proto.MIL_pb2.FLOAT8E5M2` | Object | `` |
| `proto.MIL_pb2.Function` | Object | `` |
| `proto.MIL_pb2.INT16` | Object | `` |
| `proto.MIL_pb2.INT32` | Object | `` |
| `proto.MIL_pb2.INT4` | Object | `` |
| `proto.MIL_pb2.INT64` | Object | `` |
| `proto.MIL_pb2.INT8` | Object | `` |
| `proto.MIL_pb2.ListType` | Object | `` |
| `proto.MIL_pb2.ListValue` | Object | `` |
| `proto.MIL_pb2.NamedValueType` | Object | `` |
| `proto.MIL_pb2.Operation` | Object | `` |
| `proto.MIL_pb2.Program` | Object | `` |
| `proto.MIL_pb2.STRING` | Object | `` |
| `proto.MIL_pb2.StateType` | Object | `` |
| `proto.MIL_pb2.TensorType` | Object | `` |
| `proto.MIL_pb2.TensorValue` | Object | `` |
| `proto.MIL_pb2.TupleType` | Object | `` |
| `proto.MIL_pb2.TupleValue` | Object | `` |
| `proto.MIL_pb2.UINT1` | Object | `` |
| `proto.MIL_pb2.UINT16` | Object | `` |
| `proto.MIL_pb2.UINT2` | Object | `` |
| `proto.MIL_pb2.UINT3` | Object | `` |
| `proto.MIL_pb2.UINT32` | Object | `` |
| `proto.MIL_pb2.UINT4` | Object | `` |
| `proto.MIL_pb2.UINT6` | Object | `` |
| `proto.MIL_pb2.UINT64` | Object | `` |
| `proto.MIL_pb2.UINT8` | Object | `` |
| `proto.MIL_pb2.UNUSED_TYPE` | Object | `` |
| `proto.MIL_pb2.Value` | Object | `` |
| `proto.MIL_pb2.ValueType` | Object | `` |
| `proto.Model_pb2.AcosLayerParams` | Object | `` |
| `proto.Model_pb2.AcoshLayerParams` | Object | `` |
| `proto.Model_pb2.ActivationELU` | Object | `` |
| `proto.Model_pb2.ActivationLeakyReLU` | Object | `` |
| `proto.Model_pb2.ActivationLinear` | Object | `` |
| `proto.Model_pb2.ActivationPReLU` | Object | `` |
| `proto.Model_pb2.ActivationParametricSoftplus` | Object | `` |
| `proto.Model_pb2.ActivationParams` | Object | `` |
| `proto.Model_pb2.ActivationReLU` | Object | `` |
| `proto.Model_pb2.ActivationScaledTanh` | Object | `` |
| `proto.Model_pb2.ActivationSigmoid` | Object | `` |
| `proto.Model_pb2.ActivationSigmoidHard` | Object | `` |
| `proto.Model_pb2.ActivationSoftplus` | Object | `` |
| `proto.Model_pb2.ActivationSoftsign` | Object | `` |
| `proto.Model_pb2.ActivationTanh` | Object | `` |
| `proto.Model_pb2.ActivationThresholdedReLU` | Object | `` |
| `proto.Model_pb2.AdamOptimizer` | Object | `` |
| `proto.Model_pb2.AddBroadcastableLayerParams` | Object | `` |
| `proto.Model_pb2.AddLayerParams` | Object | `` |
| `proto.Model_pb2.ArgMaxLayerParams` | Object | `` |
| `proto.Model_pb2.ArgMinLayerParams` | Object | `` |
| `proto.Model_pb2.ArgSortLayerParams` | Object | `` |
| `proto.Model_pb2.Argument` | Object | `` |
| `proto.Model_pb2.ArrayFeatureExtractor` | Object | `` |
| `proto.Model_pb2.ArrayFeatureExtractor__pb2` | Object | `` |
| `proto.Model_pb2.ArrayFeatureType` | Object | `` |
| `proto.Model_pb2.AsinLayerParams` | Object | `` |
| `proto.Model_pb2.AsinhLayerParams` | Object | `` |
| `proto.Model_pb2.AtanLayerParams` | Object | `` |
| `proto.Model_pb2.AtanhLayerParams` | Object | `` |
| `proto.Model_pb2.AudioFeaturePrint` | Object | `` |
| `proto.Model_pb2.AudioFeaturePrint__pb2` | Object | `` |
| `proto.Model_pb2.AverageLayerParams` | Object | `` |
| `proto.Model_pb2.BFLOAT16` | Object | `` |
| `proto.Model_pb2.BOOL` | Object | `` |
| `proto.Model_pb2.BatchedMatMulLayerParams` | Object | `` |
| `proto.Model_pb2.BatchnormLayerParams` | Object | `` |
| `proto.Model_pb2.BayesianProbitRegressor` | Object | `` |
| `proto.Model_pb2.BayesianProbitRegressor__pb2` | Object | `` |
| `proto.Model_pb2.BiDirectionalLSTMLayerParams` | Object | `` |
| `proto.Model_pb2.BiasLayerParams` | Object | `` |
| `proto.Model_pb2.Block` | Object | `` |
| `proto.Model_pb2.BoolParameter` | Object | `` |
| `proto.Model_pb2.BorderAmounts` | Object | `` |
| `proto.Model_pb2.BoxCoordinatesMode` | Object | `` |
| `proto.Model_pb2.BranchLayerParams` | Object | `` |
| `proto.Model_pb2.BroadcastToDynamicLayerParams` | Object | `` |
| `proto.Model_pb2.BroadcastToLikeLayerParams` | Object | `` |
| `proto.Model_pb2.BroadcastToStaticLayerParams` | Object | `` |
| `proto.Model_pb2.CategoricalCrossEntropyLossLayer` | Object | `` |
| `proto.Model_pb2.CategoricalDistributionLayerParams` | Object | `` |
| `proto.Model_pb2.CategoricalMapping` | Object | `` |
| `proto.Model_pb2.CategoricalMapping__pb2` | Object | `` |
| `proto.Model_pb2.CeilLayerParams` | Object | `` |
| `proto.Model_pb2.ClampedReLULayerParams` | Object | `` |
| `proto.Model_pb2.ClassConfidenceThresholding` | Object | `` |
| `proto.Model_pb2.ClassConfidenceThresholding__pb2` | Object | `` |
| `proto.Model_pb2.Classification_SoftMax` | Object | `` |
| `proto.Model_pb2.Classification_SoftMaxWithZeroClassReference` | Object | `` |
| `proto.Model_pb2.ClipLayerParams` | Object | `` |
| `proto.Model_pb2.Coefficients` | Object | `` |
| `proto.Model_pb2.ConcatLayerParams` | Object | `` |
| `proto.Model_pb2.ConcatNDLayerParams` | Object | `` |
| `proto.Model_pb2.ConstantPaddingLayerParams` | Object | `` |
| `proto.Model_pb2.Convolution3DLayerParams` | Object | `` |
| `proto.Model_pb2.ConvolutionLayerParams` | Object | `` |
| `proto.Model_pb2.CopyLayerParams` | Object | `` |
| `proto.Model_pb2.CosLayerParams` | Object | `` |
| `proto.Model_pb2.CoshLayerParams` | Object | `` |
| `proto.Model_pb2.CropLayerParams` | Object | `` |
| `proto.Model_pb2.CropResizeLayerParams` | Object | `` |
| `proto.Model_pb2.CumSumLayerParams` | Object | `` |
| `proto.Model_pb2.CustomLayerParams` | Object | `` |
| `proto.Model_pb2.CustomModel` | Object | `` |
| `proto.Model_pb2.CustomModel__pb2` | Object | `` |
| `proto.Model_pb2.DESCRIPTOR` | Object | `` |
| `proto.Model_pb2.DataStructures__pb2` | Object | `` |
| `proto.Model_pb2.DataType` | Object | `` |
| `proto.Model_pb2.DenseSupportVectors` | Object | `` |
| `proto.Model_pb2.DenseVector` | Object | `` |
| `proto.Model_pb2.DictVectorizer` | Object | `` |
| `proto.Model_pb2.DictVectorizer__pb2` | Object | `` |
| `proto.Model_pb2.DictionaryFeatureType` | Object | `` |
| `proto.Model_pb2.DictionaryType` | Object | `` |
| `proto.Model_pb2.DictionaryValue` | Object | `` |
| `proto.Model_pb2.Dimension` | Object | `` |
| `proto.Model_pb2.DivideBroadcastableLayerParams` | Object | `` |
| `proto.Model_pb2.DotProductLayerParams` | Object | `` |
| `proto.Model_pb2.DoubleFeatureType` | Object | `` |
| `proto.Model_pb2.DoubleParameter` | Object | `` |
| `proto.Model_pb2.DoubleRange` | Object | `` |
| `proto.Model_pb2.DoubleVector` | Object | `` |
| `proto.Model_pb2.EXACT_ARRAY_MAPPING` | Object | `` |
| `proto.Model_pb2.EmbeddingLayerParams` | Object | `` |
| `proto.Model_pb2.EmbeddingNDLayerParams` | Object | `` |
| `proto.Model_pb2.EqualLayerParams` | Object | `` |
| `proto.Model_pb2.ErfLayerParams` | Object | `` |
| `proto.Model_pb2.Exp2LayerParams` | Object | `` |
| `proto.Model_pb2.ExpandDimsLayerParams` | Object | `` |
| `proto.Model_pb2.FLOAT16` | Object | `` |
| `proto.Model_pb2.FLOAT32` | Object | `` |
| `proto.Model_pb2.FLOAT64` | Object | `` |
| `proto.Model_pb2.FLOAT8E4M3FN` | Object | `` |
| `proto.Model_pb2.FLOAT8E5M2` | Object | `` |
| `proto.Model_pb2.FeatureDescription` | Object | `` |
| `proto.Model_pb2.FeatureType` | Object | `` |
| `proto.Model_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.Model_pb2.FeatureVectorizer` | Object | `` |
| `proto.Model_pb2.FeatureVectorizer__pb2` | Object | `` |
| `proto.Model_pb2.FillDynamicLayerParams` | Object | `` |
| `proto.Model_pb2.FillLikeLayerParams` | Object | `` |
| `proto.Model_pb2.FillStaticLayerParams` | Object | `` |
| `proto.Model_pb2.FlattenLayerParams` | Object | `` |
| `proto.Model_pb2.FlattenTo2DLayerParams` | Object | `` |
| `proto.Model_pb2.FloatVector` | Object | `` |
| `proto.Model_pb2.FloorDivBroadcastableLayerParams` | Object | `` |
| `proto.Model_pb2.FloorLayerParams` | Object | `` |
| `proto.Model_pb2.Function` | Object | `` |
| `proto.Model_pb2.FunctionDescription` | Object | `` |
| `proto.Model_pb2.GLMClassifier` | Object | `` |
| `proto.Model_pb2.GLMClassifier__pb2` | Object | `` |
| `proto.Model_pb2.GLMRegressor` | Object | `` |
| `proto.Model_pb2.GLMRegressor__pb2` | Object | `` |
| `proto.Model_pb2.GRULayerParams` | Object | `` |
| `proto.Model_pb2.GatherAlongAxisLayerParams` | Object | `` |
| `proto.Model_pb2.GatherLayerParams` | Object | `` |
| `proto.Model_pb2.GatherNDLayerParams` | Object | `` |
| `proto.Model_pb2.Gazetteer` | Object | `` |
| `proto.Model_pb2.Gazetteer__pb2` | Object | `` |
| `proto.Model_pb2.GeluLayerParams` | Object | `` |
| `proto.Model_pb2.GetShapeLayerParams` | Object | `` |
| `proto.Model_pb2.GlobalPooling3DLayerParams` | Object | `` |
| `proto.Model_pb2.GreaterEqualLayerParams` | Object | `` |
| `proto.Model_pb2.GreaterThanLayerParams` | Object | `` |
| `proto.Model_pb2.INT16` | Object | `` |
| `proto.Model_pb2.INT32` | Object | `` |
| `proto.Model_pb2.INT4` | Object | `` |
| `proto.Model_pb2.INT64` | Object | `` |
| `proto.Model_pb2.INT8` | Object | `` |
| `proto.Model_pb2.Identity` | Object | `` |
| `proto.Model_pb2.Identity__pb2` | Object | `` |
| `proto.Model_pb2.ImageFeatureType` | Object | `` |
| `proto.Model_pb2.Imputer` | Object | `` |
| `proto.Model_pb2.Imputer__pb2` | Object | `` |
| `proto.Model_pb2.InnerProductLayerParams` | Object | `` |
| `proto.Model_pb2.Int64FeatureType` | Object | `` |
| `proto.Model_pb2.Int64Parameter` | Object | `` |
| `proto.Model_pb2.Int64Range` | Object | `` |
| `proto.Model_pb2.Int64Set` | Object | `` |
| `proto.Model_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.Model_pb2.Int64ToStringMap` | Object | `` |
| `proto.Model_pb2.Int64Vector` | Object | `` |
| `proto.Model_pb2.InverseDistanceWeighting` | Object | `` |
| `proto.Model_pb2.ItemSimilarityRecommender` | Object | `` |
| `proto.Model_pb2.ItemSimilarityRecommender__pb2` | Object | `` |
| `proto.Model_pb2.KNearestNeighborsClassifier` | Object | `` |
| `proto.Model_pb2.Kernel` | Object | `` |
| `proto.Model_pb2.L2NormalizeLayerParams` | Object | `` |
| `proto.Model_pb2.LRNLayerParams` | Object | `` |
| `proto.Model_pb2.LSTMParams` | Object | `` |
| `proto.Model_pb2.LSTMWeightParams` | Object | `` |
| `proto.Model_pb2.LayerNormalizationLayerParams` | Object | `` |
| `proto.Model_pb2.LessEqualLayerParams` | Object | `` |
| `proto.Model_pb2.LessThanLayerParams` | Object | `` |
| `proto.Model_pb2.LinearIndex` | Object | `` |
| `proto.Model_pb2.LinearKernel` | Object | `` |
| `proto.Model_pb2.LinearQuantizationParams` | Object | `` |
| `proto.Model_pb2.LinkedModel` | Object | `` |
| `proto.Model_pb2.LinkedModelFile` | Object | `` |
| `proto.Model_pb2.LinkedModel__pb2` | Object | `` |
| `proto.Model_pb2.ListType` | Object | `` |
| `proto.Model_pb2.ListValue` | Object | `` |
| `proto.Model_pb2.LoadConstantLayerParams` | Object | `` |
| `proto.Model_pb2.LoadConstantNDLayerParams` | Object | `` |
| `proto.Model_pb2.LogicalAndLayerParams` | Object | `` |
| `proto.Model_pb2.LogicalNotLayerParams` | Object | `` |
| `proto.Model_pb2.LogicalOrLayerParams` | Object | `` |
| `proto.Model_pb2.LogicalXorLayerParams` | Object | `` |
| `proto.Model_pb2.LookUpTableQuantizationParams` | Object | `` |
| `proto.Model_pb2.LoopBreakLayerParams` | Object | `` |
| `proto.Model_pb2.LoopContinueLayerParams` | Object | `` |
| `proto.Model_pb2.LoopLayerParams` | Object | `` |
| `proto.Model_pb2.LossLayer` | Object | `` |
| `proto.Model_pb2.LowerTriangularLayerParams` | Object | `` |
| `proto.Model_pb2.MIL__pb2` | Object | `` |
| `proto.Model_pb2.MatrixBandPartLayerParams` | Object | `` |
| `proto.Model_pb2.MaxBroadcastableLayerParams` | Object | `` |
| `proto.Model_pb2.MaxLayerParams` | Object | `` |
| `proto.Model_pb2.MeanSquaredErrorLossLayer` | Object | `` |
| `proto.Model_pb2.MeanVarianceNormalizeLayerParams` | Object | `` |
| `proto.Model_pb2.Metadata` | Object | `` |
| `proto.Model_pb2.MinBroadcastableLayerParams` | Object | `` |
| `proto.Model_pb2.MinLayerParams` | Object | `` |
| `proto.Model_pb2.ModBroadcastableLayerParams` | Object | `` |
| `proto.Model_pb2.Model` | Object | `` |
| `proto.Model_pb2.ModelDescription` | Object | `` |
| `proto.Model_pb2.MultiplyBroadcastableLayerParams` | Object | `` |
| `proto.Model_pb2.MultiplyLayerParams` | Object | `` |
| `proto.Model_pb2.NamedValueType` | Object | `` |
| `proto.Model_pb2.NearestNeighborsIndex` | Object | `` |
| `proto.Model_pb2.NearestNeighbors__pb2` | Object | `` |
| `proto.Model_pb2.NetworkUpdateParameters` | Object | `` |
| `proto.Model_pb2.NeuralNetwork` | Object | `` |
| `proto.Model_pb2.NeuralNetworkClassifier` | Object | `` |
| `proto.Model_pb2.NeuralNetworkImageScaler` | Object | `` |
| `proto.Model_pb2.NeuralNetworkImageShapeMapping` | Object | `` |
| `proto.Model_pb2.NeuralNetworkLayer` | Object | `` |
| `proto.Model_pb2.NeuralNetworkMeanImage` | Object | `` |
| `proto.Model_pb2.NeuralNetworkMultiArrayShapeMapping` | Object | `` |
| `proto.Model_pb2.NeuralNetworkPreprocessing` | Object | `` |
| `proto.Model_pb2.NeuralNetworkRegressor` | Object | `` |
| `proto.Model_pb2.NeuralNetwork__pb2` | Object | `` |
| `proto.Model_pb2.NoTransform` | Object | `` |
| `proto.Model_pb2.NonMaximumSuppression` | Object | `` |
| `proto.Model_pb2.NonMaximumSuppressionLayerParams` | Object | `` |
| `proto.Model_pb2.NonMaximumSuppression__pb2` | Object | `` |
| `proto.Model_pb2.Normalizer` | Object | `` |
| `proto.Model_pb2.Normalizer__pb2` | Object | `` |
| `proto.Model_pb2.NotEqualLayerParams` | Object | `` |
| `proto.Model_pb2.OneHotEncoder` | Object | `` |
| `proto.Model_pb2.OneHotEncoder__pb2` | Object | `` |
| `proto.Model_pb2.OneHotLayerParams` | Object | `` |
| `proto.Model_pb2.Operation` | Object | `` |
| `proto.Model_pb2.Optimizer` | Object | `` |
| `proto.Model_pb2.PaddingLayerParams` | Object | `` |
| `proto.Model_pb2.Parameters__pb2` | Object | `` |
| `proto.Model_pb2.PermuteLayerParams` | Object | `` |
| `proto.Model_pb2.Pipeline` | Object | `` |
| `proto.Model_pb2.PipelineClassifier` | Object | `` |
| `proto.Model_pb2.PipelineRegressor` | Object | `` |
| `proto.Model_pb2.PolyKernel` | Object | `` |
| `proto.Model_pb2.Pooling3DLayerParams` | Object | `` |
| `proto.Model_pb2.PoolingLayerParams` | Object | `` |
| `proto.Model_pb2.PowBroadcastableLayerParams` | Object | `` |
| `proto.Model_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.Model_pb2.Program` | Object | `` |
| `proto.Model_pb2.QuantizationParams` | Object | `` |
| `proto.Model_pb2.RANK4_IMAGE_MAPPING` | Object | `` |
| `proto.Model_pb2.RANK5_ARRAY_MAPPING` | Object | `` |
| `proto.Model_pb2.RANK5_IMAGE_MAPPING` | Object | `` |
| `proto.Model_pb2.RBFKernel` | Object | `` |
| `proto.Model_pb2.RandomBernoulliDynamicLayerParams` | Object | `` |
| `proto.Model_pb2.RandomBernoulliLikeLayerParams` | Object | `` |
| `proto.Model_pb2.RandomBernoulliStaticLayerParams` | Object | `` |
| `proto.Model_pb2.RandomNormalDynamicLayerParams` | Object | `` |
| `proto.Model_pb2.RandomNormalLikeLayerParams` | Object | `` |
| `proto.Model_pb2.RandomNormalStaticLayerParams` | Object | `` |
| `proto.Model_pb2.RandomUniformDynamicLayerParams` | Object | `` |
| `proto.Model_pb2.RandomUniformLikeLayerParams` | Object | `` |
| `proto.Model_pb2.RandomUniformStaticLayerParams` | Object | `` |
| `proto.Model_pb2.RangeDynamicLayerParams` | Object | `` |
| `proto.Model_pb2.RangeStaticLayerParams` | Object | `` |
| `proto.Model_pb2.RankPreservingReshapeLayerParams` | Object | `` |
| `proto.Model_pb2.ReduceL1LayerParams` | Object | `` |
| `proto.Model_pb2.ReduceL2LayerParams` | Object | `` |
| `proto.Model_pb2.ReduceLayerParams` | Object | `` |
| `proto.Model_pb2.ReduceLogSumExpLayerParams` | Object | `` |
| `proto.Model_pb2.ReduceLogSumLayerParams` | Object | `` |
| `proto.Model_pb2.ReduceMaxLayerParams` | Object | `` |
| `proto.Model_pb2.ReduceMeanLayerParams` | Object | `` |
| `proto.Model_pb2.ReduceMinLayerParams` | Object | `` |
| `proto.Model_pb2.ReduceProdLayerParams` | Object | `` |
| `proto.Model_pb2.ReduceSumLayerParams` | Object | `` |
| `proto.Model_pb2.ReduceSumSquareLayerParams` | Object | `` |
| `proto.Model_pb2.Regression_Logistic` | Object | `` |
| `proto.Model_pb2.ReorganizeDataLayerParams` | Object | `` |
| `proto.Model_pb2.ReshapeDynamicLayerParams` | Object | `` |
| `proto.Model_pb2.ReshapeLayerParams` | Object | `` |
| `proto.Model_pb2.ReshapeLikeLayerParams` | Object | `` |
| `proto.Model_pb2.ReshapeStaticLayerParams` | Object | `` |
| `proto.Model_pb2.ResizeBilinearLayerParams` | Object | `` |
| `proto.Model_pb2.ReverseLayerParams` | Object | `` |
| `proto.Model_pb2.ReverseSeqLayerParams` | Object | `` |
| `proto.Model_pb2.RoundLayerParams` | Object | `` |
| `proto.Model_pb2.SCATTER_ADD` | Object | `` |
| `proto.Model_pb2.SCATTER_DIV` | Object | `` |
| `proto.Model_pb2.SCATTER_MAX` | Object | `` |
| `proto.Model_pb2.SCATTER_MIN` | Object | `` |
| `proto.Model_pb2.SCATTER_MUL` | Object | `` |
| `proto.Model_pb2.SCATTER_SUB` | Object | `` |
| `proto.Model_pb2.SCATTER_UPDATE` | Object | `` |
| `proto.Model_pb2.SGDOptimizer` | Object | `` |
| `proto.Model_pb2.STRING` | Object | `` |
| `proto.Model_pb2.SVM__pb2` | Object | `` |
| `proto.Model_pb2.SamePadding` | Object | `` |
| `proto.Model_pb2.SamplingMode` | Object | `` |
| `proto.Model_pb2.ScaleLayerParams` | Object | `` |
| `proto.Model_pb2.Scaler` | Object | `` |
| `proto.Model_pb2.Scaler__pb2` | Object | `` |
| `proto.Model_pb2.ScatterAlongAxisLayerParams` | Object | `` |
| `proto.Model_pb2.ScatterLayerParams` | Object | `` |
| `proto.Model_pb2.ScatterMode` | Object | `` |
| `proto.Model_pb2.ScatterNDLayerParams` | Object | `` |
| `proto.Model_pb2.SequenceFeatureType` | Object | `` |
| `proto.Model_pb2.SequenceRepeatLayerParams` | Object | `` |
| `proto.Model_pb2.SerializedModel` | Object | `` |
| `proto.Model_pb2.SigmoidKernel` | Object | `` |
| `proto.Model_pb2.SignLayerParams` | Object | `` |
| `proto.Model_pb2.SimpleRecurrentLayerParams` | Object | `` |
| `proto.Model_pb2.SinLayerParams` | Object | `` |
| `proto.Model_pb2.SingleKdTreeIndex` | Object | `` |
| `proto.Model_pb2.SinhLayerParams` | Object | `` |
| `proto.Model_pb2.SizeRange` | Object | `` |
| `proto.Model_pb2.SliceBySizeLayerParams` | Object | `` |
| `proto.Model_pb2.SliceDynamicLayerParams` | Object | `` |
| `proto.Model_pb2.SliceLayerParams` | Object | `` |
| `proto.Model_pb2.SliceStaticLayerParams` | Object | `` |
| `proto.Model_pb2.SlidingWindowsLayerParams` | Object | `` |
| `proto.Model_pb2.SoftmaxLayerParams` | Object | `` |
| `proto.Model_pb2.SoftmaxNDLayerParams` | Object | `` |
| `proto.Model_pb2.SoundAnalysisPreprocessing` | Object | `` |
| `proto.Model_pb2.SoundAnalysisPreprocessing__pb2` | Object | `` |
| `proto.Model_pb2.SparseNode` | Object | `` |
| `proto.Model_pb2.SparseSupportVectors` | Object | `` |
| `proto.Model_pb2.SparseVector` | Object | `` |
| `proto.Model_pb2.SplitLayerParams` | Object | `` |
| `proto.Model_pb2.SplitNDLayerParams` | Object | `` |
| `proto.Model_pb2.SquaredEuclideanDistance` | Object | `` |
| `proto.Model_pb2.SqueezeLayerParams` | Object | `` |
| `proto.Model_pb2.StackLayerParams` | Object | `` |
| `proto.Model_pb2.StateFeatureType` | Object | `` |
| `proto.Model_pb2.StateType` | Object | `` |
| `proto.Model_pb2.StringFeatureType` | Object | `` |
| `proto.Model_pb2.StringParameter` | Object | `` |
| `proto.Model_pb2.StringToDoubleMap` | Object | `` |
| `proto.Model_pb2.StringToInt64Map` | Object | `` |
| `proto.Model_pb2.StringVector` | Object | `` |
| `proto.Model_pb2.SubtractBroadcastableLayerParams` | Object | `` |
| `proto.Model_pb2.SupportVectorClassifier` | Object | `` |
| `proto.Model_pb2.SupportVectorRegressor` | Object | `` |
| `proto.Model_pb2.TanLayerParams` | Object | `` |
| `proto.Model_pb2.TanhLayerParams` | Object | `` |
| `proto.Model_pb2.Tensor` | Object | `` |
| `proto.Model_pb2.TensorType` | Object | `` |
| `proto.Model_pb2.TensorValue` | Object | `` |
| `proto.Model_pb2.TextClassifier` | Object | `` |
| `proto.Model_pb2.TextClassifier__pb2` | Object | `` |
| `proto.Model_pb2.TileLayerParams` | Object | `` |
| `proto.Model_pb2.TopKLayerParams` | Object | `` |
| `proto.Model_pb2.TransposeLayerParams` | Object | `` |
| `proto.Model_pb2.TreeEnsembleClassifier` | Object | `` |
| `proto.Model_pb2.TreeEnsembleParameters` | Object | `` |
| `proto.Model_pb2.TreeEnsemblePostEvaluationTransform` | Object | `` |
| `proto.Model_pb2.TreeEnsembleRegressor` | Object | `` |
| `proto.Model_pb2.TreeEnsemble__pb2` | Object | `` |
| `proto.Model_pb2.TupleType` | Object | `` |
| `proto.Model_pb2.TupleValue` | Object | `` |
| `proto.Model_pb2.UINT1` | Object | `` |
| `proto.Model_pb2.UINT16` | Object | `` |
| `proto.Model_pb2.UINT2` | Object | `` |
| `proto.Model_pb2.UINT3` | Object | `` |
| `proto.Model_pb2.UINT32` | Object | `` |
| `proto.Model_pb2.UINT4` | Object | `` |
| `proto.Model_pb2.UINT6` | Object | `` |
| `proto.Model_pb2.UINT64` | Object | `` |
| `proto.Model_pb2.UINT8` | Object | `` |
| `proto.Model_pb2.UNUSED_TYPE` | Object | `` |
| `proto.Model_pb2.UnaryFunctionLayerParams` | Object | `` |
| `proto.Model_pb2.UniDirectionalLSTMLayerParams` | Object | `` |
| `proto.Model_pb2.UniformWeighting` | Object | `` |
| `proto.Model_pb2.UpperTriangularLayerParams` | Object | `` |
| `proto.Model_pb2.UpsampleLayerParams` | Object | `` |
| `proto.Model_pb2.ValidPadding` | Object | `` |
| `proto.Model_pb2.Value` | Object | `` |
| `proto.Model_pb2.ValueType` | Object | `` |
| `proto.Model_pb2.VisionFeaturePrint` | Object | `` |
| `proto.Model_pb2.VisionFeaturePrint__pb2` | Object | `` |
| `proto.Model_pb2.WeightParams` | Object | `` |
| `proto.Model_pb2.WhereBroadcastableLayerParams` | Object | `` |
| `proto.Model_pb2.WhereNonZeroLayerParams` | Object | `` |
| `proto.Model_pb2.WordEmbedding` | Object | `` |
| `proto.Model_pb2.WordEmbedding__pb2` | Object | `` |
| `proto.Model_pb2.WordTagger` | Object | `` |
| `proto.Model_pb2.WordTagger__pb2` | Object | `` |
| `proto.Model_pb2.enum_type_wrapper` | Object | `` |
| `proto.NamedParameters_pb2.DESCRIPTOR` | Object | `` |
| `proto.NamedParameters_pb2.FloatParameter` | Object | `` |
| `proto.NamedParameters_pb2.FloatRange` | Object | `` |
| `proto.NamedParameters_pb2.Int32Parameter` | Object | `` |
| `proto.NamedParameters_pb2.Int32Range` | Object | `` |
| `proto.NamedParameters_pb2.Int32Set` | Object | `` |
| `proto.NamedParameters_pb2.NamedParameter` | Object | `` |
| `proto.NamedParameters_pb2.Parameter` | Object | `` |
| `proto.NearestNeighbors_pb2.ArrayFeatureType` | Object | `` |
| `proto.NearestNeighbors_pb2.BoolParameter` | Object | `` |
| `proto.NearestNeighbors_pb2.DESCRIPTOR` | Object | `` |
| `proto.NearestNeighbors_pb2.DataStructures__pb2` | Object | `` |
| `proto.NearestNeighbors_pb2.DictionaryFeatureType` | Object | `` |
| `proto.NearestNeighbors_pb2.DoubleFeatureType` | Object | `` |
| `proto.NearestNeighbors_pb2.DoubleParameter` | Object | `` |
| `proto.NearestNeighbors_pb2.DoubleRange` | Object | `` |
| `proto.NearestNeighbors_pb2.DoubleVector` | Object | `` |
| `proto.NearestNeighbors_pb2.FeatureType` | Object | `` |
| `proto.NearestNeighbors_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.NearestNeighbors_pb2.FloatVector` | Object | `` |
| `proto.NearestNeighbors_pb2.ImageFeatureType` | Object | `` |
| `proto.NearestNeighbors_pb2.Int64FeatureType` | Object | `` |
| `proto.NearestNeighbors_pb2.Int64Parameter` | Object | `` |
| `proto.NearestNeighbors_pb2.Int64Range` | Object | `` |
| `proto.NearestNeighbors_pb2.Int64Set` | Object | `` |
| `proto.NearestNeighbors_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.NearestNeighbors_pb2.Int64ToStringMap` | Object | `` |
| `proto.NearestNeighbors_pb2.Int64Vector` | Object | `` |
| `proto.NearestNeighbors_pb2.InverseDistanceWeighting` | Object | `` |
| `proto.NearestNeighbors_pb2.KNearestNeighborsClassifier` | Object | `` |
| `proto.NearestNeighbors_pb2.LinearIndex` | Object | `` |
| `proto.NearestNeighbors_pb2.NearestNeighborsIndex` | Object | `` |
| `proto.NearestNeighbors_pb2.Parameters__pb2` | Object | `` |
| `proto.NearestNeighbors_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.NearestNeighbors_pb2.SequenceFeatureType` | Object | `` |
| `proto.NearestNeighbors_pb2.SingleKdTreeIndex` | Object | `` |
| `proto.NearestNeighbors_pb2.SizeRange` | Object | `` |
| `proto.NearestNeighbors_pb2.SquaredEuclideanDistance` | Object | `` |
| `proto.NearestNeighbors_pb2.StateFeatureType` | Object | `` |
| `proto.NearestNeighbors_pb2.StringFeatureType` | Object | `` |
| `proto.NearestNeighbors_pb2.StringParameter` | Object | `` |
| `proto.NearestNeighbors_pb2.StringToDoubleMap` | Object | `` |
| `proto.NearestNeighbors_pb2.StringToInt64Map` | Object | `` |
| `proto.NearestNeighbors_pb2.StringVector` | Object | `` |
| `proto.NearestNeighbors_pb2.UniformWeighting` | Object | `` |
| `proto.NeuralNetwork_pb2.AcosLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.AcoshLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationELU` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationLeakyReLU` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationLinear` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationPReLU` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationParametricSoftplus` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationReLU` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationScaledTanh` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationSigmoid` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationSigmoidHard` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationSoftplus` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationSoftsign` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationTanh` | Object | `` |
| `proto.NeuralNetwork_pb2.ActivationThresholdedReLU` | Object | `` |
| `proto.NeuralNetwork_pb2.AdamOptimizer` | Object | `` |
| `proto.NeuralNetwork_pb2.AddBroadcastableLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.AddLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ArgMaxLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ArgMinLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ArgSortLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ArrayFeatureType` | Object | `` |
| `proto.NeuralNetwork_pb2.AsinLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.AsinhLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.AtanLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.AtanhLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.AverageLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.BatchedMatMulLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.BatchnormLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.BiDirectionalLSTMLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.BiasLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.BoolParameter` | Object | `` |
| `proto.NeuralNetwork_pb2.BorderAmounts` | Object | `` |
| `proto.NeuralNetwork_pb2.BoxCoordinatesMode` | Object | `` |
| `proto.NeuralNetwork_pb2.BranchLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.BroadcastToDynamicLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.BroadcastToLikeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.BroadcastToStaticLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.CategoricalCrossEntropyLossLayer` | Object | `` |
| `proto.NeuralNetwork_pb2.CategoricalDistributionLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.CeilLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ClampedReLULayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ClipLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ConcatLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ConcatNDLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ConstantPaddingLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.Convolution3DLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ConvolutionLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.CopyLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.CosLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.CoshLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.CropLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.CropResizeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.CumSumLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.CustomLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.DESCRIPTOR` | Object | `` |
| `proto.NeuralNetwork_pb2.DataStructures__pb2` | Object | `` |
| `proto.NeuralNetwork_pb2.DictionaryFeatureType` | Object | `` |
| `proto.NeuralNetwork_pb2.DivideBroadcastableLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.DotProductLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.DoubleFeatureType` | Object | `` |
| `proto.NeuralNetwork_pb2.DoubleParameter` | Object | `` |
| `proto.NeuralNetwork_pb2.DoubleRange` | Object | `` |
| `proto.NeuralNetwork_pb2.DoubleVector` | Object | `` |
| `proto.NeuralNetwork_pb2.EXACT_ARRAY_MAPPING` | Object | `` |
| `proto.NeuralNetwork_pb2.EmbeddingLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.EmbeddingNDLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.EqualLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ErfLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.Exp2LayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ExpandDimsLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.FeatureType` | Object | `` |
| `proto.NeuralNetwork_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.NeuralNetwork_pb2.FillDynamicLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.FillLikeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.FillStaticLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.FlattenLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.FlattenTo2DLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.FloatVector` | Object | `` |
| `proto.NeuralNetwork_pb2.FloorDivBroadcastableLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.FloorLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.GRULayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.GatherAlongAxisLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.GatherLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.GatherNDLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.GeluLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.GetShapeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.GlobalPooling3DLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.GreaterEqualLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.GreaterThanLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ImageFeatureType` | Object | `` |
| `proto.NeuralNetwork_pb2.InnerProductLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.Int64FeatureType` | Object | `` |
| `proto.NeuralNetwork_pb2.Int64Parameter` | Object | `` |
| `proto.NeuralNetwork_pb2.Int64Range` | Object | `` |
| `proto.NeuralNetwork_pb2.Int64Set` | Object | `` |
| `proto.NeuralNetwork_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.NeuralNetwork_pb2.Int64ToStringMap` | Object | `` |
| `proto.NeuralNetwork_pb2.Int64Vector` | Object | `` |
| `proto.NeuralNetwork_pb2.L2NormalizeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LRNLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LSTMParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LSTMWeightParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LayerNormalizationLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LessEqualLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LessThanLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LinearQuantizationParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LoadConstantLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LoadConstantNDLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LogicalAndLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LogicalNotLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LogicalOrLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LogicalXorLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LookUpTableQuantizationParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LoopBreakLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LoopContinueLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LoopLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.LossLayer` | Object | `` |
| `proto.NeuralNetwork_pb2.LowerTriangularLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.MatrixBandPartLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.MaxBroadcastableLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.MaxLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.MeanSquaredErrorLossLayer` | Object | `` |
| `proto.NeuralNetwork_pb2.MeanVarianceNormalizeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.MinBroadcastableLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.MinLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ModBroadcastableLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.MultiplyBroadcastableLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.MultiplyLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.NetworkUpdateParameters` | Object | `` |
| `proto.NeuralNetwork_pb2.NeuralNetwork` | Object | `` |
| `proto.NeuralNetwork_pb2.NeuralNetworkClassifier` | Object | `` |
| `proto.NeuralNetwork_pb2.NeuralNetworkImageScaler` | Object | `` |
| `proto.NeuralNetwork_pb2.NeuralNetworkImageShapeMapping` | Object | `` |
| `proto.NeuralNetwork_pb2.NeuralNetworkLayer` | Object | `` |
| `proto.NeuralNetwork_pb2.NeuralNetworkMeanImage` | Object | `` |
| `proto.NeuralNetwork_pb2.NeuralNetworkMultiArrayShapeMapping` | Object | `` |
| `proto.NeuralNetwork_pb2.NeuralNetworkPreprocessing` | Object | `` |
| `proto.NeuralNetwork_pb2.NeuralNetworkRegressor` | Object | `` |
| `proto.NeuralNetwork_pb2.NonMaximumSuppressionLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.NotEqualLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.OneHotLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.Optimizer` | Object | `` |
| `proto.NeuralNetwork_pb2.PaddingLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.Parameters__pb2` | Object | `` |
| `proto.NeuralNetwork_pb2.PermuteLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.Pooling3DLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.PoolingLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.PowBroadcastableLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.NeuralNetwork_pb2.QuantizationParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RANK4_IMAGE_MAPPING` | Object | `` |
| `proto.NeuralNetwork_pb2.RANK5_ARRAY_MAPPING` | Object | `` |
| `proto.NeuralNetwork_pb2.RANK5_IMAGE_MAPPING` | Object | `` |
| `proto.NeuralNetwork_pb2.RandomBernoulliDynamicLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RandomBernoulliLikeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RandomBernoulliStaticLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RandomNormalDynamicLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RandomNormalLikeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RandomNormalStaticLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RandomUniformDynamicLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RandomUniformLikeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RandomUniformStaticLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RangeDynamicLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RangeStaticLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RankPreservingReshapeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReduceL1LayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReduceL2LayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReduceLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReduceLogSumExpLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReduceLogSumLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReduceMaxLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReduceMeanLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReduceMinLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReduceProdLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReduceSumLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReduceSumSquareLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReorganizeDataLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReshapeDynamicLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReshapeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReshapeLikeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReshapeStaticLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ResizeBilinearLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReverseLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ReverseSeqLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.RoundLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SCATTER_ADD` | Object | `` |
| `proto.NeuralNetwork_pb2.SCATTER_DIV` | Object | `` |
| `proto.NeuralNetwork_pb2.SCATTER_MAX` | Object | `` |
| `proto.NeuralNetwork_pb2.SCATTER_MIN` | Object | `` |
| `proto.NeuralNetwork_pb2.SCATTER_MUL` | Object | `` |
| `proto.NeuralNetwork_pb2.SCATTER_SUB` | Object | `` |
| `proto.NeuralNetwork_pb2.SCATTER_UPDATE` | Object | `` |
| `proto.NeuralNetwork_pb2.SGDOptimizer` | Object | `` |
| `proto.NeuralNetwork_pb2.SamePadding` | Object | `` |
| `proto.NeuralNetwork_pb2.SamplingMode` | Object | `` |
| `proto.NeuralNetwork_pb2.ScaleLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ScatterAlongAxisLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ScatterLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ScatterMode` | Object | `` |
| `proto.NeuralNetwork_pb2.ScatterNDLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SequenceFeatureType` | Object | `` |
| `proto.NeuralNetwork_pb2.SequenceRepeatLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SignLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SimpleRecurrentLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SinLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SinhLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SizeRange` | Object | `` |
| `proto.NeuralNetwork_pb2.SliceBySizeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SliceDynamicLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SliceLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SliceStaticLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SlidingWindowsLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SoftmaxLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SoftmaxNDLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SplitLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SplitNDLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.SqueezeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.StackLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.StateFeatureType` | Object | `` |
| `proto.NeuralNetwork_pb2.StringFeatureType` | Object | `` |
| `proto.NeuralNetwork_pb2.StringParameter` | Object | `` |
| `proto.NeuralNetwork_pb2.StringToDoubleMap` | Object | `` |
| `proto.NeuralNetwork_pb2.StringToInt64Map` | Object | `` |
| `proto.NeuralNetwork_pb2.StringVector` | Object | `` |
| `proto.NeuralNetwork_pb2.SubtractBroadcastableLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.TanLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.TanhLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.Tensor` | Object | `` |
| `proto.NeuralNetwork_pb2.TileLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.TopKLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.TransposeLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.UnaryFunctionLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.UniDirectionalLSTMLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.UpperTriangularLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.UpsampleLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.ValidPadding` | Object | `` |
| `proto.NeuralNetwork_pb2.WeightParams` | Object | `` |
| `proto.NeuralNetwork_pb2.WhereBroadcastableLayerParams` | Object | `` |
| `proto.NeuralNetwork_pb2.WhereNonZeroLayerParams` | Object | `` |
| `proto.NonMaximumSuppression_pb2.ArrayFeatureType` | Object | `` |
| `proto.NonMaximumSuppression_pb2.DESCRIPTOR` | Object | `` |
| `proto.NonMaximumSuppression_pb2.DataStructures__pb2` | Object | `` |
| `proto.NonMaximumSuppression_pb2.DictionaryFeatureType` | Object | `` |
| `proto.NonMaximumSuppression_pb2.DoubleFeatureType` | Object | `` |
| `proto.NonMaximumSuppression_pb2.DoubleRange` | Object | `` |
| `proto.NonMaximumSuppression_pb2.DoubleVector` | Object | `` |
| `proto.NonMaximumSuppression_pb2.FeatureType` | Object | `` |
| `proto.NonMaximumSuppression_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.NonMaximumSuppression_pb2.FloatVector` | Object | `` |
| `proto.NonMaximumSuppression_pb2.ImageFeatureType` | Object | `` |
| `proto.NonMaximumSuppression_pb2.Int64FeatureType` | Object | `` |
| `proto.NonMaximumSuppression_pb2.Int64Range` | Object | `` |
| `proto.NonMaximumSuppression_pb2.Int64Set` | Object | `` |
| `proto.NonMaximumSuppression_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.NonMaximumSuppression_pb2.Int64ToStringMap` | Object | `` |
| `proto.NonMaximumSuppression_pb2.Int64Vector` | Object | `` |
| `proto.NonMaximumSuppression_pb2.NonMaximumSuppression` | Object | `` |
| `proto.NonMaximumSuppression_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.NonMaximumSuppression_pb2.SequenceFeatureType` | Object | `` |
| `proto.NonMaximumSuppression_pb2.SizeRange` | Object | `` |
| `proto.NonMaximumSuppression_pb2.StateFeatureType` | Object | `` |
| `proto.NonMaximumSuppression_pb2.StringFeatureType` | Object | `` |
| `proto.NonMaximumSuppression_pb2.StringToDoubleMap` | Object | `` |
| `proto.NonMaximumSuppression_pb2.StringToInt64Map` | Object | `` |
| `proto.NonMaximumSuppression_pb2.StringVector` | Object | `` |
| `proto.Normalizer_pb2.DESCRIPTOR` | Object | `` |
| `proto.Normalizer_pb2.Normalizer` | Object | `` |
| `proto.OneHotEncoder_pb2.ArrayFeatureType` | Object | `` |
| `proto.OneHotEncoder_pb2.DESCRIPTOR` | Object | `` |
| `proto.OneHotEncoder_pb2.DataStructures__pb2` | Object | `` |
| `proto.OneHotEncoder_pb2.DictionaryFeatureType` | Object | `` |
| `proto.OneHotEncoder_pb2.DoubleFeatureType` | Object | `` |
| `proto.OneHotEncoder_pb2.DoubleRange` | Object | `` |
| `proto.OneHotEncoder_pb2.DoubleVector` | Object | `` |
| `proto.OneHotEncoder_pb2.FeatureType` | Object | `` |
| `proto.OneHotEncoder_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.OneHotEncoder_pb2.FloatVector` | Object | `` |
| `proto.OneHotEncoder_pb2.ImageFeatureType` | Object | `` |
| `proto.OneHotEncoder_pb2.Int64FeatureType` | Object | `` |
| `proto.OneHotEncoder_pb2.Int64Range` | Object | `` |
| `proto.OneHotEncoder_pb2.Int64Set` | Object | `` |
| `proto.OneHotEncoder_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.OneHotEncoder_pb2.Int64ToStringMap` | Object | `` |
| `proto.OneHotEncoder_pb2.Int64Vector` | Object | `` |
| `proto.OneHotEncoder_pb2.OneHotEncoder` | Object | `` |
| `proto.OneHotEncoder_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.OneHotEncoder_pb2.SequenceFeatureType` | Object | `` |
| `proto.OneHotEncoder_pb2.SizeRange` | Object | `` |
| `proto.OneHotEncoder_pb2.StateFeatureType` | Object | `` |
| `proto.OneHotEncoder_pb2.StringFeatureType` | Object | `` |
| `proto.OneHotEncoder_pb2.StringToDoubleMap` | Object | `` |
| `proto.OneHotEncoder_pb2.StringToInt64Map` | Object | `` |
| `proto.OneHotEncoder_pb2.StringVector` | Object | `` |
| `proto.Parameters_pb2.ArrayFeatureType` | Object | `` |
| `proto.Parameters_pb2.BoolParameter` | Object | `` |
| `proto.Parameters_pb2.DESCRIPTOR` | Object | `` |
| `proto.Parameters_pb2.DataStructures__pb2` | Object | `` |
| `proto.Parameters_pb2.DictionaryFeatureType` | Object | `` |
| `proto.Parameters_pb2.DoubleFeatureType` | Object | `` |
| `proto.Parameters_pb2.DoubleParameter` | Object | `` |
| `proto.Parameters_pb2.DoubleRange` | Object | `` |
| `proto.Parameters_pb2.DoubleVector` | Object | `` |
| `proto.Parameters_pb2.FeatureType` | Object | `` |
| `proto.Parameters_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.Parameters_pb2.FloatVector` | Object | `` |
| `proto.Parameters_pb2.ImageFeatureType` | Object | `` |
| `proto.Parameters_pb2.Int64FeatureType` | Object | `` |
| `proto.Parameters_pb2.Int64Parameter` | Object | `` |
| `proto.Parameters_pb2.Int64Range` | Object | `` |
| `proto.Parameters_pb2.Int64Set` | Object | `` |
| `proto.Parameters_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.Parameters_pb2.Int64ToStringMap` | Object | `` |
| `proto.Parameters_pb2.Int64Vector` | Object | `` |
| `proto.Parameters_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.Parameters_pb2.SequenceFeatureType` | Object | `` |
| `proto.Parameters_pb2.SizeRange` | Object | `` |
| `proto.Parameters_pb2.StateFeatureType` | Object | `` |
| `proto.Parameters_pb2.StringFeatureType` | Object | `` |
| `proto.Parameters_pb2.StringParameter` | Object | `` |
| `proto.Parameters_pb2.StringToDoubleMap` | Object | `` |
| `proto.Parameters_pb2.StringToInt64Map` | Object | `` |
| `proto.Parameters_pb2.StringVector` | Object | `` |
| `proto.SVM_pb2.ArrayFeatureType` | Object | `` |
| `proto.SVM_pb2.Coefficients` | Object | `` |
| `proto.SVM_pb2.DESCRIPTOR` | Object | `` |
| `proto.SVM_pb2.DataStructures__pb2` | Object | `` |
| `proto.SVM_pb2.DenseSupportVectors` | Object | `` |
| `proto.SVM_pb2.DenseVector` | Object | `` |
| `proto.SVM_pb2.DictionaryFeatureType` | Object | `` |
| `proto.SVM_pb2.DoubleFeatureType` | Object | `` |
| `proto.SVM_pb2.DoubleRange` | Object | `` |
| `proto.SVM_pb2.DoubleVector` | Object | `` |
| `proto.SVM_pb2.FeatureType` | Object | `` |
| `proto.SVM_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.SVM_pb2.FloatVector` | Object | `` |
| `proto.SVM_pb2.ImageFeatureType` | Object | `` |
| `proto.SVM_pb2.Int64FeatureType` | Object | `` |
| `proto.SVM_pb2.Int64Range` | Object | `` |
| `proto.SVM_pb2.Int64Set` | Object | `` |
| `proto.SVM_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.SVM_pb2.Int64ToStringMap` | Object | `` |
| `proto.SVM_pb2.Int64Vector` | Object | `` |
| `proto.SVM_pb2.Kernel` | Object | `` |
| `proto.SVM_pb2.LinearKernel` | Object | `` |
| `proto.SVM_pb2.PolyKernel` | Object | `` |
| `proto.SVM_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.SVM_pb2.RBFKernel` | Object | `` |
| `proto.SVM_pb2.SequenceFeatureType` | Object | `` |
| `proto.SVM_pb2.SigmoidKernel` | Object | `` |
| `proto.SVM_pb2.SizeRange` | Object | `` |
| `proto.SVM_pb2.SparseNode` | Object | `` |
| `proto.SVM_pb2.SparseSupportVectors` | Object | `` |
| `proto.SVM_pb2.SparseVector` | Object | `` |
| `proto.SVM_pb2.StateFeatureType` | Object | `` |
| `proto.SVM_pb2.StringFeatureType` | Object | `` |
| `proto.SVM_pb2.StringToDoubleMap` | Object | `` |
| `proto.SVM_pb2.StringToInt64Map` | Object | `` |
| `proto.SVM_pb2.StringVector` | Object | `` |
| `proto.SVM_pb2.SupportVectorClassifier` | Object | `` |
| `proto.SVM_pb2.SupportVectorRegressor` | Object | `` |
| `proto.Scaler_pb2.DESCRIPTOR` | Object | `` |
| `proto.Scaler_pb2.Scaler` | Object | `` |
| `proto.SoundAnalysisPreprocessing_pb2.DESCRIPTOR` | Object | `` |
| `proto.SoundAnalysisPreprocessing_pb2.SoundAnalysisPreprocessing` | Object | `` |
| `proto.TextClassifier_pb2.ArrayFeatureType` | Object | `` |
| `proto.TextClassifier_pb2.DESCRIPTOR` | Object | `` |
| `proto.TextClassifier_pb2.DataStructures__pb2` | Object | `` |
| `proto.TextClassifier_pb2.DictionaryFeatureType` | Object | `` |
| `proto.TextClassifier_pb2.DoubleFeatureType` | Object | `` |
| `proto.TextClassifier_pb2.DoubleRange` | Object | `` |
| `proto.TextClassifier_pb2.DoubleVector` | Object | `` |
| `proto.TextClassifier_pb2.FeatureType` | Object | `` |
| `proto.TextClassifier_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.TextClassifier_pb2.FloatVector` | Object | `` |
| `proto.TextClassifier_pb2.ImageFeatureType` | Object | `` |
| `proto.TextClassifier_pb2.Int64FeatureType` | Object | `` |
| `proto.TextClassifier_pb2.Int64Range` | Object | `` |
| `proto.TextClassifier_pb2.Int64Set` | Object | `` |
| `proto.TextClassifier_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.TextClassifier_pb2.Int64ToStringMap` | Object | `` |
| `proto.TextClassifier_pb2.Int64Vector` | Object | `` |
| `proto.TextClassifier_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.TextClassifier_pb2.SequenceFeatureType` | Object | `` |
| `proto.TextClassifier_pb2.SizeRange` | Object | `` |
| `proto.TextClassifier_pb2.StateFeatureType` | Object | `` |
| `proto.TextClassifier_pb2.StringFeatureType` | Object | `` |
| `proto.TextClassifier_pb2.StringToDoubleMap` | Object | `` |
| `proto.TextClassifier_pb2.StringToInt64Map` | Object | `` |
| `proto.TextClassifier_pb2.StringVector` | Object | `` |
| `proto.TextClassifier_pb2.TextClassifier` | Object | `` |
| `proto.TreeEnsemble_pb2.ArrayFeatureType` | Object | `` |
| `proto.TreeEnsemble_pb2.Classification_SoftMax` | Object | `` |
| `proto.TreeEnsemble_pb2.Classification_SoftMaxWithZeroClassReference` | Object | `` |
| `proto.TreeEnsemble_pb2.DESCRIPTOR` | Object | `` |
| `proto.TreeEnsemble_pb2.DataStructures__pb2` | Object | `` |
| `proto.TreeEnsemble_pb2.DictionaryFeatureType` | Object | `` |
| `proto.TreeEnsemble_pb2.DoubleFeatureType` | Object | `` |
| `proto.TreeEnsemble_pb2.DoubleRange` | Object | `` |
| `proto.TreeEnsemble_pb2.DoubleVector` | Object | `` |
| `proto.TreeEnsemble_pb2.FeatureType` | Object | `` |
| `proto.TreeEnsemble_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.TreeEnsemble_pb2.FloatVector` | Object | `` |
| `proto.TreeEnsemble_pb2.ImageFeatureType` | Object | `` |
| `proto.TreeEnsemble_pb2.Int64FeatureType` | Object | `` |
| `proto.TreeEnsemble_pb2.Int64Range` | Object | `` |
| `proto.TreeEnsemble_pb2.Int64Set` | Object | `` |
| `proto.TreeEnsemble_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.TreeEnsemble_pb2.Int64ToStringMap` | Object | `` |
| `proto.TreeEnsemble_pb2.Int64Vector` | Object | `` |
| `proto.TreeEnsemble_pb2.NoTransform` | Object | `` |
| `proto.TreeEnsemble_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.TreeEnsemble_pb2.Regression_Logistic` | Object | `` |
| `proto.TreeEnsemble_pb2.SequenceFeatureType` | Object | `` |
| `proto.TreeEnsemble_pb2.SizeRange` | Object | `` |
| `proto.TreeEnsemble_pb2.StateFeatureType` | Object | `` |
| `proto.TreeEnsemble_pb2.StringFeatureType` | Object | `` |
| `proto.TreeEnsemble_pb2.StringToDoubleMap` | Object | `` |
| `proto.TreeEnsemble_pb2.StringToInt64Map` | Object | `` |
| `proto.TreeEnsemble_pb2.StringVector` | Object | `` |
| `proto.TreeEnsemble_pb2.TreeEnsembleClassifier` | Object | `` |
| `proto.TreeEnsemble_pb2.TreeEnsembleParameters` | Object | `` |
| `proto.TreeEnsemble_pb2.TreeEnsemblePostEvaluationTransform` | Object | `` |
| `proto.TreeEnsemble_pb2.TreeEnsembleRegressor` | Object | `` |
| `proto.VisionFeaturePrint_pb2.DESCRIPTOR` | Object | `` |
| `proto.VisionFeaturePrint_pb2.VisionFeaturePrint` | Object | `` |
| `proto.WordEmbedding_pb2.ArrayFeatureType` | Object | `` |
| `proto.WordEmbedding_pb2.DESCRIPTOR` | Object | `` |
| `proto.WordEmbedding_pb2.DataStructures__pb2` | Object | `` |
| `proto.WordEmbedding_pb2.DictionaryFeatureType` | Object | `` |
| `proto.WordEmbedding_pb2.DoubleFeatureType` | Object | `` |
| `proto.WordEmbedding_pb2.DoubleRange` | Object | `` |
| `proto.WordEmbedding_pb2.DoubleVector` | Object | `` |
| `proto.WordEmbedding_pb2.FeatureType` | Object | `` |
| `proto.WordEmbedding_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.WordEmbedding_pb2.FloatVector` | Object | `` |
| `proto.WordEmbedding_pb2.ImageFeatureType` | Object | `` |
| `proto.WordEmbedding_pb2.Int64FeatureType` | Object | `` |
| `proto.WordEmbedding_pb2.Int64Range` | Object | `` |
| `proto.WordEmbedding_pb2.Int64Set` | Object | `` |
| `proto.WordEmbedding_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.WordEmbedding_pb2.Int64ToStringMap` | Object | `` |
| `proto.WordEmbedding_pb2.Int64Vector` | Object | `` |
| `proto.WordEmbedding_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.WordEmbedding_pb2.SequenceFeatureType` | Object | `` |
| `proto.WordEmbedding_pb2.SizeRange` | Object | `` |
| `proto.WordEmbedding_pb2.StateFeatureType` | Object | `` |
| `proto.WordEmbedding_pb2.StringFeatureType` | Object | `` |
| `proto.WordEmbedding_pb2.StringToDoubleMap` | Object | `` |
| `proto.WordEmbedding_pb2.StringToInt64Map` | Object | `` |
| `proto.WordEmbedding_pb2.StringVector` | Object | `` |
| `proto.WordEmbedding_pb2.WordEmbedding` | Object | `` |
| `proto.WordTagger_pb2.ArrayFeatureType` | Object | `` |
| `proto.WordTagger_pb2.DESCRIPTOR` | Object | `` |
| `proto.WordTagger_pb2.DataStructures__pb2` | Object | `` |
| `proto.WordTagger_pb2.DictionaryFeatureType` | Object | `` |
| `proto.WordTagger_pb2.DoubleFeatureType` | Object | `` |
| `proto.WordTagger_pb2.DoubleRange` | Object | `` |
| `proto.WordTagger_pb2.DoubleVector` | Object | `` |
| `proto.WordTagger_pb2.FeatureType` | Object | `` |
| `proto.WordTagger_pb2.FeatureTypes__pb2` | Object | `` |
| `proto.WordTagger_pb2.FloatVector` | Object | `` |
| `proto.WordTagger_pb2.ImageFeatureType` | Object | `` |
| `proto.WordTagger_pb2.Int64FeatureType` | Object | `` |
| `proto.WordTagger_pb2.Int64Range` | Object | `` |
| `proto.WordTagger_pb2.Int64Set` | Object | `` |
| `proto.WordTagger_pb2.Int64ToDoubleMap` | Object | `` |
| `proto.WordTagger_pb2.Int64ToStringMap` | Object | `` |
| `proto.WordTagger_pb2.Int64Vector` | Object | `` |
| `proto.WordTagger_pb2.PrecisionRecallCurve` | Object | `` |
| `proto.WordTagger_pb2.SequenceFeatureType` | Object | `` |
| `proto.WordTagger_pb2.SizeRange` | Object | `` |
| `proto.WordTagger_pb2.StateFeatureType` | Object | `` |
| `proto.WordTagger_pb2.StringFeatureType` | Object | `` |
| `proto.WordTagger_pb2.StringToDoubleMap` | Object | `` |
| `proto.WordTagger_pb2.StringToInt64Map` | Object | `` |
| `proto.WordTagger_pb2.StringVector` | Object | `` |
| `proto.WordTagger_pb2.WordTagger` | Object | `` |
| `target` | Class | `(...)` |
| `test.api.test_api_examples.Function` | Class | `(inputs, opset_version)` |
| `test.api.test_api_examples.MLCPUComputeDevice` | Class | `(proxy)` |
| `test.api.test_api_examples.MLComputeDevice` | Class | `(...)` |
| `test.api.test_api_examples.MLComputePlan` | Class | `(proxy)` |
| `test.api.test_api_examples.MLModelStructure` | Class | `(neuralnetwork: _Optional[MLModelStructureNeuralNetwork], program: _Optional[MLModelStructureProgram], pipeline: _Optional[MLModelStructurePipeline])` |
| `test.api.test_api_examples.MLNeuralEngineComputeDevice` | Class | `(proxy)` |
| `test.api.test_api_examples.TestGraphPassManagement` | Class | `(...)` |
| `test.api.test_api_examples.TestInputs` | Class | `(...)` |
| `test.api.test_api_examples.TestMILExamples` | Class | `(...)` |
| `test.api.test_api_examples.TestMLComputeDevice` | Class | `(...)` |
| `test.api.test_api_examples.TestMLComputePlan` | Class | `(...)` |
| `test.api.test_api_examples.TestMLModelAsset` | Class | `(...)` |
| `test.api.test_api_examples.TestMLModelStructure` | Class | `(...)` |
| `test.api.test_api_examples.TestMLProgramConverterExamples` | Class | `(...)` |
| `test.api.test_api_examples.TestMLProgramFP16Transform` | Class | `(...)` |
| `test.api.test_api_examples.TestMultipleEnumeratedShapes` | Class | `(...)` |
| `test.api.test_api_examples.get_new_symbol` | Function | `(name)` |
| `test.api.test_api_examples.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `test.api.test_api_examples.mb` | Class | `(...)` |
| `test.api.test_api_examples.mil` | Object | `` |
| `test.api.test_api_examples.proto` | Object | `` |
| `test.api.test_api_examples.types` | Object | `` |
| `test.api.test_api_visibilities.EXPECTED_MODULES` | Object | `` |
| `test.api.test_api_visibilities.TestApiVisibilities` | Class | `(...)` |
| `test.blob.test_weights.BlobReader` | Object | `` |
| `test.blob.test_weights.BlobWriter` | Object | `` |
| `test.blob.test_weights.TestWeightBlob` | Class | `(...)` |
| `test.blob.test_weights.TestWeightIDSharing` | Class | `(...)` |
| `test.blob.test_weights.mb` | Class | `(...)` |
| `test.blob.test_weights.mil` | Object | `` |
| `test.blob.test_weights.types` | Object | `` |
| `test.ml_program.experimental.test_async_wrapper.MLModelAsyncWrapper` | Class | `(spec_or_path: Union[proto.Model_pb2.Model, str], weights_dir: str, compute_units: ComputeUnit, function_name: Optional[str], optimization_hints: Optional[Dict[str, Any]])` |
| `test.ml_program.experimental.test_async_wrapper.TestMLModelAsyncWrapper` | Class | `(...)` |
| `test.ml_program.experimental.test_async_wrapper.mb` | Class | `(...)` |
| `test.ml_program.experimental.test_compute_plan_utils.ComputePlan` | Class | `(infos: _Dict[ModelStructurePath, ComputePlan.OperationOrLayerInfo])` |
| `test.ml_program.experimental.test_compute_plan_utils.IS_INTEL_MAC` | Object | `` |
| `test.ml_program.experimental.test_compute_plan_utils.MLCPUComputeDevice` | Class | `(proxy)` |
| `test.ml_program.experimental.test_compute_plan_utils.MLComputeDevice` | Class | `(...)` |
| `test.ml_program.experimental.test_compute_plan_utils.MLComputePlan` | Class | `(proxy)` |
| `test.ml_program.experimental.test_compute_plan_utils.MLGPUComputeDevice` | Class | `(proxy)` |
| `test.ml_program.experimental.test_compute_plan_utils.MLModelStructure` | Class | `(neuralnetwork: _Optional[MLModelStructureNeuralNetwork], program: _Optional[MLModelStructureProgram], pipeline: _Optional[MLModelStructurePipeline])` |
| `test.ml_program.experimental.test_compute_plan_utils.MLModelStructureProgramOperation` | Class | `(inputs: _Dict[str, MLModelStructureProgramArgument], operator_name: str, outputs: _List[MLModelStructureProgramNamedValueType], blocks: _List[MLModelStructureProgramBlock], __proxy__: _Any)` |
| `test.ml_program.experimental.test_compute_plan_utils.MLNeuralEngineComputeDevice` | Class | `(proxy)` |
| `test.ml_program.experimental.test_compute_plan_utils.ModelStructurePath` | Class | `(components: _Tuple[Component, ...])` |
| `test.ml_program.experimental.test_compute_plan_utils.TestComputePlanUtils` | Class | `(...)` |
| `test.ml_program.experimental.test_compute_plan_utils.map_model_structure_to_path` | Function | `(model_structure: MLModelStructure, components: _List[ModelStructurePath.Component]) -> _List[_Tuple[ModelStructureLayerOrOperation, ModelStructurePath]]` |
| `test.ml_program.experimental.test_compute_plan_utils.mb` | Class | `(...)` |
| `test.ml_program.experimental.test_compute_plan_utils.pytestmark` | Object | `` |
| `test.ml_program.experimental.test_debugging_utils.MLModelComparator` | Class | `(reference_model: MLModel, target_model: MLModel, function_name: Optional[str], optimization_hints: Optional[Dict[str, Any]], num_predict_intermediate_outputs: int, reference_device: Optional[Device], target_device: Optional[Device])` |
| `test.ml_program.experimental.test_debugging_utils.MLModelInspector` | Class | `(model: MLModel, compute_units: ComputeUnit, function_name: Optional[str], optimization_hints: Optional[Dict[str, Any]], device: Optional[Device])` |
| `test.ml_program.experimental.test_debugging_utils.MLModelValidator` | Class | `(model: MLModel, function_name: Optional[str], compute_units: ComputeUnit, optimization_hints: Optional[Dict[str, Any]], num_predict_intermediate_outputs: int, device: Optional[Device])` |
| `test.ml_program.experimental.test_debugging_utils.TestMLModelComparator` | Class | `(...)` |
| `test.ml_program.experimental.test_debugging_utils.TestMLModelInspector` | Class | `(...)` |
| `test.ml_program.experimental.test_debugging_utils.TestMLModelValidator` | Class | `(...)` |
| `test.ml_program.experimental.test_debugging_utils.compute_ground_truth_answer` | Function | `(input)` |
| `test.ml_program.experimental.test_debugging_utils.compute_snr_and_psnr` | Function | `(x: np.array, y: np.array) -> Tuple[float, float]` |
| `test.ml_program.experimental.test_debugging_utils.get_simple_program` | Function | `()` |
| `test.ml_program.experimental.test_debugging_utils.mb` | Class | `(...)` |
| `test.ml_program.experimental.test_debugging_utils.proto` | Object | `` |
| `test.ml_program.experimental.test_debugging_utils.skip_op_by_type` | Function | `(op: proto.MIL_pb2.Operation, op_types: Iterable[str]) -> bool` |
| `test.ml_program.experimental.test_perf_utils.IS_INTEL_MAC` | Object | `` |
| `test.ml_program.experimental.test_perf_utils.MLModelBenchmarker` | Class | `(model: MLModel, device: Optional[Device])` |
| `test.ml_program.experimental.test_perf_utils.TestMLModelBenchmarker` | Class | `(...)` |
| `test.ml_program.experimental.test_perf_utils.TestTorchMLModelBenchmarker` | Class | `(...)` |
| `test.ml_program.experimental.test_perf_utils.TorchMLModelBenchmarker` | Class | `(model: Union[ExportedProgram, torch.jit.ScriptModule], device: Optional[Device], converter_kwargs)` |
| `test.ml_program.experimental.test_perf_utils.TorchNode` | Object | `` |
| `test.ml_program.experimental.test_perf_utils.TorchScriptNodeInfo` | Class | `(source_range: str, modules: Tuple[TorchScriptModuleInfo.Key], desc: str, kind: str, input_names: Tuple[str], output_names: Tuple[str])` |
| `test.ml_program.experimental.test_perf_utils.mb` | Class | `(...)` |
| `test.ml_program.experimental.test_perf_utils.pytestmark` | Object | `` |
| `test.ml_program.experimental.test_perf_utils.types` | Object | `` |
| `test.ml_program.experimental.test_remote_device.CompiledMLModel` | Class | `(path: str, compute_units: _ComputeUnit, function_name: _Optional[str], optimization_hints: _Optional[dict], asset: _Optional[_MLModelAsset])` |
| `test.ml_program.experimental.test_remote_device.Device` | Class | `(name: str, type: DeviceType, identifier: str, udid: str, os_version: str, os_build_number: str, developer_mode_state: str, state: DeviceState, session: _Optional[_Type[_AppSession]])` |
| `test.ml_program.experimental.test_remote_device.DeviceState` | Class | `(...)` |
| `test.ml_program.experimental.test_remote_device.DeviceType` | Class | `(...)` |
| `test.ml_program.experimental.test_remote_device.IS_INTEL_MAC` | Object | `` |
| `test.ml_program.experimental.test_remote_device.MockDeviceCtlSocket` | Class | `()` |
| `test.ml_program.experimental.test_remote_device.TestModelRunnerApp` | Class | `(...)` |
| `test.ml_program.experimental.test_remote_device.TestRemoteMLModelService` | Class | `(...)` |
| `test.ml_program.experimental.test_remote_device.has_xcodebuild` | Function | `()` |
| `test.ml_program.experimental.test_remote_device.mb` | Class | `(...)` |
| `test.ml_program.experimental.test_torch_debugging_utils.TestTorchMapping` | Class | `(...)` |
| `test.ml_program.experimental.test_torch_debugging_utils.TestTorchModelComparator` | Class | `(...)` |
| `test.ml_program.experimental.test_torch_debugging_utils.TorchExportMLModelComparator` | Class | `(model: ExportedProgram, num_predict_intermediate_outputs: int, target_device: Optional[Device], converter_kwargs)` |
| `test.ml_program.experimental.test_torch_debugging_utils.TorchScriptMLModelComparator` | Class | `(model: torch.nn.Module, example_inputs: Tuple[torch.tensor], num_predict_intermediate_outputs: int, target_device: Optional[Device], converter_kwargs)` |
| `test.ml_program.experimental.test_torch_debugging_utils.convert_and_retrieve_op_mapping` | Function | `(model: Union[ExportedProgram, ScriptModule], converter_kwargs) -> Tuple[MLModel, TorchNodeToMILOperationMapping]` |
| `test.ml_program.experimental.test_torch_debugging_utils.get_stack_frame_infos` | Function | `(node: TorchNode) -> Optional[List[FrameInfo]]` |
| `test.ml_program.experimental.test_torch_debugging_utils.inline_and_annotate_module` | Function | `(model: ScriptModule, name_prefix: str) -> TorchScriptModuleAnnotator` |
| `test.ml_program.experimental.test_torch_debugging_utils.proto` | Object | `` |
| `test.ml_program.test_compression.OpCompressorConfig` | Class | `(...)` |
| `test.ml_program.test_compression.TestCompressionUtils` | Class | `(...)` |
| `test.ml_program.test_compression.affine_quantize_weights` | Function | `(mlmodel, mode, op_selector, dtype)` |
| `test.ml_program.test_compression.decompress_weights` | Function | `(mlmodel)` |
| `test.ml_program.test_compression.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `test.ml_program.test_compression.get_test_model_and_data` | Function | `(multi_layer: bool, quantize_config: Optional[OpCompressorConfig], use_linear: bool)` |
| `test.ml_program.test_compression.palettize_weights` | Function | `(mlmodel, nbits, mode, op_selector, lut_function)` |
| `test.ml_program.test_compression.sparsify_weights` | Function | `(mlmodel, mode, threshold, target_percentile, op_selector)` |
| `test.ml_program.test_utils.ArrayFeatureType` | Object | `` |
| `test.ml_program.test_utils.MultiFunctionDescriptor` | Class | `(model_path: _Optional[str])` |
| `test.ml_program.test_utils.PassPipeline` | Class | `(pass_names, pipeline_name)` |
| `test.ml_program.test_utils.PassPipelineManager` | Class | `(...)` |
| `test.ml_program.test_utils.Program` | Class | `()` |
| `test.ml_program.test_utils.TestBisectModel` | Class | `(...)` |
| `test.ml_program.test_utils.TestChangeInputOutputTensorType` | Class | `(...)` |
| `test.ml_program.test_utils.TestMILConvertCall` | Class | `(...)` |
| `test.ml_program.test_utils.TestMaterializeSymbolicShapeMLModel` | Class | `(...)` |
| `test.ml_program.test_utils.TestMultiFunctionDescriptor` | Class | `(...)` |
| `test.ml_program.test_utils.TestMultiFunctionModelEnd2End` | Class | `(...)` |
| `test.ml_program.test_utils.assert_spec_input_type` | Function | `(spec, expected_feature_type, expected_name, index)` |
| `test.ml_program.test_utils.assert_spec_output_type` | Function | `(spec, expected_feature_type, expected_name, index)` |
| `test.ml_program.test_utils.bisect_model` | Function | `(model: _Union[str, _ct.models.MLModel], output_dir: str, merge_chunks_to_pipeline: _Optional[bool], check_output_correctness: _Optional[bool])` |
| `test.ml_program.test_utils.change_input_output_tensor_type` | Function | `(ml_model: _ct.models.MLModel, from_type: _proto.FeatureTypes_pb2.ArrayFeatureType, to_type: _proto.FeatureTypes_pb2.ArrayFeatureType, function_names: _Optional[_List[str]], input_names: _Optional[_List[str]], output_names: _Optional[_List[str]]) -> _ct.models.model.MLModel` |
| `test.ml_program.test_utils.cto` | Object | `` |
| `test.ml_program.test_utils.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `test.ml_program.test_utils.load_spec` | Function | `(model_path: str) -> _proto.Model_pb2` |
| `test.ml_program.test_utils.mb` | Class | `(...)` |
| `test.ml_program.test_utils.mil` | Object | `` |
| `test.ml_program.test_utils.proto` | Object | `` |
| `test.ml_program.test_utils.save_multifunction` | Function | `(desc: MultiFunctionDescriptor, destination_path: str)` |
| `test.ml_program.test_utils.str_to_proto_feature_type` | Function | `(dtype: str) -> proto.FeatureTypes_pb2.ArrayFeatureType` |
| `test.ml_program.test_utils.types` | Object | `` |
| `test.modelpackage.test_mlmodel.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `test.modelpackage.test_mlmodel.test_mlmodel_demo` | Function | `(tmpdir)` |
| `test.modelpackage.test_modelpackage.CompiledMLModel` | Class | `(path: str, compute_units: _ComputeUnit, function_name: _Optional[str], optimization_hints: _Optional[dict], asset: _Optional[_MLModelAsset])` |
| `test.modelpackage.test_modelpackage.ComputeUnit` | Class | `(...)` |
| `test.modelpackage.test_modelpackage.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `test.modelpackage.test_modelpackage.ModelPackage` | Object | `` |
| `test.modelpackage.test_modelpackage.TestCompiledMLModel` | Class | `(...)` |
| `test.modelpackage.test_modelpackage.TestMLModel` | Class | `(...)` |
| `test.modelpackage.test_modelpackage.TestSpecAndMLModelAPIs` | Class | `(...)` |
| `test.modelpackage.test_modelpackage.mb` | Class | `(...)` |
| `test.modelpackage.test_modelpackage.proto` | Object | `` |
| `test.modelpackage.test_modelpackage.types` | Object | `` |
| `test.modelpackage.test_modelpackage.utils` | Object | `` |
| `test.neural_network.test_compiled_model.CompiledMLModel` | Class | `(path: str, compute_units: _ComputeUnit, function_name: _Optional[str], optimization_hints: _Optional[dict], asset: _Optional[_MLModelAsset])` |
| `test.neural_network.test_compiled_model.ComputeUnit` | Class | `(...)` |
| `test.neural_network.test_compiled_model.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `test.neural_network.test_compiled_model.ReshapeFrequency` | Class | `(...)` |
| `test.neural_network.test_compiled_model.SpecializationStrategy` | Class | `(...)` |
| `test.neural_network.test_compiled_model.TestCompiledModel` | Class | `(...)` |
| `test.neural_network.test_compiled_model.compile_model` | Function | `(model: _Union[_proto.Model_pb2.Model, str], destination_path: _Optional[str]) -> str` |
| `test.neural_network.test_compiled_model.load_spec` | Function | `(model_path: str) -> _proto.Model_pb2` |
| `test.neural_network.test_compiled_model.proto` | Object | `` |
| `test.neural_network.test_compiled_model.save_spec` | Function | `(spec, filename, auto_set_specification_version, weights_dir)` |
| `test.neural_network.test_compiled_model.utils` | Object | `` |
| `test.neural_network.test_custom_neural_nets.SimpleTest` | Class | `(...)` |
| `test.neural_network.test_custom_neural_nets.datatypes` | Object | `` |
| `test.neural_network.test_custom_neural_nets.neural_network` | Object | `` |
| `test.neural_network.test_model.ComputeUnit` | Class | `(...)` |
| `test.neural_network.test_model.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `test.neural_network.test_model.MLModelTest` | Class | `(...)` |
| `test.neural_network.test_model.NeuralNetworkBuilder` | Class | `(input_features, output_features, mode, spec, nn_spec, disable_rank5_shape_mapping, training_features, use_float_arraytype)` |
| `test.neural_network.test_model.convert_double_to_float_multiarray_type` | Function | `(spec)` |
| `test.neural_network.test_model.datatypes` | Object | `` |
| `test.neural_network.test_model.make_image_input` | Function | `(model, input_name, is_bgr, red_bias, blue_bias, green_bias, gray_bias, scale, image_format)` |
| `test.neural_network.test_model.make_nn_classifier` | Function | `(model, class_labels, predicted_feature_name, predicted_probabilities_output)` |
| `test.neural_network.test_model.mb` | Class | `(...)` |
| `test.neural_network.test_model.proto` | Object | `` |
| `test.neural_network.test_model.rename_feature` | Function | `(spec, current_name, new_name, rename_inputs, rename_outputs)` |
| `test.neural_network.test_model.save_spec` | Function | `(spec, filename, auto_set_specification_version, weights_dir)` |
| `test.neural_network.test_neural_networks.CustomLayerUtilsTest` | Class | `(...)` |
| `test.neural_network.test_neural_networks.proto` | Object | `` |
| `test.neural_network.test_nn_builder.BasicNumericCorrectnessTest` | Class | `(...)` |
| `test.neural_network.test_nn_builder.BasicNumericCorrectnessTest_1014NewLayers` | Class | `(...)` |
| `test.neural_network.test_nn_builder.BasicNumericCorrectnessTest_1015NewLayers` | Class | `(...)` |
| `test.neural_network.test_nn_builder.ComputeUnit` | Class | `(...)` |
| `test.neural_network.test_nn_builder.ControlFlowCorrectnessTest` | Class | `(...)` |
| `test.neural_network.test_nn_builder.LAYERS_10_14_MACOS_VERSION` | Object | `` |
| `test.neural_network.test_nn_builder.LAYERS_10_15_MACOS_VERSION` | Object | `` |
| `test.neural_network.test_nn_builder.MIN_MACOS_VERSION_REQUIRED` | Object | `` |
| `test.neural_network.test_nn_builder.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `test.neural_network.test_nn_builder.NeuralNetworkBuilder` | Class | `(input_features, output_features, mode, spec, nn_spec, disable_rank5_shape_mapping, training_features, use_float_arraytype)` |
| `test.neural_network.test_nn_builder.UseFloatArraytypeTest` | Class | `(...)` |
| `test.neural_network.test_nn_builder.datatypes` | Object | `` |
| `test.neural_network.test_nn_builder.np_val_to_py_type` | Function | `(val)` |
| `test.neural_network.test_nn_builder.quantize_weights` | Function | `(full_precision_model, nbits, quantization_mode, sample_data, kwargs)` |
| `test.neural_network.test_numpy_nn_layers.ComputeUnit` | Class | `(...)` |
| `test.neural_network.test_numpy_nn_layers.CoreML3NetworkStressTest` | Class | `(...)` |
| `test.neural_network.test_numpy_nn_layers.CorrectnessTest` | Class | `(...)` |
| `test.neural_network.test_numpy_nn_layers.IOS14SingleLayerTests` | Class | `(...)` |
| `test.neural_network.test_numpy_nn_layers.LAYERS_10_15_MACOS_VERSION` | Object | `` |
| `test.neural_network.test_numpy_nn_layers.LAYERS_11_0_MACOS_VERSION` | Object | `` |
| `test.neural_network.test_numpy_nn_layers.MIN_MACOS_VERSION_REQUIRED` | Object | `` |
| `test.neural_network.test_numpy_nn_layers.MSG_TF2_NOT_FOUND` | Object | `` |
| `test.neural_network.test_numpy_nn_layers.NewLayersSimpleTest` | Class | `(...)` |
| `test.neural_network.test_numpy_nn_layers.SimpleTest` | Class | `(...)` |
| `test.neural_network.test_numpy_nn_layers.StressTest` | Class | `(...)` |
| `test.neural_network.test_numpy_nn_layers.TestReorganizeDataTests` | Class | `(...)` |
| `test.neural_network.test_numpy_nn_layers.aggregated_pad` | Function | `(pad_type, kernel_shape, input_shape, strides, dilations, custom_pad)` |
| `test.neural_network.test_numpy_nn_layers.datatypes` | Object | `` |
| `test.neural_network.test_numpy_nn_layers.flexible_shape_utils` | Object | `` |
| `test.neural_network.test_numpy_nn_layers.get_coreml_predictions_reduce` | Function | `(X, params)` |
| `test.neural_network.test_numpy_nn_layers.get_coreml_predictions_slice` | Function | `(X, params)` |
| `test.neural_network.test_numpy_nn_layers.get_numpy_predictions_reduce` | Function | `(X, params)` |
| `test.neural_network.test_numpy_nn_layers.get_numpy_predictions_slice` | Function | `(X, params)` |
| `test.neural_network.test_numpy_nn_layers.get_size_after_stride` | Function | `(X, params)` |
| `test.neural_network.test_numpy_nn_layers.neural_network` | Object | `` |
| `test.neural_network.test_quantization.ComputeUnit` | Class | `(...)` |
| `test.neural_network.test_quantization.DynamicQuantizedInt8Int8MatMul` | Class | `(...)` |
| `test.neural_network.test_quantization.MatrixMultiplyLayerSelector` | Class | `(minimum_weight_count, minimum_input_channels, minimum_output_channels, maximum_input_channels, maximum_output_channels, include_layers_with_names)` |
| `test.neural_network.test_quantization.TestKMeansLookup` | Class | `(...)` |
| `test.neural_network.test_quantization.TestQuantizeWeightsAPI` | Class | `(...)` |
| `test.neural_network.test_quantization.activate_int8_int8_matrix_multiplications` | Function | `(spec, selector)` |
| `test.neural_network.test_quantization.datatypes` | Object | `` |
| `test.neural_network.test_quantization.neural_network` | Object | `` |
| `test.neural_network.test_quantization.quantization_utils` | Object | `` |
| `test.neural_network.test_simple_nn_inference.ComputeUnit` | Class | `(...)` |
| `test.neural_network.test_simple_nn_inference.TestNeuralNetworkPrediction` | Class | `(...)` |
| `test.neural_network.test_simple_nn_inference.datatypes` | Object | `` |
| `test.neural_network.test_simple_nn_inference.neural_network` | Object | `` |
| `test.neural_network.test_simple_nn_inference.utils` | Object | `` |
| `test.neural_network.test_tf_numeric.ComputeUnit` | Class | `(...)` |
| `test.neural_network.test_tf_numeric.CorrectnessTest` | Class | `(...)` |
| `test.neural_network.test_tf_numeric.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `test.neural_network.test_tf_numeric.MSG_TF2_NOT_FOUND` | Object | `` |
| `test.neural_network.test_tf_numeric.StressTest` | Class | `(...)` |
| `test.neural_network.test_tf_numeric.datatypes` | Object | `` |
| `test.neural_network.test_tf_numeric.neural_network` | Object | `` |
| `test.optimize.api.test_optimize_api.TestConvertingCompressedSourceModels` | Class | `(...)` |
| `test.optimize.api.test_optimize_api.TestOptimizeCoremlAPIOverview` | Class | `(...)` |
| `test.optimize.api.test_optimize_api.TestOptimizeTorchAPIOverview` | Class | `(...)` |
| `test.optimize.api.test_optimize_api.TestPostTrainingPalettization` | Class | `(...)` |
| `test.optimize.api.test_optimize_api.TestPostTrainingPruning` | Class | `(...)` |
| `test.optimize.api.test_optimize_api.TestPostTrainingQuantization` | Class | `(...)` |
| `test.optimize.api.test_optimize_api.TestTrainingTimePalettization` | Class | `(...)` |
| `test.optimize.api.test_optimize_api.TestTrainingTimePruning` | Class | `(...)` |
| `test.optimize.api.test_optimize_api.TestTrainingTimeQuantization` | Class | `(...)` |
| `test.optimize.api.test_optimize_api.create_model_and_optimizer` | Function | `()` |
| `test.optimize.api.test_optimize_api.get_mlmodel` | Function | `()` |
| `test.optimize.api.test_optimize_api.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `test.optimize.api.test_optimize_api.get_test_program` | Object | `` |
| `test.optimize.coreml.test_passes.CONSTEXPR_FUNCS` | Object | `` |
| `test.optimize.coreml.test_passes.CONSTEXPR_OPS` | Object | `` |
| `test.optimize.coreml.test_passes.PASS_REGISTRY` | Object | `` |
| `test.optimize.coreml.test_passes.PassOption` | Class | `(option_name: Text, option_val: Union[Text, Callable[[Operation], bool]])` |
| `test.optimize.coreml.test_passes.TestCompressionGraphBackwardCompatibility` | Class | `(...)` |
| `test.optimize.coreml.test_passes.TestCompressionNumerical` | Class | `(...)` |
| `test.optimize.coreml.test_passes.TestCompressionOperations` | Class | `(...)` |
| `test.optimize.coreml.test_passes.TestCompressionPasses` | Class | `(...)` |
| `test.optimize.coreml.test_passes.TestConfigurationFromDictFromYaml` | Class | `(...)` |
| `test.optimize.coreml.test_passes.TestGetActivationStats` | Class | `(...)` |
| `test.optimize.coreml.test_passes.TestInvalidConfig` | Class | `(...)` |
| `test.optimize.coreml.test_passes.TestLinearActivationQuantizer` | Class | `(...)` |
| `test.optimize.coreml.test_passes.TestLinearQuantizer` | Class | `(...)` |
| `test.optimize.coreml.test_passes.TestOptimizationConfig` | Class | `(...)` |
| `test.optimize.coreml.test_passes.TestPalettizer` | Class | `(...)` |
| `test.optimize.coreml.test_passes.TestPruner` | Class | `(...)` |
| `test.optimize.coreml.test_passes.apply_pass_and_basic_check` | Function | `(prog: Program, pass_name: Union[str, AbstractGraphPass], skip_output_name_check: Optional[bool], skip_output_type_check: Optional[bool], skip_output_shape_check: Optional[bool], skip_input_name_check: Optional[bool], skip_input_type_check: Optional[bool], skip_function_name_check: Optional[bool], func_name: Optional[str], skip_essential_scope_check: Optional[bool]) -> Tuple[Program, Block, Block]` |
| `test.optimize.coreml.test_passes.compute_snr_and_psnr` | Function | `(x, y)` |
| `test.optimize.coreml.test_passes.cto` | Object | `` |
| `test.optimize.coreml.test_passes.gen_activation_stats_for_program` | Function | `(prog)` |
| `test.optimize.coreml.test_passes.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `test.optimize.coreml.test_passes.mb` | Class | `(...)` |
| `test.optimize.coreml.test_passes.quantization` | Object | `` |
| `test.optimize.coreml.test_passes.types` | Object | `` |
| `test.optimize.coreml.test_post_training_quantization.CoreMLWeightMetaData` | Class | `(...)` |
| `test.optimize.coreml.test_post_training_quantization.MultiFunctionDescriptor` | Class | `(model_path: _Optional[str])` |
| `test.optimize.coreml.test_post_training_quantization.TestConvertMixedCompression` | Class | `(...)` |
| `test.optimize.coreml.test_post_training_quantization.TestCoreMLWeightMetaData` | Class | `(...)` |
| `test.optimize.coreml.test_post_training_quantization.TestDecompressWeights` | Class | `(...)` |
| `test.optimize.coreml.test_post_training_quantization.TestErrorHandling` | Class | `(...)` |
| `test.optimize.coreml.test_post_training_quantization.TestJointCompressWeights` | Class | `(...)` |
| `test.optimize.coreml.test_post_training_quantization.TestLinearQuantizeWeights` | Class | `(...)` |
| `test.optimize.coreml.test_post_training_quantization.TestPalettizeWeights` | Class | `(...)` |
| `test.optimize.coreml.test_post_training_quantization.TestPruneWeights` | Class | `(...)` |
| `test.optimize.coreml.test_post_training_quantization.TestPyTorchConverterExamples` | Class | `(...)` |
| `test.optimize.coreml.test_post_training_quantization.backends` | Object | `` |
| `test.optimize.coreml.test_post_training_quantization.compute_snr_and_psnr` | Function | `(x, y)` |
| `test.optimize.coreml.test_post_training_quantization.compute_units` | Object | `` |
| `test.optimize.coreml.test_post_training_quantization.create_quantize_friendly_weight` | Function | `(weight: np.ndarray, nbits: int, signed: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` |
| `test.optimize.coreml.test_post_training_quantization.create_sparse_weight` | Function | `(weight, target_sparsity)` |
| `test.optimize.coreml.test_post_training_quantization.create_unique_weight` | Function | `(weight, nbits, vector_size, vector_axis)` |
| `test.optimize.coreml.test_post_training_quantization.create_weight_with_inf` | Function | `(weight, inf_ratio, inf_val)` |
| `test.optimize.coreml.test_post_training_quantization.cto` | Object | `` |
| `test.optimize.coreml.test_post_training_quantization.decompress_weights` | Function | `(mlmodel)` |
| `test.optimize.coreml.test_post_training_quantization.get_op_types_in_program` | Function | `(prog: Program, func_name: str, skip_const_ops: bool, recurse: bool)` |
| `test.optimize.coreml.test_post_training_quantization.get_test_model_and_data` | Function | `(multi_layer: bool, quantize_config: Optional[OpCompressorConfig], use_linear: bool)` |
| `test.optimize.coreml.test_post_training_quantization.get_test_model_and_data_complex` | Function | `()` |
| `test.optimize.coreml.test_post_training_quantization.get_test_model_and_data_conv_transpose` | Function | `()` |
| `test.optimize.coreml.test_post_training_quantization.linear_quantize_weights` | Function | `(mlmodel, mode, dtype)` |
| `test.optimize.coreml.test_post_training_quantization.mb` | Class | `(...)` |
| `test.optimize.coreml.test_post_training_quantization.optimize_utils` | Object | `` |
| `test.optimize.coreml.test_post_training_quantization.palettize_weights` | Function | `(mlmodel, nbits, mode, lut_function)` |
| `test.optimize.coreml.test_post_training_quantization.prune_weights` | Function | `(mlmodel, mode, threshold, target_sparsity, block_size, n_m_ratio)` |
| `test.optimize.coreml.test_post_training_quantization.save_multifunction` | Function | `(desc: MultiFunctionDescriptor, destination_path: str)` |
| `test.optimize.coreml.test_post_training_quantization.types` | Object | `` |
| `test.optimize.coreml.test_post_training_quantization.verify_model_outputs` | Function | `(model, compressed_model, input_values, rtol, atol)` |
| `test.optimize.test_utils.TestComputeQuantizationParams` | Class | `(...)` |
| `test.optimize.test_utils.TestFindIndicesForLut` | Class | `(...)` |
| `test.optimize.test_utils.TestPackUnpackBits` | Class | `(...)` |
| `test.optimize.test_utils.constexpr_lut_to_dense` | Class | `(...)` |
| `test.optimize.test_utils.optimize_utils` | Object | `` |
| `test.optimize.test_utils.types` | Object | `` |
| `test.optimize.torch.conftest.datadir` | Function | `(request)` |
| `test.optimize.torch.conftest.get_model_and_pruner` | Function | `(mnist_model, pruner_cls, pruner_config)` |
| `test.optimize.torch.conftest.marker_names` | Function | `(item)` |
| `test.optimize.torch.conftest.mnist_dataset` | Function | `()` |
| `test.optimize.torch.conftest.mnist_example_input` | Function | `()` |
| `test.optimize.torch.conftest.mnist_example_output` | Function | `()` |
| `test.optimize.torch.conftest.mnist_model` | Function | `()` |
| `test.optimize.torch.conftest.mnist_model_conv_transpose` | Function | `()` |
| `test.optimize.torch.conftest.mnist_model_large` | Function | `()` |
| `test.optimize.torch.conftest.mnist_model_quantization` | Function | `()` |
| `test.optimize.torch.conftest.mock_name_main` | Function | `(monkeypatch)` |
| `test.optimize.torch.conftest.pytest_addoption` | Function | `(parser)` |
| `test.optimize.torch.conftest.pytest_collection_modifyitems` | Function | `(config, items)` |
| `test.optimize.torch.conftest.pytest_configure` | Function | `(config)` |
| `test.optimize.torch.conftest.residual_mnist_model` | Function | `()` |
| `test.optimize.torch.conversion.conversion_utils.compute_SNR_and_PSNR` | Function | `(x, y)` |
| `test.optimize.torch.conversion.conversion_utils.convert_and_verify` | Function | `(pytorch_model, input_data, input_as_shape, pass_pipeline, minimum_deployment_target, expected_ops)` |
| `test.optimize.torch.conversion.conversion_utils.get_converted_model` | Function | `(pytorch_model, input_data, pass_pipeline, minimum_deployment_target)` |
| `test.optimize.torch.conversion.conversion_utils.verify_model_outputs` | Function | `(pytorch_model, coreml_model, input_value, snr_thresh, psnr_thresh)` |
| `test.optimize.torch.conversion.conversion_utils.verify_ops` | Function | `(coreml_model, expected_ops)` |
| `test.optimize.torch.conversion.joint.test_joint_compression_conversion.LayerwiseCompressor` | Class | `(model: _nn.Module, config: LayerwiseCompressorConfig)` |
| `test.optimize.torch.conversion.joint.test_joint_compression_conversion.LayerwiseCompressorConfig` | Class | `(...)` |
| `test.optimize.torch.conversion.joint.test_joint_compression_conversion.LinearQuantizer` | Class | `(model: _torch.nn.Module, config: _Optional[_LinearQuantizerConfig])` |
| `test.optimize.torch.conversion.joint.test_joint_compression_conversion.LinearQuantizerConfig` | Class | `(...)` |
| `test.optimize.torch.conversion.joint.test_joint_compression_conversion.MagnitudePruner` | Class | `(model: _torch.nn.Module, config: _Optional[MagnitudePrunerConfig])` |
| `test.optimize.torch.conversion.joint.test_joint_compression_conversion.MagnitudePrunerConfig` | Class | `(...)` |
| `test.optimize.torch.conversion.joint.test_joint_compression_conversion.ct` | Object | `` |
| `test.optimize.torch.conversion.joint.test_joint_compression_conversion.test_joint_pruning_quantization` | Function | `(mnist_model, mnist_example_input)` |
| `test.optimize.torch.conversion.joint.test_joint_compression_conversion.test_sparsegpt` | Function | `(config, mnist_model, mnist_example_input, expected_ops)` |
| `test.optimize.torch.conversion.joint.test_joint_compression_conversion.util` | Object | `` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.DKMPalettizer` | Class | `(model: _nn.Module, config: _Optional[_DKMPalettizerConfig])` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.DKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.PostTrainingPalettizer` | Class | `(model: _torch.nn.Module, config: PostTrainingPalettizerConfig)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.PostTrainingPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.SKMPalettizer` | Class | `(model: _torch.nn.Module, config: _Optional[SKMPalettizerConfig])` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.SKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.count_unique_params` | Function | `(tensor)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.ct` | Object | `` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.cto` | Object | `` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.get_compressed_model` | Function | `(algorithm, mnist_model, mnist_example_input, mnist_example_output, config)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.get_compressed_model_for_dkm` | Function | `(mnist_model, mnist_example_input, config)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.get_compressed_model_for_ptp` | Function | `(mnist_model, config)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.get_compressed_model_for_skm` | Function | `(mnist_model, mnist_example_input, mnist_example_output, config)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.test_compression_for_dkm_on_non_cpu_device_with_pcs` | Function | `(mnist_model, mnist_example_input)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.test_palettization_grouped_channelwise` | Function | `(mnist_model, mnist_example_input, mnist_example_output, config, lut_shape_map, algorithm)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.test_palettization_int8_lut` | Function | `(mnist_model, mnist_example_input, mnist_example_output, config, lut_shape_map, lut_dtype, algorithm)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.test_palettization_per_channel_scale` | Function | `(mnist_model, mnist_example_input, mnist_example_output, config, lut_shape_map, algorithm)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.test_palettization_per_tensor` | Function | `(mnist_model, mnist_example_input, mnist_example_output, config, lut_shape_map, algorithm)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.test_palettization_vector` | Function | `(mnist_model, mnist_example_input, mnist_example_output, config, lut_shape_map, algorithm, vector_ch_axis)` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.util` | Object | `` |
| `test.optimize.torch.conversion.palettization.test_palettization_conversion.verify_op_constexpr_lut_to_dense` | Function | `(coreml_model, per_layer_lut_shape)` |
| `test.optimize.torch.conversion.pruning.test_pruning_conversion.MagnitudePruner` | Class | `(model: _torch.nn.Module, config: _Optional[MagnitudePrunerConfig])` |
| `test.optimize.torch.conversion.pruning.test_pruning_conversion.MagnitudePrunerConfig` | Class | `(...)` |
| `test.optimize.torch.conversion.pruning.test_pruning_conversion.ct` | Object | `` |
| `test.optimize.torch.conversion.pruning.test_pruning_conversion.get_pruned_model` | Function | `(pruner)` |
| `test.optimize.torch.conversion.pruning.test_pruning_conversion.test_magnitude_pruner` | Function | `(config, mnist_model, mnist_example_input)` |
| `test.optimize.torch.conversion.pruning.test_pruning_conversion.util` | Object | `` |
| `test.optimize.torch.conversion.quantization.test_quantization_conversion.LayerwiseCompressor` | Class | `(model: _nn.Module, config: LayerwiseCompressorConfig)` |
| `test.optimize.torch.conversion.quantization.test_quantization_conversion.LayerwiseCompressorConfig` | Class | `(...)` |
| `test.optimize.torch.conversion.quantization.test_quantization_conversion.LinearQuantizer` | Class | `(model: _torch.nn.Module, config: _Optional[_LinearQuantizerConfig])` |
| `test.optimize.torch.conversion.quantization.test_quantization_conversion.LinearQuantizerConfig` | Class | `(...)` |
| `test.optimize.torch.conversion.quantization.test_quantization_conversion.ct` | Object | `` |
| `test.optimize.torch.conversion.quantization.test_quantization_conversion.get_quantized_model` | Function | `(quantizer, example_input)` |
| `test.optimize.torch.conversion.quantization.test_quantization_conversion.test_gptq` | Function | `(config, mnist_model, mnist_example_input)` |
| `test.optimize.torch.conversion.quantization.test_quantization_conversion.test_linear_quantizer` | Function | `(config, model, mnist_example_input, request)` |
| `test.optimize.torch.conversion.quantization.test_quantization_conversion.test_ptq` | Function | `(mnist_model, mnist_example_input, config)` |
| `test.optimize.torch.conversion.quantization.test_quantization_conversion.util` | Object | `` |
| `test.optimize.torch.layerwise_compression.test_algorithms.CompressionMetadata` | Class | `(...)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.CompressionType` | Class | `(...)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.GPTQ` | Class | `(layer: _nn.Module, config: ModuleGPTQConfig)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.LayerwiseCompressionAlgorithmConfig` | Class | `(...)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.LayerwiseCompressor` | Class | `(model: _nn.Module, config: LayerwiseCompressorConfig)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.LayerwiseCompressorConfig` | Class | `(...)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.METADATA_VERSION` | Object | `` |
| `test.optimize.torch.layerwise_compression.test_algorithms.METADATA_VERSION_BUFFER` | Object | `` |
| `test.optimize.torch.layerwise_compression.test_algorithms.ModuleGPTQConfig` | Class | `(...)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.ModuleSparseGPTConfig` | Class | `(...)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.Quantizer` | Class | `(n_bits: int, per_channel: bool, symmetric: bool, enable_normal_float: bool, mse: bool, norm: float, grid: int, max_shrink: float, group_rows: int)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.test_block_size_validation_gptq` | Function | `(input_size, expectation)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.test_blockwise_compression_gptq` | Function | `(block_size)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.test_custom_obs_compression_algorithm_config` | Function | `()` |
| `test.optimize.torch.layerwise_compression.test_algorithms.test_gptq_block_size_configs` | Function | `(config)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.test_gptq_metadata` | Function | `(config, model, input_shape)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.test_gptq_static_blocking` | Function | `()` |
| `test.optimize.torch.layerwise_compression.test_algorithms.test_obs_compression_algorithm_config` | Function | `(global_config_and_class)` |
| `test.optimize.torch.layerwise_compression.test_algorithms.test_sparse_gpt_metadata` | Function | `(config)` |
| `test.optimize.torch.layerwise_compression.test_quant.Quantizer` | Class | `(n_bits: int, per_channel: bool, symmetric: bool, enable_normal_float: bool, mse: bool, norm: float, grid: int, max_shrink: float, group_rows: int)` |
| `test.optimize.torch.layerwise_compression.test_quant.test_find_params` | Function | `(quantizer, expected_scale, expected_zp)` |
| `test.optimize.torch.layerwise_compression.test_quant.test_find_params_reshape` | Function | `(input, weight, expected_shape)` |
| `test.optimize.torch.models.mnist.LeNet5` | Function | `()` |
| `test.optimize.torch.models.mnist.Residual` | Class | `(module)` |
| `test.optimize.torch.models.mnist.mnist_dataset` | Function | `()` |
| `test.optimize.torch.models.mnist.mnist_example_input` | Function | `()` |
| `test.optimize.torch.models.mnist.mnist_example_output` | Function | `()` |
| `test.optimize.torch.models.mnist.mnist_model` | Function | `()` |
| `test.optimize.torch.models.mnist.mnist_model_conv_transpose` | Function | `()` |
| `test.optimize.torch.models.mnist.mnist_model_large` | Function | `()` |
| `test.optimize.torch.models.mnist.mnist_model_quantization` | Function | `()` |
| `test.optimize.torch.models.mnist.num_classes` | Object | `` |
| `test.optimize.torch.models.mnist.residual_mnist_model` | Function | `()` |
| `test.optimize.torch.models.mnist.test_data_path` | Function | `()` |
| `test.optimize.torch.models.multi_input_net.MultiInputNet` | Class | `()` |
| `test.optimize.torch.models.multi_input_net.num_classes` | Object | `` |
| `test.optimize.torch.palettization.palettization_utils.DKMPalettizerModulesRegistry` | Class | `(...)` |
| `test.optimize.torch.palettization.test_palettization_api.DEFAULT_PALETTIZATION_SCHEME` | Object | `` |
| `test.optimize.torch.palettization.test_palettization_api.DKMPalettizer` | Class | `(model: _nn.Module, config: _Optional[_DKMPalettizerConfig])` |
| `test.optimize.torch.palettization.test_palettization_api.DKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.palettization.test_palettization_api.ModuleDKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.palettization.test_palettization_api.REGEX_YAML` | Object | `` |
| `test.optimize.torch.palettization.test_palettization_api.get_logging_capture_context_manager` | Function | `()` |
| `test.optimize.torch.palettization.test_palettization_api.simple_model` | Function | `()` |
| `test.optimize.torch.palettization.test_palettization_api.test_attach_config_only_on_specified_modules_conv` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_attach_config_only_on_specified_modules_linear` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_attach_config_simple_model_custom_palettization_config` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_attach_config_simple_model_uniform_palettization_config` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_attach_config_simple_model_weight_threshold_range_test` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_attach_config_simple_model_weight_threshold_test` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_attach_config_weight_threshold_range_different_milestone` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_config_initialization_from_module_name_configs` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_deprecated_api` | Function | `()` |
| `test.optimize.torch.palettization.test_palettization_api.test_empty_dict_for_config` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_empty_yaml_for_config` | Function | `(simple_model, tmp_path_factory)` |
| `test.optimize.torch.palettization.test_palettization_api.test_finalize_without_forward` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_inplace_false_attach_config` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_inplace_true_attach_config` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_inplace_true_prepare_palettizer` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_prepare_palettizer_different_milestone_per_module_type` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_prepare_palettizer_simple_model_block_size_mil_check` | Function | `(simple_model, block_size_expected_std_outputs)` |
| `test.optimize.torch.palettization.test_palettization_api.test_prepare_palettizer_simple_model_cluster_dim_mil_check` | Function | `(simple_model, cluster_dim_expected_std_outputs)` |
| `test.optimize.torch.palettization.test_palettization_api.test_prepare_palettizer_simple_model_custom_palettization_config` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_prepare_palettizer_simple_model_custom_palettization_config_linear_default` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_prepare_palettizer_simple_model_custom_palettization_config_milestone_1` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_prepare_palettizer_simple_model_custom_palettization_config_none_conv2d` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_prepare_palettizer_simple_model_custom_palettization_config_none_module` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_quantize_activations_flag` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_palettization_api.test_regex_module_name_configs` | Function | `(simple_model, tmp_path_factory)` |
| `test.optimize.torch.palettization.test_palettization_utils.devectorize` | Function | `(current_tensor, pad, target_size, cluster_dim, vector_ch_axis) -> _torch.Tensor` |
| `test.optimize.torch.palettization.test_palettization_utils.test_devectorize_cluster_dim_gt_1` | Function | `(cluster_dim_reshape)` |
| `test.optimize.torch.palettization.test_palettization_utils.test_vectorize_cluster_dim_gt_1` | Function | `(cluster_dim_reshape_expected_shape_expected_first_row)` |
| `test.optimize.torch.palettization.test_palettization_utils.vectorize` | Function | `(current_tensor, cluster_dim, vector_ch_axis) -> _Tuple[_torch.Tensor, _torch.Tensor]` |
| `test.optimize.torch.palettization.test_palettizer.DKMPalettizer` | Class | `(model: _nn.Module, config: _Optional[_DKMPalettizerConfig])` |
| `test.optimize.torch.palettization.test_palettizer.DKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.palettization.test_palettizer.FakePalettize` | Class | `(observer: _ObserverBase, n_bits: int, cluster_dim: int, enable_per_channel_scale: bool, group_size: _Optional[int], quant_min: int, quant_max: int, lut_dtype: str, advanced_options: dict, observer_kwargs)` |
| `test.optimize.torch.palettization.test_palettizer.ModuleDKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.palettization.test_palettizer.palettizer_config` | Function | `()` |
| `test.optimize.torch.palettization.test_palettizer.test_fake_palettize_insertion_multihead_attention` | Function | `(kdim, vdim, batch_first, palettizer_config)` |
| `test.optimize.torch.palettization.test_palettizer.test_fake_palettize_insertion_weighted_modules` | Function | `(module, palettizer_config)` |
| `test.optimize.torch.palettization.test_palettizer.test_fake_palettize_train_no_grad_fwd` | Function | `(module, palettizer_config)` |
| `test.optimize.torch.palettization.test_post_training_palettization.CompressionMetadata` | Class | `(...)` |
| `test.optimize.torch.palettization.test_post_training_palettization.PostTrainingPalettizer` | Class | `(model: _torch.nn.Module, config: PostTrainingPalettizerConfig)` |
| `test.optimize.torch.palettization.test_post_training_palettization.PostTrainingPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.palettization.test_post_training_palettization.SKMPalettizer` | Class | `(model: _torch.nn.Module, config: _Optional[SKMPalettizerConfig])` |
| `test.optimize.torch.palettization.test_post_training_palettization.SKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.palettization.test_post_training_palettization.loss_fn` | Function | `(model, input)` |
| `test.optimize.torch.palettization.test_post_training_palettization.simple_model` | Function | `()` |
| `test.optimize.torch.palettization.test_post_training_palettization.test_compute_sensitivity_single_worker_mutability` | Function | `(mnist_model, mnist_example_input)` |
| `test.optimize.torch.palettization.test_post_training_palettization.test_no_config` | Function | `(simple_model)` |
| `test.optimize.torch.palettization.test_post_training_palettization.test_post_training_palettization_dict_config` | Function | `(simple_model, config_dict, expected_output)` |
| `test.optimize.torch.palettization.test_post_training_palettization.test_post_training_vector_palettization_dict_config` | Function | `(simple_model, config_dict, expected_output)` |
| `test.optimize.torch.palettization.test_post_training_palettization.test_ptp_int_lut` | Function | `(simple_model, config_dict, lut_dtype, layer)` |
| `test.optimize.torch.palettization.test_sensitive_k_means.FSDPAutoWrapPolicy` | Class | `(...)` |
| `test.optimize.torch.palettization.test_sensitive_k_means.KMeansConfig` | Class | `(...)` |
| `test.optimize.torch.palettization.test_sensitive_k_means.ModuleSKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.palettization.test_sensitive_k_means.ModuleWrapPolicy` | Class | `(...)` |
| `test.optimize.torch.palettization.test_sensitive_k_means.SKMPalettizer` | Class | `(model: _torch.nn.Module, config: _Optional[SKMPalettizerConfig])` |
| `test.optimize.torch.palettization.test_sensitive_k_means.SKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.palettization.test_sensitive_k_means.SizeBasedWrapPolicy` | Class | `(...)` |
| `test.optimize.torch.palettization.test_sensitive_k_means.model_for_compression` | Function | `() -> torch.nn.Module` |
| `test.optimize.torch.palettization.test_sensitive_k_means.model_for_compression_custom_module` | Function | `() -> torch.nn.Module` |
| `test.optimize.torch.palettization.test_sensitive_k_means.sensitvity_dict_for_compression` | Function | `() -> Dict[str, Any]` |
| `test.optimize.torch.palettization.test_sensitive_k_means.test_compress_cluster_weights_call` | Function | `(mocker, num_kmeans_workers, model, sensitivity_dict, config, kmeans_keys, request)` |
| `test.optimize.torch.palettization.test_sensitive_k_means.test_fsdp_auto_wrap_policy_compress_call` | Function | `(mocker, num_kmeans_workers, num_sensitivity_workers, auto_wrap_policy)` |
| `test.optimize.torch.palettization.test_sensitive_k_means.test_fsdp_auto_wrap_policy_compute_sensitivity_call` | Function | `(mocker, num_sensitivity_workers, auto_wrap_policy)` |
| `test.optimize.torch.palettization.test_sensitive_k_means.test_fsdp_auto_wrap_policy_multi_worker_compute_sensitivity_call` | Function | `(mocker, auto_wrap_policy)` |
| `test.optimize.torch.pruning.pruning_utils.batch_size` | Object | `` |
| `test.optimize.torch.pruning.pruning_utils.get_compression_ratio` | Function | `(model, pruner)` |
| `test.optimize.torch.pruning.pruning_utils.get_model_and_pruner` | Function | `(mnist_model, pruner_cls, pruner_config)` |
| `test.optimize.torch.pruning.pruning_utils.train_and_eval_model` | Function | `(model, dataset, pruner, num_epochs, pass_loss)` |
| `test.optimize.torch.pruning.pruning_utils.utils` | Object | `` |
| `test.optimize.torch.pruning.pruning_utils.verify_global_pruning_amount` | Function | `(supported_modules, model, expected_sparsity)` |
| `test.optimize.torch.pruning.test_base_pruner.CompressionMetadata` | Class | `(...)` |
| `test.optimize.torch.pruning.test_base_pruner.CompressionType` | Class | `(...)` |
| `test.optimize.torch.pruning.test_base_pruner.MagnitudePruner` | Class | `(model: _torch.nn.Module, config: _Optional[MagnitudePrunerConfig])` |
| `test.optimize.torch.pruning.test_base_pruner.MagnitudePrunerConfig` | Class | `(...)` |
| `test.optimize.torch.pruning.test_base_pruner.test_compression_metadata` | Function | `(algorithm, config)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.CompressionMetadata` | Class | `(...)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.CompressionType` | Class | `(...)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.MagnitudePruner` | Class | `(model: _torch.nn.Module, config: _Optional[MagnitudePrunerConfig])` |
| `test.optimize.torch.pruning.test_magnitude_pruner.MagnitudePrunerConfig` | Class | `(...)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.ModuleMagnitudePrunerConfig` | Class | `(...)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.large_module` | Function | `()` |
| `test.optimize.torch.pruning.test_magnitude_pruner.n_m_mask` | Function | `(weights: _torch.Tensor, nm: _Tuple[int, int], dim: _Optional[int])` |
| `test.optimize.torch.pruning.test_magnitude_pruner.sample_data` | Function | `()` |
| `test.optimize.torch.pruning.test_magnitude_pruner.simple_module` | Function | `()` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_compression_metadata` | Function | `()` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_finalize` | Function | `(simple_module)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_magnitude_pruner_block_sparsity` | Function | `(out_channels, block_size)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_magnitude_pruner_cloning` | Function | `(simple_module)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_magnitude_pruner_config_global_config_set` | Function | `(config_dict)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_magnitude_pruner_morethanhalf_block_size` | Function | `(out_channels, block_size)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_magnitude_pruner_n_m_ratio_param_usage` | Function | `(options)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_magnitude_pruner_nondivisible_block_size` | Function | `(out_channels, block_size)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_magnitude_pruning_correctness` | Function | `(simple_module)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_magnitude_pruning_granularity_parameter_usage` | Function | `(simple_module, granularity)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_magnitude_pruning_training_and_validation` | Function | `(simple_module, sample_data)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_nm_pruner_mask_computation` | Function | `(weights_shape, dim)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_nm_pruner_polynomial_scheduler` | Function | `()` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_polynomial_scheduler_range_str` | Function | `(range_str)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_pruner_finalize` | Function | `(simple_module, granularity)` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_sparsity_report_block2_sparsity_not_applicable` | Function | `()` |
| `test.optimize.torch.pruning.test_magnitude_pruner.test_sparsity_report_method` | Function | `(large_module, block_size, granularity)` |
| `test.optimize.torch.pruning.test_pruning_scheduler.ConstantSparsityScheduler` | Class | `(...)` |
| `test.optimize.torch.pruning.test_pruning_scheduler.MagnitudePruner` | Class | `(model: _torch.nn.Module, config: _Optional[MagnitudePrunerConfig])` |
| `test.optimize.torch.pruning.test_pruning_scheduler.MagnitudePrunerConfig` | Class | `(...)` |
| `test.optimize.torch.pruning.test_pruning_scheduler.ModuleMagnitudePrunerConfig` | Class | `(...)` |
| `test.optimize.torch.pruning.test_pruning_scheduler.PolynomialDecayScheduler` | Class | `(...)` |
| `test.optimize.torch.pruning.test_pruning_scheduler.simple_module` | Function | `()` |
| `test.optimize.torch.pruning.test_pruning_scheduler.test_constant_sparsity_correctness` | Function | `(simple_module, step_and_target)` |
| `test.optimize.torch.pruning.test_pruning_scheduler.test_polynomial_decay_correctness` | Function | `(simple_module, steps_and_expected)` |
| `test.optimize.torch.pruning.test_pruning_scheduler.test_polynomial_decay_initialization_failure` | Function | `(steps)` |
| `test.optimize.torch.quantization.test_configure.ComplexAdd` | Class | `(activation_fn)` |
| `test.optimize.torch.quantization.test_configure.ComplexConcatAdd` | Class | `(conv_transpose, activation_fn)` |
| `test.optimize.torch.quantization.test_configure.ConcatBlock` | Class | `(conv_transpose: bool, activations: torch.nn.Module)` |
| `test.optimize.torch.quantization.test_configure.ConvBlock` | Class | `(conv_transpose, activation)` |
| `test.optimize.torch.quantization.test_configure.LearnableFakeQuantize` | Class | `(observer: _ObserverBase, dtype: _torch.dtype, qscheme: _torch.qscheme, reduce_range: bool, observer_kwargs)` |
| `test.optimize.torch.quantization.test_configure.LinearQuantizer` | Class | `(model: _torch.nn.Module, config: _Optional[_LinearQuantizerConfig])` |
| `test.optimize.torch.quantization.test_configure.LinearQuantizerConfig` | Class | `(...)` |
| `test.optimize.torch.quantization.test_configure.MySoftmax` | Class | `(dim)` |
| `test.optimize.torch.quantization.test_configure.QuantizationScheme` | Class | `(...)` |
| `test.optimize.torch.quantization.test_configure.ResidualBlock` | Class | `(conv_transpose: bool, activation: nn.Module)` |
| `test.optimize.torch.quantization.test_configure.find_module` | Function | `(model: _torch.nn.Module, node: _fx.Node)` |
| `test.optimize.torch.quantization.test_configure.get_configs_for_qscheme` | Function | `(algorithm, activation_dtype, weight_per_channel, weight_dtype) -> List[LinearQuantizerConfig]` |
| `test.optimize.torch.quantization.test_configure.get_quant_range` | Function | `(n_bits: int, dtype: _torch.dtype) -> _Tuple[int, int]` |
| `test.optimize.torch.quantization.test_configure.is_activation_post_process` | Function | `(module: _torch.nn.Module) -> bool` |
| `test.optimize.torch.quantization.test_configure.quantize_model` | Function | `(model, data, config)` |
| `test.optimize.torch.quantization.test_configure.test_addition_of_int_and_uint_for_symmetric` | Function | `(algorithm, activation_fn, conv_transpose)` |
| `test.optimize.torch.quantization.test_configure.test_addition_of_uint_and_uint_for_symmetric` | Function | `(algorithm, activation_fn, conv_transpose)` |
| `test.optimize.torch.quantization.test_configure.test_complex_add` | Function | `(algorithm, activation_fn)` |
| `test.optimize.torch.quantization.test_configure.test_complex_concat_add` | Function | `(algorithm, activation_fn, conv_transpose)` |
| `test.optimize.torch.quantization.test_configure.test_concat_uint_and_int` | Function | `(algorithm, activation_fn, conv_transpose)` |
| `test.optimize.torch.quantization.test_configure.test_conv_act_fusion` | Function | `(config, activation_fn, conv_transpose)` |
| `test.optimize.torch.quantization.test_configure.test_conv_activation_only_quantization` | Function | `(config, activation_fn, bn)` |
| `test.optimize.torch.quantization.test_configure.test_conv_bn_act_fusion` | Function | `(config, activation_fn, conv_transpose)` |
| `test.optimize.torch.quantization.test_configure.test_conv_bn_relu_fusion` | Function | `(config, model_config)` |
| `test.optimize.torch.quantization.test_configure.test_conv_relu_fusion` | Function | `(config, model_config)` |
| `test.optimize.torch.quantization.test_configure.test_conv_weight_only_quantization` | Function | `(config, activation_fn, bn, conv_transpose)` |
| `test.optimize.torch.quantization.test_configure.test_dropout_affine_input` | Function | `(algorithm, activation_fn, conv_transpose)` |
| `test.optimize.torch.quantization.test_configure.test_elementwise_op_act_fusion` | Function | `(config, activation_fn, elementwise_op, conv_transpose)` |
| `test.optimize.torch.quantization.test_configure.test_embedding_layer_quantization` | Function | `(algorithm, activation_dtype)` |
| `test.optimize.torch.quantization.test_configure.test_functional_relu_qscheme_for_symmetric` | Function | `(algorithm, activation_fn, conv_transpose)` |
| `test.optimize.torch.quantization.test_configure.test_linear_act_fusion` | Function | `(config, activation_fn)` |
| `test.optimize.torch.quantization.test_configure.test_linear_activation_only_quantization` | Function | `(config, activation_fn)` |
| `test.optimize.torch.quantization.test_configure.test_linear_relu_fusion` | Function | `(config)` |
| `test.optimize.torch.quantization.test_configure.test_linear_weight_only_quantization` | Function | `(config, activation_fn)` |
| `test.optimize.torch.quantization.test_configure.test_sequential_network_config_for_symmetric` | Function | `(mnist_model_quantization)` |
| `test.optimize.torch.quantization.test_configure.test_single_act_qscheme_for_symmetric` | Function | `(algorithm, activation_fn, layer_and_data, bn)` |
| `test.optimize.torch.quantization.test_configure.test_single_fixed_qparams_act_for_symmetric` | Function | `(config, activation_fn, layer_and_data, bn)` |
| `test.optimize.torch.quantization.test_configure.test_skipping_quantization_for_layers` | Function | `(mnist_model_quantization, algorithm, quantization_scheme, skipped_layers)` |
| `test.optimize.torch.quantization.test_configure.test_softmax_breakdown` | Function | `()` |
| `test.optimize.torch.quantization.test_coreml_quantizer.CoreMLQuantizer` | Class | `(config: _Optional[_LinearQuantizerConfig])` |
| `test.optimize.torch.quantization.test_coreml_quantizer.LinearQuantizerConfig` | Class | `(...)` |
| `test.optimize.torch.quantization.test_coreml_quantizer.QuantizationScheme` | Class | `(...)` |
| `test.optimize.torch.quantization.test_coreml_quantizer.activations` | Object | `` |
| `test.optimize.torch.quantization.test_coreml_quantizer.config` | Function | `(request) -> LinearQuantizerConfig` |
| `test.optimize.torch.quantization.test_coreml_quantizer.get_node_map` | Function | `(model: torch.fx.GraphModule) -> Dict[str, Node]` |
| `test.optimize.torch.quantization.test_coreml_quantizer.model_for_quant` | Function | `() -> torch.nn.Module` |
| `test.optimize.torch.quantization.test_coreml_quantizer.quantize_model` | Function | `(model: nn.Module, data: torch.Tensor, quantization_config: Optional[LinearQuantizerConfig], is_qat: bool)` |
| `test.optimize.torch.quantization.test_coreml_quantizer.test_weight_module_act_fusion` | Function | `(model_for_quant, is_qat, config)` |
| `test.optimize.torch.quantization.test_learnable_fake_quantize.EMAMinMaxObserver` | Class | `(dtype: _torch.dtype, qscheme: _torch.qscheme, reduce_range: bool, quant_min: _Optional[int], quant_max: _Optional[int], ch_axis: int, ema_ratio: float, factory_kwargs: _Any, is_dynamic: bool)` |
| `test.optimize.torch.quantization.test_learnable_fake_quantize.LearnableFakeQuantize` | Class | `(observer: _ObserverBase, dtype: _torch.dtype, qscheme: _torch.qscheme, reduce_range: bool, observer_kwargs)` |
| `test.optimize.torch.quantization.test_learnable_fake_quantize.call_setters` | Function | `(lfq, ch_axis, shape)` |
| `test.optimize.torch.quantization.test_learnable_fake_quantize.create_learnable_fake_quantize` | Function | `(qscheme, dtype, reduce_range, observer_cls, mode)` |
| `test.optimize.torch.quantization.test_learnable_fake_quantize.is_per_channel_quant` | Function | `(qscheme: _torch.qscheme) -> bool` |
| `test.optimize.torch.quantization.test_learnable_fake_quantize.is_symmetric_quant` | Function | `(qscheme: _torch.qscheme) -> bool` |
| `test.optimize.torch.quantization.test_learnable_fake_quantize.test_learnable_fake_quantize_checkpoint` | Function | `(qscheme, dtype, reduce_range, observer_cls, mode)` |
| `test.optimize.torch.quantization.test_learnable_fake_quantize.test_learnable_fake_quantize_init` | Function | `(qscheme, dtype, reduce_range, observer_cls, mode)` |
| `test.optimize.torch.quantization.test_learnable_fake_quantize.test_learnable_fake_quantize_phases` | Function | `(qscheme, dtype, reduce_range, observer_cls, mode)` |
| `test.optimize.torch.quantization.test_learnable_fake_quantize.test_learnable_fake_quantize_setters` | Function | `(qscheme, dtype, reduce_range, observer_cls, mode)` |
| `test.optimize.torch.quantization.test_observers.EMAMinMaxObserver` | Class | `(dtype: _torch.dtype, qscheme: _torch.qscheme, reduce_range: bool, quant_min: _Optional[int], quant_max: _Optional[int], ch_axis: int, ema_ratio: float, factory_kwargs: _Any, is_dynamic: bool)` |
| `test.optimize.torch.quantization.test_observers.generate_binary_tensor` | Function | `(shape)` |
| `test.optimize.torch.quantization.test_observers.is_per_channel_quant` | Function | `(qscheme: _torch.qscheme) -> bool` |
| `test.optimize.torch.quantization.test_observers.is_symmetric_quant` | Function | `(qscheme: _torch.qscheme) -> bool` |
| `test.optimize.torch.quantization.test_observers.test_observer_checkpoint` | Function | `(qscheme, dtype, observer_cls, mode)` |
| `test.optimize.torch.quantization.test_observers.test_observer_fake_quantize` | Function | `(qscheme, dtype, observer_cls, mode)` |
| `test.optimize.torch.quantization.test_observers.test_observer_fixed_data` | Function | `(qscheme, dtype, observer_cls, mode)` |
| `test.optimize.torch.quantization.test_observers.volume` | Function | `(shape)` |
| `test.optimize.torch.quantization.test_post_training_quantization.PostTrainingQuantizer` | Class | `(model: _torch.nn.Module, config: PostTrainingQuantizerConfig)` |
| `test.optimize.torch.quantization.test_post_training_quantization.PostTrainingQuantizerConfig` | Class | `(...)` |
| `test.optimize.torch.quantization.test_post_training_quantization.QuantizationGranularity` | Class | `(...)` |
| `test.optimize.torch.quantization.test_post_training_quantization.QuantizationScheme` | Class | `(...)` |
| `test.optimize.torch.quantization.test_post_training_quantization.ct` | Object | `` |
| `test.optimize.torch.quantization.test_post_training_quantization.get_atol_rtol` | Function | `(block_size, weight_n_bits)` |
| `test.optimize.torch.quantization.test_post_training_quantization.get_n_bits_from_dtype` | Function | `(dtype: _Union[str, _torch.dtype]) -> int` |
| `test.optimize.torch.quantization.test_post_training_quantization.get_quant_range` | Function | `(n_bits: int, dtype: _torch.dtype) -> _Tuple[int, int]` |
| `test.optimize.torch.quantization.test_post_training_quantization.get_rmse` | Function | `(a, b)` |
| `test.optimize.torch.quantization.test_post_training_quantization.test_ptq_compress_all_combinations` | Function | `(module, quantization_scheme, granularity_block_size, weight_dtype)` |
| `test.optimize.torch.quantization.test_post_training_quantization.test_ptq_compression_metadata` | Function | `(weight_dtype, n_bits, qscheme)` |
| `test.optimize.torch.quantization.test_post_training_quantization.test_ptq_config_n_bits` | Function | `(dtype, n_bits)` |
| `test.optimize.torch.quantization.test_post_training_quantization.test_ptq_default_config` | Function | `()` |
| `test.optimize.torch.quantization.test_post_training_quantization.test_ptq_post_compress_conv_linear` | Function | `(quantization_scheme, granularity_block_size, weight_dtype, module)` |
| `test.optimize.torch.quantization.test_post_training_quantization.test_ptq_post_compress_multihead` | Function | `(quantization_scheme, granularity_block_size, weight_dtype)` |
| `test.optimize.torch.quantization.test_post_training_quantization.test_ptq_unsigned` | Function | `(weight_dtype, qscheme)` |
| `test.optimize.torch.quantization.test_quantizer.CompressionMetadata` | Class | `(...)` |
| `test.optimize.torch.quantization.test_quantizer.CompressionType` | Class | `(...)` |
| `test.optimize.torch.quantization.test_quantizer.LearnableFakeQuantize` | Class | `(observer: _ObserverBase, dtype: _torch.dtype, qscheme: _torch.qscheme, reduce_range: bool, observer_kwargs)` |
| `test.optimize.torch.quantization.test_quantizer.LinearQuantizer` | Class | `(model: _torch.nn.Module, config: _Optional[_LinearQuantizerConfig])` |
| `test.optimize.torch.quantization.test_quantizer.LinearQuantizerConfig` | Class | `(...)` |
| `test.optimize.torch.quantization.test_quantizer.ModuleLinearQuantizerConfig` | Class | `(...)` |
| `test.optimize.torch.quantization.test_quantizer.ObserverType` | Class | `(...)` |
| `test.optimize.torch.quantization.test_quantizer.QuantizationScheme` | Class | `(...)` |
| `test.optimize.torch.quantization.test_quantizer.test_activation_defaults` | Function | `(algorithm, quantization_scheme, model_config)` |
| `test.optimize.torch.quantization.test_quantizer.test_config_illegal_options` | Function | `(algorithm, option_and_value)` |
| `test.optimize.torch.quantization.test_quantizer.test_linear_quantizer_compression_metadata` | Function | `(algorithm, dtype, scheme, conv_transpose)` |
| `test.optimize.torch.quantization.test_quantizer.test_linear_quantizer_config_different_config_success` | Function | `(algorithm)` |
| `test.optimize.torch.quantization.test_quantizer.test_linear_quantizer_config_failure_modes` | Function | `(config_dict)` |
| `test.optimize.torch.quantization.test_quantizer.test_linear_quantizer_config_global_config_set` | Function | `(config_dict)` |
| `test.optimize.torch.quantization.test_quantizer.test_linear_quantizer_config_n_bits` | Function | `(algorithm, dtype, n_bits)` |
| `test.optimize.torch.quantization.test_quantizer.test_linear_quantizer_fq_insert` | Function | `(mnist_model_conv_transpose, algorithm, weight_dtype, weight_per_channel, quantization_scheme)` |
| `test.optimize.torch.quantization.test_quantizer.test_linear_quantizer_observer_sync` | Function | `(mnist_model_conv_transpose, algorithm, test_config, observer_type, expected_synchronize)` |
| `test.optimize.torch.quantization.test_quantizer.test_linear_quantizer_preserved_attributes` | Function | `(algorithm, model_dict)` |
| `test.optimize.torch.quantization.test_quantizer.test_linear_quantizer_quantization_scheme_setting` | Function | `(config_dict)` |
| `test.optimize.torch.quantization.test_quantizer.test_linear_quantizer_report` | Function | `(mnist_model_conv_transpose, algorithm, weight_dtype, weight_per_channel, quantization_scheme)` |
| `test.optimize.torch.quantization.test_quantizer.test_linear_quantizer_step_mechanism` | Function | `(algorithm, quantization_scheme, model_config)` |
| `test.optimize.torch.quantization.test_utils.EMAMinMaxObserver` | Class | `(dtype: _torch.dtype, qscheme: _torch.qscheme, reduce_range: bool, quant_min: _Optional[int], quant_max: _Optional[int], ch_axis: int, ema_ratio: float, factory_kwargs: _Any, is_dynamic: bool)` |
| `test.optimize.torch.quantization.test_utils.NoopObserver` | Class | `(dtype: _torch.dtype, custom_op_name: str, factory_kwargs: _Dict[str, _Any])` |
| `test.optimize.torch.quantization.test_utils.get_n_bits_from_range` | Function | `(quant_min: int, quant_max: int) -> int` |
| `test.optimize.torch.quantization.test_utils.get_quant_range` | Function | `(n_bits: int, dtype: _torch.dtype) -> _Tuple[int, int]` |
| `test.optimize.torch.quantization.test_utils.is_per_channel_quant` | Function | `(qscheme: _torch.qscheme) -> bool` |
| `test.optimize.torch.quantization.test_utils.is_pytorch_defined_observer` | Function | `(observer: _aoquant.ObserverBase)` |
| `test.optimize.torch.quantization.test_utils.is_symmetric_quant` | Function | `(qscheme: _torch.qscheme) -> bool` |
| `test.optimize.torch.quantization.test_utils.test_quantization_utils_get_n_bits_from_range` | Function | `(dtype, n_bits)` |
| `test.optimize.torch.quantization.test_utils.test_quantization_utils_get_quant_range` | Function | `(dtype, n_bits)` |
| `test.optimize.torch.quantization.test_utils.test_quantization_utils_is_per_channel_quant` | Function | `(qscheme, expected_result)` |
| `test.optimize.torch.quantization.test_utils.test_quantization_utils_is_pytorch_defined_observer` | Function | `(observer_cls, args, expected_result)` |
| `test.optimize.torch.quantization.test_utils.test_quantization_utils_is_symmetric_quant` | Function | `(qscheme, expected_result)` |
| `test.optimize.torch.smoke_test.TestSmokeTest` | Class | `(...)` |
| `test.optimize.torch.test_api_surface.TestApiVisibilities` | Class | `(...)` |
| `test.optimize.torch.test_base_optimizer.BaseDataCalibratedModelOptimizer` | Class | `(model: _torch.nn.Module, config: _Optional[_OptimizationConfig])` |
| `test.optimize.torch.test_base_optimizer.BasePostTrainingModelOptimizer` | Class | `(model: _torch.nn.Module, config: _Optional[_OptimizationConfig])` |
| `test.optimize.torch.test_base_optimizer.DKMPalettizer` | Class | `(model: _nn.Module, config: _Optional[_DKMPalettizerConfig])` |
| `test.optimize.torch.test_base_optimizer.LinearQuantizer` | Class | `(model: _torch.nn.Module, config: _Optional[_LinearQuantizerConfig])` |
| `test.optimize.torch.test_base_optimizer.MagnitudePruner` | Class | `(model: _torch.nn.Module, config: _Optional[MagnitudePrunerConfig])` |
| `test.optimize.torch.test_base_optimizer.test_inplace_behavior_for_optimizers` | Function | `(optimizer, inplace)` |
| `test.optimize.torch.test_base_optimizer.test_report_model_train_state` | Function | `(optimizer, inplace)` |
| `test.optimize.torch.test_utils.test_fsdp_utils.ModuleWrapPolicy` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_fsdp_utils.SizeBasedWrapPolicy` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_fsdp_utils.sync_tensor` | Function | `(tensor: _torch.Tensor, reduce_op: _torch.distributed.ReduceOp)` |
| `test.optimize.torch.test_utils.test_fsdp_utils.test_fsdp_utils_module_wrap_policy` | Function | `()` |
| `test.optimize.torch.test_utils.test_fsdp_utils.test_fsdp_utils_size_based_policy` | Function | `()` |
| `test.optimize.torch.test_utils.test_fsdp_utils.test_fsdp_utils_sync_tensor` | Function | `(reduce_op)` |
| `test.optimize.torch.test_utils.test_k_means.KMeansConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_k_means.KMeansSupportedModulesRegistry` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_k_means.ParallelKMeans` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_k_means.SequentialKMeans` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_k_means.count_unique_params` | Function | `(tensor)` |
| `test.optimize.torch.test_utils.test_k_means.test_k_means_block_wise` | Function | `(mock_name_main, config, kmeans_cls)` |
| `test.optimize.torch.test_utils.test_k_means.test_k_means_masked` | Function | `(mock_name_main, importance, config, kmeans_cls)` |
| `test.optimize.torch.test_utils.test_k_means.test_k_means_mnist_per_weight` | Function | `(mock_name_main, mnist_model, config, kmeans_cls)` |
| `test.optimize.torch.test_utils.test_k_means.test_k_means_perf` | Function | `(config, num_workers)` |
| `test.optimize.torch.test_utils.test_k_means.test_k_means_rounding_no_warnings` | Function | `()` |
| `test.optimize.torch.test_utils.test_k_means.test_k_means_vector_wise` | Function | `(mock_name_main, config, kmeans_cls)` |
| `test.optimize.torch.test_utils.test_k_means.test_parameter_reshaping` | Function | `(layer, param_name, axis, expected_shape)` |
| `test.optimize.torch.test_utils.test_k_means.test_zero_per_channel_scale` | Function | `(layer, layer_config)` |
| `test.optimize.torch.test_utils.test_metadata_utils.CompressionMetadata` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_metadata_utils.CompressionType` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_metadata_utils.METADATA_VERSION` | Object | `` |
| `test.optimize.torch.test_utils.test_metadata_utils.METADATA_VERSION_BUFFER` | Object | `` |
| `test.optimize.torch.test_utils.test_metadata_utils.register_metadata_version` | Function | `(model: _torch.nn.Module)` |
| `test.optimize.torch.test_utils.test_metadata_utils.test_chaining_compression_type` | Function | `()` |
| `test.optimize.torch.test_utils.test_metadata_utils.test_metadata_from_dict` | Function | `(metadata_dict, expectation)` |
| `test.optimize.torch.test_utils.test_metadata_utils.test_metadata_from_state_dict` | Function | `(state_dict)` |
| `test.optimize.torch.test_utils.test_metadata_utils.test_register` | Function | `(metadata_dict)` |
| `test.optimize.torch.test_utils.test_metadata_utils.test_register_metadata_version` | Function | `()` |
| `test.optimize.torch.test_utils.test_optimizer_utils.LayerwiseCompressor` | Class | `(model: _nn.Module, config: LayerwiseCompressorConfig)` |
| `test.optimize.torch.test_utils.test_optimizer_utils.ModuleGPTQConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_optimizer_utils.ModuleOptimizationConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_optimizer_utils.ModulePostTrainingPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_optimizer_utils.ModuleSKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_optimizer_utils.OptimizationConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_optimizer_utils.PostTrainingPalettizer` | Class | `(model: _torch.nn.Module, config: PostTrainingPalettizerConfig)` |
| `test.optimize.torch.test_utils.test_optimizer_utils.SKMPalettizer` | Class | `(model: _torch.nn.Module, config: _Optional[SKMPalettizerConfig])` |
| `test.optimize.torch.test_utils.test_optimizer_utils.get_classes_recursively` | Function | `(module)` |
| `test.optimize.torch.test_utils.test_optimizer_utils.is_supported_module_for_config` | Function | `(module: _torch.nn.Module, module_config_cls: _Type[_ModuleOptimizationConfig]) -> bool` |
| `test.optimize.torch.test_utils.test_optimizer_utils.test_is_supported_config` | Function | `(module, config_cls, is_supported)` |
| `test.optimize.torch.test_utils.test_optimizer_utils.test_optimization_config_registry` | Function | `()` |
| `test.optimize.torch.test_utils.test_optimizer_utils.test_optimizer_config_registry` | Function | `(module_config_cls, optimizer)` |
| `test.optimize.torch.test_utils.test_optimizer_utils.test_optimizer_registry` | Function | `()` |
| `test.optimize.torch.test_utils.test_report_utils.LayerwiseCompressor` | Class | `(model: _nn.Module, config: LayerwiseCompressorConfig)` |
| `test.optimize.torch.test_utils.test_report_utils.LayerwiseCompressorConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_report_utils.PostTrainingPalettizer` | Class | `(model: _torch.nn.Module, config: PostTrainingPalettizerConfig)` |
| `test.optimize.torch.test_utils.test_report_utils.PostTrainingPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_report_utils.PostTrainingQuantizer` | Class | `(model: _torch.nn.Module, config: PostTrainingQuantizerConfig)` |
| `test.optimize.torch.test_utils.test_report_utils.PostTrainingQuantizerConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_report_utils.SKMPalettizer` | Class | `(model: _torch.nn.Module, config: _Optional[SKMPalettizerConfig])` |
| `test.optimize.torch.test_utils.test_report_utils.SKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_report_utils.model_for_compression` | Function | `(request) -> torch.nn.Module` |
| `test.optimize.torch.test_utils.test_report_utils.test_report_layerwise_compressor` | Function | `(model_for_compression, config, expected_num_columns)` |
| `test.optimize.torch.test_utils.test_report_utils.test_report_post_training_palettization` | Function | `(model_for_compression, config)` |
| `test.optimize.torch.test_utils.test_report_utils.test_report_post_training_quantization` | Function | `(model_for_compression, quantization_scheme, granularity_block_size, weight_dtype)` |
| `test.optimize.torch.test_utils.test_report_utils.test_report_skm_palettizer` | Function | `(model_for_compression, config)` |
| `test.optimize.torch.test_utils.test_torch_utils.test_torch_utils_normalize_fsdp_module_name` | Function | `(module_names)` |
| `test.optimize.torch.test_utils.test_torch_utils.test_torch_utils_transform_to_ch_axis` | Function | `(input_shape, ch_axis, expected_shape)` |
| `test.optimize.torch.test_utils.test_validation_utils.ConfigValidator` | Class | `(param_name: str, param: _torch.Tensor, module: _nn.Module, config: _Optional[_ModuleOptimizationConfig], module_level_advanced_options: _Optional[_Dict])` |
| `test.optimize.torch.test_utils.test_validation_utils.ModulePostTrainingPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_validation_utils.ModulePostTrainingQuantizerConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_validation_utils.ModuleSKMPalettizerConfig` | Class | `(...)` |
| `test.optimize.torch.test_utils.test_validation_utils.test_validate_no_check` | Function | `()` |
| `test.optimize.torch.test_utils.test_validation_utils.test_validate_palettization_cluster_dim` | Function | `(cluster_dim, expectation)` |
| `test.optimize.torch.test_utils.test_validation_utils.test_validate_palettization_group_size` | Function | `(group_size, channel_axis, expectation)` |
| `test.optimize.torch.test_utils.test_validation_utils.test_validate_param_config` | Function | `(config, expectation)` |
| `test.optimize.torch.test_utils.test_validation_utils.test_validate_quantization_block_size` | Function | `(block_size, sanitized_block_size, expectation)` |
| `test.optimize.torch.test_utils.test_validation_utils.validate_param_config` | Function | `(param_name: str, param: _torch.Tensor, module: _nn.Module, config: _Optional[_ModuleOptimizationConfig], checks_to_run: _List[str], module_level_advanced_options: _Optional[_Dict])` |
| `test.optimize.torch.utils.convert_to_coreml` | Function | `(model, example_input, minimum_deployment_target)` |
| `test.optimize.torch.utils.count_unique_params` | Function | `(tensor)` |
| `test.optimize.torch.utils.eval_model` | Function | `(model, test_loader)` |
| `test.optimize.torch.utils.get_classes_in_module` | Function | `(module)` |
| `test.optimize.torch.utils.get_classes_recursively` | Function | `(module)` |
| `test.optimize.torch.utils.get_logging_capture_context_manager` | Function | `()` |
| `test.optimize.torch.utils.get_model_size` | Function | `(model, example_input)` |
| `test.optimize.torch.utils.get_total_params` | Function | `(model)` |
| `test.optimize.torch.utils.psnr` | Function | `(pred, target)` |
| `test.optimize.torch.utils.setup_data_loaders` | Function | `(dataset, batch_size)` |
| `test.optimize.torch.utils.test_data_path` | Function | `()` |
| `test.optimize.torch.utils.train_step` | Function | `(model, optimizer, train_loader, data, target, batch_idx, epoch)` |
| `test.optimize.torch.utils.version_ge` | Function | `(module, target_version)` |
| `test.optimize.torch.utils.version_lt` | Function | `(module, target_version)` |
| `test.pipeline.test_model_updatable.AdamParams` | Class | `(lr, batch, beta1, beta2, eps)` |
| `test.pipeline.test_model_updatable.LayerSelector` | Class | `(layer_name)` |
| `test.pipeline.test_model_updatable.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `test.pipeline.test_model_updatable.MLModelUpdatableTest` | Class | `(...)` |
| `test.pipeline.test_model_updatable.NeuralNetworkBuilder` | Class | `(input_features, output_features, mode, spec, nn_spec, disable_rank5_shape_mapping, training_features, use_float_arraytype)` |
| `test.pipeline.test_model_updatable.PipelineClassifier` | Class | `(input_features, class_labels, output_features, training_features)` |
| `test.pipeline.test_model_updatable.PipelineRegressor` | Class | `(input_features, output_features, training_features)` |
| `test.pipeline.test_model_updatable.SgdParams` | Class | `(lr, batch, momentum)` |
| `test.pipeline.test_model_updatable.datatypes` | Object | `` |
| `test.pipeline.test_model_updatable.quantization_utils` | Object | `` |
| `test.pipeline.test_model_updatable.save_spec` | Function | `(spec, filename, auto_set_specification_version, weights_dir)` |
| `test.pipeline.test_pipeline.Function` | Class | `(inputs, opset_version)` |
| `test.pipeline.test_pipeline.LibSVMPipelineCreationTest` | Class | `(...)` |
| `test.pipeline.test_pipeline.LinearRegressionPipeline` | Class | `(...)` |
| `test.pipeline.test_pipeline.LinearRegressionPipelineCreationTest` | Class | `(...)` |
| `test.pipeline.test_pipeline.PipelineClassifier` | Class | `(input_features, class_labels, output_features, training_features)` |
| `test.pipeline.test_pipeline.PipelineRegressor` | Class | `(input_features, output_features, training_features)` |
| `test.pipeline.test_pipeline.Program` | Class | `()` |
| `test.pipeline.test_pipeline.TestMakePipeline` | Class | `(...)` |
| `test.pipeline.test_pipeline.converter` | Object | `` |
| `test.pipeline.test_pipeline.libsvm_converter` | Object | `` |
| `test.pipeline.test_pipeline.load_boston` | Function | `()` |
| `test.pipeline.test_pipeline.mb` | Class | `(...)` |
| `test.pipeline.test_pipeline.mil` | Object | `` |
| `test.sklearn_tests.test_NuSVC.MSG_LIBSVM_NOT_FOUND` | Object | `` |
| `test.sklearn_tests.test_NuSVC.MSG_SKLEARN_NOT_FOUND` | Object | `` |
| `test.sklearn_tests.test_NuSVC.NuSVCLibSVMTest` | Class | `(...)` |
| `test.sklearn_tests.test_NuSVC.NuSvcScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_NuSVC.evaluate_classifier` | Function | `(model, data, target, verbose)` |
| `test.sklearn_tests.test_NuSVC.evaluate_classifier_with_probabilities` | Function | `(model, data, probabilities, verbose)` |
| `test.sklearn_tests.test_NuSVC.libsvm` | Object | `` |
| `test.sklearn_tests.test_NuSVC.scikit_converter` | Object | `` |
| `test.sklearn_tests.test_NuSVR.MSG_LIBSVM_NOT_FOUND` | Object | `` |
| `test.sklearn_tests.test_NuSVR.MSG_SKLEARN_NOT_FOUND` | Object | `` |
| `test.sklearn_tests.test_NuSVR.NuSVRLibSVMTest` | Class | `(...)` |
| `test.sklearn_tests.test_NuSVR.NuSVRScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_NuSVR.evaluate_regressor` | Function | `(model, data, target, verbose)` |
| `test.sklearn_tests.test_NuSVR.libsvm` | Object | `` |
| `test.sklearn_tests.test_NuSVR.load_boston` | Function | `()` |
| `test.sklearn_tests.test_NuSVR.scikit_converter` | Object | `` |
| `test.sklearn_tests.test_SVC.CSVCLibSVMTest` | Class | `(...)` |
| `test.sklearn_tests.test_SVC.SvcScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_SVC.evaluate_classifier` | Function | `(model, data, target, verbose)` |
| `test.sklearn_tests.test_SVC.evaluate_classifier_with_probabilities` | Function | `(model, data, probabilities, verbose)` |
| `test.sklearn_tests.test_SVC.libsvm` | Object | `` |
| `test.sklearn_tests.test_SVC.scikit_converter` | Object | `` |
| `test.sklearn_tests.test_SVR.EpsilonSVRLibSVMTest` | Class | `(...)` |
| `test.sklearn_tests.test_SVR.MSG_LIBSVM_NOT_FOUND` | Object | `` |
| `test.sklearn_tests.test_SVR.MSG_SKLEARN_NOT_FOUND` | Object | `` |
| `test.sklearn_tests.test_SVR.SvrScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_SVR.evaluate_regressor` | Function | `(model, data, target, verbose)` |
| `test.sklearn_tests.test_SVR.libsvm` | Object | `` |
| `test.sklearn_tests.test_SVR.load_boston` | Function | `()` |
| `test.sklearn_tests.test_SVR.sklearn_converter` | Object | `` |
| `test.sklearn_tests.test_categorical_imputer.ImputerTestCase` | Class | `(...)` |
| `test.sklearn_tests.test_categorical_imputer.converter` | Object | `` |
| `test.sklearn_tests.test_categorical_imputer.load_boston` | Function | `()` |
| `test.sklearn_tests.test_categorical_imputer.sklearn_class` | Object | `` |
| `test.sklearn_tests.test_composite_pipelines.GradientBoostingRegressorBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.sklearn_tests.test_composite_pipelines.convert` | Function | `(sk_obj, input_features, output_feature_names)` |
| `test.sklearn_tests.test_composite_pipelines.evaluate_regressor` | Function | `(model, data, target, verbose)` |
| `test.sklearn_tests.test_composite_pipelines.evaluate_transformer` | Function | `(model, input_data, reference_output, verbose)` |
| `test.sklearn_tests.test_composite_pipelines.load_boston` | Function | `()` |
| `test.sklearn_tests.test_dict_vectorizer.DictVectorizerScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_dict_vectorizer.evaluate_classifier` | Function | `(model, data, target, verbose)` |
| `test.sklearn_tests.test_dict_vectorizer.evaluate_transformer` | Function | `(model, input_data, reference_output, verbose)` |
| `test.sklearn_tests.test_dict_vectorizer.sklearn` | Object | `` |
| `test.sklearn_tests.test_feature_names.FeatureManagementTests` | Class | `(...)` |
| `test.sklearn_tests.test_feature_names.dt` | Object | `` |
| `test.sklearn_tests.test_feature_names.fm` | Object | `` |
| `test.sklearn_tests.test_glm_classifier.GlmCassifierTest` | Class | `(...)` |
| `test.sklearn_tests.test_glm_classifier.convert` | Function | `(sk_obj, input_features, output_feature_names)` |
| `test.sklearn_tests.test_glm_classifier.evaluate_classifier` | Function | `(model, data, target, verbose)` |
| `test.sklearn_tests.test_glm_classifier.evaluate_classifier_with_probabilities` | Function | `(model, data, probabilities, verbose)` |
| `test.sklearn_tests.test_imputer.NumericalImputerTestCase` | Class | `(...)` |
| `test.sklearn_tests.test_imputer.converter` | Object | `` |
| `test.sklearn_tests.test_imputer.evaluate_transformer` | Function | `(model, input_data, reference_output, verbose)` |
| `test.sklearn_tests.test_imputer.load_boston` | Function | `()` |
| `test.sklearn_tests.test_imputer.sklearn_class` | Object | `` |
| `test.sklearn_tests.test_io_types.MSG_SKLEARN_NOT_FOUND` | Object | `` |
| `test.sklearn_tests.test_io_types.TestIODataTypes` | Class | `(...)` |
| `test.sklearn_tests.test_io_types.create_model` | Function | `(spec)` |
| `test.sklearn_tests.test_io_types.load_boston` | Function | `()` |
| `test.sklearn_tests.test_k_neighbors_classifier.KNeighborsClassifierScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_k_neighbors_classifier.sklearn` | Object | `` |
| `test.sklearn_tests.test_linear_regression.LinearRegressionScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_linear_regression.convert` | Function | `(sk_obj, input_features, output_feature_names)` |
| `test.sklearn_tests.test_linear_regression.evaluate_regressor` | Function | `(model, data, target, verbose)` |
| `test.sklearn_tests.test_linear_regression.load_boston` | Function | `()` |
| `test.sklearn_tests.test_nearest_neighbors_builder.KNearestNeighborsClassifierBuilder` | Class | `(input_name, output_name, number_of_dimensions, default_class_label, kwargs)` |
| `test.sklearn_tests.test_nearest_neighbors_builder.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `test.sklearn_tests.test_nearest_neighbors_builder.NearestNeighborsBuilderTest` | Class | `(...)` |
| `test.sklearn_tests.test_normalizer.NormalizerScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_normalizer.converter` | Object | `` |
| `test.sklearn_tests.test_normalizer.evaluate_transformer` | Function | `(model, input_data, reference_output, verbose)` |
| `test.sklearn_tests.test_normalizer.load_boston` | Function | `()` |
| `test.sklearn_tests.test_one_hot_encoder.Array` | Class | `(dimensions)` |
| `test.sklearn_tests.test_one_hot_encoder.OneHotEncoderScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_one_hot_encoder.evaluate_transformer` | Function | `(model, input_data, reference_output, verbose)` |
| `test.sklearn_tests.test_one_hot_encoder.load_boston` | Function | `()` |
| `test.sklearn_tests.test_one_hot_encoder.sklearn` | Object | `` |
| `test.sklearn_tests.test_random_forest_classifier.RandomForestBinaryClassifierScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_random_forest_classifier.RandomForestMultiClassClassifierScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_random_forest_classifier.load_boston` | Function | `()` |
| `test.sklearn_tests.test_random_forest_classifier.skl_converter` | Object | `` |
| `test.sklearn_tests.test_random_forest_classifier_numeric.RandomForestBinaryClassifierBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.sklearn_tests.test_random_forest_classifier_numeric.RandomForestClassificationBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.sklearn_tests.test_random_forest_classifier_numeric.RandomForestMultiClassClassificationBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.sklearn_tests.test_random_forest_classifier_numeric.evaluate_classifier` | Function | `(model, data, target, verbose)` |
| `test.sklearn_tests.test_random_forest_classifier_numeric.load_boston` | Function | `()` |
| `test.sklearn_tests.test_random_forest_classifier_numeric.skl_converter` | Object | `` |
| `test.sklearn_tests.test_random_forest_regression.RandomForestRegressorScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_random_forest_regression.load_boston` | Function | `()` |
| `test.sklearn_tests.test_random_forest_regression.skl_converter` | Object | `` |
| `test.sklearn_tests.test_random_forest_regression_numeric.RandomForestRegressorBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.sklearn_tests.test_random_forest_regression_numeric.evaluate_regressor` | Function | `(model, data, target, verbose)` |
| `test.sklearn_tests.test_random_forest_regression_numeric.load_boston` | Function | `()` |
| `test.sklearn_tests.test_random_forest_regression_numeric.skl_converter` | Object | `` |
| `test.sklearn_tests.test_ridge_regression.RidgeRegressionScikitTest` | Class | `(...)` |
| `test.sklearn_tests.test_ridge_regression.convert` | Function | `(sk_obj, input_features, output_feature_names)` |
| `test.sklearn_tests.test_ridge_regression.evaluate_regressor` | Function | `(model, data, target, verbose)` |
| `test.sklearn_tests.test_ridge_regression.load_boston` | Function | `()` |
| `test.sklearn_tests.test_standard_scalar.StandardScalerTestCase` | Class | `(...)` |
| `test.sklearn_tests.test_standard_scalar.converter` | Object | `` |
| `test.sklearn_tests.test_standard_scalar.evaluate_transformer` | Function | `(model, input_data, reference_output, verbose)` |
| `test.sklearn_tests.test_standard_scalar.load_boston` | Function | `()` |
| `test.sklearn_tests.test_utils.MLModel` | Class | `(model, is_temp_package, mil_program, skip_model_load, compute_units, weights_dir, function_name, optimization_hints: _Optional[dict])` |
| `test.sklearn_tests.test_utils.PipeLineRenameTests` | Class | `(...)` |
| `test.sklearn_tests.test_utils.converter` | Object | `` |
| `test.sklearn_tests.test_utils.load_boston` | Function | `()` |
| `test.sklearn_tests.test_utils.rename_feature` | Function | `(spec, current_name, new_name, rename_inputs, rename_outputs)` |
| `test.utils.load_boston` | Function | `()` |
| `test.xgboost_tests.test_boosted_trees_classifier.GradientBoostingBinaryClassifierScikitTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_classifier.GradientBoostingBinaryClassifierXGboostTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_classifier.GradientBoostingMulticlassClassifierScikitTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_classifier.GradientBoostingMulticlassClassifierXGboostTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_classifier.load_boston` | Function | `()` |
| `test.xgboost_tests.test_boosted_trees_classifier.skl_converter` | Object | `` |
| `test.xgboost_tests.test_boosted_trees_classifier.xgb_converter` | Object | `` |
| `test.xgboost_tests.test_boosted_trees_classifier_numeric.BoostedTreeBinaryClassificationBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_classifier_numeric.BoostedTreeBinaryClassificationBostonHousingXGboostNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_classifier_numeric.BoostedTreeClassificationBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_classifier_numeric.BoostedTreeClassificationBostonHousingXGboostNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_classifier_numeric.BoostedTreeMultiClassClassificationBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_classifier_numeric.BoostedTreeMultiClassClassificationBostonHousingXGboostNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_classifier_numeric.evaluate_classifier` | Function | `(model, data, target, verbose)` |
| `test.xgboost_tests.test_boosted_trees_classifier_numeric.evaluate_classifier_with_probabilities` | Function | `(model, data, probabilities, verbose)` |
| `test.xgboost_tests.test_boosted_trees_classifier_numeric.load_boston` | Function | `()` |
| `test.xgboost_tests.test_boosted_trees_classifier_numeric.skl_converter` | Object | `` |
| `test.xgboost_tests.test_boosted_trees_classifier_numeric.xgb_converter` | Object | `` |
| `test.xgboost_tests.test_boosted_trees_regression.BoostedTreeRegressorXGboostTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_regression.GradientBoostingRegressorScikitTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_regression.load_boston` | Function | `()` |
| `test.xgboost_tests.test_boosted_trees_regression.skl_converter` | Object | `` |
| `test.xgboost_tests.test_boosted_trees_regression.xgb_converter` | Object | `` |
| `test.xgboost_tests.test_boosted_trees_regression_numeric.GradientBoostingRegressorBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_regression_numeric.XGboostRegressorBostonHousingNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_regression_numeric.XgboostBoosterBostonHousingNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_boosted_trees_regression_numeric.evaluate_regressor` | Function | `(model, data, target, verbose)` |
| `test.xgboost_tests.test_boosted_trees_regression_numeric.load_boston` | Function | `()` |
| `test.xgboost_tests.test_boosted_trees_regression_numeric.skl_converter` | Object | `` |
| `test.xgboost_tests.test_boosted_trees_regression_numeric.xgb_converter` | Object | `` |
| `test.xgboost_tests.test_decision_tree_classifier.DecisionTreeBinaryClassifierScikitTest` | Class | `(...)` |
| `test.xgboost_tests.test_decision_tree_classifier.DecisionTreeMultiClassClassifierScikitTest` | Class | `(...)` |
| `test.xgboost_tests.test_decision_tree_classifier.load_boston` | Function | `()` |
| `test.xgboost_tests.test_decision_tree_classifier.skl_converter` | Function | `(sk_obj, input_features, output_feature_names)` |
| `test.xgboost_tests.test_decision_tree_classifier_numeric.DecisionTreeBinaryClassificationBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_decision_tree_classifier_numeric.DecisionTreeClassificationBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_decision_tree_classifier_numeric.DecisionTreeMultiClassClassificationBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_decision_tree_classifier_numeric.evaluate_classifier` | Function | `(model, data, target, verbose)` |
| `test.xgboost_tests.test_decision_tree_classifier_numeric.load_boston` | Function | `()` |
| `test.xgboost_tests.test_decision_tree_classifier_numeric.skl_converter` | Object | `` |
| `test.xgboost_tests.test_decision_tree_regression.DecisionTreeRegressorScikitTest` | Class | `(...)` |
| `test.xgboost_tests.test_decision_tree_regression.load_boston` | Function | `()` |
| `test.xgboost_tests.test_decision_tree_regression.skl_converter` | Object | `` |
| `test.xgboost_tests.test_decision_tree_regression_numeric.DecisionTreeRegressorBostonHousingScikitNumericTest` | Class | `(...)` |
| `test.xgboost_tests.test_decision_tree_regression_numeric.evaluate_regressor` | Function | `(model, data, target, verbose)` |
| `test.xgboost_tests.test_decision_tree_regression_numeric.load_boston` | Function | `()` |
| `transform` | Object | `` |
| `utils` | Object | `` |