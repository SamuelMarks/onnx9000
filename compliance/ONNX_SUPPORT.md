# ONNX Spec Coverage

**Coverage:** 160/200 (80.00%)

**Commit:** [`657f5abe0846f25b103e83d9e580a3bc3e0677b8`](https://github.com/onnx/onnx/commit/657f5abe0846f25b103e83d9e580a3bc3e0677b8)


## ONNX Detailed API

This table provides a more detailed view of the objects exposed by the `onnx` Python package.

| Object Name | Type | Signature |
|---|---|---|
| `AttributeProto` | Object | `` |
| `DeviceConfigurationProto` | Object | `` |
| `EXPERIMENTAL` | Object | `` |
| `FunctionProto` | Object | `` |
| `GraphProto` | Object | `` |
| `IR_VERSION` | Object | `` |
| `IR_VERSION_2017_10_10` | Object | `` |
| `IR_VERSION_2017_10_30` | Object | `` |
| `IR_VERSION_2017_11_3` | Object | `` |
| `IR_VERSION_2019_1_22` | Object | `` |
| `IR_VERSION_2019_3_18` | Object | `` |
| `IR_VERSION_2019_9_19` | Object | `` |
| `IR_VERSION_2020_5_8` | Object | `` |
| `IR_VERSION_2021_7_30` | Object | `` |
| `IR_VERSION_2023_5_5` | Object | `` |
| `IR_VERSION_2024_3_25` | Object | `` |
| `IntIntListEntryProto` | Object | `` |
| `MapProto` | Object | `` |
| `ModelProto` | Object | `` |
| `NodeDeviceConfigurationProto` | Object | `` |
| `NodeProto` | Object | `` |
| `ONNX_ML` | Object | `` |
| `OperatorProto` | Object | `` |
| `OperatorSetIdProto` | Object | `` |
| `OperatorSetProto` | Object | `` |
| `OperatorStatus` | Object | `` |
| `OptionalProto` | Object | `` |
| `STABLE` | Object | `` |
| `SequenceProto` | Object | `` |
| `ShardedDimProto` | Object | `` |
| `ShardingSpecProto` | Object | `` |
| `SimpleShardedDimProto` | Object | `` |
| `SparseTensorProto` | Object | `` |
| `StringStringEntryProto` | Object | `` |
| `TensorAnnotation` | Object | `` |
| `TensorProto` | Object | `` |
| `TensorShapeProto` | Object | `` |
| `TrainingInfoProto` | Object | `` |
| `TypeProto` | Object | `` |
| `ValueInfoProto` | Object | `` |
| `Version` | Object | `` |
| `backend.base.Backend` | Class | `(...)` |
| `backend.base.BackendRep` | Class | `(...)` |
| `backend.base.Device` | Class | `(device: str)` |
| `backend.base.DeviceType` | Class | `(...)` |
| `backend.base.IR_VERSION` | Object | `` |
| `backend.base.ModelProto` | Object | `` |
| `backend.base.NodeProto` | Object | `` |
| `backend.base.c_checker` | Object | `` |
| `backend.base.namedtupledict` | Function | `(typename: str, field_names: Sequence[str], args: Any, kwargs: Any) -> type[tuple[Any, ...]]` |
| `backend.sample.ops.abs.abs` | Function | `(input: np.ndarray) -> np.ndarray` |
| `backend.sample.ops.collect_sample_implementations` | Function | `() -> dict[str, str]` |
| `backend.test.BackendTest` | Class | `(backend: type[Backend], parent_module: str \| None, test_kwargs: dict \| None)` |
| `backend.test.case.Snippets` | Object | `` |
| `backend.test.case.base.Base` | Class | `(...)` |
| `backend.test.case.base.Snippets` | Object | `` |
| `backend.test.case.base.process_snippet` | Function | `(op_name: str, name: str, export: Any) -> tuple[str, str]` |
| `backend.test.case.collect_snippets` | Function | `() -> dict[str, list[tuple[str, str]]]` |
| `backend.test.case.import_recursive` | Function | `(package: ModuleType) -> None` |
| `backend.test.case.model.BASE_URL` | Object | `` |
| `backend.test.case.model.ModelProto` | Object | `` |
| `backend.test.case.model.TestCase` | Class | `(name: str, model_name: str, url: str \| None, model_dir: str \| None, model: onnx.ModelProto \| None, data_sets: Sequence[tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] \| None, kind: str, rtol: float, atol: float, __test__: bool)` |
| `backend.test.case.model.collect_testcases` | Function | `() -> list[TestCase]` |
| `backend.test.case.model.expand.Base` | Class | `(...)` |
| `backend.test.case.model.expand.ExpandDynamicShape` | Class | `(...)` |
| `backend.test.case.model.expand.expect` | Function | `(model: ModelProto, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], name: str \| None) -> None` |
| `backend.test.case.model.expect` | Function | `(model: ModelProto, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], name: str \| None) -> None` |
| `backend.test.case.model.gradient.AI_ONNX_PREVIEW_TRAINING_DOMAIN` | Object | `` |
| `backend.test.case.model.gradient.Base` | Class | `(...)` |
| `backend.test.case.model.gradient.Gradient` | Class | `(...)` |
| `backend.test.case.model.gradient.ONNX_DOMAIN` | Object | `` |
| `backend.test.case.model.gradient.expect` | Function | `(model: ModelProto, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], name: str \| None) -> None` |
| `backend.test.case.model.import_recursive` | Function | `(package: ModuleType) -> None` |
| `backend.test.case.model.sequence.Base` | Class | `(...)` |
| `backend.test.case.model.sequence.ConcatFromSequenceImpl` | Function | `(sequence: list[np.ndarray], axis: int, new_axis: int \| None) -> np.ndarray` |
| `backend.test.case.model.sequence.Sequence` | Class | `(...)` |
| `backend.test.case.model.sequence.SequenceAtImpl` | Function | `(sequence: list[np.ndarray], position: int) -> np.ndarray` |
| `backend.test.case.model.sequence.SequenceConstructImpl` | Function | `(tensors: np.ndarray) -> list[np.ndarray]` |
| `backend.test.case.model.sequence.SequenceEmptyImpl` | Function | `() -> list[np.ndarray \| None]` |
| `backend.test.case.model.sequence.SequenceEraseImpl` | Function | `(sequence: list[np.ndarray], position: int \| None) -> list[np.ndarray \| None]` |
| `backend.test.case.model.sequence.SequenceInsertImpl` | Function | `(sequence: list[np.ndarray], tensor: np.ndarray, position: int \| None) -> list[np.ndarray]` |
| `backend.test.case.model.sequence.SequenceLengthImpl` | Function | `(sequence: list[np.ndarray]) -> np.int64` |
| `backend.test.case.model.sequence.SplitToSequenceImpl` | Function | `(tensor: np.ndarray, split: int \| list[int] \| None, axis: int, keepdims: int) -> list[np.ndarray]` |
| `backend.test.case.model.sequence.TensorProto` | Object | `` |
| `backend.test.case.model.sequence.expect` | Function | `(model: ModelProto, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], name: str \| None) -> None` |
| `backend.test.case.model.shrink.Base` | Class | `(...)` |
| `backend.test.case.model.shrink.ShrinkTest` | Class | `(...)` |
| `backend.test.case.model.shrink.expect` | Function | `(model: ModelProto, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], name: str \| None) -> None` |
| `backend.test.case.model.sign.Base` | Class | `(...)` |
| `backend.test.case.model.sign.SingleSign` | Class | `(...)` |
| `backend.test.case.model.sign.expect` | Function | `(model: ModelProto, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], name: str \| None) -> None` |
| `backend.test.case.model.single-relu.Base` | Class | `(...)` |
| `backend.test.case.model.single-relu.SingleRelu` | Class | `(...)` |
| `backend.test.case.model.single-relu.expect` | Function | `(model: ModelProto, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], name: str \| None) -> None` |
| `backend.test.case.model.stringnormalizer.Base` | Class | `(...)` |
| `backend.test.case.model.stringnormalizer.NormalizeStrings` | Class | `(...)` |
| `backend.test.case.model.stringnormalizer.expect` | Function | `(model: ModelProto, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], name: str \| None) -> None` |
| `backend.test.case.node.AttributeProto` | Object | `` |
| `backend.test.case.node.FunctionProto` | Object | `` |
| `backend.test.case.node.GraphProto` | Object | `` |
| `backend.test.case.node.ModelProto` | Object | `` |
| `backend.test.case.node.NodeProto` | Object | `` |
| `backend.test.case.node.OperatorSetIdProto` | Object | `` |
| `backend.test.case.node.TensorProto` | Object | `` |
| `backend.test.case.node.TestCase` | Class | `(name: str, model_name: str, url: str \| None, model_dir: str \| None, model: onnx.ModelProto \| None, data_sets: Sequence[tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] \| None, kind: str, rtol: float, atol: float, __test__: bool)` |
| `backend.test.case.node.TypeProto` | Object | `` |
| `backend.test.case.node.abs.Abs` | Class | `(...)` |
| `backend.test.case.node.abs.Base` | Class | `(...)` |
| `backend.test.case.node.abs.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.acos.Acos` | Class | `(...)` |
| `backend.test.case.node.acos.Base` | Class | `(...)` |
| `backend.test.case.node.acos.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.acosh.Acosh` | Class | `(...)` |
| `backend.test.case.node.acosh.Base` | Class | `(...)` |
| `backend.test.case.node.acosh.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.adagrad.AI_ONNX_PREVIEW_TRAINING_DOMAIN` | Object | `` |
| `backend.test.case.node.adagrad.Adagrad` | Class | `(...)` |
| `backend.test.case.node.adagrad.Base` | Class | `(...)` |
| `backend.test.case.node.adagrad.apply_adagrad` | Function | `(r, t, x, g, h, norm_coefficient, epsilon, decay_factor)` |
| `backend.test.case.node.adagrad.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.adam.AI_ONNX_PREVIEW_TRAINING_DOMAIN` | Object | `` |
| `backend.test.case.node.adam.Adam` | Class | `(...)` |
| `backend.test.case.node.adam.Base` | Class | `(...)` |
| `backend.test.case.node.adam.apply_adam` | Function | `(r, t, x, g, v, h, norm_coefficient, norm_coefficient_post, alpha, beta, epsilon)` |
| `backend.test.case.node.adam.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.add.Add` | Class | `(...)` |
| `backend.test.case.node.add.Base` | Class | `(...)` |
| `backend.test.case.node.add.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.affinegrid.AffineGrid` | Class | `(...)` |
| `backend.test.case.node.affinegrid.Base` | Class | `(...)` |
| `backend.test.case.node.affinegrid.apply_affine_transform` | Function | `(theta_n, original_grid_homo)` |
| `backend.test.case.node.affinegrid.construct_original_grid` | Function | `(data_size, align_corners)` |
| `backend.test.case.node.affinegrid.create_affine_matrix_2d` | Function | `(angle1, offset_x, offset_y, shear_x, shear_y, scale_x, scale_y)` |
| `backend.test.case.node.affinegrid.create_affine_matrix_3d` | Function | `(angle1, angle2, offset_x, offset_y, offset_z, shear_x, shear_y, shear_z, scale_x, scale_y, scale_z)` |
| `backend.test.case.node.affinegrid.create_theta_2d` | Function | `()` |
| `backend.test.case.node.affinegrid.create_theta_3d` | Function | `()` |
| `backend.test.case.node.affinegrid.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.ai_onnx_ml.array_feature_extractor.ArrayFeatureExtractor` | Class | `(...)` |
| `backend.test.case.node.ai_onnx_ml.array_feature_extractor.Base` | Class | `(...)` |
| `backend.test.case.node.ai_onnx_ml.array_feature_extractor.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.ai_onnx_ml.binarizer.Base` | Class | `(...)` |
| `backend.test.case.node.ai_onnx_ml.binarizer.Binarizer` | Class | `(...)` |
| `backend.test.case.node.ai_onnx_ml.binarizer.compute_binarizer` | Function | `(x, threshold)` |
| `backend.test.case.node.ai_onnx_ml.binarizer.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.ai_onnx_ml.label_encoder.Base` | Class | `(...)` |
| `backend.test.case.node.ai_onnx_ml.label_encoder.LabelEncoder` | Class | `(...)` |
| `backend.test.case.node.ai_onnx_ml.label_encoder.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.ai_onnx_ml.label_encoder.make_tensor` | Function | `(name: str, data_type: int, dims: Sequence[int], vals: Sequence[int \| float] \| bytes \| np.ndarray, raw: bool) -> TensorProto` |
| `backend.test.case.node.ai_onnx_ml.tree_ensemble.Base` | Class | `(...)` |
| `backend.test.case.node.ai_onnx_ml.tree_ensemble.TreeEnsemble` | Class | `(...)` |
| `backend.test.case.node.ai_onnx_ml.tree_ensemble.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.ai_onnx_ml.tree_ensemble.make_tensor` | Function | `(name: str, data_type: int, dims: Sequence[int], vals: Sequence[int \| float] \| bytes \| np.ndarray, raw: bool) -> TensorProto` |
| `backend.test.case.node.and.And` | Class | `(...)` |
| `backend.test.case.node.and.Base` | Class | `(...)` |
| `backend.test.case.node.and.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.argmax.ArgMax` | Class | `(...)` |
| `backend.test.case.node.argmax.Base` | Class | `(...)` |
| `backend.test.case.node.argmax.argmax_use_numpy` | Function | `(data: np.ndarray, axis: int, keepdims: int) -> np.ndarray` |
| `backend.test.case.node.argmax.argmax_use_numpy_select_last_index` | Function | `(data: np.ndarray, axis: int, keepdims: int) -> np.ndarray` |
| `backend.test.case.node.argmax.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.argmin.ArgMin` | Class | `(...)` |
| `backend.test.case.node.argmin.Base` | Class | `(...)` |
| `backend.test.case.node.argmin.argmin_use_numpy` | Function | `(data: np.ndarray, axis: int, keepdims: int) -> np.ndarray` |
| `backend.test.case.node.argmin.argmin_use_numpy_select_last_index` | Function | `(data: np.ndarray, axis: int, keepdims: int) -> np.ndarray` |
| `backend.test.case.node.argmin.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.asin.Asin` | Class | `(...)` |
| `backend.test.case.node.asin.Base` | Class | `(...)` |
| `backend.test.case.node.asin.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.asinh.Asinh` | Class | `(...)` |
| `backend.test.case.node.asinh.Base` | Class | `(...)` |
| `backend.test.case.node.asinh.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.atan.Atan` | Class | `(...)` |
| `backend.test.case.node.atan.Base` | Class | `(...)` |
| `backend.test.case.node.atan.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.atanh.Atanh` | Class | `(...)` |
| `backend.test.case.node.atanh.Base` | Class | `(...)` |
| `backend.test.case.node.atanh.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.attention.Attention` | Class | `(...)` |
| `backend.test.case.node.attention.Base` | Class | `(...)` |
| `backend.test.case.node.attention.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.averagepool.AveragePool` | Class | `(...)` |
| `backend.test.case.node.averagepool.Base` | Class | `(...)` |
| `backend.test.case.node.averagepool.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.averagepool.get_output_shape_auto_pad` | Function | `(auto_pad: str, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int]) -> Sequence[int]` |
| `backend.test.case.node.averagepool.get_output_shape_explicit_padding` | Function | `(pads: Sequence[int] \| None, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int], dilations: Sequence[int] \| None, ceil_mode: bool) -> tuple[Sequence[int], Sequence[int]]` |
| `backend.test.case.node.averagepool.get_pad_shape` | Function | `(auto_pad: str, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int], output_spatial_shape: Sequence[int]) -> Sequence[int]` |
| `backend.test.case.node.averagepool.pool` | Function | `(padded: np.ndarray, x_shape: Sequence[int], kernel: Sequence[int], strides: Sequence[int], out_shape: Sequence[int], pooling_type: str, pads_required: Sequence[int] \| None, pads: Sequence[int] \| None, dilations: Sequence[int] \| None, count_include_pad: int, p: int) -> np.ndarray` |
| `backend.test.case.node.batchnorm.Base` | Class | `(...)` |
| `backend.test.case.node.batchnorm.BatchNormalization` | Class | `(...)` |
| `backend.test.case.node.batchnorm.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.bernoulli.Base` | Class | `(...)` |
| `backend.test.case.node.bernoulli.Bernoulli` | Class | `(...)` |
| `backend.test.case.node.bernoulli.bernoulli_reference_implementation` | Function | `(x, dtype)` |
| `backend.test.case.node.bernoulli.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.bitshift.Base` | Class | `(...)` |
| `backend.test.case.node.bitshift.BitShift` | Class | `(...)` |
| `backend.test.case.node.bitshift.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.bitwiseand.Base` | Class | `(...)` |
| `backend.test.case.node.bitwiseand.BitwiseAnd` | Class | `(...)` |
| `backend.test.case.node.bitwiseand.create_random_int` | Function | `(input_shape: tuple[int], dtype: np.dtype, seed: int) -> np.ndarray` |
| `backend.test.case.node.bitwiseand.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.bitwisenot.Base` | Class | `(...)` |
| `backend.test.case.node.bitwisenot.BitwiseNot` | Class | `(...)` |
| `backend.test.case.node.bitwisenot.create_random_int` | Function | `(input_shape: tuple[int], dtype: np.dtype, seed: int) -> np.ndarray` |
| `backend.test.case.node.bitwisenot.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.bitwiseor.Base` | Class | `(...)` |
| `backend.test.case.node.bitwiseor.BitwiseOr` | Class | `(...)` |
| `backend.test.case.node.bitwiseor.create_random_int` | Function | `(input_shape: tuple[int], dtype: np.dtype, seed: int) -> np.ndarray` |
| `backend.test.case.node.bitwiseor.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.bitwisexor.Base` | Class | `(...)` |
| `backend.test.case.node.bitwisexor.BitwiseXor` | Class | `(...)` |
| `backend.test.case.node.bitwisexor.create_random_int` | Function | `(input_shape: tuple[int], dtype: np.dtype, seed: int) -> np.ndarray` |
| `backend.test.case.node.bitwisexor.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.blackmanwindow.Base` | Class | `(...)` |
| `backend.test.case.node.blackmanwindow.BlackmanWindow` | Class | `(...)` |
| `backend.test.case.node.blackmanwindow.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.cast.Base` | Class | `(...)` |
| `backend.test.case.node.cast.Cast` | Class | `(...)` |
| `backend.test.case.node.cast.F8_TYPES` | Object | `` |
| `backend.test.case.node.cast.FOUR_BIT_TYPES` | Object | `` |
| `backend.test.case.node.cast.TWO_BIT_TYPES` | Object | `` |
| `backend.test.case.node.cast.TensorProto` | Object | `` |
| `backend.test.case.node.cast.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.cast.make_tensor` | Function | `(name: str, data_type: int, dims: Sequence[int], vals: Sequence[int \| float] \| bytes \| np.ndarray, raw: bool) -> TensorProto` |
| `backend.test.case.node.cast.tensor_dtype_to_np_dtype` | Function | `(tensor_dtype: int) -> np.dtype` |
| `backend.test.case.node.cast.to_float8e8m0` | Function | `(x: np.ndarray, saturate: bool, round_mode: str) -> np.ndarray` |
| `backend.test.case.node.castlike.Base` | Class | `(...)` |
| `backend.test.case.node.castlike.CastLike` | Class | `(...)` |
| `backend.test.case.node.castlike.F8_TYPES` | Object | `` |
| `backend.test.case.node.castlike.FOUR_BIT_TYPES` | Object | `` |
| `backend.test.case.node.castlike.TWO_BIT_TYPES` | Object | `` |
| `backend.test.case.node.castlike.TensorProto` | Object | `` |
| `backend.test.case.node.castlike.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.castlike.make_tensor` | Function | `(name: str, data_type: int, dims: Sequence[int], vals: Sequence[int \| float] \| bytes \| np.ndarray, raw: bool) -> TensorProto` |
| `backend.test.case.node.castlike.tensor_dtype_to_np_dtype` | Function | `(tensor_dtype: int) -> np.dtype` |
| `backend.test.case.node.ceil.Base` | Class | `(...)` |
| `backend.test.case.node.ceil.Ceil` | Class | `(...)` |
| `backend.test.case.node.ceil.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.celu.Base` | Class | `(...)` |
| `backend.test.case.node.celu.Celu` | Class | `(...)` |
| `backend.test.case.node.celu.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.center_crop_pad.Base` | Class | `(...)` |
| `backend.test.case.node.center_crop_pad.CenterCropPad` | Class | `(...)` |
| `backend.test.case.node.center_crop_pad.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.clip.Base` | Class | `(...)` |
| `backend.test.case.node.clip.Clip` | Class | `(...)` |
| `backend.test.case.node.clip.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.col2im.Base` | Class | `(...)` |
| `backend.test.case.node.col2im.Col2Im` | Class | `(...)` |
| `backend.test.case.node.col2im.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.collect_diff_testcases` | Function | `() -> list[TestCase]` |
| `backend.test.case.node.collect_testcases` | Function | `(op_type: str) -> list[TestCase]` |
| `backend.test.case.node.compress.Base` | Class | `(...)` |
| `backend.test.case.node.compress.Compress` | Class | `(...)` |
| `backend.test.case.node.compress.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.concat.Base` | Class | `(...)` |
| `backend.test.case.node.concat.Concat` | Class | `(...)` |
| `backend.test.case.node.concat.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.constant.Base` | Class | `(...)` |
| `backend.test.case.node.constant.Constant` | Class | `(...)` |
| `backend.test.case.node.constant.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.constantofshape.Base` | Class | `(...)` |
| `backend.test.case.node.constantofshape.ConstantOfShape` | Class | `(...)` |
| `backend.test.case.node.constantofshape.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.conv.Base` | Class | `(...)` |
| `backend.test.case.node.conv.Conv` | Class | `(...)` |
| `backend.test.case.node.conv.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.convinteger.Base` | Class | `(...)` |
| `backend.test.case.node.convinteger.ConvInteger` | Class | `(...)` |
| `backend.test.case.node.convinteger.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.convtranspose.Base` | Class | `(...)` |
| `backend.test.case.node.convtranspose.ConvTranspose` | Class | `(...)` |
| `backend.test.case.node.convtranspose.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.cos.Base` | Class | `(...)` |
| `backend.test.case.node.cos.Cos` | Class | `(...)` |
| `backend.test.case.node.cos.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.cosh.Base` | Class | `(...)` |
| `backend.test.case.node.cosh.Cosh` | Class | `(...)` |
| `backend.test.case.node.cosh.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.cumsum.Base` | Class | `(...)` |
| `backend.test.case.node.cumsum.CumSum` | Class | `(...)` |
| `backend.test.case.node.cumsum.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.deformconv.Base` | Class | `(...)` |
| `backend.test.case.node.deformconv.DeformConv` | Class | `(...)` |
| `backend.test.case.node.deformconv.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.depthtospace.Base` | Class | `(...)` |
| `backend.test.case.node.depthtospace.DepthToSpace` | Class | `(...)` |
| `backend.test.case.node.depthtospace.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.dequantizelinear.Base` | Class | `(...)` |
| `backend.test.case.node.dequantizelinear.DequantizeLinear` | Class | `(...)` |
| `backend.test.case.node.dequantizelinear.TensorProto` | Object | `` |
| `backend.test.case.node.dequantizelinear.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.dequantizelinear.make_tensor` | Function | `(name: str, data_type: int, dims: Sequence[int], vals: Sequence[int \| float] \| bytes \| np.ndarray, raw: bool) -> TensorProto` |
| `backend.test.case.node.det.Base` | Class | `(...)` |
| `backend.test.case.node.det.Det` | Class | `(...)` |
| `backend.test.case.node.det.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.dft.Base` | Class | `(...)` |
| `backend.test.case.node.dft.DFT` | Class | `(...)` |
| `backend.test.case.node.dft.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.div.Base` | Class | `(...)` |
| `backend.test.case.node.div.Div` | Class | `(...)` |
| `backend.test.case.node.div.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.dropout.Base` | Class | `(...)` |
| `backend.test.case.node.dropout.Dropout` | Class | `(...)` |
| `backend.test.case.node.dropout.dropout` | Function | `(X, drop_probability, seed, training_mode, return_mask)` |
| `backend.test.case.node.dropout.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.dropout.helper` | Object | `` |
| `backend.test.case.node.dynamicquantizelinear.Base` | Class | `(...)` |
| `backend.test.case.node.dynamicquantizelinear.DynamicQuantizeLinear` | Class | `(...)` |
| `backend.test.case.node.dynamicquantizelinear.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.einsum.Base` | Class | `(...)` |
| `backend.test.case.node.einsum.Einsum` | Class | `(...)` |
| `backend.test.case.node.einsum.einsum_reference_implementation` | Function | `(Eqn: str, Operands: tuple[np.ndarray, ...]) -> np.ndarray` |
| `backend.test.case.node.einsum.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.elu.Base` | Class | `(...)` |
| `backend.test.case.node.elu.Elu` | Class | `(...)` |
| `backend.test.case.node.elu.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.equal.Base` | Class | `(...)` |
| `backend.test.case.node.equal.Equal` | Class | `(...)` |
| `backend.test.case.node.equal.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.erf.Base` | Class | `(...)` |
| `backend.test.case.node.erf.Erf` | Class | `(...)` |
| `backend.test.case.node.erf.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.exp.Base` | Class | `(...)` |
| `backend.test.case.node.exp.Exp` | Class | `(...)` |
| `backend.test.case.node.exp.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.expand.Base` | Class | `(...)` |
| `backend.test.case.node.expand.Expand` | Class | `(...)` |
| `backend.test.case.node.expand.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.eyelike.Base` | Class | `(...)` |
| `backend.test.case.node.eyelike.EyeLike` | Class | `(...)` |
| `backend.test.case.node.eyelike.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.flatten.Base` | Class | `(...)` |
| `backend.test.case.node.flatten.Flatten` | Class | `(...)` |
| `backend.test.case.node.flatten.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.floor.Base` | Class | `(...)` |
| `backend.test.case.node.floor.Floor` | Class | `(...)` |
| `backend.test.case.node.floor.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.function_expand_helper` | Function | `(node: NodeProto, function_proto: FunctionProto, op_prefix: str) -> list[NodeProto]` |
| `backend.test.case.node.function_testcase_helper` | Function | `(node: NodeProto, input_types: list[TypeProto], name: str, opset_imports: Sequence[OperatorSetIdProto] \| None) -> tuple[list[tuple[list[NodeProto], Any]], int]` |
| `backend.test.case.node.gather.Base` | Class | `(...)` |
| `backend.test.case.node.gather.Gather` | Class | `(...)` |
| `backend.test.case.node.gather.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.gatherelements.Base` | Class | `(...)` |
| `backend.test.case.node.gatherelements.GatherElements` | Class | `(...)` |
| `backend.test.case.node.gatherelements.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.gatherelements.gather_elements` | Function | `(data, indices, axis)` |
| `backend.test.case.node.gathernd.Base` | Class | `(...)` |
| `backend.test.case.node.gathernd.GatherND` | Class | `(...)` |
| `backend.test.case.node.gathernd.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.gathernd.gather_nd_impl` | Function | `(data: np.ndarray, indices: np.ndarray, batch_dims: int) -> np.ndarray` |
| `backend.test.case.node.gelu.Base` | Class | `(...)` |
| `backend.test.case.node.gelu.Gelu` | Class | `(...)` |
| `backend.test.case.node.gelu.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.gemm.Base` | Class | `(...)` |
| `backend.test.case.node.gemm.Gemm` | Class | `(...)` |
| `backend.test.case.node.gemm.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.gemm.gemm_reference_implementation` | Function | `(A: np.ndarray, B: np.ndarray, C: np.ndarray \| None, alpha: float, beta: float, transA: int, transB: int) -> np.ndarray` |
| `backend.test.case.node.get_diff_op_types` | Function | `()` |
| `backend.test.case.node.globalaveragepool.Base` | Class | `(...)` |
| `backend.test.case.node.globalaveragepool.GlobalAveragePool` | Class | `(...)` |
| `backend.test.case.node.globalaveragepool.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.globalmaxpool.Base` | Class | `(...)` |
| `backend.test.case.node.globalmaxpool.GlobalMaxPool` | Class | `(...)` |
| `backend.test.case.node.globalmaxpool.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.greater.Base` | Class | `(...)` |
| `backend.test.case.node.greater.Greater` | Class | `(...)` |
| `backend.test.case.node.greater.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.greater_equal.Base` | Class | `(...)` |
| `backend.test.case.node.greater_equal.Greater` | Class | `(...)` |
| `backend.test.case.node.greater_equal.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.gridsample.Base` | Class | `(...)` |
| `backend.test.case.node.gridsample.GridSample` | Class | `(...)` |
| `backend.test.case.node.gridsample.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.groupnormalization.Base` | Class | `(...)` |
| `backend.test.case.node.groupnormalization.GroupNormalization` | Class | `(...)` |
| `backend.test.case.node.groupnormalization.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.gru.Base` | Class | `(...)` |
| `backend.test.case.node.gru.GRU` | Class | `(...)` |
| `backend.test.case.node.gru.GRUHelper` | Class | `(params: Any)` |
| `backend.test.case.node.gru.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.hammingwindow.Base` | Class | `(...)` |
| `backend.test.case.node.hammingwindow.HammingWindow` | Class | `(...)` |
| `backend.test.case.node.hammingwindow.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.hannwindow.Base` | Class | `(...)` |
| `backend.test.case.node.hannwindow.HannWindow` | Class | `(...)` |
| `backend.test.case.node.hannwindow.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.hardmax.Base` | Class | `(...)` |
| `backend.test.case.node.hardmax.Hardmax` | Class | `(...)` |
| `backend.test.case.node.hardmax.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.hardmax.hardmax` | Function | `(x: np.ndarray, axis: int) -> np.ndarray` |
| `backend.test.case.node.hardsigmoid.Base` | Class | `(...)` |
| `backend.test.case.node.hardsigmoid.HardSigmoid` | Class | `(...)` |
| `backend.test.case.node.hardsigmoid.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.hardswish.Base` | Class | `(...)` |
| `backend.test.case.node.hardswish.HardSwish` | Class | `(...)` |
| `backend.test.case.node.hardswish.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.hardswish.hardswish` | Function | `(x: np.ndarray) -> np.ndarray` |
| `backend.test.case.node.identity.Base` | Class | `(...)` |
| `backend.test.case.node.identity.Identity` | Class | `(...)` |
| `backend.test.case.node.identity.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.if.Base` | Class | `(...)` |
| `backend.test.case.node.if.If` | Class | `(...)` |
| `backend.test.case.node.if.compute_if_outputs` | Function | `(x, cond)` |
| `backend.test.case.node.if.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.image_decoder.Base` | Class | `(...)` |
| `backend.test.case.node.image_decoder.ImageDecoder` | Class | `(...)` |
| `backend.test.case.node.image_decoder.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.image_decoder.generate_checkerboard` | Function | `(width: int, height: int, square_size: int) -> np.ndarray` |
| `backend.test.case.node.import_recursive` | Function | `(package: ModuleType) -> None` |
| `backend.test.case.node.instancenorm.Base` | Class | `(...)` |
| `backend.test.case.node.instancenorm.InstanceNormalization` | Class | `(...)` |
| `backend.test.case.node.instancenorm.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.isinf.Base` | Class | `(...)` |
| `backend.test.case.node.isinf.IsInf` | Class | `(...)` |
| `backend.test.case.node.isinf.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.isnan.Base` | Class | `(...)` |
| `backend.test.case.node.isnan.IsNaN` | Class | `(...)` |
| `backend.test.case.node.isnan.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.layernormalization.Base` | Class | `(...)` |
| `backend.test.case.node.layernormalization.LayerNormalization` | Class | `(...)` |
| `backend.test.case.node.layernormalization.calculate_normalized_shape` | Function | `(X_shape, axis)` |
| `backend.test.case.node.layernormalization.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.leakyrelu.Base` | Class | `(...)` |
| `backend.test.case.node.leakyrelu.LeakyRelu` | Class | `(...)` |
| `backend.test.case.node.leakyrelu.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.less.Base` | Class | `(...)` |
| `backend.test.case.node.less.Less` | Class | `(...)` |
| `backend.test.case.node.less.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.less_equal.Base` | Class | `(...)` |
| `backend.test.case.node.less_equal.Less` | Class | `(...)` |
| `backend.test.case.node.less_equal.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.log.Base` | Class | `(...)` |
| `backend.test.case.node.log.Log` | Class | `(...)` |
| `backend.test.case.node.log.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.logsoftmax.Base` | Class | `(...)` |
| `backend.test.case.node.logsoftmax.LogSoftmax` | Class | `(...)` |
| `backend.test.case.node.logsoftmax.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.logsoftmax.logsoftmax` | Function | `(x: np.ndarray, axis: int) -> np.ndarray` |
| `backend.test.case.node.loop.Base` | Class | `(...)` |
| `backend.test.case.node.loop.Loop` | Class | `(...)` |
| `backend.test.case.node.loop.compute_loop_outputs` | Function | `(x, seq, trip_count)` |
| `backend.test.case.node.loop.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.lpnormalization.Base` | Class | `(...)` |
| `backend.test.case.node.lpnormalization.LpNormalization` | Class | `(...)` |
| `backend.test.case.node.lpnormalization.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.lppool.Base` | Class | `(...)` |
| `backend.test.case.node.lppool.LpPool` | Class | `(...)` |
| `backend.test.case.node.lppool.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.lppool.get_output_shape_auto_pad` | Function | `(auto_pad: str, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int]) -> Sequence[int]` |
| `backend.test.case.node.lppool.get_output_shape_explicit_padding` | Function | `(pads: Sequence[int] \| None, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int], dilations: Sequence[int] \| None, ceil_mode: bool) -> tuple[Sequence[int], Sequence[int]]` |
| `backend.test.case.node.lppool.get_pad_shape` | Function | `(auto_pad: str, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int], output_spatial_shape: Sequence[int]) -> Sequence[int]` |
| `backend.test.case.node.lppool.pool` | Function | `(padded: np.ndarray, x_shape: Sequence[int], kernel: Sequence[int], strides: Sequence[int], out_shape: Sequence[int], pooling_type: str, pads_required: Sequence[int] \| None, pads: Sequence[int] \| None, dilations: Sequence[int] \| None, count_include_pad: int, p: int) -> np.ndarray` |
| `backend.test.case.node.lrn.Base` | Class | `(...)` |
| `backend.test.case.node.lrn.LRN` | Class | `(...)` |
| `backend.test.case.node.lrn.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.lstm.Base` | Class | `(...)` |
| `backend.test.case.node.lstm.LSTM` | Class | `(...)` |
| `backend.test.case.node.lstm.LSTMHelper` | Class | `(params: Any)` |
| `backend.test.case.node.lstm.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.matmul.Base` | Class | `(...)` |
| `backend.test.case.node.matmul.MatMul` | Class | `(...)` |
| `backend.test.case.node.matmul.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.matmulinteger.Base` | Class | `(...)` |
| `backend.test.case.node.matmulinteger.MatMulInteger` | Class | `(...)` |
| `backend.test.case.node.matmulinteger.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.max.Base` | Class | `(...)` |
| `backend.test.case.node.max.Max` | Class | `(...)` |
| `backend.test.case.node.max.all_numeric_dtypes` | Object | `` |
| `backend.test.case.node.max.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.maxpool.Base` | Class | `(...)` |
| `backend.test.case.node.maxpool.MaxPool` | Class | `(...)` |
| `backend.test.case.node.maxpool.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.maxpool.get_output_shape_auto_pad` | Function | `(auto_pad: str, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int]) -> Sequence[int]` |
| `backend.test.case.node.maxpool.get_output_shape_explicit_padding` | Function | `(pads: Sequence[int] \| None, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int], dilations: Sequence[int] \| None, ceil_mode: bool) -> tuple[Sequence[int], Sequence[int]]` |
| `backend.test.case.node.maxpool.get_pad_shape` | Function | `(auto_pad: str, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int], output_spatial_shape: Sequence[int]) -> Sequence[int]` |
| `backend.test.case.node.maxpool.pool` | Function | `(padded: np.ndarray, x_shape: Sequence[int], kernel: Sequence[int], strides: Sequence[int], out_shape: Sequence[int], pooling_type: str, pads_required: Sequence[int] \| None, pads: Sequence[int] \| None, dilations: Sequence[int] \| None, count_include_pad: int, p: int) -> np.ndarray` |
| `backend.test.case.node.maxunpool.Base` | Class | `(...)` |
| `backend.test.case.node.maxunpool.MaxUnpool` | Class | `(...)` |
| `backend.test.case.node.maxunpool.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.mean.Base` | Class | `(...)` |
| `backend.test.case.node.mean.Mean` | Class | `(...)` |
| `backend.test.case.node.mean.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.meanvariancenormalization.Base` | Class | `(...)` |
| `backend.test.case.node.meanvariancenormalization.MeanVarianceNormalization` | Class | `(...)` |
| `backend.test.case.node.meanvariancenormalization.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.melweightmatrix.Base` | Class | `(...)` |
| `backend.test.case.node.melweightmatrix.MelWeightMatrix` | Class | `(...)` |
| `backend.test.case.node.melweightmatrix.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.min.Base` | Class | `(...)` |
| `backend.test.case.node.min.Min` | Class | `(...)` |
| `backend.test.case.node.min.all_numeric_dtypes` | Object | `` |
| `backend.test.case.node.min.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.mish.Base` | Class | `(...)` |
| `backend.test.case.node.mish.Mish` | Class | `(...)` |
| `backend.test.case.node.mish.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.mod.Base` | Class | `(...)` |
| `backend.test.case.node.mod.Mod` | Class | `(...)` |
| `backend.test.case.node.mod.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.momentum.AI_ONNX_PREVIEW_TRAINING_DOMAIN` | Object | `` |
| `backend.test.case.node.momentum.Base` | Class | `(...)` |
| `backend.test.case.node.momentum.Momentum` | Class | `(...)` |
| `backend.test.case.node.momentum.apply_momentum` | Function | `(r, t, x, g, v, norm_coefficient, alpha, beta)` |
| `backend.test.case.node.momentum.apply_nesterov` | Function | `(r, t, x, g, v, norm_coefficient, alpha, beta)` |
| `backend.test.case.node.momentum.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.mul.Base` | Class | `(...)` |
| `backend.test.case.node.mul.Mul` | Class | `(...)` |
| `backend.test.case.node.mul.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.neg.Base` | Class | `(...)` |
| `backend.test.case.node.neg.Neg` | Class | `(...)` |
| `backend.test.case.node.neg.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.negativeloglikelihoodloss.Base` | Class | `(...)` |
| `backend.test.case.node.negativeloglikelihoodloss.NegativeLogLikelihoodLoss` | Class | `(...)` |
| `backend.test.case.node.negativeloglikelihoodloss.compute_negative_log_likelihood_loss` | Function | `(input, target, weight, reduction, ignore_index)` |
| `backend.test.case.node.negativeloglikelihoodloss.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.nonmaxsuppression.Base` | Class | `(...)` |
| `backend.test.case.node.nonmaxsuppression.NonMaxSuppression` | Class | `(...)` |
| `backend.test.case.node.nonmaxsuppression.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.nonzero.Base` | Class | `(...)` |
| `backend.test.case.node.nonzero.NonZero` | Class | `(...)` |
| `backend.test.case.node.nonzero.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.not.Base` | Class | `(...)` |
| `backend.test.case.node.not.Not` | Class | `(...)` |
| `backend.test.case.node.not.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.onehot.Base` | Class | `(...)` |
| `backend.test.case.node.onehot.OneHot` | Class | `(...)` |
| `backend.test.case.node.onehot.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.onehot.one_hot` | Function | `(indices, depth, axis, dtype)` |
| `backend.test.case.node.optionalgetelement.Base` | Class | `(...)` |
| `backend.test.case.node.optionalgetelement.OptionalHasElement` | Class | `(...)` |
| `backend.test.case.node.optionalgetelement.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.optionalgetelement.optional_get_element_reference_implementation` | Function | `(optional: Any \| None) -> Any` |
| `backend.test.case.node.optionalhaselement.Base` | Class | `(...)` |
| `backend.test.case.node.optionalhaselement.OptionalHasElement` | Class | `(...)` |
| `backend.test.case.node.optionalhaselement.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.optionalhaselement.optional_has_element_reference_implementation` | Function | `(optional: np.ndarray \| None) -> np.ndarray` |
| `backend.test.case.node.or.Base` | Class | `(...)` |
| `backend.test.case.node.or.Or` | Class | `(...)` |
| `backend.test.case.node.or.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.pad.Base` | Class | `(...)` |
| `backend.test.case.node.pad.Pad` | Class | `(...)` |
| `backend.test.case.node.pad.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.pad.pad_impl` | Function | `(data, raw_pads, mode, constant_values, axes)` |
| `backend.test.case.node.pow.Base` | Class | `(...)` |
| `backend.test.case.node.pow.Pow` | Class | `(...)` |
| `backend.test.case.node.pow.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.pow.pow` | Function | `(x, y)` |
| `backend.test.case.node.prelu.Base` | Class | `(...)` |
| `backend.test.case.node.prelu.PRelu` | Class | `(...)` |
| `backend.test.case.node.prelu.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.qlinearconv.Base` | Class | `(...)` |
| `backend.test.case.node.qlinearconv.QLinearConv` | Class | `(...)` |
| `backend.test.case.node.qlinearconv.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.qlinearmatmul.Base` | Class | `(...)` |
| `backend.test.case.node.qlinearmatmul.QLinearMatMul` | Class | `(...)` |
| `backend.test.case.node.qlinearmatmul.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.quantizelinear.Base` | Class | `(...)` |
| `backend.test.case.node.quantizelinear.QuantizeLinear` | Class | `(...)` |
| `backend.test.case.node.quantizelinear.TensorProto` | Object | `` |
| `backend.test.case.node.quantizelinear.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.quantizelinear.make_tensor` | Function | `(name: str, data_type: int, dims: Sequence[int], vals: Sequence[int \| float] \| bytes \| np.ndarray, raw: bool) -> TensorProto` |
| `backend.test.case.node.rangeop.Base` | Class | `(...)` |
| `backend.test.case.node.rangeop.Range` | Class | `(...)` |
| `backend.test.case.node.rangeop.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reciprocal.Base` | Class | `(...)` |
| `backend.test.case.node.reciprocal.Reciprocal` | Class | `(...)` |
| `backend.test.case.node.reciprocal.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reduce_log_sum.Base` | Class | `(...)` |
| `backend.test.case.node.reduce_log_sum.ReduceLogSum` | Class | `(...)` |
| `backend.test.case.node.reduce_log_sum.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reduce_log_sum_exp.Base` | Class | `(...)` |
| `backend.test.case.node.reduce_log_sum_exp.ReduceLogSumExp` | Class | `(...)` |
| `backend.test.case.node.reduce_log_sum_exp.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reducel1.Base` | Class | `(...)` |
| `backend.test.case.node.reducel1.ReduceL1` | Class | `(...)` |
| `backend.test.case.node.reducel1.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reducel2.Base` | Class | `(...)` |
| `backend.test.case.node.reducel2.ReduceL2` | Class | `(...)` |
| `backend.test.case.node.reducel2.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reducemax.Base` | Class | `(...)` |
| `backend.test.case.node.reducemax.ReduceMax` | Class | `(...)` |
| `backend.test.case.node.reducemax.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reducemean.Base` | Class | `(...)` |
| `backend.test.case.node.reducemean.ReduceMean` | Class | `(...)` |
| `backend.test.case.node.reducemean.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reducemin.Base` | Class | `(...)` |
| `backend.test.case.node.reducemin.ReduceMin` | Class | `(...)` |
| `backend.test.case.node.reducemin.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reduceprod.Base` | Class | `(...)` |
| `backend.test.case.node.reduceprod.ReduceProd` | Class | `(...)` |
| `backend.test.case.node.reduceprod.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reducesum.Base` | Class | `(...)` |
| `backend.test.case.node.reducesum.ReduceSum` | Class | `(...)` |
| `backend.test.case.node.reducesum.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reducesumsquare.Base` | Class | `(...)` |
| `backend.test.case.node.reducesumsquare.ReduceSumSquare` | Class | `(...)` |
| `backend.test.case.node.reducesumsquare.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.regex_full_match.Base` | Class | `(...)` |
| `backend.test.case.node.regex_full_match.RegexFullMatch` | Class | `(...)` |
| `backend.test.case.node.regex_full_match.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.relu.Base` | Class | `(...)` |
| `backend.test.case.node.relu.Relu` | Class | `(...)` |
| `backend.test.case.node.relu.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reshape.Base` | Class | `(...)` |
| `backend.test.case.node.reshape.Reshape` | Class | `(...)` |
| `backend.test.case.node.reshape.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.reshape.reshape_reference_implementation` | Function | `(data: np.ndarray, shape: np.ndarray, allowzero: int) -> np.ndarray` |
| `backend.test.case.node.resize.Base` | Class | `(...)` |
| `backend.test.case.node.resize.Resize` | Class | `(...)` |
| `backend.test.case.node.resize.cubic_coeffs` | Function | `(ratio: float, scale: float \| None, A: float) -> np.ndarray` |
| `backend.test.case.node.resize.cubic_coeffs_antialias` | Function | `(ratio: float, scale: float, A: float) -> np.ndarray` |
| `backend.test.case.node.resize.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.resize.interpolate_nd` | Function | `(data: np.ndarray, get_coeffs: Callable[[float, float], np.ndarray], output_size: list[int] \| None, scale_factors: list[float] \| None, axes: list[int] \| None, roi: np.ndarray \| None, keep_aspect_ratio_policy: str \| None, exclude_outside: bool, kwargs: Any) -> np.ndarray` |
| `backend.test.case.node.resize.linear_coeffs` | Function | `(ratio: float, scale: float \| None) -> np.ndarray` |
| `backend.test.case.node.resize.linear_coeffs_antialias` | Function | `(ratio: float, scale: float) -> np.ndarray` |
| `backend.test.case.node.resize.nearest_coeffs` | Function | `(ratio: float \| int \| np.ndarray, mode: str) -> np.ndarray` |
| `backend.test.case.node.reversesequence.Base` | Class | `(...)` |
| `backend.test.case.node.reversesequence.ReverseSequence` | Class | `(...)` |
| `backend.test.case.node.reversesequence.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.rmsnormalization.Base` | Class | `(...)` |
| `backend.test.case.node.rmsnormalization.RMSNormalization` | Class | `(...)` |
| `backend.test.case.node.rmsnormalization.calculate_normalized_shape` | Function | `(x_shape, axis)` |
| `backend.test.case.node.rmsnormalization.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.rnn.Base` | Class | `(...)` |
| `backend.test.case.node.rnn.RNN` | Class | `(...)` |
| `backend.test.case.node.rnn.RNNHelper` | Class | `(params: Any)` |
| `backend.test.case.node.rnn.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.roialign.Base` | Class | `(...)` |
| `backend.test.case.node.roialign.RoiAlign` | Class | `(...)` |
| `backend.test.case.node.roialign.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.roialign.get_roi_align_input_values` | Function | `()` |
| `backend.test.case.node.rotaryembedding.Base` | Class | `(...)` |
| `backend.test.case.node.rotaryembedding.RotaryEmbedding` | Class | `(...)` |
| `backend.test.case.node.rotaryembedding.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.rotaryembedding.rotary_embedding` | Function | `(input: np.ndarray, cos_cache: np.ndarray, sin_cache: np.ndarray, position_ids: np.ndarray \| None, interleaved, rotary_embedding_dim, num_heads) -> np.ndarray` |
| `backend.test.case.node.round.Base` | Class | `(...)` |
| `backend.test.case.node.round.Round` | Class | `(...)` |
| `backend.test.case.node.round.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.scan.Base` | Class | `(...)` |
| `backend.test.case.node.scan.Scan` | Class | `(...)` |
| `backend.test.case.node.scan.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.scatter.Base` | Class | `(...)` |
| `backend.test.case.node.scatter.Scatter` | Class | `(...)` |
| `backend.test.case.node.scatter.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.scatter.helper` | Object | `` |
| `backend.test.case.node.scatter.scatter` | Function | `(data, indices, updates, axis)` |
| `backend.test.case.node.scatterelements.Base` | Class | `(...)` |
| `backend.test.case.node.scatterelements.ScatterElements` | Class | `(...)` |
| `backend.test.case.node.scatterelements.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.scatterelements.scatter_elements` | Function | `(data, indices, updates, axis, reduction)` |
| `backend.test.case.node.scatternd.Base` | Class | `(...)` |
| `backend.test.case.node.scatternd.ScatterND` | Class | `(...)` |
| `backend.test.case.node.scatternd.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.scatternd.scatter_nd_impl` | Function | `(data, indices, updates, reduction)` |
| `backend.test.case.node.selu.Base` | Class | `(...)` |
| `backend.test.case.node.selu.Selu` | Class | `(...)` |
| `backend.test.case.node.selu.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.sequence_map.Base` | Class | `(...)` |
| `backend.test.case.node.sequence_map.SequenceMap` | Class | `(...)` |
| `backend.test.case.node.sequence_map.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.sequenceinsert.Base` | Class | `(...)` |
| `backend.test.case.node.sequenceinsert.SequenceInsert` | Class | `(...)` |
| `backend.test.case.node.sequenceinsert.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.sequenceinsert.sequence_insert_reference_implementation` | Function | `(sequence: list[Any], tensor: np.ndarray, position: np.ndarray) -> list[Any]` |
| `backend.test.case.node.shape.Base` | Class | `(...)` |
| `backend.test.case.node.shape.Shape` | Class | `(...)` |
| `backend.test.case.node.shape.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.shape.shape_reference_impl` | Function | `(x, start, end)` |
| `backend.test.case.node.shape.test_shape` | Function | `(testname, xval, start, end)` |
| `backend.test.case.node.shrink.Base` | Class | `(...)` |
| `backend.test.case.node.shrink.Shrink` | Class | `(...)` |
| `backend.test.case.node.shrink.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.sigmoid.Base` | Class | `(...)` |
| `backend.test.case.node.sigmoid.Sigmoid` | Class | `(...)` |
| `backend.test.case.node.sigmoid.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.sign.Base` | Class | `(...)` |
| `backend.test.case.node.sign.Sign` | Class | `(...)` |
| `backend.test.case.node.sign.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.sin.Base` | Class | `(...)` |
| `backend.test.case.node.sin.Sin` | Class | `(...)` |
| `backend.test.case.node.sin.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.sinh.Base` | Class | `(...)` |
| `backend.test.case.node.sinh.Sinh` | Class | `(...)` |
| `backend.test.case.node.sinh.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.size.Base` | Class | `(...)` |
| `backend.test.case.node.size.Size` | Class | `(...)` |
| `backend.test.case.node.size.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.slice.Base` | Class | `(...)` |
| `backend.test.case.node.slice.Slice` | Class | `(...)` |
| `backend.test.case.node.slice.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.softmax.Base` | Class | `(...)` |
| `backend.test.case.node.softmax.Softmax` | Class | `(...)` |
| `backend.test.case.node.softmax.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.softmax.softmax` | Function | `(x: np.ndarray, axis: int) -> np.ndarray` |
| `backend.test.case.node.softmaxcrossentropy.Base` | Class | `(...)` |
| `backend.test.case.node.softmaxcrossentropy.SoftmaxCrossEntropyLoss` | Class | `(...)` |
| `backend.test.case.node.softmaxcrossentropy.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.softmaxcrossentropy.softmaxcrossentropy` | Function | `(x, target, weight, reduction, ignore_index, get_log_prob)` |
| `backend.test.case.node.softplus.Base` | Class | `(...)` |
| `backend.test.case.node.softplus.Softplus` | Class | `(...)` |
| `backend.test.case.node.softplus.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.softsign.Base` | Class | `(...)` |
| `backend.test.case.node.softsign.Softsign` | Class | `(...)` |
| `backend.test.case.node.softsign.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.spacetodepth.Base` | Class | `(...)` |
| `backend.test.case.node.spacetodepth.SpaceToDepth` | Class | `(...)` |
| `backend.test.case.node.spacetodepth.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.split.Base` | Class | `(...)` |
| `backend.test.case.node.split.Split` | Class | `(...)` |
| `backend.test.case.node.split.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.splittosequence.Base` | Class | `(...)` |
| `backend.test.case.node.splittosequence.SplitToSequence` | Class | `(...)` |
| `backend.test.case.node.splittosequence.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.sqrt.Base` | Class | `(...)` |
| `backend.test.case.node.sqrt.Sqrt` | Class | `(...)` |
| `backend.test.case.node.sqrt.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.squeeze.Base` | Class | `(...)` |
| `backend.test.case.node.squeeze.Squeeze` | Class | `(...)` |
| `backend.test.case.node.squeeze.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.stft.Base` | Class | `(...)` |
| `backend.test.case.node.stft.STFT` | Class | `(...)` |
| `backend.test.case.node.stft.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.string_concat.Base` | Class | `(...)` |
| `backend.test.case.node.string_concat.StringConcat` | Class | `(...)` |
| `backend.test.case.node.string_concat.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.string_split.Base` | Class | `(...)` |
| `backend.test.case.node.string_split.StringSplit` | Class | `(...)` |
| `backend.test.case.node.string_split.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.stringnormalizer.Base` | Class | `(...)` |
| `backend.test.case.node.stringnormalizer.StringNormalizer` | Class | `(...)` |
| `backend.test.case.node.stringnormalizer.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.sub.Base` | Class | `(...)` |
| `backend.test.case.node.sub.Sub` | Class | `(...)` |
| `backend.test.case.node.sub.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.sum.Base` | Class | `(...)` |
| `backend.test.case.node.sum.Sum` | Class | `(...)` |
| `backend.test.case.node.sum.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.swish.Base` | Class | `(...)` |
| `backend.test.case.node.swish.Swish` | Class | `(...)` |
| `backend.test.case.node.swish.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.swish.swish` | Function | `(x: np.ndarray, alpha: float) -> np.ndarray` |
| `backend.test.case.node.tan.Base` | Class | `(...)` |
| `backend.test.case.node.tan.Tan` | Class | `(...)` |
| `backend.test.case.node.tan.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.tanh.Base` | Class | `(...)` |
| `backend.test.case.node.tanh.Tanh` | Class | `(...)` |
| `backend.test.case.node.tanh.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.tensorscatter.Base` | Class | `(...)` |
| `backend.test.case.node.tensorscatter.TensorScatter` | Class | `(...)` |
| `backend.test.case.node.tensorscatter.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.tfidfvectorizer.Base` | Class | `(...)` |
| `backend.test.case.node.tfidfvectorizer.NodeProto` | Object | `` |
| `backend.test.case.node.tfidfvectorizer.TfIdfVectorizer` | Class | `(...)` |
| `backend.test.case.node.tfidfvectorizer.TfIdfVectorizerHelper` | Class | `(params: Any)` |
| `backend.test.case.node.tfidfvectorizer.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.thresholdedrelu.Base` | Class | `(...)` |
| `backend.test.case.node.thresholdedrelu.ThresholdedRelu` | Class | `(...)` |
| `backend.test.case.node.thresholdedrelu.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.tile.Base` | Class | `(...)` |
| `backend.test.case.node.tile.Tile` | Class | `(...)` |
| `backend.test.case.node.tile.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.topk.Base` | Class | `(...)` |
| `backend.test.case.node.topk.TopK` | Class | `(...)` |
| `backend.test.case.node.topk.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.topk.topk_sorted_implementation` | Function | `(X, k, axis, largest)` |
| `backend.test.case.node.transpose.Base` | Class | `(...)` |
| `backend.test.case.node.transpose.Transpose` | Class | `(...)` |
| `backend.test.case.node.transpose.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.trilu.Base` | Class | `(...)` |
| `backend.test.case.node.trilu.Trilu` | Class | `(...)` |
| `backend.test.case.node.trilu.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.trilu.tril_reference_implementation` | Function | `(x, k)` |
| `backend.test.case.node.trilu.triu_reference_implementation` | Function | `(x, k)` |
| `backend.test.case.node.unique.Base` | Class | `(...)` |
| `backend.test.case.node.unique.Unique` | Class | `(...)` |
| `backend.test.case.node.unique.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.unique.specify_int64` | Function | `(indices, inverse_indices, counts)` |
| `backend.test.case.node.unsqueeze.Base` | Class | `(...)` |
| `backend.test.case.node.unsqueeze.Unsqueeze` | Class | `(...)` |
| `backend.test.case.node.unsqueeze.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.upsample.Base` | Class | `(...)` |
| `backend.test.case.node.upsample.Upsample` | Class | `(...)` |
| `backend.test.case.node.upsample.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.upsample.helper` | Object | `` |
| `backend.test.case.node.where.Base` | Class | `(...)` |
| `backend.test.case.node.where.Where` | Class | `(...)` |
| `backend.test.case.node.where.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.node.xor.Base` | Class | `(...)` |
| `backend.test.case.node.xor.Xor` | Class | `(...)` |
| `backend.test.case.node.xor.expect` | Function | `(node_op: onnx.NodeProto, inputs: Sequence[np.ndarray \| TensorProto], outputs: Sequence[np.ndarray \| TensorProto], name: str, kwargs: Any) -> None` |
| `backend.test.case.test_case.TestCase` | Class | `(name: str, model_name: str, url: str \| None, model_dir: str \| None, model: onnx.ModelProto \| None, data_sets: Sequence[tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] \| None, kind: str, rtol: float, atol: float, __test__: bool)` |
| `backend.test.case.utils.ONNX_ML` | Object | `` |
| `backend.test.case.utils.all_numeric_dtypes` | Object | `` |
| `backend.test.case.utils.import_recursive` | Function | `(package: ModuleType) -> None` |
| `backend.test.cmd_tools.DATA_DIR` | Object | `` |
| `backend.test.cmd_tools.ONNX_ML` | Object | `` |
| `backend.test.cmd_tools.TOP_DIR` | Object | `` |
| `backend.test.cmd_tools.TensorProto` | Object | `` |
| `backend.test.cmd_tools.generate_data` | Function | `(args: argparse.Namespace) -> None` |
| `backend.test.cmd_tools.main` | Function | `() -> None` |
| `backend.test.cmd_tools.model_test` | Object | `` |
| `backend.test.cmd_tools.node_test` | Object | `` |
| `backend.test.cmd_tools.numpy_helper` | Object | `` |
| `backend.test.cmd_tools.parse_args` | Function | `() -> argparse.Namespace` |
| `backend.test.loader.DATA_DIR` | Object | `` |
| `backend.test.loader.TestCase` | Class | `(name: str, model_name: str, url: str \| None, model_dir: str \| None, model: onnx.ModelProto \| None, data_sets: Sequence[tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] \| None, kind: str, rtol: float, atol: float, __test__: bool)` |
| `backend.test.loader.load_model_tests` | Function | `(data_dir: str, kind: str \| None) -> list[TestCase]` |
| `backend.test.report.Coverage` | Class | `()` |
| `backend.test.report.base.ReporterBase` | Class | `(...)` |
| `backend.test.report.coverage.AttrCoverage` | Class | `()` |
| `backend.test.report.coverage.Coverage` | Class | `()` |
| `backend.test.report.coverage.GraphProto` | Object | `` |
| `backend.test.report.coverage.ModelCoverage` | Class | `()` |
| `backend.test.report.coverage.NodeCoverage` | Class | `()` |
| `backend.test.report.coverage.defs` | Object | `` |
| `backend.test.report.coverage.helper` | Object | `` |
| `backend.test.report.pytest_runtest_call` | Function | `(item: _pytest.nodes.Item) -> None` |
| `backend.test.report.pytest_runtest_logreport` | Function | `(report: Any) -> None` |
| `backend.test.report.pytest_terminal_summary` | Function | `(terminalreporter: _pytest.terminal.TerminalReporter, exitstatus: int) -> None` |
| `backend.test.runner.Backend` | Class | `(...)` |
| `backend.test.runner.BackendIsNotSupposedToImplementIt` | Class | `(...)` |
| `backend.test.runner.ModelProto` | Object | `` |
| `backend.test.runner.NodeProto` | Object | `` |
| `backend.test.runner.ONNX_ML` | Object | `` |
| `backend.test.runner.Runner` | Class | `(backend: type[Backend], parent_module: str \| None, test_kwargs: dict \| None)` |
| `backend.test.runner.TestCase` | Class | `(name: str, model_name: str, url: str \| None, model_dir: str \| None, model: onnx.ModelProto \| None, data_sets: Sequence[tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] \| None, kind: str, rtol: float, atol: float, __test__: bool)` |
| `backend.test.runner.TestItem` | Class | `(func: Callable[..., Any], proto: list[ModelProto \| NodeProto \| None])` |
| `backend.test.runner.TypeProto` | Object | `` |
| `backend.test.runner.ValueInfoProto` | Object | `` |
| `backend.test.runner.item.ModelProto` | Object | `` |
| `backend.test.runner.item.NodeProto` | Object | `` |
| `backend.test.runner.item.TestItem` | Class | `(func: Callable[..., Any], proto: list[ModelProto \| NodeProto \| None])` |
| `backend.test.runner.load_model_tests` | Function | `(data_dir: str, kind: str \| None) -> list[TestCase]` |
| `backend.test.runner.numpy_helper` | Object | `` |
| `backend.test.runner.retry_execute` | Function | `(times: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]` |
| `backend.test.stat_coverage.AttributeProto` | Object | `` |
| `backend.test.stat_coverage.Runner` | Class | `(backend: type[Backend], parent_module: str \| None, test_kwargs: dict \| None)` |
| `backend.test.stat_coverage.collect_snippets` | Function | `() -> dict[str, list[tuple[str, str]]]` |
| `backend.test.stat_coverage.common_covered` | Object | `` |
| `backend.test.stat_coverage.defs` | Object | `` |
| `backend.test.stat_coverage.experimental_covered` | Object | `` |
| `backend.test.stat_coverage.gen_model_test_coverage` | Function | `(schemas: Sequence[defs.OpSchema], f: IO[Any], ml: bool) -> None` |
| `backend.test.stat_coverage.gen_node_test_coverage` | Function | `(schemas: Sequence[defs.OpSchema], f: IO[Any], ml: bool) -> None` |
| `backend.test.stat_coverage.gen_outlines` | Function | `(f: IO[Any], ml: bool) -> None` |
| `backend.test.stat_coverage.gen_overall_test_coverage` | Function | `(f: IO[Any]) -> None` |
| `backend.test.stat_coverage.gen_spdx` | Function | `(f: IO[Any]) -> None` |
| `backend.test.stat_coverage.is_ml` | Function | `(schemas: Sequence[defs.OpSchema]) -> bool` |
| `backend.test.stat_coverage.load` | Object | `` |
| `backend.test.stat_coverage.load_model_tests` | Function | `(data_dir: str, kind: str \| None) -> list[TestCase]` |
| `backend.test.stat_coverage.main` | Function | `() -> None` |
| `bin.checker.NodeProto` | Object | `` |
| `bin.checker.check_model` | Function | `() -> None` |
| `bin.checker.check_node` | Function | `() -> None` |
| `bin.checker.checker` | Object | `` |
| `bin.checker.load` | Object | `` |
| `checker.C` | Object | `` |
| `checker.DEFAULT_CONTEXT` | Object | `` |
| `checker.IR_VERSION` | Object | `` |
| `checker.LEXICAL_SCOPE_CONTEXT` | Object | `` |
| `checker.MAXIMUM_PROTOBUF` | Object | `` |
| `checker.ValidationError` | Object | `` |
| `checker.check_attribute` | Function | `(attr: onnx.AttributeProto, ctx: C.CheckerContext, lexical_scope_ctx: C.LexicalScopeContext) -> None` |
| `checker.check_function` | Function | `(function: onnx.FunctionProto, ctx: C.CheckerContext \| None, lexical_scope_ctx: C.LexicalScopeContext) -> None` |
| `checker.check_graph` | Function | `(graph: onnx.GraphProto, ctx: C.CheckerContext, lexical_scope_ctx: C.LexicalScopeContext) -> None` |
| `checker.check_model` | Function | `(model: onnx.ModelProto \| str \| bytes \| os.PathLike, full_check: bool, skip_opset_compatibility_check: bool, check_custom_domain: bool) -> None` |
| `checker.check_node` | Function | `(node: onnx.NodeProto, ctx: C.CheckerContext, lexical_scope_ctx: C.LexicalScopeContext) -> None` |
| `checker.check_sparse_tensor` | Function | `(sparse: onnx.SparseTensorProto, ctx: C.CheckerContext) -> None` |
| `checker.check_tensor` | Function | `(tensor: onnx.TensorProto, ctx: C.CheckerContext) -> None` |
| `checker.check_value_info` | Function | `(value_info: onnx.ValueInfoProto, ctx: C.CheckerContext) -> None` |
| `compose.AttributeProto` | Object | `` |
| `compose.GraphProto` | Object | `` |
| `compose.ModelProto` | Object | `` |
| `compose.TensorProto` | Object | `` |
| `compose.add_prefix` | Function | `(model: ModelProto, prefix: str, rename_nodes: bool \| None, rename_edges: bool \| None, rename_inputs: bool \| None, rename_outputs: bool \| None, rename_initializers: bool \| None, rename_value_infos: bool \| None, rename_functions: bool \| None, inplace: bool \| None) -> ModelProto` |
| `compose.add_prefix_graph` | Function | `(graph: GraphProto, prefix: str, rename_nodes: bool \| None, rename_edges: bool \| None, rename_inputs: bool \| None, rename_outputs: bool \| None, rename_initializers: bool \| None, rename_value_infos: bool \| None, inplace: bool \| None, name_map: dict[str, str] \| None) -> GraphProto` |
| `compose.check_overlapping_names` | Function | `(g1: GraphProto, g2: GraphProto, io_map: list[tuple[str, str]] \| None) -> list[tuple[str, list[str]]]` |
| `compose.checker` | Object | `` |
| `compose.expand_out_dim` | Function | `(model: ModelProto, dim_idx: int, inplace: bool \| None) -> ModelProto` |
| `compose.expand_out_dim_graph` | Function | `(graph: GraphProto, dim_idx: int, inplace: bool \| None) -> GraphProto` |
| `compose.helper` | Object | `` |
| `compose.merge_graphs` | Function | `(g1: GraphProto, g2: GraphProto, io_map: list[tuple[str, str]], inputs: list[str] \| None, outputs: list[str] \| None, prefix1: str \| None, prefix2: str \| None, name: str \| None, doc_string: str \| None) -> GraphProto` |
| `compose.merge_models` | Function | `(m1: ModelProto, m2: ModelProto, io_map: list[tuple[str, str]], inputs: list[str] \| None, outputs: list[str] \| None, prefix1: str \| None, prefix2: str \| None, name: str \| None, doc_string: str \| None, producer_name: str \| None, producer_version: str \| None, domain: str \| None, model_version: int \| None) -> ModelProto` |
| `compose.utils` | Object | `` |
| `convert_model_to_external_data` | Function | `(model: ModelProto, all_tensors_to_one_file: bool, location: str \| None, size_threshold: int, convert_attribute: bool) -> None` |
| `defs.AI_ONNX_PREVIEW_TRAINING_DOMAIN` | Object | `` |
| `defs.C` | Object | `` |
| `defs.ONNX_DOMAIN` | Object | `` |
| `defs.ONNX_ML_DOMAIN` | Object | `` |
| `defs.OpSchema` | Object | `` |
| `defs.SchemaError` | Object | `` |
| `defs.deregister_schema` | Object | `` |
| `defs.gen_doc.Args` | Class | `(...)` |
| `defs.gen_doc.ONNX_ML` | Object | `` |
| `defs.gen_doc.ONNX_ML_DOMAIN` | Object | `` |
| `defs.gen_doc.OpSchema` | Object | `` |
| `defs.gen_doc.SAMPLE_IMPLEMENTATIONS` | Object | `` |
| `defs.gen_doc.SNIPPETS` | Object | `` |
| `defs.gen_doc.collect_sample_implementations` | Function | `() -> dict[str, str]` |
| `defs.gen_doc.collect_snippets` | Function | `() -> dict[str, list[tuple[str, str]]]` |
| `defs.gen_doc.defs` | Object | `` |
| `defs.gen_doc.display_attr_type` | Function | `(v: OpSchema.AttrType) -> str` |
| `defs.gen_doc.display_domain` | Function | `(domain: str) -> str` |
| `defs.gen_doc.display_domain_short` | Function | `(domain: str) -> str` |
| `defs.gen_doc.display_number` | Function | `(v: int) -> str` |
| `defs.gen_doc.display_schema` | Function | `(schema: OpSchema, versions: Sequence[OpSchema], changelog: str) -> str` |
| `defs.gen_doc.display_version_link` | Function | `(name: str, version: int, changelog: str) -> str` |
| `defs.gen_doc.format_function_versions` | Function | `(function_versions: Sequence[int]) -> str` |
| `defs.gen_doc.format_name_with_domain` | Function | `(domain: str, schema_name: str) -> str` |
| `defs.gen_doc.format_versions` | Function | `(versions: Sequence[OpSchema], changelog: str) -> str` |
| `defs.gen_doc.generate_formal_parameter_tags` | Function | `(formal_parameter: OpSchema.FormalParameter) -> str` |
| `defs.gen_doc.helper` | Object | `` |
| `defs.gen_doc.main` | Function | `(args: Args) -> None` |
| `defs.gen_doc.should_render_domain` | Function | `(domain: str, output: str) -> bool` |
| `defs.gen_doc.support_level_str` | Function | `(level: OpSchema.SupportType) -> str` |
| `defs.gen_shape_inference_information.defs` | Object | `` |
| `defs.gen_shape_inference_information.main` | Function | `() -> None` |
| `defs.get_all_schemas` | Object | `` |
| `defs.get_all_schemas_with_history` | Object | `` |
| `defs.get_function_ops` | Function | `() -> list[OpSchema]` |
| `defs.get_schema` | Object | `` |
| `defs.has` | Object | `` |
| `defs.onnx_ml_opset_version` | Function | `() -> int` |
| `defs.onnx_opset_version` | Function | `() -> int` |
| `defs.register_schema` | Function | `(schema: OpSchema) -> None` |
| `external_data_helper.AttributeProto` | Object | `` |
| `external_data_helper.ExternalDataInfo` | Class | `(tensor: TensorProto)` |
| `external_data_helper.FunctionProto` | Object | `` |
| `external_data_helper.GraphProto` | Object | `` |
| `external_data_helper.ModelProto` | Object | `` |
| `external_data_helper.TensorProto` | Object | `` |
| `external_data_helper.c_checker` | Object | `` |
| `external_data_helper.convert_model_from_external_data` | Function | `(model: ModelProto) -> None` |
| `external_data_helper.convert_model_to_external_data` | Function | `(model: ModelProto, all_tensors_to_one_file: bool, location: str \| None, size_threshold: int, convert_attribute: bool) -> None` |
| `external_data_helper.load_external_data_for_model` | Function | `(model: ModelProto, base_dir: str) -> None` |
| `external_data_helper.load_external_data_for_tensor` | Function | `(tensor: TensorProto, base_dir: str) -> None` |
| `external_data_helper.remove_external_data_field` | Function | `(tensor: TensorProto, field_key: str) -> None` |
| `external_data_helper.save_external_data` | Function | `(tensor: TensorProto, base_path: str) -> None` |
| `external_data_helper.set_external_data` | Function | `(tensor: TensorProto, location: str, offset: int \| None, length: int \| None, checksum: str \| None, basepath: str \| None) -> None` |
| `external_data_helper.uses_external_data` | Function | `(tensor: TensorProto) -> bool` |
| `external_data_helper.write_external_data_tensors` | Function | `(model: ModelProto, filepath: str) -> ModelProto` |
| `gen_proto.DEFAULT_PACKAGE_NAME` | Object | `` |
| `gen_proto.ELSE_ONNX_ML_REGEX` | Object | `` |
| `gen_proto.ENDIF_ONNX_ML_REGEX` | Object | `` |
| `gen_proto.IF_ONNX_ML_REGEX` | Object | `` |
| `gen_proto.IMPORT_REGEX` | Object | `` |
| `gen_proto.LITE_OPTION` | Object | `` |
| `gen_proto.ML_REGEX` | Object | `` |
| `gen_proto.OPTIONAL_REGEX` | Object | `` |
| `gen_proto.PACKAGE_NAME_REGEX` | Object | `` |
| `gen_proto.PROTO_SYNTAX_REGEX` | Object | `` |
| `gen_proto.autogen_header` | Object | `` |
| `gen_proto.convert` | Function | `(stem: str, package_name: str, output: str, do_onnx_ml: bool, lite: bool, protoc_path: str) -> None` |
| `gen_proto.convert_to_proto3` | Function | `(lines: Iterable[str]) -> Iterable[str]` |
| `gen_proto.gen_proto3_code` | Function | `(protoc_path: str, proto3_path: str, include_path: str, cpp_out: str, python_out: str) -> None` |
| `gen_proto.main` | Function | `() -> None` |
| `gen_proto.process_ifs` | Function | `(lines: Iterable[str], onnx_ml: bool) -> Iterable[str]` |
| `gen_proto.process_package_name` | Function | `(lines: Iterable[str], package_name: str) -> Iterable[str]` |
| `gen_proto.qualify` | Function | `(f: str, pardir: str \| None) -> str` |
| `gen_proto.translate` | Function | `(source: str, proto: int, onnx_ml: bool, package_name: str) -> str` |
| `helper.AssignmentBindingType` | Object | `` |
| `helper.AttributeProto` | Object | `` |
| `helper.FunctionProto` | Object | `` |
| `helper.GraphProto` | Object | `` |
| `helper.MapProto` | Object | `` |
| `helper.ModelProto` | Object | `` |
| `helper.NodeProto` | Object | `` |
| `helper.OP_SET_ID_VERSION_MAP` | Object | `` |
| `helper.OperatorSetIdProto` | Object | `` |
| `helper.OptionalProto` | Object | `` |
| `helper.SequenceProto` | Object | `` |
| `helper.TensorProto` | Object | `` |
| `helper.TensorShapeProto` | Object | `` |
| `helper.TrainingInfoProto` | Object | `` |
| `helper.TypeProto` | Object | `` |
| `helper.VERSION_TABLE` | Object | `` |
| `helper.ValueInfoProto` | Object | `` |
| `helper.VersionMapType` | Object | `` |
| `helper.VersionRowType` | Object | `` |
| `helper.VersionTableType` | Object | `` |
| `helper.defs` | Object | `` |
| `helper.find_min_ir_version_for` | Function | `(opsetidlist: Sequence[OperatorSetIdProto], ignore_unknown: bool) -> int` |
| `helper.get_all_tensor_dtypes` | Function | `() -> KeysView[int]` |
| `helper.get_attribute_value` | Function | `(attr: AttributeProto) -> Any` |
| `helper.get_node_attr_value` | Function | `(node: NodeProto, attr_name: str) -> Any` |
| `helper.make_attribute` | Function | `(key: str, value: Any, doc_string: str \| None, attr_type: int \| None) -> AttributeProto` |
| `helper.make_attribute_ref` | Function | `(name: str, attr_type: AttributeProto.AttributeType, doc_string: str \| None) -> AttributeProto` |
| `helper.make_empty_tensor_value_info` | Function | `(name: str) -> ValueInfoProto` |
| `helper.make_function` | Function | `(domain: str, fname: str, inputs: Sequence[str], outputs: Sequence[str], nodes: Sequence[NodeProto], opset_imports: Sequence[OperatorSetIdProto], attributes: Sequence[str] \| None, attribute_protos: Sequence[AttributeProto] \| None, doc_string: str \| None, overload: str \| None, value_info: Sequence[ValueInfoProto] \| None) -> FunctionProto` |
| `helper.make_graph` | Function | `(nodes: Sequence[NodeProto], name: str, inputs: Sequence[ValueInfoProto], outputs: Sequence[ValueInfoProto], initializer: Sequence[TensorProto] \| None, doc_string: str \| None, value_info: Sequence[ValueInfoProto] \| None, sparse_initializer: Sequence[onnx.SparseTensorProto] \| None) -> GraphProto` |
| `helper.make_map` | Function | `(name: str, key_type: int, keys: list[Any], values: SequenceProto) -> MapProto` |
| `helper.make_map_type_proto` | Function | `(key_type: int, value_type: TypeProto) -> TypeProto` |
| `helper.make_model` | Function | `(graph: GraphProto, kwargs: Any) -> ModelProto` |
| `helper.make_model_gen_version` | Function | `(graph: GraphProto, kwargs: Any) -> ModelProto` |
| `helper.make_node` | Function | `(op_type: str, inputs: Sequence[str], outputs: Sequence[str], name: str \| None, doc_string: str \| None, domain: str \| None, overload: str \| None, kwargs: Any) -> NodeProto` |
| `helper.make_operatorsetid` | Function | `(domain: str, version: int) -> OperatorSetIdProto` |
| `helper.make_opsetid` | Function | `(domain: str, version: int) -> OperatorSetIdProto` |
| `helper.make_optional` | Function | `(name: str, elem_type: OptionalProto.DataType, value: google.protobuf.message.Message \| None) -> OptionalProto` |
| `helper.make_optional_type_proto` | Function | `(inner_type_proto: TypeProto) -> TypeProto` |
| `helper.make_sequence` | Function | `(name: str, elem_type: SequenceProto.DataType, values: Sequence[Any]) -> SequenceProto` |
| `helper.make_sequence_type_proto` | Function | `(inner_type_proto: TypeProto) -> TypeProto` |
| `helper.make_sparse_tensor` | Function | `(values: TensorProto, indices: TensorProto, dims: Sequence[int]) -> onnx.SparseTensorProto` |
| `helper.make_sparse_tensor_type_proto` | Function | `(elem_type: int, shape: Sequence[str \| int \| None] \| None, shape_denotation: list[str] \| None) -> TypeProto` |
| `helper.make_sparse_tensor_value_info` | Function | `(name: str, elem_type: int, shape: Sequence[str \| int \| None] \| None, doc_string: str, shape_denotation: list[str] \| None) -> ValueInfoProto` |
| `helper.make_tensor` | Function | `(name: str, data_type: int, dims: Sequence[int], vals: Sequence[int \| float] \| bytes \| np.ndarray, raw: bool) -> TensorProto` |
| `helper.make_tensor_sequence_value_info` | Function | `(name: str, elem_type: int, shape: Sequence[str \| int \| None] \| None, doc_string: str, elem_shape_denotation: list[str] \| None) -> ValueInfoProto` |
| `helper.make_tensor_type_proto` | Function | `(elem_type: int, shape: Sequence[str \| int \| None] \| None, shape_denotation: list[str] \| None) -> TypeProto` |
| `helper.make_tensor_value_info` | Function | `(name: str, elem_type: int, shape: Sequence[str \| int \| None] \| None, doc_string: str, shape_denotation: list[str] \| None) -> ValueInfoProto` |
| `helper.make_training_info` | Function | `(algorithm: GraphProto, algorithm_bindings: AssignmentBindingType, initialization: GraphProto \| None, initialization_bindings: AssignmentBindingType \| None) -> TrainingInfoProto` |
| `helper.make_value_info` | Function | `(name: str, type_proto: TypeProto, doc_string: str) -> ValueInfoProto` |
| `helper.np_dtype_to_tensor_dtype` | Function | `(np_dtype: np.dtype) -> TensorProto.DataType` |
| `helper.printable_attribute` | Function | `(attr: AttributeProto, subgraphs: bool) -> str \| tuple[str, list[GraphProto]]` |
| `helper.printable_dim` | Function | `(dim: TensorShapeProto.Dimension) -> str` |
| `helper.printable_graph` | Function | `(graph: GraphProto, prefix: str) -> str` |
| `helper.printable_node` | Function | `(node: NodeProto, prefix: str, subgraphs: bool) -> str \| tuple[str, list[GraphProto]]` |
| `helper.printable_tensor_proto` | Function | `(t: TensorProto) -> str` |
| `helper.printable_type` | Function | `(t: TypeProto) -> str` |
| `helper.printable_value_info` | Function | `(v: ValueInfoProto) -> str` |
| `helper.set_metadata_props` | Function | `(proto: ModelProto \| GraphProto \| FunctionProto \| NodeProto \| TensorProto \| ValueInfoProto, dict_value: dict[str, str]) -> None` |
| `helper.set_model_props` | Function | `(model: ModelProto, dict_value: dict[str, str]) -> None` |
| `helper.strip_doc_string` | Function | `(proto: google.protobuf.message.Message) -> None` |
| `helper.tensor_dtype_to_field` | Function | `(tensor_dtype: int) -> str` |
| `helper.tensor_dtype_to_np_dtype` | Function | `(tensor_dtype: int) -> np.dtype` |
| `helper.tensor_dtype_to_storage_tensor_dtype` | Function | `(tensor_dtype: int) -> int` |
| `helper.tensor_dtype_to_string` | Function | `(tensor_dtype: int) -> str` |
| `hub.ModelInfo` | Class | `(raw_model_info: dict[str, Any])` |
| `hub.download_model_with_test_data` | Function | `(model: str, repo: str, opset: int \| None, force_reload: bool, silent: bool) -> str \| None` |
| `hub.get_dir` | Function | `() -> str` |
| `hub.get_model_info` | Function | `(model: str, repo: str, opset: int \| None) -> ModelInfo` |
| `hub.list_models` | Function | `(repo: str, model: str \| None, tags: list[str] \| None) -> list[ModelInfo]` |
| `hub.load` | Function | `(model: str, repo: str, opset: int \| None, force_reload: bool, silent: bool) -> onnx.ModelProto \| None` |
| `hub.load_composite_model` | Function | `(network_model: str, preprocessing_model: str, network_repo: str, preprocessing_repo: str, opset: int \| None, force_reload: bool, silent: bool) -> onnx.ModelProto \| None` |
| `hub.set_dir` | Function | `(new_dir: str) -> None` |
| `inliner.C` | Object | `` |
| `inliner.inline_local_functions` | Function | `(model: onnx.ModelProto, convert_version: bool) -> onnx.ModelProto` |
| `inliner.inline_selected_functions` | Function | `(model: onnx.ModelProto, function_ids: list[tuple[str, str]], exclude: bool, inline_schema_functions: bool) -> onnx.ModelProto` |
| `load` | Object | `` |
| `load_external_data_for_model` | Function | `(model: ModelProto, base_dir: str) -> None` |
| `load_from_string` | Object | `` |
| `load_model` | Function | `(f: IO[bytes] \| str \| os.PathLike, format: _SupportedFormat \| None, load_external_data: bool) -> ModelProto` |
| `load_model_from_string` | Function | `(s: bytes \| str, format: _SupportedFormat) -> ModelProto` |
| `load_tensor` | Function | `(f: IO[bytes] \| str \| os.PathLike, format: _SupportedFormat \| None) -> TensorProto` |
| `load_tensor_from_string` | Function | `(s: bytes, format: _SupportedFormat) -> TensorProto` |
| `model_container.ModelContainer` | Class | `()` |
| `model_container.c_checker` | Object | `` |
| `model_container.ext_data` | Object | `` |
| `model_container.make_large_model` | Function | `(graph: onnx.GraphProto, large_initializers: dict[str, np.ndarray] \| None, kwargs: Any) -> ModelContainer` |
| `model_container.make_large_tensor_proto` | Function | `(location: str, tensor_name: str, tensor_type: int, shape: tuple[int, ...]) -> onnx.TensorProto` |
| `numpy_helper.create_random_int` | Function | `(input_shape: tuple[int], dtype: np.dtype, seed: int) -> np.ndarray` |
| `numpy_helper.from_array` | Function | `(array: np.ndarray, name: str \| None) -> onnx.TensorProto` |
| `numpy_helper.from_dict` | Function | `(dict_: dict[Any, Any], name: str \| None) -> onnx.MapProto` |
| `numpy_helper.from_list` | Function | `(lst: list[Any], name: str \| None, dtype: int \| None) -> onnx.SequenceProto` |
| `numpy_helper.from_optional` | Function | `(opt: Any \| None, name: str \| None, dtype: int \| None) -> onnx.OptionalProto` |
| `numpy_helper.helper` | Object | `` |
| `numpy_helper.saturate_cast` | Function | `(x: np.ndarray, dtype: np.dtype) -> np.ndarray` |
| `numpy_helper.to_array` | Function | `(tensor: onnx.TensorProto, base_dir: str) -> np.ndarray` |
| `numpy_helper.to_dict` | Function | `(map_proto: onnx.MapProto) -> dict[Any, Any]` |
| `numpy_helper.to_float8e8m0` | Function | `(x: np.ndarray, saturate: bool, round_mode: str) -> np.ndarray` |
| `numpy_helper.to_list` | Function | `(sequence: onnx.SequenceProto) -> list[Any]` |
| `numpy_helper.to_optional` | Function | `(optional: onnx.OptionalProto) -> Any \| None` |
| `numpy_helper.tobytes_little_endian` | Function | `(array: np.ndarray) -> bytes` |
| `onnx_cpp2py_export.ONNX_ML` | Object | `` |
| `onnx_cpp2py_export.checker.CheckerContext` | Class | `(...)` |
| `onnx_cpp2py_export.checker.LexicalScopeContext` | Class | `(...)` |
| `onnx_cpp2py_export.checker.ValidationError` | Class | `(...)` |
| `onnx_cpp2py_export.checker.check_attribute` | Function | `(bytes: bytes, checker_context: CheckerContext, lexical_scope_context: LexicalScopeContext) -> None` |
| `onnx_cpp2py_export.checker.check_function` | Function | `(bytes: bytes, checker_context: CheckerContext, lexical_scope_context: LexicalScopeContext) -> None` |
| `onnx_cpp2py_export.checker.check_graph` | Function | `(bytes: bytes, checker_context: CheckerContext, lexical_scope_context: LexicalScopeContext) -> None` |
| `onnx_cpp2py_export.checker.check_model` | Function | `(bytes: bytes, full_check: bool, skip_opset_compatibility_check: bool, check_custom_domain: bool) -> None` |
| `onnx_cpp2py_export.checker.check_model_path` | Function | `(path: str, full_check: bool, skip_opset_compatibility_check: bool, check_custom_domain: bool) -> None` |
| `onnx_cpp2py_export.checker.check_node` | Function | `(bytes: bytes, checker_context: CheckerContext, lexical_scope_context: LexicalScopeContext) -> None` |
| `onnx_cpp2py_export.checker.check_sparse_tensor` | Function | `(bytes: bytes, checker_context: CheckerContext) -> None` |
| `onnx_cpp2py_export.checker.check_tensor` | Function | `(bytes: bytes, checker_context: CheckerContext) -> None` |
| `onnx_cpp2py_export.checker.check_value_info` | Function | `(bytes: bytes, checker_context: CheckerContext) -> None` |
| `onnx_cpp2py_export.defs.AttributeProto` | Object | `` |
| `onnx_cpp2py_export.defs.FunctionProto` | Object | `` |
| `onnx_cpp2py_export.defs.InferenceContext` | Object | `` |
| `onnx_cpp2py_export.defs.OpSchema` | Class | `(name: str, domain: str, since_version: int, doc: str, inputs: Sequence[OpSchema.FormalParameter], outputs: Sequence[OpSchema.FormalParameter], type_constraints: Sequence[tuple[str, Sequence[str], str]], attributes: Sequence[OpSchema.Attribute])` |
| `onnx_cpp2py_export.defs.SchemaError` | Class | `(...)` |
| `onnx_cpp2py_export.defs.deregister_schema` | Function | `(op_type: str, version: int, domain: str) -> None` |
| `onnx_cpp2py_export.defs.get_all_schemas` | Function | `() -> Sequence[OpSchema]` |
| `onnx_cpp2py_export.defs.get_all_schemas_with_history` | Function | `() -> Sequence[OpSchema]` |
| `onnx_cpp2py_export.defs.register_schema` | Function | `(schema: OpSchema) -> None` |
| `onnx_cpp2py_export.defs.schema_version_map` | Function | `() -> dict[str, tuple[int, int]]` |
| `onnx_cpp2py_export.defs.set_domain_to_version` | Function | `(domain: str, min_version: int, max_version: int, last_release_version: int) -> None` |
| `onnx_cpp2py_export.inliner.inline_local_functions` | Function | `(model: bytes, convert_version: bool) -> bytes` |
| `onnx_cpp2py_export.inliner.inline_selected_functions` | Function | `(model: bytes, function_ids: list[tuple[str, str]], exclude: bool) -> bytes` |
| `onnx_cpp2py_export.inliner.inline_selected_functions2` | Function | `(model: bytes, function_ids: list[tuple[str, str]], exclude: bool) -> bytes` |
| `onnx_cpp2py_export.parser.parse_function` | Function | `(function: str) -> tuple[bool, bytes, bytes]` |
| `onnx_cpp2py_export.parser.parse_graph` | Function | `(graph: str) -> tuple[bool, bytes, bytes]` |
| `onnx_cpp2py_export.parser.parse_model` | Function | `(model: str) -> tuple[bool, bytes, bytes]` |
| `onnx_cpp2py_export.parser.parse_node` | Function | `(node: str) -> tuple[bool, bytes, bytes]` |
| `onnx_cpp2py_export.printer.function_to_text` | Function | `(serialized_function_proto: bytes) -> str` |
| `onnx_cpp2py_export.printer.graph_to_text` | Function | `(serialized_graph_proto: bytes) -> str` |
| `onnx_cpp2py_export.printer.model_to_text` | Function | `(serialized_model_proto: bytes) -> str` |
| `onnx_cpp2py_export.shape_inference.AttributeProto` | Object | `` |
| `onnx_cpp2py_export.shape_inference.GraphInferencer` | Class | `(...)` |
| `onnx_cpp2py_export.shape_inference.InferenceContext` | Class | `(...)` |
| `onnx_cpp2py_export.shape_inference.InferenceError` | Class | `(...)` |
| `onnx_cpp2py_export.shape_inference.SparseTensorProto` | Object | `` |
| `onnx_cpp2py_export.shape_inference.TensorProto` | Object | `` |
| `onnx_cpp2py_export.shape_inference.TensorShapeProto` | Object | `` |
| `onnx_cpp2py_export.shape_inference.TypeProto` | Object | `` |
| `onnx_cpp2py_export.shape_inference.infer_function_output_types` | Function | `(b: bytes, input_types: list[bytes], attributes: list[bytes]) -> list[bytes]` |
| `onnx_cpp2py_export.shape_inference.infer_shapes` | Function | `(b: bytes, check_type: bool, strict_mode: bool, data_prop: bool) -> bytes` |
| `onnx_cpp2py_export.shape_inference.infer_shapes_path` | Function | `(model_path: str, output_path: str, check_type: bool, strict_mode: bool, data_prop: bool) -> None` |
| `onnx_cpp2py_export.version_converter.ConvertError` | Class | `(...)` |
| `onnx_cpp2py_export.version_converter.convert_version` | Function | `(bytes: bytes, target: int) -> bytes` |
| `onnx_data_pb.DESCRIPTOR` | Object | `` |
| `onnx_data_pb.onnx_dot_onnx__ml__pb2` | Object | `` |
| `onnx_data_pb2.DESCRIPTOR` | Object | `` |
| `onnx_data_pb2.MapProto` | Class | `(name: _Optional[str], key_type: _Optional[int], keys: _Optional[_Iterable[int]], string_keys: _Optional[_Iterable[bytes]], values: _Optional[_Union[SequenceProto, _Mapping]])` |
| `onnx_data_pb2.OptionalProto` | Class | `(name: _Optional[str], elem_type: _Optional[int], tensor_value: _Optional[_Union[_onnx_ml_pb2.TensorProto, _Mapping]], sparse_tensor_value: _Optional[_Union[_onnx_ml_pb2.SparseTensorProto, _Mapping]], sequence_value: _Optional[_Union[SequenceProto, _Mapping]], map_value: _Optional[_Union[MapProto, _Mapping]], optional_value: _Optional[_Union[OptionalProto, _Mapping]])` |
| `onnx_data_pb2.SequenceProto` | Class | `(name: _Optional[str], elem_type: _Optional[int], tensor_values: _Optional[_Iterable[_Union[_onnx_ml_pb2.TensorProto, _Mapping]]], sparse_tensor_values: _Optional[_Iterable[_Union[_onnx_ml_pb2.SparseTensorProto, _Mapping]]], sequence_values: _Optional[_Iterable[_Union[SequenceProto, _Mapping]]], map_values: _Optional[_Iterable[_Union[MapProto, _Mapping]]], optional_values: _Optional[_Iterable[_Union[OptionalProto, _Mapping]]])` |
| `onnx_data_pb2.onnx_dot_onnx__ml__pb2` | Object | `` |
| `onnx_ml_pb2.AttributeProto` | Class | `(name: _Optional[str], ref_attr_name: _Optional[str], doc_string: _Optional[str], type: _Optional[_Union[AttributeProto.AttributeType, str]], f: _Optional[float], i: _Optional[int], s: _Optional[bytes], t: _Optional[_Union[TensorProto, _Mapping]], g: _Optional[_Union[GraphProto, _Mapping]], sparse_tensor: _Optional[_Union[SparseTensorProto, _Mapping]], tp: _Optional[_Union[TypeProto, _Mapping]], floats: _Optional[_Iterable[float]], ints: _Optional[_Iterable[int]], strings: _Optional[_Iterable[bytes]], tensors: _Optional[_Iterable[_Union[TensorProto, _Mapping]]], graphs: _Optional[_Iterable[_Union[GraphProto, _Mapping]]], sparse_tensors: _Optional[_Iterable[_Union[SparseTensorProto, _Mapping]]], type_protos: _Optional[_Iterable[_Union[TypeProto, _Mapping]]])` |
| `onnx_ml_pb2.DESCRIPTOR` | Object | `` |
| `onnx_ml_pb2.DeviceConfigurationProto` | Class | `(name: _Optional[str], num_devices: _Optional[int], device: _Optional[_Iterable[str]])` |
| `onnx_ml_pb2.EXPERIMENTAL` | Object | `` |
| `onnx_ml_pb2.FunctionProto` | Class | `(name: _Optional[str], input: _Optional[_Iterable[str]], output: _Optional[_Iterable[str]], attribute: _Optional[_Iterable[str]], attribute_proto: _Optional[_Iterable[_Union[AttributeProto, _Mapping]]], node: _Optional[_Iterable[_Union[NodeProto, _Mapping]]], doc_string: _Optional[str], opset_import: _Optional[_Iterable[_Union[OperatorSetIdProto, _Mapping]]], domain: _Optional[str], overload: _Optional[str], value_info: _Optional[_Iterable[_Union[ValueInfoProto, _Mapping]]], metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]])` |
| `onnx_ml_pb2.GraphProto` | Class | `(node: _Optional[_Iterable[_Union[NodeProto, _Mapping]]], name: _Optional[str], initializer: _Optional[_Iterable[_Union[TensorProto, _Mapping]]], sparse_initializer: _Optional[_Iterable[_Union[SparseTensorProto, _Mapping]]], doc_string: _Optional[str], input: _Optional[_Iterable[_Union[ValueInfoProto, _Mapping]]], output: _Optional[_Iterable[_Union[ValueInfoProto, _Mapping]]], value_info: _Optional[_Iterable[_Union[ValueInfoProto, _Mapping]]], quantization_annotation: _Optional[_Iterable[_Union[TensorAnnotation, _Mapping]]], metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]])` |
| `onnx_ml_pb2.IR_VERSION` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2017_10_10` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2017_10_30` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2017_11_3` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2019_1_22` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2019_3_18` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2019_9_19` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2020_5_8` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2021_7_30` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2023_5_5` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2024_3_25` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2025_05_12` | Object | `` |
| `onnx_ml_pb2.IR_VERSION_2025_08_26` | Object | `` |
| `onnx_ml_pb2.IntIntListEntryProto` | Class | `(key: _Optional[int], value: _Optional[_Iterable[int]])` |
| `onnx_ml_pb2.ModelProto` | Class | `(ir_version: _Optional[int], opset_import: _Optional[_Iterable[_Union[OperatorSetIdProto, _Mapping]]], producer_name: _Optional[str], producer_version: _Optional[str], domain: _Optional[str], model_version: _Optional[int], doc_string: _Optional[str], graph: _Optional[_Union[GraphProto, _Mapping]], metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]], training_info: _Optional[_Iterable[_Union[TrainingInfoProto, _Mapping]]], functions: _Optional[_Iterable[_Union[FunctionProto, _Mapping]]], configuration: _Optional[_Iterable[_Union[DeviceConfigurationProto, _Mapping]]])` |
| `onnx_ml_pb2.NodeDeviceConfigurationProto` | Class | `(configuration_id: _Optional[str], sharding_spec: _Optional[_Iterable[_Union[ShardingSpecProto, _Mapping]]], pipeline_stage: _Optional[int])` |
| `onnx_ml_pb2.NodeProto` | Class | `(input: _Optional[_Iterable[str]], output: _Optional[_Iterable[str]], name: _Optional[str], op_type: _Optional[str], domain: _Optional[str], overload: _Optional[str], attribute: _Optional[_Iterable[_Union[AttributeProto, _Mapping]]], doc_string: _Optional[str], metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]], device_configurations: _Optional[_Iterable[_Union[NodeDeviceConfigurationProto, _Mapping]]])` |
| `onnx_ml_pb2.OperatorSetIdProto` | Class | `(domain: _Optional[str], version: _Optional[int])` |
| `onnx_ml_pb2.OperatorStatus` | Class | `(...)` |
| `onnx_ml_pb2.STABLE` | Object | `` |
| `onnx_ml_pb2.ShardedDimProto` | Class | `(axis: _Optional[int], simple_sharding: _Optional[_Iterable[_Union[SimpleShardedDimProto, _Mapping]]])` |
| `onnx_ml_pb2.ShardingSpecProto` | Class | `(tensor_name: _Optional[str], device: _Optional[_Iterable[int]], index_to_device_group_map: _Optional[_Iterable[_Union[IntIntListEntryProto, _Mapping]]], sharded_dim: _Optional[_Iterable[_Union[ShardedDimProto, _Mapping]]])` |
| `onnx_ml_pb2.SimpleShardedDimProto` | Class | `(dim_value: _Optional[int], dim_param: _Optional[str], num_shards: _Optional[int])` |
| `onnx_ml_pb2.SparseTensorProto` | Class | `(values: _Optional[_Union[TensorProto, _Mapping]], indices: _Optional[_Union[TensorProto, _Mapping]], dims: _Optional[_Iterable[int]])` |
| `onnx_ml_pb2.StringStringEntryProto` | Class | `(key: _Optional[str], value: _Optional[str])` |
| `onnx_ml_pb2.TensorAnnotation` | Class | `(tensor_name: _Optional[str], quant_parameter_tensor_names: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]])` |
| `onnx_ml_pb2.TensorProto` | Class | `(dims: _Optional[_Iterable[int]], data_type: _Optional[int], segment: _Optional[_Union[TensorProto.Segment, _Mapping]], float_data: _Optional[_Iterable[float]], int32_data: _Optional[_Iterable[int]], string_data: _Optional[_Iterable[bytes]], int64_data: _Optional[_Iterable[int]], name: _Optional[str], doc_string: _Optional[str], raw_data: _Optional[bytes], external_data: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]], data_location: _Optional[_Union[TensorProto.DataLocation, str]], double_data: _Optional[_Iterable[float]], uint64_data: _Optional[_Iterable[int]], metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]])` |
| `onnx_ml_pb2.TensorShapeProto` | Class | `(dim: _Optional[_Iterable[_Union[TensorShapeProto.Dimension, _Mapping]]])` |
| `onnx_ml_pb2.TrainingInfoProto` | Class | `(initialization: _Optional[_Union[GraphProto, _Mapping]], algorithm: _Optional[_Union[GraphProto, _Mapping]], initialization_binding: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]], update_binding: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]])` |
| `onnx_ml_pb2.TypeProto` | Class | `(tensor_type: _Optional[_Union[TypeProto.Tensor, _Mapping]], sequence_type: _Optional[_Union[TypeProto.Sequence, _Mapping]], map_type: _Optional[_Union[TypeProto.Map, _Mapping]], optional_type: _Optional[_Union[TypeProto.Optional, _Mapping]], sparse_tensor_type: _Optional[_Union[TypeProto.SparseTensor, _Mapping]], opaque_type: _Optional[_Union[TypeProto.Opaque, _Mapping]], denotation: _Optional[str])` |
| `onnx_ml_pb2.ValueInfoProto` | Class | `(name: _Optional[str], type: _Optional[_Union[TypeProto, _Mapping]], doc_string: _Optional[str], metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]])` |
| `onnx_ml_pb2.Version` | Class | `(...)` |
| `onnx_operators_ml_pb2.DESCRIPTOR` | Object | `` |
| `onnx_operators_ml_pb2.OperatorProto` | Class | `(op_type: _Optional[str], since_version: _Optional[int], status: _Optional[_Union[_onnx_ml_pb2.OperatorStatus, str]], doc_string: _Optional[str])` |
| `onnx_operators_ml_pb2.OperatorSetProto` | Class | `(magic: _Optional[str], ir_version: _Optional[int], ir_version_prerelease: _Optional[str], ir_build_metadata: _Optional[str], domain: _Optional[str], opset_version: _Optional[int], doc_string: _Optional[str], operator: _Optional[_Iterable[_Union[OperatorProto, _Mapping]]], functions: _Optional[_Iterable[_Union[_onnx_ml_pb2.FunctionProto, _Mapping]]])` |
| `onnx_operators_ml_pb2.onnx_dot_onnx__ml__pb2` | Object | `` |
| `onnx_operators_pb.DESCRIPTOR` | Object | `` |
| `onnx_operators_pb.onnx_dot_onnx__ml__pb2` | Object | `` |
| `onnx_pb.DESCRIPTOR` | Object | `` |
| `parser.C` | Object | `` |
| `parser.ParseError` | Class | `(...)` |
| `parser.parse_function` | Function | `(function_text: str) -> onnx.FunctionProto` |
| `parser.parse_graph` | Function | `(graph_text: str) -> onnx.GraphProto` |
| `parser.parse_model` | Function | `(model_text: str) -> onnx.ModelProto` |
| `parser.parse_node` | Function | `(node_text: str) -> onnx.NodeProto` |
| `printer.C` | Object | `` |
| `printer.to_text` | Function | `(proto: onnx.ModelProto \| onnx.FunctionProto \| onnx.GraphProto) -> str` |
| `reference.ReferenceEvaluator` | Class | `(proto: Any, opsets: dict[str, int] \| None, functions: list[ReferenceEvaluator \| FunctionProto] \| None, verbose: int, new_ops: list[type[op_run.OpRun]] \| None, optimized: bool)` |
| `reference.op_run.DefaultNone` | Class | `(...)` |
| `reference.op_run.Graph` | Class | `(g: onnx.GraphProto)` |
| `reference.op_run.OnnxType` | Class | `(type_proto: onnx.TypeProto)` |
| `reference.op_run.OpFunction` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any] \| None, impl: Any, attributes: dict[str, Any] \| None)` |
| `reference.op_run.OpFunctionContextDependant` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any] \| None, parent: Any)` |
| `reference.op_run.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.op_run.OpRunExpand` | Class | `(args, kwargs)` |
| `reference.op_run.RefAttrName` | Class | `(name: str)` |
| `reference.op_run.RuntimeContextError` | Class | `(...)` |
| `reference.op_run.RuntimeImplementationError` | Class | `(...)` |
| `reference.op_run.RuntimeTypeError` | Class | `(...)` |
| `reference.op_run.SparseTensor` | Class | `(values: np.ndarray, indices: np.ndarray, shape: tuple[int])` |
| `reference.op_run.to_sparse_tensor` | Function | `(att: onnx.AttributeProto) -> SparseTensor` |
| `reference.ops.aionnx_preview_training.OpRunTraining` | Class | `(...)` |
| `reference.ops.aionnx_preview_training.load_op` | Function | `(domain: str, op_type: str, version: None \| int, custom: Any) -> Any` |
| `reference.ops.aionnx_preview_training.op_adagrad.Adagrad` | Class | `(...)` |
| `reference.ops.aionnx_preview_training.op_adagrad.OpRunTraining` | Class | `(...)` |
| `reference.ops.aionnx_preview_training.op_adam.Adam` | Class | `(...)` |
| `reference.ops.aionnx_preview_training.op_adam.OpRunTraining` | Class | `(...)` |
| `reference.ops.aionnx_preview_training.op_momentum.Momentum` | Class | `(...)` |
| `reference.ops.aionnx_preview_training.op_momentum.OpRunTraining` | Class | `(...)` |
| `reference.ops.aionnxml.load_op` | Function | `(domain: str, op_type: str, version: None \| int, custom: Any) -> Any` |
| `reference.ops.aionnxml.op_array_feature_extractor.ArrayFeatureExtractor` | Class | `(...)` |
| `reference.ops.aionnxml.op_array_feature_extractor.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_binarizer.Binarizer` | Class | `(...)` |
| `reference.ops.aionnxml.op_binarizer.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_binarizer.compute_binarizer` | Function | `(x, threshold)` |
| `reference.ops.aionnxml.op_dict_vectorizer.DictVectorizer` | Class | `(...)` |
| `reference.ops.aionnxml.op_dict_vectorizer.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_feature_vectorizer.FeatureVectorizer` | Class | `(...)` |
| `reference.ops.aionnxml.op_feature_vectorizer.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_imputer.Imputer` | Class | `(...)` |
| `reference.ops.aionnxml.op_imputer.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_label_encoder.LabelEncoder` | Class | `(...)` |
| `reference.ops.aionnxml.op_label_encoder.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_linear_classifier.LinearClassifier` | Class | `(...)` |
| `reference.ops.aionnxml.op_linear_classifier.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_linear_classifier.compute_probit` | Function | `(val: float) -> float` |
| `reference.ops.aionnxml.op_linear_classifier.compute_softmax_zero` | Function | `(values: np.ndarray) -> np.ndarray` |
| `reference.ops.aionnxml.op_linear_classifier.expit` | Function | `(x: np.ndarray) -> np.ndarray` |
| `reference.ops.aionnxml.op_linear_regressor.LinearRegressor` | Class | `(...)` |
| `reference.ops.aionnxml.op_linear_regressor.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_normalizer.Normalizer` | Class | `(...)` |
| `reference.ops.aionnxml.op_normalizer.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_one_hot_encoder.OneHotEncoder` | Class | `(...)` |
| `reference.ops.aionnxml.op_one_hot_encoder.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_scaler.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_scaler.Scaler` | Class | `(...)` |
| `reference.ops.aionnxml.op_svm_classifier.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_svm_classifier.SVMClassifier` | Class | `(...)` |
| `reference.ops.aionnxml.op_svm_classifier.SVMCommon` | Class | `(kwargs)` |
| `reference.ops.aionnxml.op_svm_classifier.compute_logistic` | Function | `(val: float) -> float` |
| `reference.ops.aionnxml.op_svm_classifier.compute_probit` | Function | `(val: float) -> float` |
| `reference.ops.aionnxml.op_svm_classifier.compute_softmax_zero` | Function | `(values: np.ndarray) -> np.ndarray` |
| `reference.ops.aionnxml.op_svm_classifier.logistic` | Object | `` |
| `reference.ops.aionnxml.op_svm_classifier.multiclass_probability` | Function | `(k, R)` |
| `reference.ops.aionnxml.op_svm_classifier.set_score_svm` | Function | `(max_weight, maxclass, has_proba, weights_are_all_positive_, classlabels, posclass, negclass)` |
| `reference.ops.aionnxml.op_svm_classifier.sigmoid_probability` | Function | `(score, proba, probb)` |
| `reference.ops.aionnxml.op_svm_classifier.softmax` | Function | `(values: np.ndarray) -> np.ndarray` |
| `reference.ops.aionnxml.op_svm_classifier.softmax_zero` | Function | `(values: np.ndarray) -> np.ndarray` |
| `reference.ops.aionnxml.op_svm_classifier.write_scores` | Function | `(n_classes, scores, post_transform, add_second_class)` |
| `reference.ops.aionnxml.op_svm_helper.SVMAttributes` | Class | `()` |
| `reference.ops.aionnxml.op_svm_helper.SVMCommon` | Class | `(kwargs)` |
| `reference.ops.aionnxml.op_svm_regressor.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_svm_regressor.SVMCommon` | Class | `(kwargs)` |
| `reference.ops.aionnxml.op_svm_regressor.SVMRegressor` | Class | `(...)` |
| `reference.ops.aionnxml.op_tree_ensemble.AggregationFunction` | Class | `(...)` |
| `reference.ops.aionnxml.op_tree_ensemble.Leaf` | Class | `(weight: float, target_id: int)` |
| `reference.ops.aionnxml.op_tree_ensemble.Mode` | Class | `(...)` |
| `reference.ops.aionnxml.op_tree_ensemble.Node` | Class | `(mode: Mode, value: float \| set[float], feature: int, missing_tracks_true: bool)` |
| `reference.ops.aionnxml.op_tree_ensemble.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_tree_ensemble.PostTransform` | Class | `(...)` |
| `reference.ops.aionnxml.op_tree_ensemble.TreeEnsemble` | Class | `(...)` |
| `reference.ops.aionnxml.op_tree_ensemble_classifier.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_tree_ensemble_classifier.TreeEnsemble` | Class | `(kwargs)` |
| `reference.ops.aionnxml.op_tree_ensemble_classifier.TreeEnsembleClassifier` | Class | `(...)` |
| `reference.ops.aionnxml.op_tree_ensemble_classifier.logistic` | Object | `` |
| `reference.ops.aionnxml.op_tree_ensemble_classifier.probit` | Object | `` |
| `reference.ops.aionnxml.op_tree_ensemble_classifier.softmax` | Function | `(values: np.ndarray) -> np.ndarray` |
| `reference.ops.aionnxml.op_tree_ensemble_classifier.softmax_zero` | Function | `(values: np.ndarray) -> np.ndarray` |
| `reference.ops.aionnxml.op_tree_ensemble_helper.TreeEnsemble` | Class | `(kwargs)` |
| `reference.ops.aionnxml.op_tree_ensemble_helper.TreeEnsembleAttributes` | Class | `()` |
| `reference.ops.aionnxml.op_tree_ensemble_regressor.OpRunAiOnnxMl` | Class | `(...)` |
| `reference.ops.aionnxml.op_tree_ensemble_regressor.TreeEnsemble` | Class | `(kwargs)` |
| `reference.ops.aionnxml.op_tree_ensemble_regressor.TreeEnsembleRegressor` | Class | `(...)` |
| `reference.ops.experimental.load_op` | Function | `(domain: str, op_type: str, version: None \| int, custom: Any) -> Any` |
| `reference.ops.experimental.op_im2col.Im2Col` | Class | `(...)` |
| `reference.ops.experimental.op_im2col.OpRunExperimental` | Class | `(...)` |
| `reference.ops.experimental.op_im2col.im2col_fast` | Function | `(X, kernel_shape, pads, strides)` |
| `reference.ops.load_op` | Function | `(domain: str, op_type: str, version: None \| int, custom: Any, node: None \| NodeProto, input_types: None \| list[TypeProto], expand: bool, evaluator_cls: type \| None) -> Any` |
| `reference.ops.op_abs.Abs` | Class | `(...)` |
| `reference.ops.op_abs.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_acos.Acos` | Class | `(...)` |
| `reference.ops.op_acos.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_acosh.Acosh` | Class | `(...)` |
| `reference.ops.op_acosh.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_add.Add` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_add.OpRunBinaryNumpy` | Class | `(numpy_fct: Any, onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_affine_grid.AffineGrid` | Class | `(...)` |
| `reference.ops.op_affine_grid.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_affine_grid.apply_affine_transform` | Function | `(theta_n, original_grid_homo)` |
| `reference.ops.op_affine_grid.construct_original_grid` | Function | `(data_size, align_corners)` |
| `reference.ops.op_and.And` | Class | `(...)` |
| `reference.ops.op_and.OpRunBinary` | Class | `(...)` |
| `reference.ops.op_argmax.ArgMax_1` | Class | `(...)` |
| `reference.ops.op_argmax.ArgMax_12` | Class | `(...)` |
| `reference.ops.op_argmax.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_argmin.ArgMin_1` | Class | `(...)` |
| `reference.ops.op_argmin.ArgMin_12` | Class | `(...)` |
| `reference.ops.op_argmin.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_asin.Asin` | Class | `(...)` |
| `reference.ops.op_asin.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_asinh.Asinh` | Class | `(...)` |
| `reference.ops.op_asinh.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_atan.Atan` | Class | `(...)` |
| `reference.ops.op_atan.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_atanh.Atanh` | Class | `(...)` |
| `reference.ops.op_atanh.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_attention.Attention` | Class | `(...)` |
| `reference.ops.op_attention.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_attribute_has_value.AttributeHasValue` | Class | `(...)` |
| `reference.ops.op_attribute_has_value.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_average_pool.AveragePool_1` | Class | `(...)` |
| `reference.ops.op_average_pool.AveragePool_11` | Class | `(...)` |
| `reference.ops.op_average_pool.AveragePool_19` | Class | `(...)` |
| `reference.ops.op_average_pool.AveragePool_7` | Class | `(...)` |
| `reference.ops.op_average_pool.CommonPool` | Class | `(...)` |
| `reference.ops.op_batch_normalization.BatchNormalization_14` | Class | `(...)` |
| `reference.ops.op_batch_normalization.BatchNormalization_6` | Class | `(...)` |
| `reference.ops.op_batch_normalization.BatchNormalization_9` | Class | `(...)` |
| `reference.ops.op_batch_normalization.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_bernoulli.Bernoulli` | Class | `(...)` |
| `reference.ops.op_bernoulli.np_dtype_to_tensor_dtype` | Function | `(np_dtype: np.dtype) -> TensorProto.DataType` |
| `reference.ops.op_bitshift.BitShift` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_bitshift.OpRunBinaryNumpy` | Class | `(numpy_fct: Any, onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_bitwise_and.BitwiseAnd` | Class | `(...)` |
| `reference.ops.op_bitwise_and.OpRunBinary` | Class | `(...)` |
| `reference.ops.op_bitwise_not.BitwiseNot` | Class | `(...)` |
| `reference.ops.op_bitwise_not.OpRunUnary` | Class | `(...)` |
| `reference.ops.op_bitwise_or.BitwiseOr` | Class | `(...)` |
| `reference.ops.op_bitwise_or.OpRunBinary` | Class | `(...)` |
| `reference.ops.op_bitwise_xor.BitwiseXor` | Class | `(...)` |
| `reference.ops.op_bitwise_xor.OpRunBinary` | Class | `(...)` |
| `reference.ops.op_blackman_window.BlackmanWindow` | Class | `(...)` |
| `reference.ops.op_cast.Cast_1` | Class | `(...)` |
| `reference.ops.op_cast.Cast_19` | Class | `(...)` |
| `reference.ops.op_cast.Cast_24` | Class | `(...)` |
| `reference.ops.op_cast.Cast_25` | Class | `(...)` |
| `reference.ops.op_cast.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_cast.cast_to` | Function | `(x: np.ndarray, to: onnx.TensorProto.DataType, saturate: bool, round_mode: str)` |
| `reference.ops.op_cast_like.CastLike_15` | Class | `(...)` |
| `reference.ops.op_cast_like.CastLike_19` | Class | `(...)` |
| `reference.ops.op_cast_like.CastLike_25` | Class | `(...)` |
| `reference.ops.op_cast_like.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_cast_like.cast_to` | Function | `(x: np.ndarray, to: onnx.TensorProto.DataType, saturate: bool, round_mode: str)` |
| `reference.ops.op_cast_like.np_dtype_to_tensor_dtype` | Function | `(np_dtype: np.dtype) -> TensorProto.DataType` |
| `reference.ops.op_ceil.Ceil` | Class | `(...)` |
| `reference.ops.op_ceil.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_celu.Celu` | Class | `(...)` |
| `reference.ops.op_celu.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_center_crop_pad.CenterCropPad` | Class | `(...)` |
| `reference.ops.op_center_crop_pad.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_clip.Clip_11` | Class | `(...)` |
| `reference.ops.op_clip.Clip_6` | Class | `(...)` |
| `reference.ops.op_clip.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_col2im.Col2Im` | Class | `(...)` |
| `reference.ops.op_col2im.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_col2im.col2im_naive_implementation` | Function | `(data, image_shape, kernel_shape, dilations, pads, strides)` |
| `reference.ops.op_compress.Compress` | Class | `(...)` |
| `reference.ops.op_compress.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_concat.Concat` | Class | `(...)` |
| `reference.ops.op_concat.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_concat_from_sequence.ConcatFromSequence` | Class | `(...)` |
| `reference.ops.op_concat_from_sequence.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_constant.ConstantCommon` | Class | `(...)` |
| `reference.ops.op_constant.Constant_1` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_constant.Constant_11` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_constant.Constant_12` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_constant.Constant_9` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_constant.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_constant.RefAttrName` | Class | `(name: str)` |
| `reference.ops.op_constant_of_shape.ConstantOfShape` | Class | `(...)` |
| `reference.ops.op_constant_of_shape.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_conv.Conv` | Class | `(...)` |
| `reference.ops.op_conv.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_conv_integer.ConvInteger` | Class | `(...)` |
| `reference.ops.op_conv_integer.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_conv_transpose.ConvTranspose` | Class | `(...)` |
| `reference.ops.op_conv_transpose.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_conv_transpose.col2im_naive_implementation` | Function | `(data, image_shape, kernel_shape, dilations, pads, strides)` |
| `reference.ops.op_cos.Cos` | Class | `(...)` |
| `reference.ops.op_cos.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_cosh.Cosh` | Class | `(...)` |
| `reference.ops.op_cosh.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_cum_sum.CumSum` | Class | `(...)` |
| `reference.ops.op_cum_sum.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_deform_conv.DeformConv` | Class | `(...)` |
| `reference.ops.op_deform_conv.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_deform_conv.op_grid_sample` | Object | `` |
| `reference.ops.op_depth_to_space.DepthToSpace` | Class | `(...)` |
| `reference.ops.op_depth_to_space.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_dequantize_linear.DequantizeLinear_19` | Class | `(...)` |
| `reference.ops.op_dequantize_linear.DequantizeLinear_21` | Class | `(...)` |
| `reference.ops.op_dequantize_linear.DequantizeLinear_23` | Class | `(...)` |
| `reference.ops.op_dequantize_linear.DequantizeLinear_25` | Class | `(...)` |
| `reference.ops.op_dequantize_linear.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_dequantize_linear.TensorProto` | Object | `` |
| `reference.ops.op_dequantize_linear.np_dtype_to_tensor_dtype` | Function | `(np_dtype: np.dtype) -> TensorProto.DataType` |
| `reference.ops.op_dequantize_linear.tensor_dtype_to_np_dtype` | Function | `(tensor_dtype: int) -> np.dtype` |
| `reference.ops.op_det.Det` | Class | `(...)` |
| `reference.ops.op_det.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_dft.DFT_17` | Class | `(...)` |
| `reference.ops.op_dft.DFT_20` | Class | `(...)` |
| `reference.ops.op_dft.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_div.Div` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_div.OpRunBinaryNumpy` | Class | `(numpy_fct: Any, onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_dropout.DropoutBase` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_dropout.Dropout_12` | Class | `(...)` |
| `reference.ops.op_dropout.Dropout_7` | Class | `(...)` |
| `reference.ops.op_dropout.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_dynamic_quantize_linear.DynamicQuantizeLinear` | Class | `(...)` |
| `reference.ops.op_dynamic_quantize_linear.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_einsum.Einsum` | Class | `(...)` |
| `reference.ops.op_einsum.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_elu.Elu` | Class | `(...)` |
| `reference.ops.op_elu.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_equal.Equal` | Class | `(...)` |
| `reference.ops.op_equal.OpRunBinaryComparison` | Class | `(...)` |
| `reference.ops.op_erf.Erf` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_erf.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_exp.Exp` | Class | `(...)` |
| `reference.ops.op_exp.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_expand.Expand` | Class | `(...)` |
| `reference.ops.op_expand.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_expand.common_reference_implementation` | Function | `(data: np.ndarray, shape: np.ndarray) -> np.ndarray` |
| `reference.ops.op_eyelike.EyeLike` | Class | `(...)` |
| `reference.ops.op_eyelike.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_eyelike.TensorProto` | Object | `` |
| `reference.ops.op_eyelike.tensor_dtype_to_np_dtype` | Function | `(tensor_dtype: int) -> np.dtype` |
| `reference.ops.op_flatten.Flatten` | Class | `(...)` |
| `reference.ops.op_flatten.OpRunUnary` | Class | `(...)` |
| `reference.ops.op_floor.Floor` | Class | `(...)` |
| `reference.ops.op_floor.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_gather.Gather` | Class | `(...)` |
| `reference.ops.op_gather.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_gather_elements.GatherElements` | Class | `(...)` |
| `reference.ops.op_gather_elements.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_gather_elements.gather_numpy` | Function | `(self: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray` |
| `reference.ops.op_gather_elements.gather_numpy_2` | Function | `(self: np.ndarray, index: np.ndarray) -> np.ndarray` |
| `reference.ops.op_gathernd.GatherND` | Class | `(...)` |
| `reference.ops.op_gathernd.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_gemm.Gemm_6` | Class | `(...)` |
| `reference.ops.op_gemm.Gemm_7` | Class | `(...)` |
| `reference.ops.op_gemm.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_global_average_pool.GlobalAveragePool` | Class | `(...)` |
| `reference.ops.op_global_average_pool.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_global_max_pool.GlobalMaxPool` | Class | `(...)` |
| `reference.ops.op_global_max_pool.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_greater.Greater` | Class | `(...)` |
| `reference.ops.op_greater.OpRunBinaryComparison` | Class | `(...)` |
| `reference.ops.op_greater_or_equal.GreaterOrEqual` | Class | `(...)` |
| `reference.ops.op_greater_or_equal.OpRunBinaryComparison` | Class | `(...)` |
| `reference.ops.op_grid_sample.GridSample` | Class | `(...)` |
| `reference.ops.op_grid_sample.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_gru.CommonGRU` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_gru.GRU` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_gru.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_hamming_window.HammingWindow` | Class | `(...)` |
| `reference.ops.op_hann_window.HannWindow` | Class | `(...)` |
| `reference.ops.op_hard_sigmoid.HardSigmoid` | Class | `(...)` |
| `reference.ops.op_hard_sigmoid.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_hardmax.Hardmax` | Class | `(...)` |
| `reference.ops.op_hardmax.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_identity.Identity` | Class | `(...)` |
| `reference.ops.op_identity.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_if.If` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_if.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_image_decoder.ImageDecoder` | Class | `(...)` |
| `reference.ops.op_image_decoder.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_instance_normalization.InstanceNormalization` | Class | `(...)` |
| `reference.ops.op_instance_normalization.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_isinf.IsInf` | Class | `(...)` |
| `reference.ops.op_isinf.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_isnan.IsNaN` | Class | `(...)` |
| `reference.ops.op_isnan.OpRunUnary` | Class | `(...)` |
| `reference.ops.op_layer_normalization.LayerNormalization` | Class | `(...)` |
| `reference.ops.op_layer_normalization.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_leaky_relu.LeakyRelu` | Class | `(...)` |
| `reference.ops.op_leaky_relu.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_less.Less` | Class | `(...)` |
| `reference.ops.op_less.OpRunBinaryComparison` | Class | `(...)` |
| `reference.ops.op_less_or_equal.LessOrEqual` | Class | `(...)` |
| `reference.ops.op_less_or_equal.OpRunBinaryComparison` | Class | `(...)` |
| `reference.ops.op_log.Log` | Class | `(...)` |
| `reference.ops.op_log.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_log_softmax.LogSoftmax` | Class | `(...)` |
| `reference.ops.op_log_softmax.Softmax` | Class | `(...)` |
| `reference.ops.op_loop.Loop` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_loop.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_lp_normalization.LpNormalization` | Class | `(...)` |
| `reference.ops.op_lp_normalization.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_lp_pool.CommonPool` | Class | `(...)` |
| `reference.ops.op_lp_pool.LpPool` | Class | `(...)` |
| `reference.ops.op_lrn.LRN` | Class | `(...)` |
| `reference.ops.op_lrn.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_lstm.CommonLSTM` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_lstm.LSTM` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_lstm.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_matmul.MatMul` | Class | `(...)` |
| `reference.ops.op_matmul.OpRunBinaryNum` | Class | `(...)` |
| `reference.ops.op_matmul.numpy_matmul` | Function | `(a, b)` |
| `reference.ops.op_matmul_integer.MatMulInteger` | Class | `(...)` |
| `reference.ops.op_matmul_integer.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_max.Max` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_max.OpRunBinaryNumpy` | Class | `(numpy_fct: Any, onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_max_pool.CommonPool` | Class | `(...)` |
| `reference.ops.op_max_pool.MaxPool` | Class | `(...)` |
| `reference.ops.op_max_unpool.MaxUnpool` | Class | `(...)` |
| `reference.ops.op_max_unpool.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_mean.Mean` | Class | `(...)` |
| `reference.ops.op_mean.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_mel_weight_matrix.MelWeightMatrix` | Class | `(...)` |
| `reference.ops.op_mel_weight_matrix.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_mel_weight_matrix.tensor_dtype_to_np_dtype` | Function | `(tensor_dtype: int) -> np.dtype` |
| `reference.ops.op_min.Min` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_min.OpRunBinaryNumpy` | Class | `(numpy_fct: Any, onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_mod.Mod` | Class | `(...)` |
| `reference.ops.op_mod.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_mul.Mul` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_mul.OpRunBinaryNumpy` | Class | `(numpy_fct: Any, onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_neg.Neg` | Class | `(...)` |
| `reference.ops.op_neg.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_negative_log_likelihood_loss.NegativeLogLikelihoodLoss` | Class | `(...)` |
| `reference.ops.op_negative_log_likelihood_loss.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_non_max_suppression.BoxInfo` | Class | `(score: float, idx: int)` |
| `reference.ops.op_non_max_suppression.NonMaxSuppression` | Class | `(...)` |
| `reference.ops.op_non_max_suppression.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_non_max_suppression.PrepareContext` | Class | `(boxes_data_: np.ndarray \| None, boxes_size_: int, scores_data_: np.ndarray \| None, scores_size_: int, max_output_boxes_per_class_: np.ndarray \| None, score_threshold_: np.ndarray \| None, iou_threshold_: np.ndarray \| None, num_batches_: int, num_classes_: int, num_boxes_: int)` |
| `reference.ops.op_non_max_suppression.SelectedIndex` | Class | `(batch_index: int, class_index: int, box_index: int)` |
| `reference.ops.op_non_max_suppression.max_min` | Function | `(lhs: float, rhs: float) -> tuple[float, float]` |
| `reference.ops.op_non_max_suppression.suppress_by_iou` | Function | `(boxes_data: np.ndarray, box_index1: int, box_index2: int, center_point_box: int, iou_threshold: float) -> bool` |
| `reference.ops.op_non_zero.NonZero` | Class | `(...)` |
| `reference.ops.op_non_zero.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_not.Not` | Class | `(...)` |
| `reference.ops.op_not.OpRunUnary` | Class | `(...)` |
| `reference.ops.op_one_hot.OneHot` | Class | `(...)` |
| `reference.ops.op_one_hot.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_optional.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_optional.Optional` | Class | `(...)` |
| `reference.ops.op_optional.tensor_dtype_to_np_dtype` | Function | `(tensor_dtype: int) -> np.dtype` |
| `reference.ops.op_optional_get_element.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_optional_get_element.OptionalGetElement` | Class | `(...)` |
| `reference.ops.op_optional_has_element.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_optional_has_element.OptionalHasElement` | Class | `(...)` |
| `reference.ops.op_or.OpRunBinary` | Class | `(...)` |
| `reference.ops.op_or.Or` | Class | `(...)` |
| `reference.ops.op_pad.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_pad.Pad_1` | Class | `(...)` |
| `reference.ops.op_pad.Pad_11` | Class | `(...)` |
| `reference.ops.op_pad.Pad_18` | Class | `(...)` |
| `reference.ops.op_pad.Pad_2` | Class | `(...)` |
| `reference.ops.op_pool_common.CommonPool` | Class | `(...)` |
| `reference.ops.op_pool_common.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_pool_common.get_output_shape_auto_pad` | Function | `(auto_pad: str, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int]) -> Sequence[int]` |
| `reference.ops.op_pool_common.get_output_shape_explicit_padding` | Function | `(pads: Sequence[int] \| None, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int], dilations: Sequence[int] \| None, ceil_mode: bool) -> tuple[Sequence[int], Sequence[int]]` |
| `reference.ops.op_pool_common.get_pad_shape` | Function | `(auto_pad: str, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int], output_spatial_shape: Sequence[int]) -> Sequence[int]` |
| `reference.ops.op_pool_common.get_pad_with_auto_pad` | Function | `(auto_pad: str, pad_shape: Sequence[int]) -> Sequence[int]` |
| `reference.ops.op_pool_common.pool` | Function | `(padded: np.ndarray, x_shape: Sequence[int], kernel: Sequence[int], strides: Sequence[int], out_shape: Sequence[int], pooling_type: str, pads_required: Sequence[int] \| None, pads: Sequence[int] \| None, dilations: Sequence[int] \| None, count_include_pad: int, p: int) -> np.ndarray` |
| `reference.ops.op_pow.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_pow.Pow` | Class | `(...)` |
| `reference.ops.op_prelu.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_prelu.PRelu` | Class | `(...)` |
| `reference.ops.op_qlinear_conv.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_qlinear_conv.QLinearConv` | Class | `(...)` |
| `reference.ops.op_qlinear_matmul.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_qlinear_matmul.QLinearMatMul` | Class | `(...)` |
| `reference.ops.op_quantize_linear.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_quantize_linear.QuantizeLinear_10` | Class | `(...)` |
| `reference.ops.op_quantize_linear.QuantizeLinear_19` | Class | `(...)` |
| `reference.ops.op_quantize_linear.QuantizeLinear_21` | Class | `(...)` |
| `reference.ops.op_quantize_linear.QuantizeLinear_23` | Class | `(...)` |
| `reference.ops.op_quantize_linear.QuantizeLinear_25` | Class | `(...)` |
| `reference.ops.op_quantize_linear.TensorProto` | Object | `` |
| `reference.ops.op_quantize_linear.np_dtype_to_tensor_dtype` | Function | `(np_dtype: np.dtype) -> TensorProto.DataType` |
| `reference.ops.op_quantize_linear.tensor_dtype_to_np_dtype` | Function | `(tensor_dtype: int) -> np.dtype` |
| `reference.ops.op_random_normal.RandomNormal` | Class | `(...)` |
| `reference.ops.op_random_normal_like.RandomNormalLike` | Class | `(...)` |
| `reference.ops.op_random_normal_like.np_dtype_to_tensor_dtype` | Function | `(np_dtype: np.dtype) -> TensorProto.DataType` |
| `reference.ops.op_random_uniform.RandomUniform` | Class | `(...)` |
| `reference.ops.op_random_uniform_like.RandomUniformLike` | Class | `(...)` |
| `reference.ops.op_random_uniform_like.np_dtype_to_tensor_dtype` | Function | `(np_dtype: np.dtype) -> TensorProto.DataType` |
| `reference.ops.op_range.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_range.Range` | Class | `(...)` |
| `reference.ops.op_reciprocal.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_reciprocal.Reciprocal` | Class | `(...)` |
| `reference.ops.op_reduce_l1.OpRunReduceNumpy` | Class | `(onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_reduce_l1.ReduceL1_1` | Class | `(...)` |
| `reference.ops.op_reduce_l1.ReduceL1_18` | Class | `(...)` |
| `reference.ops.op_reduce_l2.OpRunReduceNumpy` | Class | `(onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_reduce_l2.ReduceL2_1` | Class | `(...)` |
| `reference.ops.op_reduce_l2.ReduceL2_18` | Class | `(...)` |
| `reference.ops.op_reduce_log_sum.OpRunReduceNumpy` | Class | `(onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_reduce_log_sum.ReduceLogSum_1` | Class | `(...)` |
| `reference.ops.op_reduce_log_sum.ReduceLogSum_18` | Class | `(...)` |
| `reference.ops.op_reduce_log_sum_exp.OpRunReduceNumpy` | Class | `(onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_reduce_log_sum_exp.ReduceLogSumExp_1` | Class | `(...)` |
| `reference.ops.op_reduce_log_sum_exp.ReduceLogSumExp_18` | Class | `(...)` |
| `reference.ops.op_reduce_log_sum_exp.compute_log_sum_exp` | Function | `(data, axes, keepdims)` |
| `reference.ops.op_reduce_max.OpRunReduceNumpy` | Class | `(onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_reduce_max.ReduceMax_1` | Class | `(...)` |
| `reference.ops.op_reduce_max.ReduceMax_18` | Class | `(...)` |
| `reference.ops.op_reduce_mean.OpRunReduceNumpy` | Class | `(onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_reduce_mean.ReduceMean_1` | Class | `(...)` |
| `reference.ops.op_reduce_mean.ReduceMean_18` | Class | `(...)` |
| `reference.ops.op_reduce_min.OpRunReduceNumpy` | Class | `(onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_reduce_min.ReduceMin_1` | Class | `(...)` |
| `reference.ops.op_reduce_min.ReduceMin_11` | Class | `(...)` |
| `reference.ops.op_reduce_min.ReduceMin_18` | Class | `(...)` |
| `reference.ops.op_reduce_prod.OpRunReduceNumpy` | Class | `(onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_reduce_prod.ReduceProd_1` | Class | `(...)` |
| `reference.ops.op_reduce_prod.ReduceProd_18` | Class | `(...)` |
| `reference.ops.op_reduce_sum.OpRunReduceNumpy` | Class | `(onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_reduce_sum.ReduceSum_1` | Class | `(...)` |
| `reference.ops.op_reduce_sum.ReduceSum_13` | Class | `(...)` |
| `reference.ops.op_reduce_sum_square.OpRunReduceNumpy` | Class | `(onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_reduce_sum_square.ReduceSumSquare_1` | Class | `(...)` |
| `reference.ops.op_reduce_sum_square.ReduceSumSquare_18` | Class | `(...)` |
| `reference.ops.op_regex_full_match.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_regex_full_match.RegexFullMatch` | Class | `(...)` |
| `reference.ops.op_relu.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_relu.Relu` | Class | `(...)` |
| `reference.ops.op_reshape.CommonReshape` | Class | `(...)` |
| `reference.ops.op_reshape.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_reshape.Reshape_14` | Class | `(...)` |
| `reference.ops.op_reshape.Reshape_5` | Class | `(...)` |
| `reference.ops.op_reshape.reshape_reference_implementation` | Function | `(data: np.ndarray, shape: np.ndarray, allowzero: int) -> np.ndarray` |
| `reference.ops.op_resize.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_resize.Resize` | Class | `(...)` |
| `reference.ops.op_reverse_sequence.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_reverse_sequence.ReverseSequence` | Class | `(...)` |
| `reference.ops.op_rms_normalization.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_rms_normalization.RMSNormalization` | Class | `(...)` |
| `reference.ops.op_rnn.CommonRNN` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_rnn.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_rnn.RNN_14` | Class | `(...)` |
| `reference.ops.op_rnn.RNN_7` | Class | `(...)` |
| `reference.ops.op_roi_align.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_roi_align.PreCalc` | Class | `(pos1, pos2, pos3, pos4, w1, w2, w3, w4)` |
| `reference.ops.op_roi_align.RoiAlign` | Class | `(...)` |
| `reference.ops.op_rotary_embedding.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_rotary_embedding.RotaryEmbedding` | Class | `(...)` |
| `reference.ops.op_rotary_embedding.rotary_embedding` | Function | `(input: np.ndarray, cos_cache: np.ndarray, sin_cache: np.ndarray, position_ids: np.ndarray \| None, interleaved, rotary_embedding_dim, num_heads) -> np.ndarray` |
| `reference.ops.op_round.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_round.Round` | Class | `(...)` |
| `reference.ops.op_scan.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_scan.Scan` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_scatter_elements.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_scatter_elements.ScatterElements` | Class | `(...)` |
| `reference.ops.op_scatter_elements.scatter_elements` | Function | `(data, indices, updates, axis, reduction)` |
| `reference.ops.op_scatternd.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_scatternd.ScatterND` | Class | `(...)` |
| `reference.ops.op_selu.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_selu.Selu` | Class | `(...)` |
| `reference.ops.op_sequence_at.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_sequence_at.SequenceAt` | Class | `(...)` |
| `reference.ops.op_sequence_construct.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_sequence_construct.SequenceConstruct` | Class | `(...)` |
| `reference.ops.op_sequence_empty.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_sequence_empty.SequenceEmpty` | Class | `(...)` |
| `reference.ops.op_sequence_erase.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_sequence_erase.SequenceErase` | Class | `(...)` |
| `reference.ops.op_sequence_insert.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_sequence_insert.SequenceInsert` | Class | `(...)` |
| `reference.ops.op_sequence_insert.sequence_insert_reference_implementation` | Function | `(sequence: list[Any] \| np.ndarray, tensor: np.ndarray, position: np.ndarray \| None) -> list[Any]` |
| `reference.ops.op_sequence_length.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_sequence_length.SequenceLength` | Class | `(...)` |
| `reference.ops.op_sequence_map.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_sequence_map.SequenceMap` | Class | `(...)` |
| `reference.ops.op_shape.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_shape.Shape_1` | Class | `(...)` |
| `reference.ops.op_shape.Shape_15` | Class | `(...)` |
| `reference.ops.op_shrink.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_shrink.Shrink` | Class | `(...)` |
| `reference.ops.op_sigmoid.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_sigmoid.Sigmoid` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_sigmoid.sigmoid` | Function | `(x)` |
| `reference.ops.op_sign.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_sign.Sign` | Class | `(...)` |
| `reference.ops.op_sin.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_sin.Sin` | Class | `(...)` |
| `reference.ops.op_sinh.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_sinh.Sinh` | Class | `(...)` |
| `reference.ops.op_size.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_size.Size` | Class | `(...)` |
| `reference.ops.op_slice.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_slice.SliceCommon` | Class | `(...)` |
| `reference.ops.op_slice.Slice_1` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_slice.Slice_10` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_softmax.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_softmax.Softmax` | Class | `(...)` |
| `reference.ops.op_softmax_cross_entropy_loss.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_softmax_cross_entropy_loss.SoftmaxCrossEntropyLoss` | Class | `(...)` |
| `reference.ops.op_softmax_cross_entropy_loss.softmaxcrossentropy` | Function | `(x, target, weight, reduction, ignore_index, get_log_prob)` |
| `reference.ops.op_softplus.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_softplus.Softplus` | Class | `(...)` |
| `reference.ops.op_softsign.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_softsign.Softsign` | Class | `(...)` |
| `reference.ops.op_space_to_depth.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_space_to_depth.SpaceToDepth` | Class | `(...)` |
| `reference.ops.op_split.CommonSplit` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_split.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_split.Split_11` | Class | `(...)` |
| `reference.ops.op_split.Split_13` | Class | `(...)` |
| `reference.ops.op_split.Split_18` | Class | `(...)` |
| `reference.ops.op_split.Split_2` | Class | `(...)` |
| `reference.ops.op_split_to_sequence.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_split_to_sequence.SplitToSequence` | Class | `(...)` |
| `reference.ops.op_sqrt.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_sqrt.Sqrt` | Class | `(...)` |
| `reference.ops.op_squeeze.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_squeeze.Squeeze_1` | Class | `(...)` |
| `reference.ops.op_squeeze.Squeeze_11` | Class | `(...)` |
| `reference.ops.op_squeeze.Squeeze_13` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_stft.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_stft.STFT` | Class | `(...)` |
| `reference.ops.op_string_concat.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_string_concat.StringConcat` | Class | `(...)` |
| `reference.ops.op_string_normalizer.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_string_normalizer.RuntimeTypeError` | Class | `(...)` |
| `reference.ops.op_string_normalizer.StringNormalizer` | Class | `(...)` |
| `reference.ops.op_string_split.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_string_split.StringSplit` | Class | `(...)` |
| `reference.ops.op_string_split.pad_empty_string` | Function | `(split_lists: list \| np.ndarray, padding_requirement: list \| int) -> list` |
| `reference.ops.op_string_split.split_with_padding` | Function | `(x, separator, maxsplit)` |
| `reference.ops.op_sub.OpRunBinaryNumpy` | Class | `(numpy_fct: Any, onnx_node: NodeProto, run_params: dict[str, Any])` |
| `reference.ops.op_sub.Sub` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_sum.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_sum.Sum` | Class | `(...)` |
| `reference.ops.op_swish.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_swish.Swish` | Class | `(...)` |
| `reference.ops.op_tan.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_tan.Tan` | Class | `(...)` |
| `reference.ops.op_tanh.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_tanh.Tanh` | Class | `(...)` |
| `reference.ops.op_tensor_scatter.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_tensor_scatter.TensorScatter` | Class | `(...)` |
| `reference.ops.op_tfidf_vectorizer.IntMap` | Class | `()` |
| `reference.ops.op_tfidf_vectorizer.NgramPart` | Class | `(nid: int)` |
| `reference.ops.op_tfidf_vectorizer.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_tfidf_vectorizer.TfIdfVectorizer` | Class | `(onnx_node, run_params)` |
| `reference.ops.op_tfidf_vectorizer.WeightingCriteria` | Class | `(...)` |
| `reference.ops.op_tfidf_vectorizer.populate_grams` | Function | `(els, els_index, n_ngrams: int, ngram_size: int, ngram_id: int, c)` |
| `reference.ops.op_thresholded_relu.OpRunUnaryNum` | Class | `(...)` |
| `reference.ops.op_thresholded_relu.ThresholdedRelu` | Class | `(...)` |
| `reference.ops.op_tile.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_tile.Tile` | Class | `(...)` |
| `reference.ops.op_topk.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_topk.TopK_1` | Class | `(...)` |
| `reference.ops.op_topk.TopK_10` | Class | `(...)` |
| `reference.ops.op_topk.TopK_11` | Class | `(...)` |
| `reference.ops.op_topk.topk_sorted_implementation` | Function | `(X, k, axis, largest)` |
| `reference.ops.op_transpose.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_transpose.Transpose` | Class | `(...)` |
| `reference.ops.op_trilu.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_trilu.Trilu` | Class | `(...)` |
| `reference.ops.op_unique.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_unique.Unique` | Class | `(...)` |
| `reference.ops.op_unsqueeze.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_unsqueeze.Unsqueeze_1` | Class | `(...)` |
| `reference.ops.op_unsqueeze.Unsqueeze_11` | Class | `(...)` |
| `reference.ops.op_unsqueeze.Unsqueeze_13` | Class | `(...)` |
| `reference.ops.op_upsample.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_upsample.Upsample` | Class | `(...)` |
| `reference.ops.op_where.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops.op_where.Where` | Class | `(...)` |
| `reference.ops.op_xor.OpRunBinary` | Class | `(...)` |
| `reference.ops.op_xor.Xor` | Class | `(...)` |
| `reference.ops_optimized.Conv` | Class | `(...)` |
| `reference.ops_optimized.op_conv_optimized.Conv` | Class | `(...)` |
| `reference.ops_optimized.op_conv_optimized.OpRun` | Class | `(onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any)` |
| `reference.ops_optimized.op_conv_optimized.im2col_fast` | Function | `(X, kernel_shape, pads, strides)` |
| `reference.ops_optimized.optimized_operators` | Object | `` |
| `reference.reference_evaluator.FunctionProto` | Object | `` |
| `reference.reference_evaluator.GraphProto` | Object | `` |
| `reference.reference_evaluator.ModelProto` | Object | `` |
| `reference.reference_evaluator.NodeProto` | Object | `` |
| `reference.reference_evaluator.ReferenceEvaluator` | Class | `(proto: Any, opsets: dict[str, int] \| None, functions: list[ReferenceEvaluator \| FunctionProto] \| None, verbose: int, new_ops: list[type[op_run.OpRun]] \| None, optimized: bool)` |
| `reference.reference_evaluator.TensorProto` | Object | `` |
| `reference.reference_evaluator.TypeProto` | Object | `` |
| `reference.reference_evaluator.op_run` | Object | `` |
| `reference.reference_evaluator.optimized_operators` | Object | `` |
| `save` | Object | `` |
| `save_model` | Function | `(proto: ModelProto \| bytes, f: IO[bytes] \| str \| os.PathLike, format: _SupportedFormat \| None, save_as_external_data: bool, all_tensors_to_one_file: bool, location: str \| None, size_threshold: int, convert_attribute: bool) -> None` |
| `save_tensor` | Function | `(proto: TensorProto, f: IO[bytes] \| str \| os.PathLike, format: _SupportedFormat \| None) -> None` |
| `serialization.ProtoSerializer` | Class | `(...)` |
| `serialization.registry` | Object | `` |
| `shape_inference.AttributeProto` | Object | `` |
| `shape_inference.C` | Object | `` |
| `shape_inference.FunctionProto` | Object | `` |
| `shape_inference.GraphInferencer` | Object | `` |
| `shape_inference.IR_VERSION` | Object | `` |
| `shape_inference.InferenceContext` | Object | `` |
| `shape_inference.InferenceError` | Object | `` |
| `shape_inference.ModelProto` | Object | `` |
| `shape_inference.TypeProto` | Object | `` |
| `shape_inference.infer_function_output_types` | Function | `(function: FunctionProto, input_types: Sequence[TypeProto], attributes: Sequence[AttributeProto]) -> list[TypeProto]` |
| `shape_inference.infer_node_outputs` | Function | `(schema: onnx.defs.OpSchema, node: onnx.NodeProto, input_types: dict[str, onnx.TypeProto], input_data: dict[str, onnx.TensorProto] \| None, input_sparse_data: dict[str, onnx.SparseTensorProto] \| None, opset_imports: list[onnx.OperatorSetIdProto] \| None, ir_version: int) -> dict[str, onnx.TypeProto]` |
| `shape_inference.infer_shapes` | Function | `(model: ModelProto \| bytes, check_type: bool, strict_mode: bool, data_prop: bool) -> ModelProto` |
| `shape_inference.infer_shapes_path` | Function | `(model_path: str \| os.PathLike, output_path: str \| os.PathLike, check_type: bool, strict_mode: bool, data_prop: bool) -> None` |
| `tools.net_drawer.BLOB_STYLE` | Object | `` |
| `tools.net_drawer.GetOpNodeProducer` | Function | `(embed_docstring: bool, kwargs: Any) -> _NodeProducer` |
| `tools.net_drawer.GetPydotGraph` | Function | `(graph: GraphProto, name: str \| None, rankdir: str, node_producer: _NodeProducer \| None, embed_docstring: bool) -> pydot.Dot` |
| `tools.net_drawer.GraphProto` | Object | `` |
| `tools.net_drawer.ModelProto` | Object | `` |
| `tools.net_drawer.NodeProto` | Object | `` |
| `tools.net_drawer.OP_STYLE` | Object | `` |
| `tools.net_drawer.main` | Function | `() -> None` |
| `tools.replace_constants.AttributeProto` | Object | `` |
| `tools.replace_constants.FunctionProto` | Object | `` |
| `tools.replace_constants.GraphProto` | Object | `` |
| `tools.replace_constants.ModelProto` | Object | `` |
| `tools.replace_constants.NodeProto` | Object | `` |
| `tools.replace_constants.SparseTensorProto` | Object | `` |
| `tools.replace_constants.TensorProto` | Object | `` |
| `tools.replace_constants.from_array` | Function | `(array: np.ndarray, name: str \| None) -> onnx.TensorProto` |
| `tools.replace_constants.make_attribute` | Function | `(key: str, value: Any, doc_string: str \| None, attr_type: int \| None) -> AttributeProto` |
| `tools.replace_constants.make_function` | Function | `(domain: str, fname: str, inputs: Sequence[str], outputs: Sequence[str], nodes: Sequence[NodeProto], opset_imports: Sequence[OperatorSetIdProto], attributes: Sequence[str] \| None, attribute_protos: Sequence[AttributeProto] \| None, doc_string: str \| None, overload: str \| None, value_info: Sequence[ValueInfoProto] \| None) -> FunctionProto` |
| `tools.replace_constants.make_graph` | Function | `(nodes: Sequence[NodeProto], name: str, inputs: Sequence[ValueInfoProto], outputs: Sequence[ValueInfoProto], initializer: Sequence[TensorProto] \| None, doc_string: str \| None, value_info: Sequence[ValueInfoProto] \| None, sparse_initializer: Sequence[onnx.SparseTensorProto] \| None) -> GraphProto` |
| `tools.replace_constants.make_model` | Function | `(graph: GraphProto, kwargs: Any) -> ModelProto` |
| `tools.replace_constants.make_node` | Function | `(op_type: str, inputs: Sequence[str], outputs: Sequence[str], name: str \| None, doc_string: str \| None, domain: str \| None, overload: str \| None, kwargs: Any) -> NodeProto` |
| `tools.replace_constants.make_tensor` | Function | `(name: str, data_type: int, dims: Sequence[int], vals: Sequence[int \| float] \| bytes \| np.ndarray, raw: bool) -> TensorProto` |
| `tools.replace_constants.make_tensor_value_info` | Function | `(name: str, elem_type: int, shape: Sequence[str \| int \| None] \| None, doc_string: str, shape_denotation: list[str] \| None) -> ValueInfoProto` |
| `tools.replace_constants.replace_initializer_by_constant_of_shape` | Function | `(onx: FunctionProto \| GraphProto \| ModelProto, threshold: int, ir_version: int \| None, use_range: bool, value_constant_of_shape: float)` |
| `tools.replace_constants.set_model_props` | Function | `(model: ModelProto, dict_value: dict[str, str]) -> None` |
| `tools.replace_constants.tensor_dtype_to_np_dtype` | Function | `(tensor_dtype: int) -> np.dtype` |
| `tools.update_model_dims.ModelProto` | Object | `` |
| `tools.update_model_dims.ValueInfoProto` | Object | `` |
| `tools.update_model_dims.update_inputs_outputs_dims` | Function | `(model: ModelProto, input_dims: dict[str, list[Any]], output_dims: dict[str, list[Any]]) -> ModelProto` |
| `utils.Extractor` | Class | `(model: ModelProto)` |
| `utils.FunctionProto` | Object | `` |
| `utils.ModelProto` | Object | `` |
| `utils.NodeProto` | Object | `` |
| `utils.TensorProto` | Object | `` |
| `utils.ValueInfoProto` | Object | `` |
| `utils.extract_model` | Function | `(input_path: str \| os.PathLike, output_path: str \| os.PathLike, input_names: list[str], output_names: list[str], check_model: bool, infer_shapes: bool) -> None` |
| `version.git_version` | Object | `` |
| `version.version` | Object | `` |
| `version_converter.C` | Object | `` |
| `version_converter.ConvertError` | Object | `` |
| `version_converter.ModelProto` | Object | `` |
| `version_converter.convert_version` | Function | `(model: ModelProto, target_version: int) -> ModelProto` |
| `write_external_data_tensors` | Function | `(model: ModelProto, filepath: str) -> ModelProto` |

## Detailed Operators

| ONNX Operator | ONNX9000 |
|---|---|
| Abs | ✅ |
| Acos | ✅ |
| Acosh | ✅ |
| Add | ✅ |
| AffineGrid | ✅ |
| And | ✅ |
| ArgMax | ✅ |
| ArgMin | ✅ |
| Asin | ✅ |
| Asinh | ✅ |
| Atan | ✅ |
| Atanh | ✅ |
| Attention | ✅ |
| AveragePool | ✅ |
| BatchNormalization | ❌ |
| Bernoulli | ✅ |
| BitCast | ❌ |
| BitShift | ✅ |
| BitwiseAnd | ✅ |
| BitwiseNot | ✅ |
| BitwiseOr | ✅ |
| BitwiseXor | ✅ |
| BlackmanWindow | ✅ |
| Cast | ✅ |
| CastLike | ✅ |
| Ceil | ✅ |
| Celu | ✅ |
| CenterCropPad | ✅ |
| Clip | ✅ |
| Col2Im | ✅ |
| Compress | ✅ |
| Concat | ✅ |
| ConcatFromSequence | ✅ |
| Constant | ✅ |
| ConstantOfShape | ✅ |
| Conv | ✅ |
| ConvInteger | ✅ |
| ConvTranspose | ✅ |
| Cos | ✅ |
| Cosh | ✅ |
| CumProd | ❌ |
| CumSum | ✅ |
| DFT | ✅ |
| DeformConv | ✅ |
| DepthToSpace | ✅ |
| DequantizeLinear | ✅ |
| Det | ✅ |
| Div | ✅ |
| Dropout | ✅ |
| DynamicQuantizeLinear | ✅ |
| Einsum | ✅ |
| Elu | ✅ |
| Equal | ✅ |
| Erf | ✅ |
| Exp | ✅ |
| Expand | ✅ |
| EyeLike | ✅ |
| Flatten | ❌ |
| Floor | ✅ |
| GRU | ✅ |
| Gather | ✅ |
| GatherElements | ✅ |
| GatherND | ✅ |
| Gelu | ❌ |
| Gemm | ✅ |
| GlobalAveragePool | ✅ |
| GlobalLpPool | ❌ |
| GlobalMaxPool | ✅ |
| Greater | ✅ |
| GreaterOrEqual | ✅ |
| GridSample | ❌ |
| GroupNormalization | ✅ |
| HammingWindow | ❌ |
| HannWindow | ❌ |
| HardSigmoid | ✅ |
| HardSwish | ✅ |
| Hardmax | ✅ |
| Identity | ❌ |
| If | ❌ |
| ImageDecoder | ❌ |
| InstanceNormalization | ✅ |
| IsInf | ✅ |
| IsNaN | ✅ |
| LRN | ✅ |
| LSTM | ✅ |
| LayerNormalization | ✅ |
| LeakyRelu | ✅ |
| Less | ✅ |
| LessOrEqual | ✅ |
| Log | ❌ |
| LogSoftmax | ✅ |
| Loop | ❌ |
| LpNormalization | ✅ |
| LpPool | ✅ |
| MatMul | ✅ |
| MatMulInteger | ❌ |
| Max | ✅ |
| MaxPool | ✅ |
| MaxRoiPool | ❌ |
| MaxUnpool | ❌ |
| Mean | ✅ |
| MeanVarianceNormalization | ✅ |
| MelWeightMatrix | ✅ |
| Min | ✅ |
| Mish | ✅ |
| Mod | ✅ |
| Mul | ✅ |
| Multinomial | ✅ |
| Neg | ✅ |
| NegativeLogLikelihoodLoss | ❌ |
| NonMaxSuppression | ✅ |
| NonZero | ✅ |
| Not | ✅ |
| OneHot | ✅ |
| Optional | ❌ |
| OptionalGetElement | ❌ |
| OptionalHasElement | ❌ |
| Or | ✅ |
| PRelu | ✅ |
| Pad | ❌ |
| Pow | ✅ |
| QLinearConv | ❌ |
| QLinearMatMul | ❌ |
| QuantizeLinear | ✅ |
| RMSNormalization | ❌ |
| RNN | ✅ |
| RandomNormal | ✅ |
| RandomNormalLike | ✅ |
| RandomUniform | ✅ |
| RandomUniformLike | ✅ |
| Range | ❌ |
| Reciprocal | ✅ |
| ReduceL1 | ✅ |
| ReduceL2 | ✅ |
| ReduceLogSum | ✅ |
| ReduceLogSumExp | ✅ |
| ReduceMax | ✅ |
| ReduceMean | ✅ |
| ReduceMin | ✅ |
| ReduceProd | ✅ |
| ReduceSum | ✅ |
| ReduceSumSquare | ✅ |
| RegexFullMatch | ✅ |
| Relu | ✅ |
| Reshape | ✅ |
| Resize | ✅ |
| ReverseSequence | ✅ |
| RoiAlign | ❌ |
| RotaryEmbedding | ❌ |
| Round | ✅ |
| STFT | ❌ |
| Scan | ❌ |
| Scatter | ✅ |
| ScatterElements | ✅ |
| ScatterND | ✅ |
| Selu | ✅ |
| SequenceAt | ✅ |
| SequenceConstruct | ✅ |
| SequenceEmpty | ✅ |
| SequenceErase | ✅ |
| SequenceInsert | ✅ |
| SequenceLength | ✅ |
| SequenceMap | ✅ |
| Shape | ❌ |
| Shrink | ✅ |
| Sigmoid | ✅ |
| Sign | ✅ |
| Sin | ✅ |
| Sinh | ✅ |
| Size | ✅ |
| Slice | ✅ |
| Softmax | ✅ |
| SoftmaxCrossEntropyLoss | ❌ |
| Softplus | ✅ |
| Softsign | ✅ |
| SpaceToDepth | ✅ |
| Split | ❌ |
| SplitToSequence | ✅ |
| Sqrt | ❌ |
| Squeeze | ❌ |
| StringConcat | ✅ |
| StringNormalizer | ✅ |
| StringSplit | ✅ |
| Sub | ✅ |
| Sum | ✅ |
| Swish | ✅ |
| Tan | ✅ |
| Tanh | ✅ |
| TensorScatter | ❌ |
| TfIdfVectorizer | ❌ |
| ThresholdedRelu | ✅ |
| Tile | ✅ |
| TopK | ✅ |
| Transpose | ✅ |
| Trilu | ✅ |
| Unique | ✅ |
| Unsqueeze | ❌ |
| Upsample | ❌ |
| Where | ✅ |
| Xor | ❌ |