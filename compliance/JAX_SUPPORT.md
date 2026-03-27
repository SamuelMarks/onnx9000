# Jax Support Coverage

Tracking exhaustive coverage of the `jax` python package API.


## Detailed API

| Object Name | Type | Signature |
|---|---|---|
| `Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `Device` | Object | `` |
| `NamedSharding` | Class | `(mesh: mesh_lib.Mesh \| mesh_lib.AbstractMesh, spec: PartitionSpec, memory_kind: str \| None, _logical_device_ids)` |
| `P` | Class | `(partitions, unreduced, reduced)` |
| `Ref` | Class | `(aval, refs)` |
| `ShapeDtypeStruct` | Class | `(shape, dtype, sharding, weak_type, vma, is_ref)` |
| `Shard` | Class | `(device: Device, sharding: Sharding, global_shape: Shape, data: None \| ArrayImpl \| PRNGKeyArray)` |
| `ad` | Object | `` |
| `ad_checkpoint.Offloadable` | Class | `(...)` |
| `ad_checkpoint.Recompute` | Object | `` |
| `ad_checkpoint.Saveable` | Object | `` |
| `ad_checkpoint.checkpoint` | Object | `` |
| `ad_checkpoint.checkpoint_name` | Function | `(x, name)` |
| `ad_checkpoint.checkpoint_policies` | Object | `` |
| `ad_checkpoint.print_saved_residuals` | Function | `(f, args, kwargs)` |
| `ad_checkpoint.remat` | Object | `` |
| `allow_f16_reductions` | Object | `` |
| `api_util.argnums_partial` | Function | `(f: lu.WrappedFun, dyn_argnums: int \| Sequence[int], args: Sequence, require_static_args_hashable)` |
| `api_util.debug_info` | Function | `(traced_for: str, fun: Callable, args: Sequence[Any], kwargs: dict[str, Any], static_argnums: Sequence[int], static_argnames: Sequence[str], result_paths_thunk: Callable[[], tuple[str, ...]] \| core.InitialResultPaths, sourceinfo: str \| None, signature: inspect.Signature \| None) -> core.DebugInfo` |
| `api_util.donation_vector` | Function | `(donate_argnums, donate_argnames, in_tree, kws: bool) -> tuple[bool, ...]` |
| `api_util.flatten_axes` | Function | `(name, treedef, axis_tree, kws, tupled_args)` |
| `api_util.flatten_fun` | Function | `(f: Callable, store: lu.Store, in_tree: PyTreeDef, args_flat)` |
| `api_util.flatten_fun_nokwargs` | Function | `(f: Callable, store: lu.Store, in_tree: PyTreeDef, args_flat)` |
| `api_util.rebase_donate_argnums` | Function | `(donate_argnums, static_argnums) -> tuple[int, ...]` |
| `api_util.safe_map` | Function | `(f, args)` |
| `api_util.shaped_abstractify` | Function | `(x)` |
| `array_garbage_collection_guard` | Object | `` |
| `batching` | Object | `` |
| `block_until_ready` | Function | `(x)` |
| `check_tracer_leaks` | Object | `` |
| `checking_leaks` | Object | `` |
| `checkpoint` | Function | `(fun: Callable, prevent_cse: bool \| Sequence[bool], policy: Callable[..., bool] \| None, static_argnums: int \| tuple[int, ...], concrete: bool \| DeprecatedArg) -> Callable` |
| `checkpoint_policies` | Object | `` |
| `clear_caches` | Function | `()` |
| `closure_convert` | Function | `(fun: Callable, example_args) -> tuple[Callable, list[Any]]` |
| `cloud_tpu_init.cloud_tpu_init` | Object | `` |
| `collect_profile.DEFAULT_NUM_TRACING_ATTEMPTS` | Object | `` |
| `collect_profile.collect_profile` | Function | `(port: int, duration_in_ms: int, host: str, log_dir: os.PathLike \| str \| None, no_perfetto_link: bool, xprof_options: dict[str, Any] \| None)` |
| `collect_profile.jax_profiler` | Object | `` |
| `collect_profile.main` | Function | `(known_args, unknown_flags)` |
| `collect_profile.parser` | Object | `` |
| `config` | Object | `` |
| `copy_to_host_async` | Function | `(x)` |
| `core.AbstractToken` | Object | `` |
| `core.AbstractValue` | Class | `(...)` |
| `core.Atom` | Object | `` |
| `core.CallPrimitive` | Class | `(...)` |
| `core.DebugInfo` | Object | `` |
| `core.DropVar` | Class | `(aval: AbstractValue)` |
| `core.Effect` | Object | `` |
| `core.Effects` | Object | `` |
| `core.InconclusiveDimensionOperation` | Class | `(...)` |
| `core.JaxprPpContext` | Class | `()` |
| `core.JaxprPpSettings` | Class | `(...)` |
| `core.JaxprTypeError` | Class | `(...)` |
| `core.OutputType` | Object | `` |
| `core.ParamDict` | Object | `` |
| `core.ShapedArray` | Class | `(shape, dtype, weak_type, sharding, vma: frozenset[AxisName], memory_space: MemorySpace)` |
| `core.Trace` | Class | `()` |
| `core.TraceTag` | Object | `` |
| `core.Tracer` | Class | `(trace: TraceType)` |
| `core.Value` | Object | `` |
| `core.abstract_token` | Object | `` |
| `core.aval_mapping_handlers` | Object | `` |
| `core.call` | Object | `` |
| `core.call_impl` | Object | `` |
| `core.check_jaxpr` | Function | `(jaxpr: Jaxpr)` |
| `core.concrete_or_error` | Function | `(force: Any, val: Any, context)` |
| `core.concretization_function_error` | Function | `(fun, suggest_astype)` |
| `core.custom_typechecks` | Object | `` |
| `core.ensure_compile_time_eval` | Function | `()` |
| `core.eval_context` | Function | `()` |
| `core.eval_jaxpr` | Function | `(jaxpr: Jaxpr, consts, args, propagate_source_info) -> list[Any]` |
| `core.find_top_trace` | Function | `(_)` |
| `core.gensym` | Object | `` |
| `core.get_aval` | Object | `` |
| `core.get_opaque_trace_state` | Function | `(convention)` |
| `core.is_concrete` | Function | `(x)` |
| `core.is_constant_dim` | Function | `(d: DimSize) -> bool` |
| `core.is_constant_shape` | Function | `(s: Shape) -> bool` |
| `core.jaxprs_in_params` | Function | `(params) -> Iterator[Jaxpr]` |
| `core.literalable_types` | Object | `` |
| `core.mapped_aval` | Object | `` |
| `core.max_dim` | Function | `(d1: DimSize, d2: DimSize) -> DimSize` |
| `core.min_dim` | Function | `(d1: DimSize, d2: DimSize) -> DimSize` |
| `core.new_jaxpr_eqn` | Function | `(invars, outvars, primitive, params, effects, source_info, ctx) -> JaxprEqn` |
| `core.no_axis_name` | Object | `` |
| `core.no_effects` | Object | `` |
| `core.nonempty_axis_env_DO_NOT_USE` | Function | `() -> bool` |
| `core.primal_dtype_to_tangent_dtype` | Function | `(primal_dtype)` |
| `core.pytype_aval_mappings` | Object | `` |
| `core.set_current_trace` | Object | `` |
| `core.subjaxprs` | Object | `` |
| `core.take_current_trace` | Object | `` |
| `core.trace_ctx` | Object | `` |
| `core.traverse_jaxpr_params` | Object | `` |
| `core.unmapped_aval` | Object | `` |
| `core.unsafe_am_i_under_a_jit_DO_NOT_USE` | Function | `() -> bool` |
| `core.unsafe_am_i_under_a_vmap_DO_NOT_USE` | Function | `() -> bool` |
| `core.unsafe_get_axis_names_DO_NOT_USE` | Function | `() -> list[Any]` |
| `core.valid_jaxtype` | Function | `(x) -> bool` |
| `custom_batching.custom_vmap` | Class | `(fun: Callable[..., Any])` |
| `custom_batching.sequential_vmap` | Function | `(f)` |
| `custom_derivatives.CustomVJPPrimal` | Class | `(value: Any, perturbed: bool)` |
| `custom_derivatives.SymbolicZero` | Class | `(aval: core.AbstractValue)` |
| `custom_derivatives.closure_convert` | Function | `(fun: Callable, example_args) -> tuple[Callable, list[Any]]` |
| `custom_derivatives.custom_gradient` | Function | `(fun)` |
| `custom_derivatives.custom_jvp` | Class | `(fun: Callable[..., ReturnValue], nondiff_argnums: Sequence[int], nondiff_argnames: Sequence[str])` |
| `custom_derivatives.custom_jvp_call_p` | Object | `` |
| `custom_derivatives.custom_vjp` | Class | `(fun: Callable[..., ReturnValue], nondiff_argnums: Sequence[int], nondiff_argnames: Sequence[str])` |
| `custom_derivatives.custom_vjp_call_p` | Object | `` |
| `custom_derivatives.custom_vjp_primal_tree_values` | Function | `(tree)` |
| `custom_derivatives.linear_call` | Function | `(fun: Callable, fun_transpose: Callable, residual_args, linear_args)` |
| `custom_derivatives.remat_opt_p` | Object | `` |
| `custom_derivatives.zero_from_primal` | Function | `(val, symbolic_zeros)` |
| `custom_gradient` | Function | `(fun)` |
| `custom_jvp` | Class | `(fun: Callable[..., ReturnValue], nondiff_argnums: Sequence[int], nondiff_argnames: Sequence[str])` |
| `custom_transpose.custom_transpose` | Class | `(fun: Callable)` |
| `custom_vjp` | Class | `(fun: Callable[..., ReturnValue], nondiff_argnums: Sequence[int], nondiff_argnames: Sequence[str])` |
| `debug.DebugEffect` | Class | `(...)` |
| `debug.OrderedDebugEffect` | Class | `(...)` |
| `debug.breakpoint` | Function | `(backend: str \| None, filter_frames: bool, num_frames: int \| None, ordered: bool, token, kwargs)` |
| `debug.callback` | Function | `(callback: Callable[..., None], args: Any, ordered: bool, partitioned: bool, kwargs: Any) -> None` |
| `debug.inspect_array_sharding` | Function | `(value, callback: Callable[[Sharding], None])` |
| `debug.log` | Object | `` |
| `debug.print` | Function | `(fmt: str, args, ordered: bool, partitioned: bool, skip_format_check: bool, _use_logging: bool, kwargs) -> None` |
| `debug.visualize_array_sharding` | Function | `(arr, kwargs)` |
| `debug.visualize_sharding` | Function | `(shape: Sequence[int], sharding: Sharding, use_color: bool, scale: float, min_width: int, max_width: int, color_map: ColorMap \| None)` |
| `debug_infs` | Object | `` |
| `debug_key_reuse` | Object | `` |
| `debug_nans` | Object | `` |
| `default_backend` | Function | `() -> str` |
| `default_device` | Object | `` |
| `default_matmul_precision` | Object | `` |
| `default_prng_impl` | Object | `` |
| `device_count` | Function | `(backend: str \| xla_client.Client \| None) -> int` |
| `device_get` | Function | `(x: Any)` |
| `device_put` | Function | `(x, device: None \| xc.Device \| Sharding \| P \| Format \| Any, src: None \| xc.Device \| Sharding \| P \| Format \| Any, donate: bool \| Any, may_alias: bool \| None \| Any)` |
| `device_put_replicated` | Object | `` |
| `device_put_sharded` | Object | `` |
| `devices` | Function | `(backend: str \| xla_client.Client \| None) -> list[xla_client.Device]` |
| `disable_jit` | Function | `(disable: bool)` |
| `distributed.initialize` | Function | `(coordinator_address: str \| None, num_processes: int \| None, process_id: int \| None, local_device_ids: int \| Sequence[int] \| None, cluster_detection_method: str \| None, initialization_timeout: int, heartbeat_timeout_seconds: int, shutdown_timeout_seconds: int, coordinator_bind_address: str \| None, slice_index: int \| None, partition_index: int \| None)` |
| `distributed.is_initialized` | Function | `() -> bool` |
| `distributed.shutdown` | Function | `()` |
| `dlpack.from_dlpack` | Function | `(external_array, device: _jax.Device \| Sharding \| None, copy: bool \| None)` |
| `dlpack.is_supported_dtype` | Function | `(dtype: DTypeLike) -> bool` |
| `ds` | Object | `` |
| `dtypes.bfloat16` | Object | `` |
| `dtypes.canonicalize_dtype` | Function | `(dtype: Any, allow_extended_dtype: bool) -> DType \| ExtendedDType` |
| `dtypes.extended` | Class | `(...)` |
| `dtypes.finfo` | Object | `` |
| `dtypes.float0` | Object | `` |
| `dtypes.iinfo` | Object | `` |
| `dtypes.issubdtype` | Function | `(a: DTypeLike \| ExtendedDType \| None, b: DTypeLike \| ExtendedDType \| None) -> bool` |
| `dtypes.itemsize_bits` | Function | `(dtype: DTypeLike) -> int` |
| `dtypes.prng_key` | Class | `(...)` |
| `dtypes.result_type` | Function | `(args: Any, return_weak_type_flag: bool) -> DType \| tuple[DType, bool]` |
| `dtypes.scalar_type_of` | Function | `(x: Any) -> type` |
| `effects_barrier` | Function | `()` |
| `empty_ref` | Function | `(ty, memory_space)` |
| `enable_checks` | Object | `` |
| `enable_custom_prng` | Object | `` |
| `enable_custom_vjp_by_custom_transpose` | Object | `` |
| `enable_x64` | Object | `` |
| `ensure_compile_time_eval` | Function | `()` |
| `errors.ConcretizationTypeError` | Class | `(tracer: core.Tracer, context: str)` |
| `errors.JAXIndexError` | Class | `(...)` |
| `errors.JAXTypeError` | Class | `(...)` |
| `errors.JaxRuntimeError` | Object | `` |
| `errors.KeyReuseError` | Class | `(...)` |
| `errors.NonConcreteBooleanIndexError` | Class | `(tracer: core.Tracer)` |
| `errors.TracerArrayConversionError` | Class | `(tracer: core.Tracer)` |
| `errors.TracerBoolConversionError` | Class | `(tracer: core.Tracer)` |
| `errors.TracerIntegerConversionError` | Class | `(tracer: core.Tracer)` |
| `errors.UnexpectedTracerError` | Class | `(msg: str)` |
| `eval_shape` | Function | `(fun: Callable, args, kwargs)` |
| `example_libraries.optimizers.Array` | Object | `` |
| `example_libraries.optimizers.InitFn` | Object | `` |
| `example_libraries.optimizers.JoinPoint` | Class | `(subtree)` |
| `example_libraries.optimizers.Optimizer` | Class | `(...)` |
| `example_libraries.optimizers.OptimizerState` | Object | `` |
| `example_libraries.optimizers.Params` | Object | `` |
| `example_libraries.optimizers.ParamsFn` | Object | `` |
| `example_libraries.optimizers.Schedule` | Object | `` |
| `example_libraries.optimizers.State` | Object | `` |
| `example_libraries.optimizers.Step` | Object | `` |
| `example_libraries.optimizers.UpdateFn` | Object | `` |
| `example_libraries.optimizers.Updates` | Object | `` |
| `example_libraries.optimizers.adagrad` | Function | `(step_size, momentum)` |
| `example_libraries.optimizers.adam` | Function | `(step_size, b1, b2, eps)` |
| `example_libraries.optimizers.adamax` | Function | `(step_size, b1, b2, eps)` |
| `example_libraries.optimizers.clip_grads` | Function | `(grad_tree, max_norm)` |
| `example_libraries.optimizers.constant` | Function | `(step_size) -> Schedule` |
| `example_libraries.optimizers.exponential_decay` | Function | `(step_size, decay_steps, decay_rate)` |
| `example_libraries.optimizers.inverse_time_decay` | Function | `(step_size, decay_steps, decay_rate, staircase)` |
| `example_libraries.optimizers.jnp` | Object | `` |
| `example_libraries.optimizers.l2_norm` | Function | `(tree)` |
| `example_libraries.optimizers.make_schedule` | Function | `(scalar_or_schedule: float \| Schedule) -> Schedule` |
| `example_libraries.optimizers.map` | Object | `` |
| `example_libraries.optimizers.momentum` | Function | `(step_size: Schedule, mass: float)` |
| `example_libraries.optimizers.nesterov` | Function | `(step_size: Schedule, mass: float)` |
| `example_libraries.optimizers.optimizer` | Function | `(opt_maker: Callable[..., tuple[Callable[[Params], State], Callable[[Step, Updates, Params], Params], Callable[[State], Params]]]) -> Callable[..., Optimizer]` |
| `example_libraries.optimizers.pack_optimizer_state` | Function | `(marked_pytree)` |
| `example_libraries.optimizers.piecewise_constant` | Function | `(boundaries: Any, values: Any)` |
| `example_libraries.optimizers.polynomial_decay` | Function | `(step_size, decay_steps, final_step_size, power)` |
| `example_libraries.optimizers.rmsprop` | Function | `(step_size, gamma, eps)` |
| `example_libraries.optimizers.rmsprop_momentum` | Function | `(step_size, gamma, eps, momentum)` |
| `example_libraries.optimizers.safe_map` | Function | `(f, args)` |
| `example_libraries.optimizers.safe_zip` | Function | `(args)` |
| `example_libraries.optimizers.sgd` | Function | `(step_size)` |
| `example_libraries.optimizers.sm3` | Function | `(step_size, momentum)` |
| `example_libraries.optimizers.unpack_optimizer_state` | Function | `(opt_state)` |
| `example_libraries.optimizers.unzip2` | Function | `(xys: Iterable[tuple[T1, T2]]) -> tuple[tuple[T1, ...], tuple[T2, ...]]` |
| `example_libraries.optimizers.zip` | Object | `` |
| `example_libraries.stax.AvgPool` | Object | `` |
| `example_libraries.stax.BatchNorm` | Function | `(axis, epsilon, center, scale, beta_init, gamma_init)` |
| `example_libraries.stax.Conv` | Object | `` |
| `example_libraries.stax.Conv1DTranspose` | Object | `` |
| `example_libraries.stax.ConvTranspose` | Object | `` |
| `example_libraries.stax.Dense` | Function | `(out_dim, W_init, b_init)` |
| `example_libraries.stax.Dropout` | Function | `(rate, mode)` |
| `example_libraries.stax.Elu` | Object | `` |
| `example_libraries.stax.Exp` | Object | `` |
| `example_libraries.stax.FanInConcat` | Function | `(axis)` |
| `example_libraries.stax.FanInSum` | Object | `` |
| `example_libraries.stax.FanOut` | Function | `(num)` |
| `example_libraries.stax.Flatten` | Object | `` |
| `example_libraries.stax.Gelu` | Object | `` |
| `example_libraries.stax.GeneralConv` | Function | `(dimension_numbers, out_chan, filter_shape, strides, padding, W_init, b_init)` |
| `example_libraries.stax.GeneralConvTranspose` | Function | `(dimension_numbers, out_chan, filter_shape, strides, padding, W_init, b_init)` |
| `example_libraries.stax.Identity` | Object | `` |
| `example_libraries.stax.LeakyRelu` | Object | `` |
| `example_libraries.stax.LogSoftmax` | Object | `` |
| `example_libraries.stax.MaxPool` | Object | `` |
| `example_libraries.stax.Relu` | Object | `` |
| `example_libraries.stax.Selu` | Object | `` |
| `example_libraries.stax.Sigmoid` | Object | `` |
| `example_libraries.stax.Softmax` | Object | `` |
| `example_libraries.stax.Softplus` | Object | `` |
| `example_libraries.stax.SumPool` | Object | `` |
| `example_libraries.stax.Tanh` | Object | `` |
| `example_libraries.stax.elementwise` | Function | `(fun, fun_kwargs)` |
| `example_libraries.stax.elu` | Function | `(x: ArrayLike, alpha: ArrayLike) -> Array` |
| `example_libraries.stax.gelu` | Function | `(x: ArrayLike, approximate: bool) -> Array` |
| `example_libraries.stax.glorot` | Object | `` |
| `example_libraries.stax.glorot_normal` | Function | `(in_axis: int \| Sequence[int], out_axis: int \| Sequence[int], batch_axis: int \| Sequence[int], dtype: DTypeLikeInexact \| None) -> Initializer` |
| `example_libraries.stax.jnp` | Object | `` |
| `example_libraries.stax.lax` | Object | `` |
| `example_libraries.stax.leaky_relu` | Function | `(x: ArrayLike, negative_slope: ArrayLike) -> Array` |
| `example_libraries.stax.log_softmax` | Function | `(x: ArrayLike, axis: Axis, where: ArrayLike \| None) -> Array` |
| `example_libraries.stax.logsoftmax` | Object | `` |
| `example_libraries.stax.normal` | Function | `(stddev: RealNumeric, dtype: DTypeLikeInexact \| None) -> Initializer` |
| `example_libraries.stax.ones` | Function | `(key: Array, shape: core.Shape, dtype: DTypeLikeInexact \| None, out_sharding: OutShardingType) -> Array` |
| `example_libraries.stax.parallel` | Function | `(layers)` |
| `example_libraries.stax.randn` | Object | `` |
| `example_libraries.stax.random` | Object | `` |
| `example_libraries.stax.relu` | Function | `(x: ArrayLike) -> Array` |
| `example_libraries.stax.selu` | Function | `(x: ArrayLike) -> Array` |
| `example_libraries.stax.serial` | Function | `(layers)` |
| `example_libraries.stax.shape_dependent` | Function | `(make_layer)` |
| `example_libraries.stax.sigmoid` | Function | `(x: ArrayLike) -> Array` |
| `example_libraries.stax.softmax` | Function | `(x: ArrayLike, axis: Axis, where: ArrayLike \| None) -> Array` |
| `example_libraries.stax.softplus` | Function | `(x: ArrayLike) -> Array` |
| `example_libraries.stax.standardize` | Function | `(x: ArrayLike, axis: Axis, mean: ArrayLike \| None, variance: ArrayLike \| None, epsilon: ArrayLike, where: ArrayLike \| None) -> Array` |
| `example_libraries.stax.zeros` | Function | `(key: Array, shape: core.Shape, dtype: DTypeLikeInexact \| None, out_sharding: OutShardingType) -> Array` |
| `experimental.EArray` | Class | `(aval, data)` |
| `experimental.array_serialization.pytree_serialization.Format` | Class | `(layout: LayoutOptions, sharding: ShardingOptions)` |
| `experimental.array_serialization.pytree_serialization.PyTreeT` | Object | `` |
| `experimental.array_serialization.pytree_serialization.distributed` | Object | `` |
| `experimental.array_serialization.pytree_serialization.flatten_axes` | Function | `(name, treedef, axis_tree, kws, tupled_args)` |
| `experimental.array_serialization.pytree_serialization.load` | Function | `(directory: str \| PathLike[str], shardings: PyTreeT, mask: PyTreeT \| None, ts_specs: PyTreeT \| None) -> PyTreeT` |
| `experimental.array_serialization.pytree_serialization.load_pytreedef` | Function | `(directory: str \| PathLike[str]) -> PyTreeT` |
| `experimental.array_serialization.pytree_serialization.logger` | Object | `` |
| `experimental.array_serialization.pytree_serialization.multihost_utils` | Object | `` |
| `experimental.array_serialization.pytree_serialization.nonblocking_load` | Function | `(directory: str \| PathLike[str], shardings: PyTreeT, mask: PyTreeT \| None, ts_specs: PyTreeT \| None) -> utils.PyTreeFuture` |
| `experimental.array_serialization.pytree_serialization.nonblocking_save` | Function | `(data: PyTreeT, directory: str \| PathLike[str], overwrite: bool, ts_specs: PyTreeT \| None) -> utils.PyTreeFuture` |
| `experimental.array_serialization.pytree_serialization.pathlib` | Object | `` |
| `experimental.array_serialization.pytree_serialization.save` | Function | `(data: PyTreeT, directory: str \| PathLike[str], overwrite: bool, ts_specs: PyTreeT \| None) -> None` |
| `experimental.array_serialization.pytree_serialization.ts_impl` | Object | `` |
| `experimental.array_serialization.pytree_serialization.utils` | Object | `` |
| `experimental.array_serialization.pytree_serialization_utils.PickleModule` | Object | `` |
| `experimental.array_serialization.pytree_serialization_utils.PyTreeFuture` | Class | `(future: Future[Any])` |
| `experimental.array_serialization.pytree_serialization_utils.T` | Object | `` |
| `experimental.array_serialization.pytree_serialization_utils.deserialize_pytreedef` | Function | `(pytreedef_repr: dict[str, Any])` |
| `experimental.array_serialization.pytree_serialization_utils.flatbuffers` | Object | `` |
| `experimental.array_serialization.pytree_serialization_utils.logger` | Object | `` |
| `experimental.array_serialization.pytree_serialization_utils.register_pytree_node_serialization` | Function | `(nodetype: type[T], serialized_name: str, serialize_auxdata: _SerializeAuxData, deserialize_auxdata: _DeserializeAuxData, from_children: _BuildFromChildren \| None) -> type[T]` |
| `experimental.array_serialization.pytree_serialization_utils.ser_flatbuf` | Object | `` |
| `experimental.array_serialization.pytree_serialization_utils.serialize_pytreedef` | Function | `(node) -> dict[str, Any]` |
| `experimental.array_serialization.serialization.AsyncManager` | Class | `(timeout_secs)` |
| `experimental.array_serialization.serialization.BarrierTimeoutError` | Class | `(...)` |
| `experimental.array_serialization.serialization.Format` | Class | `(layout: LayoutOptions, sharding: ShardingOptions)` |
| `experimental.array_serialization.serialization.GlobalAsyncCheckpointManager` | Class | `(...)` |
| `experimental.array_serialization.serialization.GlobalAsyncCheckpointManagerBase` | Class | `(...)` |
| `experimental.array_serialization.serialization.TS_CONTEXT` | Object | `` |
| `experimental.array_serialization.serialization.array` | Object | `` |
| `experimental.array_serialization.serialization.async_deserialize` | Function | `(in_type: jax.sharding.Sharding \| Format \| jax.ShapeDtypeStruct, tensorstore_spec: ts.Spec \| dict[str, Any], global_shape: Sequence[int] \| None, dtype, byte_limiter: _LimitInFlightBytes \| None, context, chunk_layout, assume_metadata: bool)` |
| `experimental.array_serialization.serialization.async_serialize` | Function | `(arr_inp, tensorstore_spec, commit_future, context, chunk_layout, primary_host: int \| None, replica_id: int, transaction: ts.Transaction \| None)` |
| `experimental.array_serialization.serialization.distributed` | Object | `` |
| `experimental.array_serialization.serialization.get_tensorstore_spec` | Object | `` |
| `experimental.array_serialization.serialization.is_remote_storage` | Function | `(tspec: dict[str, Any] \| str) -> bool` |
| `experimental.array_serialization.serialization.logger` | Object | `` |
| `experimental.array_serialization.serialization.run_deserialization` | Function | `(shardings: Sequence[jax.sharding.Sharding \| Format], tensorstore_specs: Sequence[dict[str, Any] \| ts.Spec], global_shapes: Sequence[array.Shape] \| None, dtypes: Sequence[typing.DTypeLike] \| None, concurrent_gb: int)` |
| `experimental.array_serialization.serialization.run_serialization` | Function | `(arrays, tensorstore_specs)` |
| `experimental.array_serialization.serialization.sharding` | Object | `` |
| `experimental.array_serialization.serialization.ts_impl` | Object | `` |
| `experimental.array_serialization.serialization.typing` | Object | `` |
| `experimental.array_serialization.serialization.util` | Object | `` |
| `experimental.array_serialization.tensorstore_impl.Format` | Class | `(layout: LayoutOptions, sharding: ShardingOptions)` |
| `experimental.array_serialization.tensorstore_impl.array` | Object | `` |
| `experimental.array_serialization.tensorstore_impl.async_deserialize` | Function | `(in_type: jax.sharding.Sharding \| Format \| jax.ShapeDtypeStruct, tensorstore_spec: ts.Spec \| dict[str, Any], global_shape: Sequence[int] \| None, dtype, byte_limiter: _LimitInFlightBytes \| None, context, chunk_layout, assume_metadata: bool)` |
| `experimental.array_serialization.tensorstore_impl.async_serialize` | Function | `(arr_inp, tensorstore_spec, commit_future, context, chunk_layout, primary_host: int \| None, replica_id: int, transaction: ts.Transaction \| None)` |
| `experimental.array_serialization.tensorstore_impl.combine_kvstores` | Function | `(combined_kvstore: dict[str, Any], kvstores: list[dict[str, Any]], context: ts.Context \| dict[str, Any]) -> None` |
| `experimental.array_serialization.tensorstore_impl.estimate_read_memory_footprint` | Function | `(t: ts.TensorStore, domain: ts.IndexDomain) -> int` |
| `experimental.array_serialization.tensorstore_impl.get_tensorstore_spec` | Function | `(ckpt_path: str \| PathLike[str], ocdbt: bool, process_idx: int \| None, arr: jax.Array \| None, driver: str) -> dict[str, Any]` |
| `experimental.array_serialization.tensorstore_impl.is_tensorstore_spec_leaf` | Function | `(leaf: Any)` |
| `experimental.array_serialization.tensorstore_impl.jnp` | Object | `` |
| `experimental.array_serialization.tensorstore_impl.logger` | Object | `` |
| `experimental.array_serialization.tensorstore_impl.merge_nested_ts_specs` | Function | `(dict1: dict[Any, Any], dict2: dict[Any, Any] \| None)` |
| `experimental.array_serialization.tensorstore_impl.typing` | Object | `` |
| `experimental.array_serialization.tensorstore_impl.verify_tensorstore_spec` | Function | `(spec: dict[str, Any], arr: jax.Array \| None, path: str \| os.PathLike[str], ocdbt: bool, check_metadata: bool) -> None` |
| `experimental.buffer_callback.Buffer` | Object | `` |
| `experimental.buffer_callback.ExecutionContext` | Object | `` |
| `experimental.buffer_callback.ExecutionStage` | Object | `` |
| `experimental.buffer_callback.buffer_callback` | Function | `(callback: Callable[..., None], result_shape_dtypes: object, has_side_effect: bool, vmap_method: str \| None, input_output_aliases: dict[int, int] \| None, command_buffer_compatible: bool)` |
| `experimental.checkify.Error` | Class | `(_pred: dict[ErrorEffect, Bool], _code: dict[ErrorEffect, Int], _metadata: dict[Int, PyTreeDef], _payload: dict[ErrorEffect, Payload])` |
| `experimental.checkify.ErrorCategory` | Object | `` |
| `experimental.checkify.JaxRuntimeError` | Class | `(...)` |
| `experimental.checkify.all_checks` | Object | `` |
| `experimental.checkify.automatic_checks` | Object | `` |
| `experimental.checkify.check` | Function | `(pred: Bool, msg: str, fmt_args, debug: bool, fmt_kwargs) -> None` |
| `experimental.checkify.check_error` | Function | `(error: Error) -> None` |
| `experimental.checkify.checkify` | Function | `(f: Callable[..., Out], errors: frozenset[ErrorCategory]) -> Callable[..., tuple[Error, Out]]` |
| `experimental.checkify.debug_check` | Function | `(pred: Bool, msg: str, fmt_args, fmt_kwargs) -> None` |
| `experimental.checkify.div_checks` | Object | `` |
| `experimental.checkify.float_checks` | Object | `` |
| `experimental.checkify.index_checks` | Object | `` |
| `experimental.checkify.init_error` | Object | `` |
| `experimental.checkify.nan_checks` | Object | `` |
| `experimental.checkify.user_checks` | Object | `` |
| `experimental.colocated_python.api.api_util` | Object | `` |
| `experimental.colocated_python.api.colocated_cpu_devices` | Function | `(devices_or_mesh)` |
| `experimental.colocated_python.api.colocated_python` | Function | `(fun: Callable[..., Any])` |
| `experimental.colocated_python.api.colocated_python_class` | Function | `(cls: type[object]) -> type[object]` |
| `experimental.colocated_python.api.make_callable` | Function | `(fun: Callable[..., Any], fun_sourceinfo: str \| None, fun_signature: inspect.Signature \| None)` |
| `experimental.colocated_python.api.util` | Object | `` |
| `experimental.colocated_python.api.wrap_class` | Function | `(cls: type[object], cls_sourceinfo: str \| None) -> type[object]` |
| `experimental.colocated_python.colocated_cpu_devices` | Function | `(devices_or_mesh)` |
| `experimental.colocated_python.colocated_python` | Function | `(fun: Callable[..., Any])` |
| `experimental.colocated_python.colocated_python_class` | Function | `(cls: type[object]) -> type[object]` |
| `experimental.colocated_python.func.FunctionInfo` | Class | `(fun: Callable[..., Any], fun_sourceinfo: str \| None, fun_signature: inspect.Signature \| None)` |
| `experimental.colocated_python.func.ShapeDtypeStructTree` | Object | `` |
| `experimental.colocated_python.func.Specialization` | Class | `(in_specs_treedef: tree_util.PyTreeDef \| None, in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] \| None, out_specs_fn: Callable[..., ShapeDtypeStructTree] \| None, out_specs_treedef: tree_util.PyTreeDef \| None, out_specs_leaves: tuple[api.ShapeDtypeStruct, ...] \| None, devices: xc.DeviceList \| None)` |
| `experimental.colocated_python.func.api` | Object | `` |
| `experimental.colocated_python.func.api_boundary` | Function | `(fun: C, repro_api_name: str \| None, repro_user_func: bool) -> C` |
| `experimental.colocated_python.func.func_backend` | Object | `` |
| `experimental.colocated_python.func.ifrt_programs` | Object | `` |
| `experimental.colocated_python.func.jax_register_backend_cache` | Object | `` |
| `experimental.colocated_python.func.make_callable` | Function | `(fun: Callable[..., Any], fun_sourceinfo: str \| None, fun_signature: inspect.Signature \| None)` |
| `experimental.colocated_python.func.pxla` | Object | `` |
| `experimental.colocated_python.func.tree_util` | Object | `` |
| `experimental.colocated_python.func.util` | Object | `` |
| `experimental.colocated_python.func.wraps` | Function | `(wrapped: Callable, namestr: str \| None, docstr: str \| None, kwargs) -> Callable[[T], T]` |
| `experimental.colocated_python.func.xc` | Object | `` |
| `experimental.colocated_python.func_backend.SINGLETON_RESULT_STORE` | Object | `` |
| `experimental.colocated_python.obj.SINGLETON_INSTANCE_REGISTRY` | Object | `` |
| `experimental.colocated_python.obj.api_boundary` | Function | `(fun: C, repro_api_name: str \| None, repro_user_func: bool) -> C` |
| `experimental.colocated_python.obj.api_util` | Object | `` |
| `experimental.colocated_python.obj.config` | Object | `` |
| `experimental.colocated_python.obj.func` | Object | `` |
| `experimental.colocated_python.obj.obj_backend` | Object | `` |
| `experimental.colocated_python.obj.tree_util` | Object | `` |
| `experimental.colocated_python.obj.util` | Object | `` |
| `experimental.colocated_python.obj.wrap_class` | Function | `(cls: type[object], cls_sourceinfo: str \| None) -> type[object]` |
| `experimental.colocated_python.obj_backend.SINGLETON_OBJECT_STORE` | Object | `` |
| `experimental.colocated_python.serialization.DeviceList` | Object | `` |
| `experimental.colocated_python.serialization.api` | Object | `` |
| `experimental.colocated_python.serialization.tree_util` | Object | `` |
| `experimental.colocated_python.serialization.util` | Object | `` |
| `experimental.colocated_python.serialization.xb` | Object | `` |
| `experimental.colocated_python.serialization.xc` | Object | `` |
| `experimental.compilation_cache.compilation_cache.reset_cache` | Function | `() -> None` |
| `experimental.compilation_cache.compilation_cache.set_cache_dir` | Function | `(path) -> None` |
| `experimental.compute_on.compute_on` | Function | `(compute_type: str)` |
| `experimental.cur_qdd` | Function | `(x)` |
| `experimental.custom_dce.custom_dce` | Class | `(fun: Callable[..., Any], static_argnums: Sequence[int])` |
| `experimental.custom_dce.custom_dce_p` | Object | `` |
| `experimental.custom_partitioning.ArrayMapping` | Class | `(dim_mappings)` |
| `experimental.custom_partitioning.BATCHING` | Object | `` |
| `experimental.custom_partitioning.CompoundFactor` | Class | `(factors)` |
| `experimental.custom_partitioning.SdyShardingRule` | Class | `(operand_mappings: tuple[ArrayMapping, ...], result_mappings: tuple[ArrayMapping, ...], reduction_factors: tuple[str, ...], need_replication_factors: tuple[str, ...], permutation_factors: tuple[str, ...], factor_sizes: int)` |
| `experimental.custom_partitioning.custom_partitioning` | Class | `(fun, static_argnums)` |
| `experimental.custom_partitioning.custom_partitioning_p` | Object | `` |
| `experimental.fused.ad` | Object | `` |
| `experimental.fused.batching` | Object | `` |
| `experimental.fused.core` | Object | `` |
| `experimental.fused.debug_info` | Function | `(traced_for: str, fun: Callable, args: Sequence[Any], kwargs: dict[str, Any], static_argnums: Sequence[int], static_argnames: Sequence[str], result_paths_thunk: Callable[[], tuple[str, ...]] \| core.InitialResultPaths, sourceinfo: str \| None, signature: inspect.Signature \| None) -> core.DebugInfo` |
| `experimental.fused.dispatch` | Object | `` |
| `experimental.fused.flatten_fun_nokwargs` | Function | `(f: Callable, store: lu.Store, in_tree: PyTreeDef, args_flat)` |
| `experimental.fused.fused` | Function | `(out_spaces)` |
| `experimental.fused.fused_p` | Object | `` |
| `experimental.fused.ir` | Object | `` |
| `experimental.fused.lu` | Object | `` |
| `experimental.fused.mlir` | Object | `` |
| `experimental.fused.pe` | Object | `` |
| `experimental.fused.safe_map` | Function | `(f, args)` |
| `experimental.fused.safe_zip` | Function | `(args)` |
| `experimental.fused.tree_flatten` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> tuple[list[Leaf], PyTreeDef]` |
| `experimental.fused.tree_unflatten` | Function | `(treedef: PyTreeDef, leaves: Iterable[Leaf]) -> Any` |
| `experimental.fused.typeof` | Function | `(x: Any) -> Any` |
| `experimental.fused.unzip2` | Function | `(xys: Iterable[tuple[T1, T2]]) -> tuple[tuple[T1, ...], tuple[T2, ...]]` |
| `experimental.fused.weakref_lru_cache` | Function | `(f: Callable[P, R] \| None, maxsize: int \| None, trace_context_in_key: bool, explain: Callable \| None)` |
| `experimental.hijax.AbstractRef` | Class | `(inner_aval: core.AbstractValue, memory_space: Any, kind: Any)` |
| `experimental.hijax.AbstractValue` | Class | `(...)` |
| `experimental.hijax.AvalMutableQDD` | Class | `(aval: AbstractValue, mutable_qdd: MutableQuasiDynamicData)` |
| `experimental.hijax.AvalQDD` | Class | `(aval: AbstractValue, qdd: QuasiDynamicData \| None)` |
| `experimental.hijax.HiPrimitive` | Class | `(name)` |
| `experimental.hijax.HiPspec` | Class | `(...)` |
| `experimental.hijax.HiType` | Class | `(...)` |
| `experimental.hijax.MutableHiType` | Class | `(...)` |
| `experimental.hijax.ShapedArray` | Class | `(shape, dtype, weak_type, sharding, vma: frozenset[AxisName], memory_space: MemorySpace)` |
| `experimental.hijax.TransformedRef` | Class | `(ref: Any, transforms: tuple[Transform, ...])` |
| `experimental.hijax.VJPHiPrimitive` | Class | `()` |
| `experimental.hijax.Zero` | Class | `(aval: core.AbstractValue)` |
| `experimental.hijax.aval_method` | Object | `` |
| `experimental.hijax.aval_property` | Object | `` |
| `experimental.hijax.control_flow_allowed_effects` | Object | `` |
| `experimental.hijax.instantiate_zeros` | Function | `(tangent)` |
| `experimental.hijax.is_undefined_primal` | Function | `(x)` |
| `experimental.hijax.register_hitype` | Function | `(val_cls, typeof_fn) -> None` |
| `experimental.io_callback` | Function | `(callback: Callable[..., Any], result_shape_dtypes: Any, args: Any, sharding: SingleDeviceSharding \| None, ordered: bool, kwargs: Any)` |
| `experimental.jax2tf.DisabledSafetyCheck` | Object | `` |
| `experimental.jax2tf.PolyShape` | Object | `` |
| `experimental.jax2tf.call_tf.CallTfEffect` | Class | `()` |
| `experimental.jax2tf.call_tf.CallTfOrderedEffect` | Class | `(...)` |
| `experimental.jax2tf.call_tf.TfConcreteFunction` | Object | `` |
| `experimental.jax2tf.call_tf.TfVal` | Object | `` |
| `experimental.jax2tf.call_tf.UnspecifiedOutputShapeDtype` | Class | `(...)` |
| `experimental.jax2tf.call_tf.ad_util` | Object | `` |
| `experimental.jax2tf.call_tf.add_to_call_tf_concrete_function_list` | Function | `(concrete_tf_fn: Any, call_tf_concrete_function_list: list[Any]) -> int` |
| `experimental.jax2tf.call_tf.call_tf` | Function | `(callable_tf: Callable, has_side_effects, ordered, output_shape_dtype, call_tf_graph) -> Callable` |
| `experimental.jax2tf.call_tf.call_tf_effect` | Object | `` |
| `experimental.jax2tf.call_tf.call_tf_ordered_effect` | Object | `` |
| `experimental.jax2tf.call_tf.call_tf_p` | Object | `` |
| `experimental.jax2tf.call_tf.check_tf_result` | Function | `(idx: int, r_tf: TfVal, r_aval: core.ShapedArray \| None) -> TfVal` |
| `experimental.jax2tf.call_tf.core` | Object | `` |
| `experimental.jax2tf.call_tf.dlpack` | Object | `` |
| `experimental.jax2tf.call_tf.dtypes` | Object | `` |
| `experimental.jax2tf.call_tf.effects` | Object | `` |
| `experimental.jax2tf.call_tf.emit_tf_embedded_graph_custom_call` | Function | `(ctx: mlir.LoweringRuleContext, concrete_function_flat_tf, operands: Sequence[ir.Value], has_side_effects, ordered, output_avals)` |
| `experimental.jax2tf.call_tf.func_dialect` | Object | `` |
| `experimental.jax2tf.call_tf.hlo` | Object | `` |
| `experimental.jax2tf.call_tf.ir` | Object | `` |
| `experimental.jax2tf.call_tf.jax2tf_internal` | Object | `` |
| `experimental.jax2tf.call_tf.jnp` | Object | `` |
| `experimental.jax2tf.call_tf.literals` | Object | `` |
| `experimental.jax2tf.call_tf.map` | Object | `` |
| `experimental.jax2tf.call_tf.mlir` | Object | `` |
| `experimental.jax2tf.call_tf.roofline` | Object | `` |
| `experimental.jax2tf.call_tf.tree_util` | Object | `` |
| `experimental.jax2tf.call_tf.util` | Object | `` |
| `experimental.jax2tf.call_tf.zip` | Object | `` |
| `experimental.jax2tf.convert` | Function | `(fun_jax: Callable, polymorphic_shapes: str \| PolyShape \| None \| Sequence[str \| PolyShape \| None], polymorphic_constraints: Sequence[str], with_gradient: bool, enable_xla: bool, native_serialization: bool \| _DefaultNativeSerialization, native_serialization_platforms: Sequence[str] \| None, native_serialization_disabled_checks: Sequence[DisabledSafetyCheck]) -> Callable` |
| `experimental.jax2tf.dtype_of_val` | Function | `(val: TfVal) -> DType` |
| `experimental.jax2tf.eval_polymorphic_shape` | Function | `(fun_jax: Callable, polymorphic_shapes) -> Callable` |
| `experimental.jax2tf.jax2tf.DEFAULT_NATIVE_SERIALIZATION` | Object | `` |
| `experimental.jax2tf.jax2tf.DType` | Object | `` |
| `experimental.jax2tf.jax2tf.DisabledSafetyCheck` | Object | `` |
| `experimental.jax2tf.jax2tf.NameStack` | Object | `` |
| `experimental.jax2tf.jax2tf.NativeSerializationImpl` | Class | `(fun_jax, args_specs, kwargs_specs, native_serialization_platforms: Sequence[str] \| None, native_serialization_disabled_checks: Sequence[DisabledSafetyCheck])` |
| `experimental.jax2tf.jax2tf.PartitionsOrReplicated` | Object | `` |
| `experimental.jax2tf.jax2tf.PolyShape` | Object | `` |
| `experimental.jax2tf.jax2tf.PrecisionType` | Object | `` |
| `experimental.jax2tf.jax2tf.TfVal` | Object | `` |
| `experimental.jax2tf.jax2tf.api` | Object | `` |
| `experimental.jax2tf.jax2tf.api_util` | Object | `` |
| `experimental.jax2tf.jax2tf.config` | Object | `` |
| `experimental.jax2tf.jax2tf.convert` | Function | `(fun_jax: Callable, polymorphic_shapes: str \| PolyShape \| None \| Sequence[str \| PolyShape \| None], polymorphic_constraints: Sequence[str], with_gradient: bool, enable_xla: bool, native_serialization: bool \| _DefaultNativeSerialization, native_serialization_platforms: Sequence[str] \| None, native_serialization_disabled_checks: Sequence[DisabledSafetyCheck]) -> Callable` |
| `experimental.jax2tf.jax2tf.core` | Object | `` |
| `experimental.jax2tf.jax2tf.dtype_of_val` | Function | `(val: TfVal) -> DType` |
| `experimental.jax2tf.jax2tf.dtypes` | Object | `` |
| `experimental.jax2tf.jax2tf.eval_polymorphic_shape` | Function | `(fun_jax: Callable, polymorphic_shapes) -> Callable` |
| `experimental.jax2tf.jax2tf.export` | Object | `` |
| `experimental.jax2tf.jax2tf.get_thread_local_state_call_tf_concrete_function_list` | Function | `() -> list[Any] \| None` |
| `experimental.jax2tf.jax2tf.inside_call_tf` | Function | `()` |
| `experimental.jax2tf.jax2tf.map` | Object | `` |
| `experimental.jax2tf.jax2tf.op_shardings` | Object | `` |
| `experimental.jax2tf.jax2tf.preprocess_arg_tf` | Function | `(arg_idx: int, arg_tf: TfVal) -> TfVal` |
| `experimental.jax2tf.jax2tf.shape_poly` | Object | `` |
| `experimental.jax2tf.jax2tf.source_info_util` | Object | `` |
| `experimental.jax2tf.jax2tf.split_to_logical_devices` | Function | `(tensor: TfVal, partition_dimensions: PartitionsOrReplicated)` |
| `experimental.jax2tf.jax2tf.tree_util` | Object | `` |
| `experimental.jax2tf.jax2tf.util` | Object | `` |
| `experimental.jax2tf.jax2tf.xla_client` | Object | `` |
| `experimental.jax2tf.jax2tf.zip` | Object | `` |
| `experimental.jax2tf.split_to_logical_devices` | Function | `(tensor: TfVal, partition_dimensions: PartitionsOrReplicated)` |
| `experimental.jet.JetTrace` | Class | `(tag, parent_trace, order)` |
| `experimental.jet.JetTracer` | Class | `(trace, primal, terms)` |
| `experimental.jet.ZeroSeries` | Class | `(...)` |
| `experimental.jet.ZeroTerm` | Class | `(...)` |
| `experimental.jet.ad_util` | Object | `` |
| `experimental.jet.api_util` | Object | `` |
| `experimental.jet.call_param_updaters` | Object | `` |
| `experimental.jet.core` | Object | `` |
| `experimental.jet.def_comp` | Function | `(prim, comp, kwargs)` |
| `experimental.jet.def_deriv` | Function | `(prim, deriv)` |
| `experimental.jet.deflinear` | Function | `(prim)` |
| `experimental.jet.defzero` | Function | `(prim)` |
| `experimental.jet.deriv_prop` | Function | `(prim, deriv, primals_in, series_in)` |
| `experimental.jet.dispatch` | Object | `` |
| `experimental.jet.fact` | Function | `(n)` |
| `experimental.jet.jet` | Function | `(fun, primals, series, factorial_scaled, _)` |
| `experimental.jet.jet2` | Object | `` |
| `experimental.jet.jet_fun` | Function | `(f, order, primals, series)` |
| `experimental.jet.jet_rules` | Object | `` |
| `experimental.jet.jet_subtrace` | Function | `(f, tag, order, primals, series)` |
| `experimental.jet.jnp` | Object | `` |
| `experimental.jet.lax` | Object | `` |
| `experimental.jet.lax_internal` | Object | `` |
| `experimental.jet.linear_prop` | Function | `(prim, primals_in, series_in, params)` |
| `experimental.jet.lu` | Object | `` |
| `experimental.jet.pe` | Object | `` |
| `experimental.jet.pjit` | Object | `` |
| `experimental.jet.register_pytree_node` | Function | `(nodetype: type[T], flatten_func: Callable[[T], tuple[_Children, _AuxData]], unflatten_func: Callable[[_AuxData, _Children], T], flatten_with_keys_func: Callable[[T], tuple[KeyLeafPairs, _AuxData]] \| None) -> None` |
| `experimental.jet.safe_zip` | Function | `(args)` |
| `experimental.jet.sharding_impls` | Object | `` |
| `experimental.jet.traceable` | Function | `(f, store, in_tree_def, primals_and_series)` |
| `experimental.jet.tree_flatten` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> tuple[list[Leaf], PyTreeDef]` |
| `experimental.jet.tree_structure` | Function | `(tree: Any, is_leaf: None \| Callable[[Any], bool]) -> PyTreeDef` |
| `experimental.jet.tree_unflatten` | Function | `(treedef: PyTreeDef, leaves: Iterable[Leaf]) -> Any` |
| `experimental.jet.treedef_is_leaf` | Function | `(treedef: PyTreeDef) -> bool` |
| `experimental.jet.unzip2` | Function | `(xys: Iterable[tuple[T1, T2]]) -> tuple[tuple[T1, ...], tuple[T2, ...]]` |
| `experimental.jet.weakref_lru_cache` | Function | `(f: Callable[P, R] \| None, maxsize: int \| None, trace_context_in_key: bool, explain: Callable \| None)` |
| `experimental.jet.zero_prop` | Function | `(prim, primals_in, series_in, params)` |
| `experimental.jet.zero_series` | Object | `` |
| `experimental.jet.zero_term` | Object | `` |
| `experimental.layout.Format` | Class | `(layout: LayoutOptions, sharding: ShardingOptions)` |
| `experimental.layout.Layout` | Class | `(major_to_minor: tuple[int, ...], tiling: tuple[tuple[int, ...], ...] \| None, sub_byte_element_size_in_bits: int)` |
| `experimental.layout.with_layout_constraint` | Function | `(x, layouts)` |
| `experimental.mesh_utils.create_device_mesh` | Function | `(mesh_shape: Sequence[int], devices: Sequence[Any] \| None, contiguous_submeshes: bool, allow_split_physical_axes: bool) -> np.ndarray` |
| `experimental.mesh_utils.create_hybrid_device_mesh` | Function | `(mesh_shape: Sequence[int], dcn_mesh_shape: Sequence[int], devices: Sequence[Any] \| None, process_is_granule: bool, should_sort_granules_by_key: bool, allow_split_physical_axes: bool) -> np.ndarray` |
| `experimental.mesh_utils.device_kind_handler_dict` | Object | `` |
| `experimental.mosaic.Tiling` | Class | `(...)` |
| `experimental.mosaic.as_tpu_kernel` | Function | `(module: ir.Module, out_type: Any, cost_estimate: CostEstimate \| None, kernel_name: str \| None, vmem_limit_bytes: int \| None, flags: dict[str, bool \| int \| float] \| None, allow_input_fusion: Sequence[bool] \| None, input_output_aliases: tuple[tuple[int, int], ...], internal_scratch_in_bytes: int \| None, collective_id: int \| None, has_side_effects: TpuSideEffectType, serialization_format: int \| None, output_memory_spaces: tuple[MemorySpace \| None, ...] \| None, disable_bounds_checks: bool, disable_semaphore_checks: bool, input_memory_spaces: tuple[MemorySpace \| None, ...] \| None, shape_invariant_numerics: bool, needs_layout_passes: bool \| None, metadata: Any \| None, tiling: Tiling \| None, _ir_version: int \| None) -> Callable[..., Any]` |
| `experimental.mosaic.dialects.tpu` | Object | `` |
| `experimental.mosaic.gpu.AsyncCopyImplementation` | Class | `(...)` |
| `experimental.mosaic.gpu.Barrier` | Class | `(arrival_count: int, num_barriers: int)` |
| `experimental.mosaic.gpu.BarrierRef` | Class | `(base_address: ir.Value, offset: ir.Value, phases: ir.Value, num_barriers: int)` |
| `experimental.mosaic.gpu.ClusterBarrier` | Class | `(collective_dims: Sequence[gpu.Dimension], arrival_count: int, num_barriers: int)` |
| `experimental.mosaic.gpu.CollectiveBarrierRef` | Class | `(barrier: BarrierRef, cluster_mask: ir.Value \| None)` |
| `experimental.mosaic.gpu.CopyPartition` | Class | `(...)` |
| `experimental.mosaic.gpu.DialectBarrierRef` | Class | `(barrier_ref: BarrierRef)` |
| `experimental.mosaic.gpu.DynamicSlice` | Class | `(base: ir.Value \| int, length: int)` |
| `experimental.mosaic.gpu.FragmentedArray` | Class | `(_registers: np.ndarray, _layout: FragmentedLayout, _is_signed: bool \| None)` |
| `experimental.mosaic.gpu.FragmentedLayout` | Object | `` |
| `experimental.mosaic.gpu.GLOBAL_BROADCAST` | Object | `` |
| `experimental.mosaic.gpu.LaunchContext` | Class | `(module: ir.Module, scratch: Scratch, cluster_size: tuple[int, int, int], profiler: OnDeviceProfiler \| None, device_collective_metadata: ir.Value \| None, host_collective_metadata: ir.Value \| None, num_peers: int, is_device_collective: bool)` |
| `experimental.mosaic.gpu.LoweringSemantics` | Class | `(...)` |
| `experimental.mosaic.gpu.MMALayouts` | Class | `(element_type: ir.Type)` |
| `experimental.mosaic.gpu.MemRefTransform` | Class | `()` |
| `experimental.mosaic.gpu.MultimemReductionOp` | Object | `` |
| `experimental.mosaic.gpu.Partition` | Class | `(elements: tuple[int, ...], partition: tuple[int \| None, ...], base_offset: tuple[ir.Value, ...] \| None, num_chunks: tuple[int, ...] \| None, chunk_size: tuple[int, ...] \| None)` |
| `experimental.mosaic.gpu.Partition1D` | Class | `(elements: int, base_offset: ir.Value \| None, num_chunks: int \| None, chunk_size: int \| None)` |
| `experimental.mosaic.gpu.Rounding` | Class | `(...)` |
| `experimental.mosaic.gpu.SemaphoreRef` | Class | `(ptr: ir.Value)` |
| `experimental.mosaic.gpu.ShapeDtypeStruct` | Class | `(shape, dtype, sharding, weak_type, vma, is_ref)` |
| `experimental.mosaic.gpu.TCGEN05_COL_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.TCGEN05_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.TCGEN05_ROW_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.TCGEN05_TRANSPOSED_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.TMABarrier` | Class | `(num_barriers: int)` |
| `experimental.mosaic.gpu.TMAReductionOp` | Object | `` |
| `experimental.mosaic.gpu.TMA_GATHER_INDICES_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.TMEM` | Class | `(shape: tuple[int, int], dtype: Any, layout: tcgen05.TMEMLayout \| None, collective: bool, packing: int \| None)` |
| `experimental.mosaic.gpu.TMEM_NATIVE_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.ThreadSubset` | Class | `(...)` |
| `experimental.mosaic.gpu.TileTransform` | Class | `(tiling: tuple[int, ...], rounding: Rounding \| None)` |
| `experimental.mosaic.gpu.TiledLayout` | Object | `` |
| `experimental.mosaic.gpu.TransposeTransform` | Class | `(permutation: tuple[int, ...])` |
| `experimental.mosaic.gpu.Union` | Class | `(members: Sequence[T])` |
| `experimental.mosaic.gpu.WGMMAAccumulator` | Class | `(_value: fa.FragmentedArray, _original_layout: fa.FragmentedLayout, _sync: bool)` |
| `experimental.mosaic.gpu.WGMMA_COL_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.WGMMA_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.WGMMA_LAYOUT_8BIT` | Object | `` |
| `experimental.mosaic.gpu.WGMMA_LAYOUT_UPCAST_2X` | Object | `` |
| `experimental.mosaic.gpu.WGMMA_LAYOUT_UPCAST_4X` | Object | `` |
| `experimental.mosaic.gpu.WGMMA_ROW_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.WGMMA_TRANSPOSED_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.WGSplatFragLayout` | Class | `(shape: tuple[int, ...])` |
| `experimental.mosaic.gpu.WGStridedFragLayout` | Class | `(shape: tuple[int, ...], vec_size: int)` |
| `experimental.mosaic.gpu.as_gpu_kernel` | Function | `(body, grid: tuple[int, int, int], block: tuple[int, int, int], in_shape, out_shape, smem_scratch_shape: ShapeTree \| Union[ShapeTree], prof_spec: profiler.ProfilerSpec \| None, cluster: tuple[int, int, int], module_name: str, kernel_name: str \| None, ir_version: int \| None, thread_semantics: LoweringSemantics, inout_shape)` |
| `experimental.mosaic.gpu.as_torch_gpu_kernel` | Function | `(body, grid: tuple[int, int, int], block: tuple[int, int, int], in_shape, out_shape, smem_scratch_shape: ShapeTree \| Union[ShapeTree], prof_spec: profiler.ProfilerSpec \| None, cluster: tuple[int, int, int], module_name: str, kernel_name: str \| None, thread_semantics: LoweringSemantics, inout_shape)` |
| `experimental.mosaic.gpu.bitwidth` | Function | `(ty: ir.Type)` |
| `experimental.mosaic.gpu.bytewidth` | Function | `(ty: ir.Type)` |
| `experimental.mosaic.gpu.c` | Function | `(val: int \| float, ty)` |
| `experimental.mosaic.gpu.commit_shared` | Function | `()` |
| `experimental.mosaic.gpu.constraints.Constant` | Class | `(...)` |
| `experimental.mosaic.gpu.constraints.Constraint` | Object | `` |
| `experimental.mosaic.gpu.constraints.ConstraintSystem` | Class | `(assignments: dict[Variable, Constant], constraints: Sequence[Constraint])` |
| `experimental.mosaic.gpu.constraints.Divides` | Class | `(expr: Expression, tiling_multiple: tuple[int, ...])` |
| `experimental.mosaic.gpu.constraints.Equals` | Class | `(lhs: Expression, rhs: Expression)` |
| `experimental.mosaic.gpu.constraints.Expression` | Object | `` |
| `experimental.mosaic.gpu.constraints.IsSupportedBroadcast` | Class | `(src: Expression, dst: Expression, dims: tuple[int, ...])` |
| `experimental.mosaic.gpu.constraints.IsTransferable` | Class | `(source: Expression, target: Expression, shape: tuple[int, ...], bitwidth: int)` |
| `experimental.mosaic.gpu.constraints.IsValidMmaTiling` | Class | `(expr: Expression, bitwidth: int)` |
| `experimental.mosaic.gpu.constraints.NotOfType` | Class | `(expr: Expression, type: type[fa.FragmentedLayout])` |
| `experimental.mosaic.gpu.constraints.Reduce` | Class | `(expression: Expression, axes: tuple[int, ...], rank: int, keep_dims: bool)` |
| `experimental.mosaic.gpu.constraints.RegisterLayout` | Class | `(value: fa.FragmentedLayout)` |
| `experimental.mosaic.gpu.constraints.Relayout` | Class | `(source: Expression, target: Expression, bitwidth: int)` |
| `experimental.mosaic.gpu.constraints.Reshape` | Class | `(expression: Expression, source_shape: tuple[int, ...], target_shape: tuple[int, ...])` |
| `experimental.mosaic.gpu.constraints.SMEMTiling` | Class | `(value: lc.TileTransform \| None)` |
| `experimental.mosaic.gpu.constraints.TMEMLayout` | Class | `(value: tcgen05.TMEMLayout)` |
| `experimental.mosaic.gpu.constraints.Transpose` | Class | `(expression: Expression)` |
| `experimental.mosaic.gpu.constraints.Unsatisfiable` | Class | `(...)` |
| `experimental.mosaic.gpu.constraints.Variable` | Class | `(key: VariableKey)` |
| `experimental.mosaic.gpu.constraints.VariableKey` | Object | `` |
| `experimental.mosaic.gpu.constraints.compute_transitively_equal_vars` | Function | `(system: ConstraintSystem) -> dict[Variable, list[Variable]]` |
| `experimental.mosaic.gpu.constraints.fa` | Object | `` |
| `experimental.mosaic.gpu.constraints.inference_utils` | Object | `` |
| `experimental.mosaic.gpu.constraints.layouts_lib` | Object | `` |
| `experimental.mosaic.gpu.constraints.lc` | Object | `` |
| `experimental.mosaic.gpu.constraints.merge_divides_constraints` | Function | `(constraints: Sequence[Constraint]) -> list[Constraint]` |
| `experimental.mosaic.gpu.constraints.non_splat_variables` | Function | `(constraints: Sequence[Constraint]) -> set[Variable]` |
| `experimental.mosaic.gpu.constraints.reduce` | Function | `(constraint_system: ConstraintSystem) -> ConstraintSystem \| Unsatisfiable` |
| `experimental.mosaic.gpu.constraints.reduce_constraint` | Function | `(constraint: Constraint, assignments: dict[Variable, Constant]) -> Constraint \| Unsatisfiable` |
| `experimental.mosaic.gpu.constraints.reduce_expression` | Function | `(expr: Expression, assignments: dict[Variable, Constant]) -> Expression \| Unsatisfiable` |
| `experimental.mosaic.gpu.constraints.reduce_reduce_expression` | Function | `(expr: Reduce, assignments: dict[Variable, Constant]) -> Expression \| Unsatisfiable` |
| `experimental.mosaic.gpu.constraints.reduce_reshape_expression` | Function | `(reshape: Reshape, assignments: dict[Variable, Constant]) -> Expression \| Unsatisfiable` |
| `experimental.mosaic.gpu.constraints.reduce_transpose_expression` | Function | `(transpose: Transpose, assignments: dict[Variable, Constant]) -> Expression \| Unsatisfiable` |
| `experimental.mosaic.gpu.constraints.saturate_distinct_from_splat` | Function | `(constraint_system: ConstraintSystem) -> ConstraintSystem \| Unsatisfiable` |
| `experimental.mosaic.gpu.constraints.saturate_divides_constraints_for_equal_vars` | Function | `(system: ConstraintSystem) -> ConstraintSystem` |
| `experimental.mosaic.gpu.constraints.tcgen05` | Object | `` |
| `experimental.mosaic.gpu.constraints.utils` | Object | `` |
| `experimental.mosaic.gpu.copy_tiled` | Function | `(src: ir.Value, dst: ir.Value, swizzle: int)` |
| `experimental.mosaic.gpu.core.Barrier` | Class | `(arrival_count: int, num_barriers: int)` |
| `experimental.mosaic.gpu.core.ClusterBarrier` | Class | `(collective_dims: Sequence[gpu.Dimension], arrival_count: int, num_barriers: int)` |
| `experimental.mosaic.gpu.core.FWD_COMPAT_IR_VERSION` | Object | `` |
| `experimental.mosaic.gpu.core.KNOWN_KERNELS` | Object | `` |
| `experimental.mosaic.gpu.core.LoweringSemantics` | Class | `(...)` |
| `experimental.mosaic.gpu.core.PYTHON_RUNFILES` | Object | `` |
| `experimental.mosaic.gpu.core.RUNTIME_PATH` | Object | `` |
| `experimental.mosaic.gpu.core.RefTree` | Object | `` |
| `experimental.mosaic.gpu.core.ShapeTree` | Object | `` |
| `experimental.mosaic.gpu.core.T` | Object | `` |
| `experimental.mosaic.gpu.core.TMABarrier` | Class | `(num_barriers: int)` |
| `experimental.mosaic.gpu.core.TMEM` | Class | `(shape: tuple[int, int], dtype: Any, layout: tcgen05.TMEMLayout \| None, collective: bool, packing: int \| None)` |
| `experimental.mosaic.gpu.core.Union` | Class | `(members: Sequence[T])` |
| `experimental.mosaic.gpu.core.artificial_shared_memory_limit` | Function | `(limit)` |
| `experimental.mosaic.gpu.core.as_gpu_kernel` | Function | `(body, grid: tuple[int, int, int], block: tuple[int, int, int], in_shape, out_shape, smem_scratch_shape: ShapeTree \| Union[ShapeTree], prof_spec: profiler.ProfilerSpec \| None, cluster: tuple[int, int, int], module_name: str, kernel_name: str \| None, ir_version: int \| None, thread_semantics: LoweringSemantics, inout_shape)` |
| `experimental.mosaic.gpu.core.as_torch_gpu_kernel` | Function | `(body, grid: tuple[int, int, int], block: tuple[int, int, int], in_shape, out_shape, smem_scratch_shape: ShapeTree \| Union[ShapeTree], prof_spec: profiler.ProfilerSpec \| None, cluster: tuple[int, int, int], module_name: str, kernel_name: str \| None, thread_semantics: LoweringSemantics, inout_shape)` |
| `experimental.mosaic.gpu.core.c` | Object | `` |
| `experimental.mosaic.gpu.core.cuda_root` | Object | `` |
| `experimental.mosaic.gpu.core.dialect` | Object | `` |
| `experimental.mosaic.gpu.core.dialect_lowering` | Object | `` |
| `experimental.mosaic.gpu.core.dtypes` | Object | `` |
| `experimental.mosaic.gpu.core.is_nvshmem_available` | Function | `()` |
| `experimental.mosaic.gpu.core.is_single_process_multi_device_topology` | Function | `()` |
| `experimental.mosaic.gpu.core.jax_core` | Object | `` |
| `experimental.mosaic.gpu.core.jax_util` | Object | `` |
| `experimental.mosaic.gpu.core.jex_backend` | Object | `` |
| `experimental.mosaic.gpu.core.launch_context` | Object | `` |
| `experimental.mosaic.gpu.core.layout_inference` | Object | `` |
| `experimental.mosaic.gpu.core.layouts` | Object | `` |
| `experimental.mosaic.gpu.core.lib` | Object | `` |
| `experimental.mosaic.gpu.core.libdevice_path` | Object | `` |
| `experimental.mosaic.gpu.core.mesh_lib` | Object | `` |
| `experimental.mosaic.gpu.core.mlir` | Object | `` |
| `experimental.mosaic.gpu.core.mosaic_gpu_lib` | Object | `` |
| `experimental.mosaic.gpu.core.mosaic_gpu_p` | Object | `` |
| `experimental.mosaic.gpu.core.profiler` | Object | `` |
| `experimental.mosaic.gpu.core.sharding_impls` | Object | `` |
| `experimental.mosaic.gpu.core.supports_cross_device_collectives` | Function | `()` |
| `experimental.mosaic.gpu.core.tcgen05` | Object | `` |
| `experimental.mosaic.gpu.core.utils` | Object | `` |
| `experimental.mosaic.gpu.debug_print` | Function | `(fmt, args, uniform, scope)` |
| `experimental.mosaic.gpu.dialect` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.CMPF_IMPLS` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.CMPI_IMPLS` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.LoweringContext` | Class | `(launch_context: lc.LaunchContext \| None, single_thread_per_block_predicate: ir.Value \| None, single_thread_per_warpgroup_predicate: ir.Value \| None, single_warp_per_block_predicate: ir.Value \| None, auto_barriers: bool, smem_requested_bytes: int, lowered_operations: set[ir.Operation \| ir.OpView])` |
| `experimental.mosaic.gpu.dialect_lowering.MlirLoweringRule` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.MlirLoweringRuleResult` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.RECURSED` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.Recursed` | Class | `(...)` |
| `experimental.mosaic.gpu.dialect_lowering.arith` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.builtin` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.fa` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.fragmented_array_to_ir` | Function | `(fragmented_array: fa.FragmentedArray, ty: ir.Type) -> ir.Value` |
| `experimental.mosaic.gpu.dialect_lowering.func` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.gpu` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.inference_utils` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.ir` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.layouts_lib` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.lc` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.lower_mgpu_dialect` | Function | `(module: ir.Module, launch_context: lc.LaunchContext \| None, auto_barriers: bool)` |
| `experimental.mosaic.gpu.dialect_lowering.memref` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.mgpu` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.mlir_interpreter` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.mlir_math` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.nvvm` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.pprint_layout` | Function | `(v: fa.FragmentedArray \| tcgen05.TMEMRef) -> str` |
| `experimental.mosaic.gpu.dialect_lowering.scf` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.swizzle_and_transforms_from_transforms_attr` | Function | `(transforms: ir.ArrayAttr) -> tuple[mgpu.SwizzlingMode, tuple[lc.MemRefTransform, ...]]` |
| `experimental.mosaic.gpu.dialect_lowering.tcgen05` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.tile_offset` | Function | `(offsets: tuple[int, ...], tiling: tuple[int, ...]) -> tuple[int, ...]` |
| `experimental.mosaic.gpu.dialect_lowering.tile_strides` | Function | `(strides: tuple[int, ...], tiling: tuple[int, ...]) -> tuple[int, ...]` |
| `experimental.mosaic.gpu.dialect_lowering.transform_type` | Function | `(ref_ty: ir.MemRefType, transforms: tuple[lc.MemRefTransform, ...]) -> ir.MemRefType` |
| `experimental.mosaic.gpu.dialect_lowering.unwrap_transformed_memref` | Function | `(ref: ir.Value, expected_transforms: ir.ArrayAttr) -> ir.Value` |
| `experimental.mosaic.gpu.dialect_lowering.utils` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.vector` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.wgmma` | Object | `` |
| `experimental.mosaic.gpu.dialect_lowering.wrap_transformed_memref` | Function | `(transformed_memref: ir.Value, logical_type: ir.Type, transforms: ir.ArrayAttr) -> ir.Value` |
| `experimental.mosaic.gpu.ds` | Object | `` |
| `experimental.mosaic.gpu.fori` | Function | `(bound, carrys)` |
| `experimental.mosaic.gpu.fragmented_array.FragmentedArray` | Class | `(_registers: np.ndarray, _layout: FragmentedLayout, _is_signed: bool \| None)` |
| `experimental.mosaic.gpu.fragmented_array.FragmentedLayout` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.IndexTransform` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.ReductionKind` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.Replicated` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.Rounding` | Class | `(...)` |
| `experimental.mosaic.gpu.fragmented_array.SMEM_BANKS` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.SMEM_BANK_BYTES` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.StaggeredTransferPlan` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.StaggeredTransferPlanImpl` | Class | `(stagger: int, dim: int, size: int, group_pred: ir.Value)` |
| `experimental.mosaic.gpu.fragmented_array.T` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.TCGEN05_COL_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.TCGEN05_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.TCGEN05_ROW_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.TCGEN05_TRANSPOSED_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.TMA_GATHER_INDICES_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.TMEM_NATIVE_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.TiledLayout` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.TiledLayoutImpl` | Class | `(tiling: Tiling, warp_dims: tuple[int \| Replicated, ...], lane_dims: tuple[int \| Replicated, ...], vector_dim: int, _check_canonical: dataclasses.InitVar[bool])` |
| `experimental.mosaic.gpu.fragmented_array.Tiling` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.TransferPlan` | Class | `(...)` |
| `experimental.mosaic.gpu.fragmented_array.TrivialTransferPlan` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.TrivialTransferPlanImpl` | Class | `()` |
| `experimental.mosaic.gpu.fragmented_array.WARPGROUP_SIZE` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.WARPS_IN_WARPGROUP` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.WARP_SIZE` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.WGMMA_COL_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.WGMMA_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.WGMMA_LAYOUT_8BIT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.WGMMA_LAYOUT_ACC_32BIT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.WGMMA_LAYOUT_UPCAST_2X` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.WGMMA_LAYOUT_UPCAST_4X` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.WGMMA_ROW_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.WGMMA_TRANSPOSED_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.WGSplatFragLayout` | Class | `(shape: tuple[int, ...])` |
| `experimental.mosaic.gpu.fragmented_array.WGStridedFragLayout` | Class | `(shape: tuple[int, ...], vec_size: int)` |
| `experimental.mosaic.gpu.fragmented_array.addf` | Function | `(a: ir.Value, b: ir.Value)` |
| `experimental.mosaic.gpu.fragmented_array.c` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.can_relayout_wgmma_2x_to_wgmma` | Function | `(bitwidth: int) -> bool` |
| `experimental.mosaic.gpu.fragmented_array.can_relayout_wgmma_4x_to_wgmma_2x` | Function | `(bitwidth: int) -> bool` |
| `experimental.mosaic.gpu.fragmented_array.copy_tiled` | Function | `(src: ir.Value, dst: ir.Value, swizzle: int)` |
| `experimental.mosaic.gpu.fragmented_array.enumerate_negative` | Function | `(elems: Sequence[T]) -> Iterable[tuple[int, T]]` |
| `experimental.mosaic.gpu.fragmented_array.is_supported_strided_layout_broadcast` | Function | `(src: WGStridedFragLayout, dst: WGStridedFragLayout, dims: tuple[int, ...]) -> bool` |
| `experimental.mosaic.gpu.fragmented_array.mgpu` | Object | `` |
| `experimental.mosaic.gpu.fragmented_array.mulf` | Function | `(a: ir.Value, b: ir.Value)` |
| `experimental.mosaic.gpu.fragmented_array.optimization_barrier` | Function | `(arrays)` |
| `experimental.mosaic.gpu.fragmented_array.plan_tiled_transfer` | Function | `(tiles_shape: Sequence[int], tiles_strides: Sequence[int], warp_shape: Sequence[int], warp_strides: Sequence[int], lane_shape: Sequence[int], lane_strides: Sequence[int], vector_length: int, element_bits: int, swizzle: int) -> TransferPlan` |
| `experimental.mosaic.gpu.fragmented_array.subf` | Function | `(a: ir.Value, b: ir.Value)` |
| `experimental.mosaic.gpu.fragmented_array.tiled_copy_smem_gmem_layout` | Function | `(row_tiles: int, col_tiles: int, swizzle: int, bitwidth: int) -> TiledLayout` |
| `experimental.mosaic.gpu.fragmented_array.tmem_native_layout` | Function | `(vector_length: int)` |
| `experimental.mosaic.gpu.fragmented_array.utils` | Object | `` |
| `experimental.mosaic.gpu.get_cluster_ref` | Function | `(ref: ir.Value, dim: gpu.Dimension, idx: ir.Value, generic: bool)` |
| `experimental.mosaic.gpu.infer_layout` | Function | `(module: ir.Module, fuel: int)` |
| `experimental.mosaic.gpu.inference_utils.MlirOperation` | Object | `` |
| `experimental.mosaic.gpu.inference_utils.attr_element` | Function | `(attr_name: str, op: MlirOperation, index: int) -> ir.Attribute \| None` |
| `experimental.mosaic.gpu.inference_utils.fa` | Object | `` |
| `experimental.mosaic.gpu.inference_utils.has_any_layout_set` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.has_in_layouts_set` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.has_in_tmem_layouts_set` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.has_in_transforms_set` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.has_out_layouts_set` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.has_out_tmem_layouts_set` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.has_out_transforms_set` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.in_layout_for_operand` | Object | `` |
| `experimental.mosaic.gpu.inference_utils.in_layouts` | Function | `(op: MlirOperation) -> Sequence[ir.Attribute]` |
| `experimental.mosaic.gpu.inference_utils.in_tmem_layouts` | Function | `(op: MlirOperation) -> Sequence[ir.Attribute]` |
| `experimental.mosaic.gpu.inference_utils.in_transforms` | Function | `(op: MlirOperation) -> Sequence[ir.Attribute]` |
| `experimental.mosaic.gpu.inference_utils.in_transforms_for_operand` | Object | `` |
| `experimental.mosaic.gpu.inference_utils.ir` | Object | `` |
| `experimental.mosaic.gpu.inference_utils.is_mma_layout` | Function | `(layout: fa.FragmentedLayout) -> bool` |
| `experimental.mosaic.gpu.inference_utils.is_transformable_smem_memref` | Function | `(v: ir.Value) -> bool` |
| `experimental.mosaic.gpu.inference_utils.out_layouts` | Function | `(op: MlirOperation) -> Sequence[ir.Attribute]` |
| `experimental.mosaic.gpu.inference_utils.out_tmem_layouts` | Function | `(op: MlirOperation) -> Sequence[ir.Attribute]` |
| `experimental.mosaic.gpu.inference_utils.out_transforms` | Function | `(op: MlirOperation) -> Sequence[ir.Attribute]` |
| `experimental.mosaic.gpu.inference_utils.should_have_in_layout` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.should_have_in_tmem_layout` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.should_have_in_transforms` | Function | `(op: ir.OpView) -> bool` |
| `experimental.mosaic.gpu.inference_utils.should_have_layout` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.should_have_out_layout` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.should_have_out_tmem_layout` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.should_have_out_transforms` | Function | `(op: ir.OpView) -> bool` |
| `experimental.mosaic.gpu.inference_utils.should_have_tmem_layout` | Function | `(op: MlirOperation) -> bool` |
| `experimental.mosaic.gpu.inference_utils.should_have_transforms` | Function | `(op: ir.OpView) -> bool` |
| `experimental.mosaic.gpu.inference_utils.tcgen05` | Object | `` |
| `experimental.mosaic.gpu.inference_utils.utils` | Object | `` |
| `experimental.mosaic.gpu.is_known_divisible` | Function | `(value: ir.Value, divisor: int, max_depth) -> bool` |
| `experimental.mosaic.gpu.launch_context.AsyncCopyImplementation` | Class | `(...)` |
| `experimental.mosaic.gpu.launch_context.COLLECTIVE_ATTR` | Object | `` |
| `experimental.mosaic.gpu.launch_context.COLLECTIVE_METADATA_SIZE` | Object | `` |
| `experimental.mosaic.gpu.launch_context.CollapseLeadingIndicesTransform` | Class | `(strides: tuple[int, ...])` |
| `experimental.mosaic.gpu.launch_context.CopyPartition` | Class | `(...)` |
| `experimental.mosaic.gpu.launch_context.DEVICE_ID_ATTR` | Object | `` |
| `experimental.mosaic.gpu.launch_context.GLOBAL_BROADCAST` | Object | `` |
| `experimental.mosaic.gpu.launch_context.GlobalBroadcast` | Class | `(...)` |
| `experimental.mosaic.gpu.launch_context.KERNEL_ARG_ID_ATTR` | Object | `` |
| `experimental.mosaic.gpu.launch_context.LaunchContext` | Class | `(module: ir.Module, scratch: Scratch, cluster_size: tuple[int, int, int], profiler: OnDeviceProfiler \| None, device_collective_metadata: ir.Value \| None, host_collective_metadata: ir.Value \| None, num_peers: int, is_device_collective: bool)` |
| `experimental.mosaic.gpu.launch_context.MOSAIC_GPU_SMEM_ALLOC_ATTR` | Object | `` |
| `experimental.mosaic.gpu.launch_context.MemRefTransform` | Class | `()` |
| `experimental.mosaic.gpu.launch_context.ORIGINAL_KERNEL_ARG_ATTR` | Object | `` |
| `experimental.mosaic.gpu.launch_context.OnDeviceProfiler` | Object | `` |
| `experimental.mosaic.gpu.launch_context.ReplicationError` | Class | `(...)` |
| `experimental.mosaic.gpu.launch_context.Rounding` | Class | `(...)` |
| `experimental.mosaic.gpu.launch_context.Scratch` | Class | `(gpu_launch_op: gpu.LaunchOp)` |
| `experimental.mosaic.gpu.launch_context.TMAReductionOp` | Object | `` |
| `experimental.mosaic.gpu.launch_context.TMA_DESCRIPTOR_ALIGNMENT` | Object | `` |
| `experimental.mosaic.gpu.launch_context.TMA_DESCRIPTOR_BYTES` | Object | `` |
| `experimental.mosaic.gpu.launch_context.TileTransform` | Class | `(tiling: tuple[int, ...], rounding: Rounding \| None)` |
| `experimental.mosaic.gpu.launch_context.TransposeTransform` | Class | `(permutation: tuple[int, ...])` |
| `experimental.mosaic.gpu.launch_context.USES_MULTIMEM_ATTR` | Object | `` |
| `experimental.mosaic.gpu.launch_context.c` | Object | `` |
| `experimental.mosaic.gpu.launch_context.fa` | Object | `` |
| `experimental.mosaic.gpu.launch_context.jaxlib_extension_version` | Object | `` |
| `experimental.mosaic.gpu.launch_context.mgpu_dialect` | Object | `` |
| `experimental.mosaic.gpu.launch_context.profiler` | Object | `` |
| `experimental.mosaic.gpu.launch_context.uses_collective_metadata` | Function | `(module)` |
| `experimental.mosaic.gpu.launch_context.utils` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.ConstraintSystemDerivationRule` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.ConstraintSystemDerivationRuleResult` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.DerivationContext` | Class | `()` |
| `experimental.mosaic.gpu.layout_inference.MemorySpace` | Class | `(...)` |
| `experimental.mosaic.gpu.layout_inference.ValueSite` | Class | `(operation: ir.OpView, type: VariableType, index: int, region_index: int \| None)` |
| `experimental.mosaic.gpu.layout_inference.ValueSitesForVariable` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.VariableType` | Class | `(...)` |
| `experimental.mosaic.gpu.layout_inference.arith` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.assign_layouts` | Function | `(solution: dict[ValueSite, cs.Constant]) -> None` |
| `experimental.mosaic.gpu.layout_inference.check_layout_assignment` | Function | `(v: ValueSite, layout: cs.Constant) -> None` |
| `experimental.mosaic.gpu.layout_inference.conjure_assignment` | Function | `(unknowns: Sequence[cs.Variable], constraint_system: cs.ConstraintSystem) -> Iterator[tuple[cs.Variable, cs.Constant]]` |
| `experimental.mosaic.gpu.layout_inference.consumer_operands` | Function | `(result: ValueSite) -> Sequence[ValueSite]` |
| `experimental.mosaic.gpu.layout_inference.cs` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.derive_relayout_constraints` | Function | `(value_sites_for_variable: ValueSitesForVariable) -> list[cs.Relayout]` |
| `experimental.mosaic.gpu.layout_inference.dynamic_gcd` | Function | `(a: int, b: ir.Value) -> int` |
| `experimental.mosaic.gpu.layout_inference.extract_assignment_candidates_from_reduce_equation` | Function | `(small: cs.RegisterLayout, large: cs.Variable, reduction_dims: tuple[int, ...], keep_dims: bool) -> Iterator[cs.RegisterLayout]` |
| `experimental.mosaic.gpu.layout_inference.fa` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.find_assignments_for` | Function | `(unknowns: Sequence[cs.Variable], constraint_system: cs.ConstraintSystem, fuel: int) -> tuple[dict[cs.Variable, cs.Constant] \| cs.Unsatisfiable, int]` |
| `experimental.mosaic.gpu.layout_inference.infer_layout` | Function | `(module: ir.Module, fuel: int)` |
| `experimental.mosaic.gpu.layout_inference.inference_utils` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.ir` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.is_terminator` | Function | `(op: ir.OpView) -> bool` |
| `experimental.mosaic.gpu.layout_inference.is_valid_register_layout_assignment` | Function | `(shape: tuple[int, ...], layout: fa.FragmentedLayout) -> bool` |
| `experimental.mosaic.gpu.layout_inference.is_valid_smem_layout_assignment` | Function | `(shape: tuple[int, ...], tiling: lc.TileTransform) -> bool` |
| `experimental.mosaic.gpu.layout_inference.is_valid_tmem_layout_assignment` | Function | `(shape: tuple[int, ...], layout: tcgen05.TMEMLayout) -> bool` |
| `experimental.mosaic.gpu.layout_inference.is_vector` | Function | `(v: ir.Value) -> bool` |
| `experimental.mosaic.gpu.layout_inference.layouts_lib` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.lc` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.memref` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.mgpu` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.mlir_math` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.prime_decomposition` | Function | `(n: int) -> list[int]` |
| `experimental.mosaic.gpu.layout_inference.producer_result` | Function | `(operand: ValueSite) -> ValueSite` |
| `experimental.mosaic.gpu.layout_inference.scf` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.tcgen05` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.traverse_op` | Function | `(op: ir.OpView, callback: Callable[[ir.OpView], None])` |
| `experimental.mosaic.gpu.layout_inference.utils` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.vector` | Object | `` |
| `experimental.mosaic.gpu.layout_inference.vector_value_sites` | Function | `(op: ir.OpView) -> list[ValueSite]` |
| `experimental.mosaic.gpu.layouts.fa` | Object | `` |
| `experimental.mosaic.gpu.layouts.from_layout_attr` | Function | `(attr: ir.Attribute) -> fa.FragmentedLayout` |
| `experimental.mosaic.gpu.layouts.from_splat_fragmented_layout_attr` | Function | `(attr: ir.Attribute) -> fa.WGSplatFragLayout` |
| `experimental.mosaic.gpu.layouts.from_strided_fragmented_layout_attr` | Function | `(attr: ir.Attribute) -> fa.WGStridedFragLayout` |
| `experimental.mosaic.gpu.layouts.from_tiled_layout_attr` | Function | `(attr: ir.Attribute) -> fa.TiledLayout` |
| `experimental.mosaic.gpu.layouts.from_transform_attr` | Function | `(transform: ir.Attribute) -> launch_context.MemRefTransform \| mgpu.SwizzlingMode` |
| `experimental.mosaic.gpu.layouts.ir` | Object | `` |
| `experimental.mosaic.gpu.layouts.is_splat_fragmented_layout` | Function | `(attr: ir.Attribute) -> bool` |
| `experimental.mosaic.gpu.layouts.is_strided_fragmented_layout` | Function | `(attr: ir.Attribute) -> bool` |
| `experimental.mosaic.gpu.layouts.is_swizzle_transform` | Function | `(attr: ir.Attribute) -> bool` |
| `experimental.mosaic.gpu.layouts.is_tile_transform` | Function | `(attr: ir.Attribute) -> bool` |
| `experimental.mosaic.gpu.layouts.is_tiled_layout` | Function | `(attr: ir.Attribute) -> bool` |
| `experimental.mosaic.gpu.layouts.is_transpose_transform` | Function | `(attr: ir.Attribute) -> bool` |
| `experimental.mosaic.gpu.layouts.launch_context` | Object | `` |
| `experimental.mosaic.gpu.layouts.mgpu` | Object | `` |
| `experimental.mosaic.gpu.layouts.splat_is_compatible_with_tiled` | Function | `(l1: fa.WGSplatFragLayout, l2: fa.TiledLayout) -> bool` |
| `experimental.mosaic.gpu.layouts.to_layout_attr` | Function | `(layout: fa.FragmentedLayout) -> ir.Attribute` |
| `experimental.mosaic.gpu.layouts.to_splat_fragmented_layout_attr` | Function | `(layout: fa.WGSplatFragLayout) -> ir.Attribute` |
| `experimental.mosaic.gpu.layouts.to_strided_fragmented_layout_attr` | Function | `(layout: fa.WGStridedFragLayout) -> ir.Attribute` |
| `experimental.mosaic.gpu.layouts.to_tiled_layout_attr` | Function | `(layout: fa.TiledLayout) -> ir.Attribute` |
| `experimental.mosaic.gpu.layouts.to_transform_attr` | Function | `(transform: launch_context.MemRefTransform \| mgpu.SwizzlingMode) -> ir.Attribute` |
| `experimental.mosaic.gpu.lower_mgpu_dialect` | Function | `(module: ir.Module, launch_context: lc.LaunchContext \| None, auto_barriers: bool)` |
| `experimental.mosaic.gpu.memref_fold` | Function | `(ref: ir.Value \| MultimemRef, dim, fold_rank) -> ir.Value \| MultimemRef` |
| `experimental.mosaic.gpu.memref_reshape` | Function | `(ref: ir.Value \| MultimemRef, shape: tuple[int, ...]) -> ir.Value \| MultimemRef` |
| `experimental.mosaic.gpu.memref_slice` | Function | `(ref: ir.Value, index) -> ir.Value` |
| `experimental.mosaic.gpu.memref_transpose` | Function | `(ref: ir.Value, permutation: Sequence[int]) -> ir.Value` |
| `experimental.mosaic.gpu.memref_unfold` | Function | `(ref: ir.Value, dim, factors) -> ir.Value` |
| `experimental.mosaic.gpu.memref_unsqueeze` | Function | `(ref: ir.Value, dim) -> ir.Value` |
| `experimental.mosaic.gpu.mma.MMALayouts` | Class | `(element_type: ir.Type)` |
| `experimental.mosaic.gpu.mma.SUPPORTED_F8_TYPES` | Object | `` |
| `experimental.mosaic.gpu.mma.fa` | Object | `` |
| `experimental.mosaic.gpu.mma.mma` | Function | `(acc: fa.FragmentedArray, a: fa.FragmentedArray, b: fa.FragmentedArray) -> fa.FragmentedArray` |
| `experimental.mosaic.gpu.mma.utils` | Object | `` |
| `experimental.mosaic.gpu.mma_utils.Dim` | Class | `(...)` |
| `experimental.mosaic.gpu.mma_utils.create_descriptor` | Function | `(ref: ir.Value, swizzle: int, group_size: tuple[int, int], logical_k_major: bool, large_tile: tuple[int, int] \| None, mma_bytewidth_k: int, split_const: bool)` |
| `experimental.mosaic.gpu.mma_utils.encode_addr` | Function | `(x: int)` |
| `experimental.mosaic.gpu.mma_utils.encode_descriptor` | Function | `(ref_arg, leading_byte_offset: int, stride_byte_offset: int, swizzle: int \| mgpu_dialect.SwizzlingMode \| None, const_init: int, split_const: bool)` |
| `experimental.mosaic.gpu.mma_utils.mgpu_dialect` | Object | `` |
| `experimental.mosaic.gpu.mma_utils.tiled_memref_shape` | Function | `(ref: ir.Value)` |
| `experimental.mosaic.gpu.mma_utils.utils` | Object | `` |
| `experimental.mosaic.gpu.nanosleep` | Function | `(nanos: ir.Value)` |
| `experimental.mosaic.gpu.optimization_barrier` | Function | `(arrays)` |
| `experimental.mosaic.gpu.profiler.Any` | Object | `` |
| `experimental.mosaic.gpu.profiler.Arch` | Class | `(major: int, minor: int)` |
| `experimental.mosaic.gpu.profiler.BarrierRef` | Class | `(base_address: ir.Value, offset: ir.Value, phases: ir.Value, num_barriers: int)` |
| `experimental.mosaic.gpu.profiler.CollectiveBarrierRef` | Class | `(barrier: BarrierRef, cluster_mask: ir.Value \| None)` |
| `experimental.mosaic.gpu.profiler.Cupti` | Class | `(...)` |
| `experimental.mosaic.gpu.profiler.DYNAMIC` | Object | `` |
| `experimental.mosaic.gpu.profiler.DYNAMIC32` | Object | `` |
| `experimental.mosaic.gpu.profiler.DialectBarrierRef` | Class | `(barrier_ref: BarrierRef)` |
| `experimental.mosaic.gpu.profiler.DynamicSlice` | Class | `(base: ir.Value \| int, length: int)` |
| `experimental.mosaic.gpu.profiler.ForResult` | Class | `(op: scf.ForOp, results: tuple[Any, ...])` |
| `experimental.mosaic.gpu.profiler.Iterator` | Object | `` |
| `experimental.mosaic.gpu.profiler.Literal` | Object | `` |
| `experimental.mosaic.gpu.profiler.MBARRIER_BYTES` | Object | `` |
| `experimental.mosaic.gpu.profiler.MultimemReductionOp` | Object | `` |
| `experimental.mosaic.gpu.profiler.MultimemRef` | Class | `(ref: ir.Value)` |
| `experimental.mosaic.gpu.profiler.OnDeviceProfiler` | Class | `(spec: ProfilerSpec, smem_buffer: ir.Value, gmem_buffer: ir.Value, wrap_in_custom_primitive: bool)` |
| `experimental.mosaic.gpu.profiler.P` | Object | `` |
| `experimental.mosaic.gpu.profiler.Partition` | Class | `(elements: tuple[int, ...], partition: tuple[int \| None, ...], base_offset: tuple[ir.Value, ...] \| None, num_chunks: tuple[int, ...] \| None, chunk_size: tuple[int, ...] \| None)` |
| `experimental.mosaic.gpu.profiler.Partition1D` | Class | `(elements: int, base_offset: ir.Value \| None, num_chunks: int \| None, chunk_size: int \| None)` |
| `experimental.mosaic.gpu.profiler.ProfilerSpec` | Class | `(entries_per_warpgroup: int, dump_path: str)` |
| `experimental.mosaic.gpu.profiler.ReductionKind` | Object | `` |
| `experimental.mosaic.gpu.profiler.SemaphoreRef` | Class | `(ptr: ir.Value)` |
| `experimental.mosaic.gpu.profiler.Sequence` | Object | `` |
| `experimental.mosaic.gpu.profiler.T` | Object | `` |
| `experimental.mosaic.gpu.profiler.ThreadSubset` | Class | `(...)` |
| `experimental.mosaic.gpu.profiler.WARPGROUP_SIZE` | Object | `` |
| `experimental.mosaic.gpu.profiler.WARPS_IN_WARPGROUP` | Object | `` |
| `experimental.mosaic.gpu.profiler.WARP_SIZE` | Object | `` |
| `experimental.mosaic.gpu.profiler.WORKGROUP_NVPTX_ADDRESS_SPACE` | Object | `` |
| `experimental.mosaic.gpu.profiler.arith` | Object | `` |
| `experimental.mosaic.gpu.profiler.bitcast` | Function | `(x: ir.Value, new_type: ir.Type)` |
| `experimental.mosaic.gpu.profiler.bitwidth` | Function | `(ty: ir.Type)` |
| `experimental.mosaic.gpu.profiler.bitwidth_impl` | Function | `(ty: ir.Type)` |
| `experimental.mosaic.gpu.profiler.block_idx` | Object | `` |
| `experimental.mosaic.gpu.profiler.builtin` | Object | `` |
| `experimental.mosaic.gpu.profiler.bytewidth` | Function | `(ty: ir.Type)` |
| `experimental.mosaic.gpu.profiler.c` | Function | `(val: int \| float, ty)` |
| `experimental.mosaic.gpu.profiler.ceil_div` | Function | `(x: int, y: int)` |
| `experimental.mosaic.gpu.profiler.clock` | Function | `()` |
| `experimental.mosaic.gpu.profiler.cluster_collective_mask` | Function | `(cluster_shape: tuple[int, int, int], collective: Sequence[gpu.Dimension] \| gpu.Dimension)` |
| `experimental.mosaic.gpu.profiler.cluster_idx` | Function | `(dim: gpu.Dimension \| Sequence[gpu.Dimension] \| None, dim_idx: ir.Value \| Sequence[ir.Value] \| None) -> ir.Value` |
| `experimental.mosaic.gpu.profiler.commit_shared` | Function | `()` |
| `experimental.mosaic.gpu.profiler.contextlib` | Object | `` |
| `experimental.mosaic.gpu.profiler.dataclasses` | Object | `` |
| `experimental.mosaic.gpu.profiler.debug_print` | Function | `(fmt, args, uniform, scope)` |
| `experimental.mosaic.gpu.profiler.dialect` | Object | `` |
| `experimental.mosaic.gpu.profiler.ds` | Object | `` |
| `experimental.mosaic.gpu.profiler.dtype_to_ir_type` | Function | `(dtype: jax.typing.DTypeLike) -> ir.Type` |
| `experimental.mosaic.gpu.profiler.dyn_dot` | Function | `(x, y)` |
| `experimental.mosaic.gpu.profiler.enum` | Object | `` |
| `experimental.mosaic.gpu.profiler.fence_release_sys` | Function | `()` |
| `experimental.mosaic.gpu.profiler.fori` | Function | `(bound, carrys)` |
| `experimental.mosaic.gpu.profiler.functools` | Object | `` |
| `experimental.mosaic.gpu.profiler.get_arch` | Function | `() -> Arch` |
| `experimental.mosaic.gpu.profiler.get_cluster_ptr` | Function | `(ptr: ir.Value, cluster_block: ir.Value, generic: bool)` |
| `experimental.mosaic.gpu.profiler.get_cluster_ref` | Function | `(ref: ir.Value, dim: gpu.Dimension, idx: ir.Value, generic: bool)` |
| `experimental.mosaic.gpu.profiler.get_contiguous_strides` | Function | `(xs)` |
| `experimental.mosaic.gpu.profiler.getelementptr` | Function | `(ptr: ir.Value, indices: Sequence[ir.Value \| int], dtype: ir.Type) -> ir.Value` |
| `experimental.mosaic.gpu.profiler.globaltimer` | Function | `(kind: Literal['low', 'high'] \| None)` |
| `experimental.mosaic.gpu.profiler.gpu` | Object | `` |
| `experimental.mosaic.gpu.profiler.gpu_address_space_to_nvptx` | Function | `(address_space: gpu.AddressSpace) -> int` |
| `experimental.mosaic.gpu.profiler.ir` | Object | `` |
| `experimental.mosaic.gpu.profiler.is_known_divisible` | Function | `(value: ir.Value, divisor: int, max_depth) -> bool` |
| `experimental.mosaic.gpu.profiler.is_memref_transposed` | Function | `(ref: ir.MemRefType) -> bool` |
| `experimental.mosaic.gpu.profiler.is_signed` | Function | `(dtype: jax.typing.DTypeLike) -> bool \| None` |
| `experimental.mosaic.gpu.profiler.is_smem_ref` | Function | `(ref: ir.Value \| ir.Type) -> bool` |
| `experimental.mosaic.gpu.profiler.is_tmem_ref` | Function | `(ref: ir.Value \| ir.Type) -> bool` |
| `experimental.mosaic.gpu.profiler.jaxlib_extension_version` | Object | `` |
| `experimental.mosaic.gpu.profiler.jnp` | Object | `` |
| `experimental.mosaic.gpu.profiler.llvm` | Object | `` |
| `experimental.mosaic.gpu.profiler.math` | Object | `` |
| `experimental.mosaic.gpu.profiler.measure` | Function | `(f, aggregate: bool, iterations: int)` |
| `experimental.mosaic.gpu.profiler.memref` | Object | `` |
| `experimental.mosaic.gpu.profiler.memref_fold` | Function | `(ref: ir.Value \| MultimemRef, dim, fold_rank) -> ir.Value \| MultimemRef` |
| `experimental.mosaic.gpu.profiler.memref_ptr` | Function | `(memref_arg, memory_space)` |
| `experimental.mosaic.gpu.profiler.memref_reshape` | Function | `(ref: ir.Value \| MultimemRef, shape: tuple[int, ...]) -> ir.Value \| MultimemRef` |
| `experimental.mosaic.gpu.profiler.memref_slice` | Function | `(ref: ir.Value, index) -> ir.Value` |
| `experimental.mosaic.gpu.profiler.memref_transpose` | Function | `(ref: ir.Value, permutation: Sequence[int]) -> ir.Value` |
| `experimental.mosaic.gpu.profiler.memref_unfold` | Function | `(ref: ir.Value, dim, factors) -> ir.Value` |
| `experimental.mosaic.gpu.profiler.memref_unsqueeze` | Function | `(ref: ir.Value, dim) -> ir.Value` |
| `experimental.mosaic.gpu.profiler.mlir` | Object | `` |
| `experimental.mosaic.gpu.profiler.mosaic_gpu_lib` | Object | `` |
| `experimental.mosaic.gpu.profiler.multimem_load_reduce` | Function | `(ty: ir.Type, ptr: ir.Value, reduction: MultimemReductionOp, is_signed: bool \| None)` |
| `experimental.mosaic.gpu.profiler.multimem_store` | Function | `(ptr: ir.Value, value: ir.Value)` |
| `experimental.mosaic.gpu.profiler.nanosleep` | Function | `(nanos: ir.Value)` |
| `experimental.mosaic.gpu.profiler.np` | Object | `` |
| `experimental.mosaic.gpu.profiler.nvvm` | Object | `` |
| `experimental.mosaic.gpu.profiler.nvvm_mbarrier_arrive_expect_tx` | Function | `(barrier: ir.Value, expect_tx: ir.Value, predicate: ir.Value \| None)` |
| `experimental.mosaic.gpu.profiler.overload` | Object | `` |
| `experimental.mosaic.gpu.profiler.pack_array` | Function | `(values)` |
| `experimental.mosaic.gpu.profiler.parse_indices` | Function | `(index, shape: tuple[int, ...], check_oob: bool) -> tuple[list[ir.Value \| int], list[int], list[bool]]` |
| `experimental.mosaic.gpu.profiler.prmt` | Function | `(high: ir.Value, low: ir.Value, permutation: ir.Value)` |
| `experimental.mosaic.gpu.profiler.ptr_as_memref` | Function | `(ptr, memref_ty: ir.MemRefType, ptr_memory_space: int \| None)` |
| `experimental.mosaic.gpu.profiler.query_cluster_cancel` | Function | `(result_ref) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]` |
| `experimental.mosaic.gpu.profiler.reduce_shape` | Function | `(shape: Sequence[int], axes: Sequence[int], keep_dims: bool) -> tuple[int, ...]` |
| `experimental.mosaic.gpu.profiler.redux` | Function | `(x: ir.Value, mask: ir.Value, kind: ReductionKind)` |
| `experimental.mosaic.gpu.profiler.scf` | Object | `` |
| `experimental.mosaic.gpu.profiler.shfl_bfly` | Function | `(x: ir.Value, distance: int \| ir.Value)` |
| `experimental.mosaic.gpu.profiler.single_thread` | Function | `(scope: ThreadSubset)` |
| `experimental.mosaic.gpu.profiler.single_thread_predicate` | Function | `(scope: ThreadSubset)` |
| `experimental.mosaic.gpu.profiler.smem` | Function | `() -> ir.Attribute` |
| `experimental.mosaic.gpu.profiler.smid` | Function | `()` |
| `experimental.mosaic.gpu.profiler.stages` | Object | `` |
| `experimental.mosaic.gpu.profiler.thread_idx` | Object | `` |
| `experimental.mosaic.gpu.profiler.tile_shape` | Function | `(shape, tiling)` |
| `experimental.mosaic.gpu.profiler.tmem` | Function | `() -> ir.Attribute` |
| `experimental.mosaic.gpu.profiler.try_cluster_cancel` | Function | `(result_ref, barrier: BarrierRef, predicate: ir.Value \| None)` |
| `experimental.mosaic.gpu.profiler.typing` | Object | `` |
| `experimental.mosaic.gpu.profiler.util` | Object | `` |
| `experimental.mosaic.gpu.profiler.vector` | Object | `` |
| `experimental.mosaic.gpu.profiler.vector_concat` | Function | `(vectors: Sequence[ir.Value]) -> ir.Value` |
| `experimental.mosaic.gpu.profiler.vector_slice` | Function | `(v: ir.Value, s: slice)` |
| `experimental.mosaic.gpu.profiler.warp_barrier` | Function | `()` |
| `experimental.mosaic.gpu.profiler.warp_idx` | Function | `(sync)` |
| `experimental.mosaic.gpu.profiler.warp_tree_reduce` | Function | `(value, op, group_size)` |
| `experimental.mosaic.gpu.profiler.warpgroup_barrier` | Function | `()` |
| `experimental.mosaic.gpu.profiler.warpgroup_idx` | Function | `(sync)` |
| `experimental.mosaic.gpu.profiler.when` | Function | `(cond)` |
| `experimental.mosaic.gpu.query_cluster_cancel` | Function | `(result_ref) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]` |
| `experimental.mosaic.gpu.single_thread` | Function | `(scope: ThreadSubset)` |
| `experimental.mosaic.gpu.single_thread_predicate` | Function | `(scope: ThreadSubset)` |
| `experimental.mosaic.gpu.supports_cross_device_collectives` | Function | `()` |
| `experimental.mosaic.gpu.tcgen05.COL_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.tcgen05.LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.tcgen05.LaunchContext` | Class | `(module: ir.Module, scratch: Scratch, cluster_size: tuple[int, int, int], profiler: OnDeviceProfiler \| None, device_collective_metadata: ir.Value \| None, host_collective_metadata: ir.Value \| None, num_peers: int, is_device_collective: bool)` |
| `experimental.mosaic.gpu.tcgen05.ROW_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.tcgen05.TCGEN05_SMEM_DESCRIPTOR_BIT` | Object | `` |
| `experimental.mosaic.gpu.tcgen05.TMEMLayout` | Class | `(...)` |
| `experimental.mosaic.gpu.tcgen05.TMEMRef` | Class | `(address: ir.Value, shape: tuple[int, int], dtype: ir.Type, layout: TMEMLayout)` |
| `experimental.mosaic.gpu.tcgen05.TMEM_MAX_COLS` | Object | `` |
| `experimental.mosaic.gpu.tcgen05.TMEM_NATIVE_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.tcgen05.TMEM_ROWS` | Object | `` |
| `experimental.mosaic.gpu.tcgen05.TRANSPOSED_LAYOUT` | Object | `` |
| `experimental.mosaic.gpu.tcgen05.async_copy_scales_smem_to_tmem` | Function | `(smem_ref: ir.Value, tmem_ref: TMEMRef, collective: bool) -> None` |
| `experimental.mosaic.gpu.tcgen05.async_copy_smem_to_tmem` | Function | `(smem_ref: ir.Value, tmem_ref: TMEMRef, swizzle: int, collective: bool) -> None` |
| `experimental.mosaic.gpu.tcgen05.async_copy_sparse_metadata_smem_to_tmem` | Function | `(smem_ref: ir.Value, tmem_ref: TMEMRef, collective: bool) -> None` |
| `experimental.mosaic.gpu.tcgen05.commit_arrive` | Function | `(barrier: utils.BarrierRef \| ir.Value, collective: bool, ctx: LaunchContext \| None) -> None` |
| `experimental.mosaic.gpu.tcgen05.commit_tmem` | Function | `() -> None` |
| `experimental.mosaic.gpu.tcgen05.create_instr_descriptor` | Function | `(m: int, n: int, acc_dtype, input_dtype, transpose_a: bool, transpose_b: bool, sparsity_selector: int \| None) -> ir.Value` |
| `experimental.mosaic.gpu.tcgen05.create_scaled_f4_instr_descriptor` | Function | `(args, kwargs) -> ir.Value` |
| `experimental.mosaic.gpu.tcgen05.create_scaled_f8f6f4_instr_descriptor` | Function | `(args, kwargs) -> ir.Value` |
| `experimental.mosaic.gpu.tcgen05.fa` | Object | `` |
| `experimental.mosaic.gpu.tcgen05.fa_m64_collective_layout` | Function | `(columns: int) -> fa.TiledLayout` |
| `experimental.mosaic.gpu.tcgen05.mma` | Function | `(d: TMEMRef, a: ir.Value \| TMEMRef, b: ir.Value, a_swizzle: int, b_swizzle: int, a_scale: TMEMRef \| None, b_scale: TMEMRef \| None, a_sparse_metadata: TMEMRef \| None, accumulate: ir.Value \| bool, collective: bool) -> None` |
| `experimental.mosaic.gpu.tcgen05.mma_utils` | Object | `` |
| `experimental.mosaic.gpu.tcgen05.scales_layout` | Function | `() -> TMEMLayout` |
| `experimental.mosaic.gpu.tcgen05.sparse_meta_layout` | Function | `() -> TMEMLayout` |
| `experimental.mosaic.gpu.tcgen05.tmem_alloc` | Function | `(tmem_addr: ir.Value, ncols: int, collective: bool, exact: bool) -> tuple[ir.Value, int]` |
| `experimental.mosaic.gpu.tcgen05.tmem_alloc_exact_ncols` | Function | `(ncols: int, exact: bool) -> int` |
| `experimental.mosaic.gpu.tcgen05.tmem_dealloc` | Function | `(tmem_addr: ir.Value, ncols: int, collective: bool, exact: bool) -> None` |
| `experimental.mosaic.gpu.tcgen05.tmem_default_layout` | Function | `(packing: int) -> TMEMLayout` |
| `experimental.mosaic.gpu.tcgen05.tmem_half_lane_layout` | Function | `(columns, packing: int) -> TMEMLayout` |
| `experimental.mosaic.gpu.tcgen05.tmem_m64_collective_layout` | Function | `(columns: int, packing: int) -> TMEMLayout` |
| `experimental.mosaic.gpu.tcgen05.tmem_relinquish_alloc_permit` | Function | `(collective: bool) -> None` |
| `experimental.mosaic.gpu.tcgen05.utils` | Object | `` |
| `experimental.mosaic.gpu.tcgen05.wait_load_tmem` | Function | `() -> None` |
| `experimental.mosaic.gpu.thread_idx` | Object | `` |
| `experimental.mosaic.gpu.tile_shape` | Function | `(shape, tiling)` |
| `experimental.mosaic.gpu.tmem_native_layout` | Function | `(vector_length: int)` |
| `experimental.mosaic.gpu.to_layout_attr` | Function | `(layout: fa.FragmentedLayout) -> ir.Attribute` |
| `experimental.mosaic.gpu.try_cluster_cancel` | Function | `(result_ref, barrier: BarrierRef, predicate: ir.Value \| None)` |
| `experimental.mosaic.gpu.utils.Arch` | Class | `(major: int, minor: int)` |
| `experimental.mosaic.gpu.utils.BarrierRef` | Class | `(base_address: ir.Value, offset: ir.Value, phases: ir.Value, num_barriers: int)` |
| `experimental.mosaic.gpu.utils.CollectiveBarrierRef` | Class | `(barrier: BarrierRef, cluster_mask: ir.Value \| None)` |
| `experimental.mosaic.gpu.utils.DYNAMIC` | Object | `` |
| `experimental.mosaic.gpu.utils.DYNAMIC32` | Object | `` |
| `experimental.mosaic.gpu.utils.DialectBarrierRef` | Class | `(barrier_ref: BarrierRef)` |
| `experimental.mosaic.gpu.utils.DynamicSlice` | Class | `(base: ir.Value \| int, length: int)` |
| `experimental.mosaic.gpu.utils.ForResult` | Class | `(op: scf.ForOp, results: tuple[Any, ...])` |
| `experimental.mosaic.gpu.utils.MBARRIER_BYTES` | Object | `` |
| `experimental.mosaic.gpu.utils.MultimemReductionOp` | Object | `` |
| `experimental.mosaic.gpu.utils.MultimemRef` | Class | `(ref: ir.Value)` |
| `experimental.mosaic.gpu.utils.Partition` | Class | `(elements: tuple[int, ...], partition: tuple[int \| None, ...], base_offset: tuple[ir.Value, ...] \| None, num_chunks: tuple[int, ...] \| None, chunk_size: tuple[int, ...] \| None)` |
| `experimental.mosaic.gpu.utils.Partition1D` | Class | `(elements: int, base_offset: ir.Value \| None, num_chunks: int \| None, chunk_size: int \| None)` |
| `experimental.mosaic.gpu.utils.ReductionKind` | Object | `` |
| `experimental.mosaic.gpu.utils.SemaphoreRef` | Class | `(ptr: ir.Value)` |
| `experimental.mosaic.gpu.utils.ThreadSubset` | Class | `(...)` |
| `experimental.mosaic.gpu.utils.WARPGROUP_SIZE` | Object | `` |
| `experimental.mosaic.gpu.utils.WARPS_IN_WARPGROUP` | Object | `` |
| `experimental.mosaic.gpu.utils.WARP_SIZE` | Object | `` |
| `experimental.mosaic.gpu.utils.WORKGROUP_NVPTX_ADDRESS_SPACE` | Object | `` |
| `experimental.mosaic.gpu.utils.bitcast` | Function | `(x: ir.Value, new_type: ir.Type)` |
| `experimental.mosaic.gpu.utils.bitwidth` | Function | `(ty: ir.Type)` |
| `experimental.mosaic.gpu.utils.bitwidth_impl` | Function | `(ty: ir.Type)` |
| `experimental.mosaic.gpu.utils.block_idx` | Object | `` |
| `experimental.mosaic.gpu.utils.bytewidth` | Function | `(ty: ir.Type)` |
| `experimental.mosaic.gpu.utils.c` | Function | `(val: int \| float, ty)` |
| `experimental.mosaic.gpu.utils.ceil_div` | Function | `(x: int, y: int)` |
| `experimental.mosaic.gpu.utils.clock` | Function | `()` |
| `experimental.mosaic.gpu.utils.cluster_collective_mask` | Function | `(cluster_shape: tuple[int, int, int], collective: Sequence[gpu.Dimension] \| gpu.Dimension)` |
| `experimental.mosaic.gpu.utils.cluster_idx` | Function | `(dim: gpu.Dimension \| Sequence[gpu.Dimension] \| None, dim_idx: ir.Value \| Sequence[ir.Value] \| None) -> ir.Value` |
| `experimental.mosaic.gpu.utils.commit_shared` | Function | `()` |
| `experimental.mosaic.gpu.utils.debug_print` | Function | `(fmt, args, uniform, scope)` |
| `experimental.mosaic.gpu.utils.dialect` | Object | `` |
| `experimental.mosaic.gpu.utils.ds` | Object | `` |
| `experimental.mosaic.gpu.utils.dtype_to_ir_type` | Function | `(dtype: jax.typing.DTypeLike) -> ir.Type` |
| `experimental.mosaic.gpu.utils.dyn_dot` | Function | `(x, y)` |
| `experimental.mosaic.gpu.utils.fence_release_sys` | Function | `()` |
| `experimental.mosaic.gpu.utils.fori` | Function | `(bound, carrys)` |
| `experimental.mosaic.gpu.utils.get_arch` | Function | `() -> Arch` |
| `experimental.mosaic.gpu.utils.get_cluster_ptr` | Function | `(ptr: ir.Value, cluster_block: ir.Value, generic: bool)` |
| `experimental.mosaic.gpu.utils.get_cluster_ref` | Function | `(ref: ir.Value, dim: gpu.Dimension, idx: ir.Value, generic: bool)` |
| `experimental.mosaic.gpu.utils.get_contiguous_strides` | Function | `(xs)` |
| `experimental.mosaic.gpu.utils.getelementptr` | Function | `(ptr: ir.Value, indices: Sequence[ir.Value \| int], dtype: ir.Type) -> ir.Value` |
| `experimental.mosaic.gpu.utils.globaltimer` | Function | `(kind: Literal['low', 'high'] \| None)` |
| `experimental.mosaic.gpu.utils.gpu_address_space_to_nvptx` | Function | `(address_space: gpu.AddressSpace) -> int` |
| `experimental.mosaic.gpu.utils.is_known_divisible` | Function | `(value: ir.Value, divisor: int, max_depth) -> bool` |
| `experimental.mosaic.gpu.utils.is_memref_transposed` | Function | `(ref: ir.MemRefType) -> bool` |
| `experimental.mosaic.gpu.utils.is_signed` | Function | `(dtype: jax.typing.DTypeLike) -> bool \| None` |
| `experimental.mosaic.gpu.utils.is_smem_ref` | Function | `(ref: ir.Value \| ir.Type) -> bool` |
| `experimental.mosaic.gpu.utils.is_tmem_ref` | Function | `(ref: ir.Value \| ir.Type) -> bool` |
| `experimental.mosaic.gpu.utils.jaxlib_extension_version` | Object | `` |
| `experimental.mosaic.gpu.utils.jnp` | Object | `` |
| `experimental.mosaic.gpu.utils.memref_fold` | Function | `(ref: ir.Value \| MultimemRef, dim, fold_rank) -> ir.Value \| MultimemRef` |
| `experimental.mosaic.gpu.utils.memref_ptr` | Function | `(memref_arg, memory_space)` |
| `experimental.mosaic.gpu.utils.memref_reshape` | Function | `(ref: ir.Value \| MultimemRef, shape: tuple[int, ...]) -> ir.Value \| MultimemRef` |
| `experimental.mosaic.gpu.utils.memref_slice` | Function | `(ref: ir.Value, index) -> ir.Value` |
| `experimental.mosaic.gpu.utils.memref_transpose` | Function | `(ref: ir.Value, permutation: Sequence[int]) -> ir.Value` |
| `experimental.mosaic.gpu.utils.memref_unfold` | Function | `(ref: ir.Value, dim, factors) -> ir.Value` |
| `experimental.mosaic.gpu.utils.memref_unsqueeze` | Function | `(ref: ir.Value, dim) -> ir.Value` |
| `experimental.mosaic.gpu.utils.mlir` | Object | `` |
| `experimental.mosaic.gpu.utils.multimem_load_reduce` | Function | `(ty: ir.Type, ptr: ir.Value, reduction: MultimemReductionOp, is_signed: bool \| None)` |
| `experimental.mosaic.gpu.utils.multimem_store` | Function | `(ptr: ir.Value, value: ir.Value)` |
| `experimental.mosaic.gpu.utils.nanosleep` | Function | `(nanos: ir.Value)` |
| `experimental.mosaic.gpu.utils.nvvm_mbarrier_arrive_expect_tx` | Function | `(barrier: ir.Value, expect_tx: ir.Value, predicate: ir.Value \| None)` |
| `experimental.mosaic.gpu.utils.pack_array` | Function | `(values)` |
| `experimental.mosaic.gpu.utils.parse_indices` | Function | `(index, shape: tuple[int, ...], check_oob: bool) -> tuple[list[ir.Value \| int], list[int], list[bool]]` |
| `experimental.mosaic.gpu.utils.prmt` | Function | `(high: ir.Value, low: ir.Value, permutation: ir.Value)` |
| `experimental.mosaic.gpu.utils.ptr_as_memref` | Function | `(ptr, memref_ty: ir.MemRefType, ptr_memory_space: int \| None)` |
| `experimental.mosaic.gpu.utils.query_cluster_cancel` | Function | `(result_ref) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]` |
| `experimental.mosaic.gpu.utils.reduce_shape` | Function | `(shape: Sequence[int], axes: Sequence[int], keep_dims: bool) -> tuple[int, ...]` |
| `experimental.mosaic.gpu.utils.redux` | Function | `(x: ir.Value, mask: ir.Value, kind: ReductionKind)` |
| `experimental.mosaic.gpu.utils.shfl_bfly` | Function | `(x: ir.Value, distance: int \| ir.Value)` |
| `experimental.mosaic.gpu.utils.single_thread` | Function | `(scope: ThreadSubset)` |
| `experimental.mosaic.gpu.utils.single_thread_predicate` | Function | `(scope: ThreadSubset)` |
| `experimental.mosaic.gpu.utils.smem` | Function | `() -> ir.Attribute` |
| `experimental.mosaic.gpu.utils.smid` | Function | `()` |
| `experimental.mosaic.gpu.utils.thread_idx` | Object | `` |
| `experimental.mosaic.gpu.utils.tile_shape` | Function | `(shape, tiling)` |
| `experimental.mosaic.gpu.utils.tmem` | Function | `() -> ir.Attribute` |
| `experimental.mosaic.gpu.utils.try_cluster_cancel` | Function | `(result_ref, barrier: BarrierRef, predicate: ir.Value \| None)` |
| `experimental.mosaic.gpu.utils.vector_concat` | Function | `(vectors: Sequence[ir.Value]) -> ir.Value` |
| `experimental.mosaic.gpu.utils.vector_slice` | Function | `(v: ir.Value, s: slice)` |
| `experimental.mosaic.gpu.utils.warp_barrier` | Function | `()` |
| `experimental.mosaic.gpu.utils.warp_idx` | Function | `(sync)` |
| `experimental.mosaic.gpu.utils.warp_tree_reduce` | Function | `(value, op, group_size)` |
| `experimental.mosaic.gpu.utils.warpgroup_barrier` | Function | `()` |
| `experimental.mosaic.gpu.utils.warpgroup_idx` | Function | `(sync)` |
| `experimental.mosaic.gpu.utils.when` | Function | `(cond)` |
| `experimental.mosaic.gpu.warp_idx` | Function | `(sync)` |
| `experimental.mosaic.gpu.warpgroup_barrier` | Function | `()` |
| `experimental.mosaic.gpu.warpgroup_idx` | Function | `(sync)` |
| `experimental.mosaic.gpu.wgmma.WGMMAAccumulator` | Class | `(_value: fa.FragmentedArray, _original_layout: fa.FragmentedLayout, _sync: bool)` |
| `experimental.mosaic.gpu.wgmma.bytewidth` | Object | `` |
| `experimental.mosaic.gpu.wgmma.c` | Object | `` |
| `experimental.mosaic.gpu.wgmma.fa` | Object | `` |
| `experimental.mosaic.gpu.wgmma.mma_utils` | Object | `` |
| `experimental.mosaic.gpu.wgmma.utils` | Object | `` |
| `experimental.mosaic.gpu.wgmma.wgmma` | Function | `(acc: WGMMAAccumulator, a: fa.FragmentedArray \| ir.Value, b: ir.Value, swizzle: int)` |
| `experimental.mosaic.gpu.wgmma.wgmma_fence` | Function | `(array: fa.FragmentedArray) -> fa.FragmentedArray` |
| `experimental.mosaic.gpu.wgmma.wgmma_m64` | Function | `(acc: np.ndarray, a, b_descriptor: ir.Value, a_transpose: bool \| None, b_transpose: bool, a_k_stride: int \| None, b_k_stride: int, n: int, swizzle: int, element_type: ir.Type)` |
| `experimental.mosaic.gpu.when` | Function | `(cond)` |
| `experimental.mosaic.lower_module_to_custom_call` | Function | `(ctx: mlir.LoweringRuleContext, in_nodes: ir.Value, module: ir.Module, out_type: Any, kernel_name: str, cost_estimate: CostEstimate \| None, vmem_limit_bytes: int \| None, flags: dict[str, bool \| int \| float] \| None, allow_input_fusion: Sequence[bool] \| None, input_output_aliases: tuple[tuple[int, int], ...], internal_scratch_in_bytes: int \| None, collective_id: int \| None, has_side_effects: bool \| TpuSideEffectType, serialization_format: int \| None, output_memory_spaces: tuple[MemorySpace \| None, ...] \| None, disable_bounds_checks: bool, disable_semaphore_checks: bool, input_memory_spaces: tuple[MemorySpace \| None, ...] \| None, metadata: Any \| None, skip_device_barrier: bool, allow_collective_id_without_custom_barrier: bool, shape_invariant_numerics: bool, needs_layout_passes: bool \| None, tiling: Tiling \| None) -> Sequence[ir.Value]` |
| `experimental.multihost_utils.P` | Object | `` |
| `experimental.multihost_utils.ad` | Object | `` |
| `experimental.multihost_utils.array` | Object | `` |
| `experimental.multihost_utils.assert_equal` | Function | `(in_tree, fail_message: str)` |
| `experimental.multihost_utils.batching` | Object | `` |
| `experimental.multihost_utils.broadcast_one_to_all` | Function | `(in_tree: Any, is_source: bool \| None) -> Any` |
| `experimental.multihost_utils.core` | Object | `` |
| `experimental.multihost_utils.distributed` | Object | `` |
| `experimental.multihost_utils.dtypes` | Object | `` |
| `experimental.multihost_utils.global_array_to_host_local_array` | Function | `(global_inputs: Any, global_mesh: jax.sharding.Mesh, pspecs: Any)` |
| `experimental.multihost_utils.global_array_to_host_local_array_impl` | Function | `(arr: Any, global_mesh: jax.sharding.Mesh, pspec: Any)` |
| `experimental.multihost_utils.global_array_to_host_local_array_p` | Object | `` |
| `experimental.multihost_utils.gtl_abstract_eval` | Function | `(arr, global_mesh, pspec)` |
| `experimental.multihost_utils.host_local_array_to_global_array` | Function | `(local_inputs: Any, global_mesh: jax.sharding.Mesh, pspecs: Any)` |
| `experimental.multihost_utils.host_local_array_to_global_array_impl` | Function | `(arr: Any, global_mesh: jax.sharding.Mesh, pspec: Any)` |
| `experimental.multihost_utils.host_local_array_to_global_array_p` | Object | `` |
| `experimental.multihost_utils.jnp` | Object | `` |
| `experimental.multihost_utils.live_devices` | Object | `` |
| `experimental.multihost_utils.ltg_abstract_eval` | Function | `(arr, global_mesh, pspec)` |
| `experimental.multihost_utils.ltg_batcher` | Function | `(insert_axis, axis_data, vals_in, dims_in, global_mesh, pspec)` |
| `experimental.multihost_utils.mlir` | Object | `` |
| `experimental.multihost_utils.pjit_lib` | Object | `` |
| `experimental.multihost_utils.prng` | Object | `` |
| `experimental.multihost_utils.process_allgather` | Function | `(in_tree: Any, tiled: bool) -> Any` |
| `experimental.multihost_utils.pxla` | Object | `` |
| `experimental.multihost_utils.reached_preemption_sync_point` | Function | `(step_id: int) -> bool` |
| `experimental.multihost_utils.safe_zip` | Function | `(args)` |
| `experimental.multihost_utils.sharding_impls` | Object | `` |
| `experimental.multihost_utils.sync_global_devices` | Function | `(name: str)` |
| `experimental.multihost_utils.tree_flatten` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> tuple[list[Leaf], PyTreeDef]` |
| `experimental.multihost_utils.tree_unflatten` | Function | `(treedef: PyTreeDef, leaves: Iterable[Leaf]) -> Any` |
| `experimental.multihost_utils.xla_bridge` | Object | `` |
| `experimental.multihost_utils.xla_client` | Object | `` |
| `experimental.ode.abs2` | Function | `(x)` |
| `experimental.ode.api_util` | Object | `` |
| `experimental.ode.core` | Object | `` |
| `experimental.ode.custom_derivatives` | Object | `` |
| `experimental.ode.fit_4th_order_polynomial` | Function | `(y0, y1, y_mid, dy0, dy1, dt)` |
| `experimental.ode.initial_step_size` | Function | `(fun, t0, y0, order, rtol, atol, f0)` |
| `experimental.ode.interp_fit_dopri` | Function | `(y0, y1, k, dt)` |
| `experimental.ode.jnp` | Object | `` |
| `experimental.ode.lax` | Object | `` |
| `experimental.ode.lu` | Object | `` |
| `experimental.ode.map` | Object | `` |
| `experimental.ode.mean_error_ratio` | Function | `(error_estimate, rtol, atol, y0, y1)` |
| `experimental.ode.odeint` | Function | `(func, y0, t, args, rtol, atol, mxstep, hmax)` |
| `experimental.ode.optimal_step_size` | Function | `(last_step, mean_error_ratio, safety, ifactor, dfactor, order)` |
| `experimental.ode.promote_dtypes_inexact` | Function | `(args: ArrayLike) -> list[Array]` |
| `experimental.ode.ravel_first_arg` | Function | `(f: Callable, unravel, debug_info: core.DebugInfo)` |
| `experimental.ode.ravel_first_arg_` | Function | `(f, unravel, y_flat, args)` |
| `experimental.ode.ravel_pytree` | Function | `(pytree: Any) -> tuple[Array, Callable[[Array], Any]]` |
| `experimental.ode.runge_kutta_step` | Function | `(func, y0, f0, t0, dt)` |
| `experimental.ode.safe_map` | Function | `(f, args)` |
| `experimental.ode.safe_zip` | Function | `(args)` |
| `experimental.ode.tree_leaves` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> list[Leaf]` |
| `experimental.ode.tree_map` | Function | `(f: Callable[..., Any], tree: Any, rest: Any, is_leaf: Callable[[Any], bool] \| None) -> Any` |
| `experimental.ode.zip` | Object | `` |
| `experimental.pallas.ANY` | Object | `` |
| `experimental.pallas.BlockDim` | Object | `` |
| `experimental.pallas.BlockSpec` | Class | `(block_shape: Sequence[BlockDim \| int \| None] \| None, index_map: Callable[..., Any] \| None, pipeline_mode: Buffered \| None, memory_space: Any \| None)` |
| `experimental.pallas.Blocked` | Class | `(block_size: int)` |
| `experimental.pallas.BoundedSlice` | Class | `(block_size: int)` |
| `experimental.pallas.Buffered` | Class | `(buffer_count: int, use_lookahead: bool, revisit: Optional[RevisitMode])` |
| `experimental.pallas.CompilerParams` | Class | `(...)` |
| `experimental.pallas.CostEstimate` | Class | `(flops: int, transcendentals: int, bytes_accessed: int, remote_bytes_transferred: int)` |
| `experimental.pallas.DeviceIdType` | Class | `(...)` |
| `experimental.pallas.Element` | Class | `(block_size: int, padding: tuple[int, int])` |
| `experimental.pallas.GridSpec` | Class | `(grid: Grid, in_specs: BlockSpecTree, out_specs: BlockSpecTree, scratch_shapes: ScratchShapeTree)` |
| `experimental.pallas.HOST` | Object | `` |
| `experimental.pallas.MemoryRef` | Class | `(inner_aval: jax_core.AbstractValue, memory_space: Any)` |
| `experimental.pallas.MemorySpace` | Class | `(...)` |
| `experimental.pallas.RevisitMode` | Class | `(...)` |
| `experimental.pallas.Slice` | Class | `(start: int \| Array, size: int \| Array, stride: int)` |
| `experimental.pallas.Squeezed` | Class | `()` |
| `experimental.pallas.broadcast_to` | Function | `(a: Array, shape: tuple[int, ...]) -> Array` |
| `experimental.pallas.cdiv` | Function | `(a: int \| jax_typing.Array, b: int \| jax_typing.Array) -> int \| jax_typing.Array` |
| `experimental.pallas.core_map` | Function | `(mesh, compiler_params: Any \| None, interpret: bool, debug: bool, cost_estimate: CostEstimate \| None, name: str \| None, metadata: dict[str, str] \| None, scratch_shapes: ScratchShapeTree)` |
| `experimental.pallas.debug_check` | Function | `(condition, message)` |
| `experimental.pallas.debug_checks_enabled` | Function | `() -> bool` |
| `experimental.pallas.debug_print` | Function | `(fmt: str, args: jax_typing.ArrayLike)` |
| `experimental.pallas.delay` | Function | `(nanos: int \| jax_typing.Array) -> None` |
| `experimental.pallas.dot` | Function | `(a, b, trans_a: bool, trans_b: bool, allow_tf32: bool \| None, precision)` |
| `experimental.pallas.ds` | Object | `` |
| `experimental.pallas.dslice` | Function | `(start: int \| Array \| None, size: int \| Array \| None, stride: int \| None) -> slice \| Slice` |
| `experimental.pallas.empty` | Object | `` |
| `experimental.pallas.empty_like` | Function | `(x: object)` |
| `experimental.pallas.empty_ref_like` | Function | `(x: object) -> jax_typing.Array` |
| `experimental.pallas.enable_debug_checks` | Object | `` |
| `experimental.pallas.estimate_cost` | Function | `(fun, args, kwargs) -> pallas_core.CostEstimate` |
| `experimental.pallas.fuser.Fusion` | Class | `(func: Callable[A, K], in_type: tuple[tuple[Any, ...], dict[str, Any]], out_type: Any)` |
| `experimental.pallas.fuser.custom_fusion` | Class | `(fun: Callable[..., Any])` |
| `experimental.pallas.fuser.evaluate` | Function | `(f, allow_transpose: bool)` |
| `experimental.pallas.fuser.fuse` | Function | `(f, resolve_fusion_dtypes: bool, debug: bool)` |
| `experimental.pallas.fuser.fusible` | Function | `(f, output_fusion_prefix: Any)` |
| `experimental.pallas.fuser.get_fusion_values` | Function | `(fusion: Callable, args, kwargs) -> tuple[Callable, tuple[typing.SupportsShape, ...], tuple[typing.SupportsShape, ...]]` |
| `experimental.pallas.fuser.make_scalar_prefetch_handler` | Function | `(args)` |
| `experimental.pallas.fuser.pull_block_spec` | Function | `(f: Callable, out_block_specs: pallas_core.BlockSpec \| tuple[pallas_core.BlockSpec, ...], scalar_prefetch_handler: Any \| None, grid_len: int \| None)` |
| `experimental.pallas.fuser.push_block_spec` | Function | `(f: Callable, in_spec_args, in_spec_kwargs)` |
| `experimental.pallas.get_global` | Function | `(what: pallas_core.ScratchShape) -> jax_typing.Array` |
| `experimental.pallas.kernel` | Function | `(body: Callable \| api.NotSpecified, out_shape: object \| None, mesh: pl_core.Mesh, scratch_shapes: pl_core.ScratchShapeTree, compiler_params: pl_core.CompilerParams \| None, interpret: bool, cost_estimate: pl_core.CostEstimate \| None, debug: bool, name: str \| None, metadata: dict[str, str] \| None)` |
| `experimental.pallas.loop` | Function | `(lower: jax_typing.ArrayLike, upper: jax_typing.ArrayLike, step: jax_typing.ArrayLike, unroll: int \| bool \| None) -> Callable[[Callable[[jax_typing.Array], None]], None]` |
| `experimental.pallas.lower_as_mlir` | Function | `(f, args, dynamic_shapes, device, static_argnames, platforms, kwargs) -> mlir.ir.Module` |
| `experimental.pallas.mosaic_gpu.ACC` | Class | `(shape: tuple[int, int], dtype: jnp.dtype, _init: Any)` |
| `experimental.pallas.mosaic_gpu.Barrier` | Class | `(num_arrivals: int, num_barriers: int, orders_tensor_core: bool)` |
| `experimental.pallas.mosaic_gpu.BlockSpec` | Class | `(block_shape: Sequence[BlockDim \| int \| None] \| None, index_map: Callable[..., Any] \| None, pipeline_mode: Buffered \| None, transforms: Sequence[state_types.Transform], delay_release: int, collective_axes: tuple[Hashable, ...], memory_space: Any \| None)` |
| `experimental.pallas.mosaic_gpu.ClusterBarrier` | Class | `(collective_axes: tuple[str \| tuple[str, ...], ...], num_barriers: int, num_arrivals: int, orders_tensor_core: bool)` |
| `experimental.pallas.mosaic_gpu.CompilerParams` | Class | `(approx_math: bool, dimension_semantics: Sequence[DimensionSemantics] \| None, max_concurrent_steps: int, unsafe_no_auto_barriers: bool, reduction_scratch_bytes: int, profile_space: int, profile_dir: str, lowering_semantics: mgpu.core.LoweringSemantics)` |
| `experimental.pallas.mosaic_gpu.CopyPartition` | Class | `(...)` |
| `experimental.pallas.mosaic_gpu.GMEM` | Object | `` |
| `experimental.pallas.mosaic_gpu.Layout` | Class | `(...)` |
| `experimental.pallas.mosaic_gpu.LoweringSemantics` | Class | `(...)` |
| `experimental.pallas.mosaic_gpu.MemorySpace` | Class | `(...)` |
| `experimental.pallas.mosaic_gpu.Mesh` | Class | `(grid: Sequence[int], grid_names: Sequence[str], cluster: Sequence[int], cluster_names: Sequence[str], num_threads: int \| None, thread_name: str \| None, kernel_name: str \| None)` |
| `experimental.pallas.mosaic_gpu.NDLoopInfo` | Class | `(index: tuple[jax.Array, ...], local_index: jax.Array \| int, num_local_steps: jax.Array \| int \| None)` |
| `experimental.pallas.mosaic_gpu.PeerMemRef` | Class | `(device_id: Any, device_id_type: pallas_primitives.DeviceIdType)` |
| `experimental.pallas.mosaic_gpu.PipelinePipeline` | Class | `(...)` |
| `experimental.pallas.mosaic_gpu.REGS` | Object | `` |
| `experimental.pallas.mosaic_gpu.RefType` | Class | `(transforms: tuple[state_types.Transform, ...])` |
| `experimental.pallas.mosaic_gpu.RefUnion` | Class | `(refs: _GPUMemoryRefTree)` |
| `experimental.pallas.mosaic_gpu.Replicated` | Object | `` |
| `experimental.pallas.mosaic_gpu.SMEM` | Object | `` |
| `experimental.pallas.mosaic_gpu.SemaphoreSignal` | Class | `(ref: _Ref, device_id: pallas_primitives.DeviceId \| None, inc: int \| jax.Array)` |
| `experimental.pallas.mosaic_gpu.SemaphoreType` | Class | `(...)` |
| `experimental.pallas.mosaic_gpu.ShapeDtypeStruct` | Class | `(shape: tuple[int, ...], dtype: jnp.dtype, layout: SomeLayout)` |
| `experimental.pallas.mosaic_gpu.SwizzleTransform` | Class | `(swizzle: int)` |
| `experimental.pallas.mosaic_gpu.TMEM` | Object | `` |
| `experimental.pallas.mosaic_gpu.TMEMLayout` | Class | `(...)` |
| `experimental.pallas.mosaic_gpu.Tiling` | Object | `` |
| `experimental.pallas.mosaic_gpu.TilingTransform` | Class | `(tiling: tuple[int, ...])` |
| `experimental.pallas.mosaic_gpu.Transform` | Class | `(...)` |
| `experimental.pallas.mosaic_gpu.TransposeTransform` | Object | `` |
| `experimental.pallas.mosaic_gpu.TryClusterCancelResult` | Function | `(num_buffers: int \| None) -> pallas_core.MemoryRef` |
| `experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef` | Class | `(shape: tuple[int, int], dtype: jnp.dtype, _init: Any)` |
| `experimental.pallas.mosaic_gpu.WarpMesh` | Class | `(axis_name: str)` |
| `experimental.pallas.mosaic_gpu.as_torch_kernel` | Function | `(fn)` |
| `experimental.pallas.mosaic_gpu.async_copy_scales_to_tmem` | Function | `(smem_ref: _Ref, tmem_ref: _Ref, collective_axis: AxisName \| None)` |
| `experimental.pallas.mosaic_gpu.async_copy_smem_to_tmem` | Function | `(smem_ref: _Ref, tmem_ref: _Ref, collective_axis: AxisName \| None)` |
| `experimental.pallas.mosaic_gpu.async_copy_sparse_metadata_to_tmem` | Function | `(smem_ref: _Ref, tmem_ref: _Ref, collective_axis: AxisName \| None)` |
| `experimental.pallas.mosaic_gpu.async_load_tmem` | Function | `(src: _Ref, layout: SomeLayout \| None) -> jax.Array` |
| `experimental.pallas.mosaic_gpu.async_prefetch` | Function | `(ref: _Ref, collective_axes: str \| tuple[str, ...] \| None, leader_tracked: CopyPartition \| None) -> None` |
| `experimental.pallas.mosaic_gpu.async_store_tmem` | Function | `(ref: _Ref, value)` |
| `experimental.pallas.mosaic_gpu.atomic_add` | Function | `(ref: _Ref, val) -> None` |
| `experimental.pallas.mosaic_gpu.atomic_and` | Function | `(ref: _Ref, val) -> None` |
| `experimental.pallas.mosaic_gpu.atomic_max` | Function | `(ref: _Ref, val) -> None` |
| `experimental.pallas.mosaic_gpu.atomic_min` | Function | `(ref: _Ref, val) -> None` |
| `experimental.pallas.mosaic_gpu.atomic_or` | Function | `(ref: _Ref, val) -> None` |
| `experimental.pallas.mosaic_gpu.atomic_xor` | Function | `(ref: _Ref, val) -> None` |
| `experimental.pallas.mosaic_gpu.barrier_arrive` | Function | `(barrier: state.AbstractRef) -> None` |
| `experimental.pallas.mosaic_gpu.barrier_wait` | Function | `(barrier: state.AbstractRef) -> None` |
| `experimental.pallas.mosaic_gpu.broadcasted_iota` | Function | `(dtype: jax.typing.DTypeLike, shape: Sequence[int], dimension: int, layout: SomeLayout \| None) -> jax.Array` |
| `experimental.pallas.mosaic_gpu.commit_smem` | Function | `()` |
| `experimental.pallas.mosaic_gpu.commit_smem_to_gmem_group` | Function | `() -> None` |
| `experimental.pallas.mosaic_gpu.commit_tmem` | Function | `()` |
| `experimental.pallas.mosaic_gpu.copy_gmem_to_smem` | Function | `(src: _Ref, dst: _Ref, barrier: _Ref, collective_axes: str \| tuple[str, ...] \| None, leader_tracked: CopyPartition \| None) -> None` |
| `experimental.pallas.mosaic_gpu.copy_smem_to_gmem` | Function | `(src: _Ref, dst: _Ref, predicate: jax.Array \| None, commit_group: bool, reduction_op: mgpu.TMAReductionOp \| None) -> None` |
| `experimental.pallas.mosaic_gpu.dynamic_scheduling_loop` | Function | `(grid_names, thread_axis, init_carry)` |
| `experimental.pallas.mosaic_gpu.emit_pipeline` | Function | `(body: Callable[..., T], grid: pallas_core.TupleGrid, in_specs: Sequence[pallas_core.BlockSpec], out_specs: Sequence[pallas_core.BlockSpec], max_concurrent_steps: int, init_carry: T \| None)` |
| `experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized` | Function | `(body: Callable[..., None], grid: pallas_core.TupleGrid, memory_registers: int, in_specs: BlockSpecPytree, out_specs: BlockSpecPytree, max_concurrent_steps: int, wg_axis: str, num_compute_wgs: int, pipeline_state: jax.Array \| PipelinePipeline \| None, manual_consumed_barriers: bool, compute_context: ComputeContext \| None, memory_thread_idx: int \| None) -> WarpSpecializedPipeline` |
| `experimental.pallas.mosaic_gpu.find_swizzle` | Function | `(minor_dim_bits: int, what: str)` |
| `experimental.pallas.mosaic_gpu.format_tcgen05_sparse_metadata` | Function | `(meta)` |
| `experimental.pallas.mosaic_gpu.inline_mgpu` | Function | `(arg_types, return_type)` |
| `experimental.pallas.mosaic_gpu.kernel` | Function | `(body: Callable[..., None], out_shape: object, scratch_shapes: pallas_core.ScratchShapeTree, compiler_params: pallas_core.CompilerParams \| None, grid: tuple[int, ...], grid_names: tuple[str, ...], cluster: tuple[int, ...], cluster_names: tuple[str, ...], num_threads: int \| None, thread_name: str \| None, interpret: Any, mesh_kwargs: Any)` |
| `experimental.pallas.mosaic_gpu.layout_cast` | Function | `(x: Any, new_layout: SomeLayout)` |
| `experimental.pallas.mosaic_gpu.load` | Function | `(src: _Ref, idx, layout: SomeLayout \| None, optimized: bool) -> jax.Array` |
| `experimental.pallas.mosaic_gpu.multicast_ref` | Function | `(ref: _Ref, collective_axes: Hashable \| tuple[Hashable, ...]) -> pallas_core.TransformedRef` |
| `experimental.pallas.mosaic_gpu.multimem_load_reduce` | Function | `(ref: _Ref, collective_axes: Hashable \| tuple[Hashable, ...], reduction_op: mgpu.MultimemReductionOp) -> jax.Array` |
| `experimental.pallas.mosaic_gpu.multimem_store` | Function | `(source: jax.Array, ref: _Ref, collective_axes: Hashable \| tuple[Hashable, ...])` |
| `experimental.pallas.mosaic_gpu.nd_loop` | Function | `(grid, collective_axes, tiling, init_carry)` |
| `experimental.pallas.mosaic_gpu.planar_snake` | Function | `(lin_idx: jax.Array, shape: tuple[int, int], minor_dim: int, tile_width: int)` |
| `experimental.pallas.mosaic_gpu.print_layout` | Function | `(fmt: str, x: jax.typing.ArrayLike \| _Ref) -> None` |
| `experimental.pallas.mosaic_gpu.query_cluster_cancel` | Function | `(result_ref: _Ref, grid_names: Sequence[Hashable]) -> tuple[tuple[jax.Array, ...], jax.Array]` |
| `experimental.pallas.mosaic_gpu.remote_ref` | Function | `(ref: _Ref, device_id: jax.typing.ArrayLike, device_id_type: pallas_primitives.DeviceIdType) -> pallas_core.TransformedRef` |
| `experimental.pallas.mosaic_gpu.semaphore_signal_multicast` | Function | `(semaphore, value: int \| jax.Array, collective_axes: Hashable \| tuple[Hashable, ...])` |
| `experimental.pallas.mosaic_gpu.semaphore_signal_parallel` | Function | `(signals: SemaphoreSignal)` |
| `experimental.pallas.mosaic_gpu.set_max_registers` | Function | `(n: int, action: Literal['increase', 'decrease'])` |
| `experimental.pallas.mosaic_gpu.tcgen05_commit_arrive` | Function | `(barrier: _Ref, collective_axis: str \| None)` |
| `experimental.pallas.mosaic_gpu.tcgen05_mma` | Function | `(acc: _Ref, a: _Ref, b: _Ref, barrier: _Ref \| None, a_scale: _Ref \| None, b_scale: _Ref \| None, a_sparse_metadata: _Ref \| None, accumulate: bool \| jax.Array, collective_axis: str \| None)` |
| `experimental.pallas.mosaic_gpu.transform_ref` | Function | `(ref: pallas_core.TransformedRef, transform: state_types.Transform) -> pallas_core.TransformedRef` |
| `experimental.pallas.mosaic_gpu.transpose_ref` | Function | `(ref: pallas_core.TransformedRef \| Any, permutation: tuple[int, ...]) -> pallas_core.TransformedRef` |
| `experimental.pallas.mosaic_gpu.try_cluster_cancel` | Function | `(result_ref: _Ref, barrier: _Ref) -> None` |
| `experimental.pallas.mosaic_gpu.unswizzle_ref` | Function | `(ref, swizzle: int) -> pallas_core.TransformedRef` |
| `experimental.pallas.mosaic_gpu.untile_ref` | Function | `(ref, tiling: tuple[int, ...]) -> pallas_core.TransformedRef` |
| `experimental.pallas.mosaic_gpu.wait_load_tmem` | Function | `()` |
| `experimental.pallas.mosaic_gpu.wait_smem_to_gmem` | Function | `(n: int, wait_read_only: bool) -> None` |
| `experimental.pallas.mosaic_gpu.wgmma` | Function | `(acc: gpu_core.WGMMAAbstractAccumulatorRef, a, b) -> None` |
| `experimental.pallas.mosaic_gpu.wgmma_accumulator_load` | Function | `(acc, wait_n: int \| None)` |
| `experimental.pallas.mosaic_gpu.wgmma_wait` | Function | `(n: int)` |
| `experimental.pallas.multiple_of` | Function | `(x: jax_typing.Array, values: Sequence[int] \| int) -> jax_typing.Array` |
| `experimental.pallas.next_power_of_2` | Function | `(x: int) -> int` |
| `experimental.pallas.no_block_spec` | Object | `` |
| `experimental.pallas.num_programs` | Function | `(axis: int) -> int \| jax_typing.Array` |
| `experimental.pallas.ops.gpu.all_gather_mgpu.all_gather` | Function | `(x: jax.Array, axis_name: Hashable, gather_dimension: int, num_blocks: int \| None, tile_size: int \| None, vec_size: int \| None) -> jax.Array` |
| `experimental.pallas.ops.gpu.all_gather_mgpu.backend` | Object | `` |
| `experimental.pallas.ops.gpu.all_gather_mgpu.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.all_gather_mgpu.jt_multiprocess` | Object | `` |
| `experimental.pallas.ops.gpu.all_gather_mgpu.lax` | Object | `` |
| `experimental.pallas.ops.gpu.all_gather_mgpu.multihost_utils` | Object | `` |
| `experimental.pallas.ops.gpu.all_gather_mgpu.pl` | Object | `` |
| `experimental.pallas.ops.gpu.all_gather_mgpu.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.all_gather_mgpu.profiler` | Object | `` |
| `experimental.pallas.ops.gpu.attention.BlockSizes` | Class | `(block_q: int, block_k: int, block_q_dkv: int \| None, block_kv_dkv: int \| None, block_q_dq: int \| None, block_kv_dq: int \| None)` |
| `experimental.pallas.ops.gpu.attention.DEFAULT_MASK_VALUE` | Object | `` |
| `experimental.pallas.ops.gpu.attention.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.attention.lax` | Object | `` |
| `experimental.pallas.ops.gpu.attention.mha` | Function | `(q, k, v, segment_ids: jnp.ndarray \| None, sm_scale: float, causal: bool, block_sizes: BlockSizes, backward_pass_impl: str, num_warps: int \| None, num_stages: int, grid: tuple[int, ...] \| None, interpret: bool, debug: bool, return_residuals: bool)` |
| `experimental.pallas.ops.gpu.attention.mha_backward_kernel` | Function | `(q_ref, k_ref, v_ref, segment_ids_ref: jax.Array \| None, out_ref, do_scaled_ref, lse_ref, delta_ref, dq_ref, dk_ref, dv_ref, sm_scale: float, causal: bool, block_q_dkv: int, block_kv_dkv: int, block_q_dq: int, block_kv_dq: int, head_dim: int)` |
| `experimental.pallas.ops.gpu.attention.mha_forward_kernel` | Function | `(q_ref, k_ref, v_ref, segment_ids_ref: jax.Array \| None, o_ref: Any, residual_refs: Any, sm_scale: float, causal: bool, block_q: int, block_k: int, head_dim: int)` |
| `experimental.pallas.ops.gpu.attention.mha_reference` | Function | `(q, k, v, segment_ids: jnp.ndarray \| None, sm_scale, causal: bool)` |
| `experimental.pallas.ops.gpu.attention.pl` | Object | `` |
| `experimental.pallas.ops.gpu.attention.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.attention.segment_mask` | Function | `(q_segment_ids: jax.Array, kv_segment_ids: jax.Array)` |
| `experimental.pallas.ops.gpu.attention_mgpu.PipelineCallback` | Class | `(...)` |
| `experimental.pallas.ops.gpu.attention_mgpu.T` | Object | `` |
| `experimental.pallas.ops.gpu.attention_mgpu.TuningConfig` | Class | `(block_q: int, block_kv: int, max_concurrent_steps: int, use_schedule_barrier: bool, causal: bool, compute_wgs_bwd: int, block_q_dkv: int \| None, block_kv_dkv: int \| None, block_q_dq: int \| None, block_kv_dq: int \| None)` |
| `experimental.pallas.ops.gpu.attention_mgpu.attention` | Function | `(q, k, v, config: TuningConfig, save_residuals: bool)` |
| `experimental.pallas.ops.gpu.attention_mgpu.attention_reference` | Function | `(q, k, v, causal, save_residuals)` |
| `experimental.pallas.ops.gpu.attention_mgpu.attention_with_pipeline_emitter` | Function | `(q, k, v, config: TuningConfig, save_residuals)` |
| `experimental.pallas.ops.gpu.attention_mgpu.cuda_versions` | Object | `` |
| `experimental.pallas.ops.gpu.attention_mgpu.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.attention_mgpu.jtu` | Object | `` |
| `experimental.pallas.ops.gpu.attention_mgpu.lax` | Object | `` |
| `experimental.pallas.ops.gpu.attention_mgpu.main` | Function | `(unused_argv)` |
| `experimental.pallas.ops.gpu.attention_mgpu.pl` | Object | `` |
| `experimental.pallas.ops.gpu.attention_mgpu.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.attention_mgpu.profiler` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_matmul_mgpu.MatmulDimension` | Class | `(...)` |
| `experimental.pallas.ops.gpu.blackwell_matmul_mgpu.TuningConfig` | Class | `(tile_m: int, tile_n: int, tile_k: int, max_concurrent_steps: int, collective: bool, epilogue_tile_n: int, grid_minor_dim: MatmulDimension, grid_tile_width: int)` |
| `experimental.pallas.ops.gpu.blackwell_matmul_mgpu.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_matmul_mgpu.jtu` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_matmul_mgpu.lax` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_matmul_mgpu.main` | Function | `(_) -> None` |
| `experimental.pallas.ops.gpu.blackwell_matmul_mgpu.matmul_kernel` | Function | `(a, b, config: TuningConfig)` |
| `experimental.pallas.ops.gpu.blackwell_matmul_mgpu.pl` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_matmul_mgpu.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_matmul_mgpu.profiler` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.TuningConfig` | Class | `(tile_m: int, tile_n: int, tile_k: int, max_concurrent_steps: int, collective: bool, grid_tile_width: int, grid_minor_dim: blackwell_matmul_mgpu.MatmulDimension, epilogue_tile_n: int)` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.blackwell_matmul_mgpu` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.do_matmul` | Function | `(a_gmem, b_gmem, out_gmem, grid_indices: Sequence[jax.Array], wg_axis: str, collective_axes: tuple[str, ...], local_index: jax.Array, config: TuningConfig, group_info: ragged_dot_mgpu.GroupInfo, a_smem, b_smem, acc_tmem, acc_smem, a_tma_barrier, b_tma_barrier, store_done_barrier, mma_done_barrier, consumed_barrier)` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.jtu` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.lax` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.main` | Function | `(_) -> None` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.pl` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.profiler` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.ragged_dot_kernel` | Function | `(a, b, group_sizes, config: TuningConfig)` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.ragged_dot_mgpu` | Object | `` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.ragged_dot_reference` | Function | `(a, b, g)` |
| `experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu.sample_group_sizes` | Function | `(key: jax.Array, num_groups: int, num_elements: int, alpha: float)` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.MatmulDimension` | Object | `` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.TuningConfig` | Object | `` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.all_gather_lhs_matmul` | Function | `(lhs: jax.Array, rhs: jax.Array, axis_name, config: hopper_matmul_mgpu.TuningConfig, dtype: jnp.dtype) -> jax.Array` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.hopper_matmul_mgpu` | Object | `` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.is_nvshmem_used` | Function | `() -> bool` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.jax_config` | Object | `` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.jt_multiprocess` | Object | `` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.lax` | Object | `` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.multihost_utils` | Object | `` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.pl` | Object | `` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.collective_matmul_mgpu.profiler` | Object | `` |
| `experimental.pallas.ops.gpu.decode_attention.attn_forward_kernel` | Function | `(q_ref, k_ref, v_ref, start_idx_ref, kv_seq_len_ref, o_ref: Any, residual_refs: Any, sm_scale: float, block_k: int, block_h: int, num_heads: int)` |
| `experimental.pallas.ops.gpu.decode_attention.decode_attn_unbatched` | Function | `(q, k, v, start_idx, kv_seq_len, sm_scale: float, block_h: int, block_k: int, k_splits: int, num_warps: int \| None, num_stages: int, grid: tuple[int, ...] \| None, interpret: bool, debug: bool, return_residuals: bool, normalize_output: bool)` |
| `experimental.pallas.ops.gpu.decode_attention.gqa` | Function | `(q, k, v, start_idx, kv_seq_len, sm_scale: float \| None, block_h: int, block_k: int, k_splits: int, num_warps: int \| None, num_stages: int, grid: tuple[int, ...] \| None, interpret: bool, debug: bool, return_residuals: bool, normalize_output: bool)` |
| `experimental.pallas.ops.gpu.decode_attention.gqa_reference` | Function | `(q, k, v, start_idx, kv_seq_len, sm_scale, return_residuals, normalize_output)` |
| `experimental.pallas.ops.gpu.decode_attention.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.decode_attention.lax` | Object | `` |
| `experimental.pallas.ops.gpu.decode_attention.mha_reference` | Function | `(q, k, v, start_idx, kv_seq_len, sm_scale)` |
| `experimental.pallas.ops.gpu.decode_attention.mqa` | Function | `(q, k, v, start_idx, kv_seq_len, sm_scale: float \| None, block_h: int, block_k: int, k_splits: int, num_warps: int \| None, num_stages: int, grid: tuple[int, ...] \| None, interpret: bool, debug: bool, return_residuals: bool, normalize_output: bool)` |
| `experimental.pallas.ops.gpu.decode_attention.mqa_reference` | Function | `(q, k, v, start_idx, kv_seq_len, sm_scale, return_residuals, normalize_output)` |
| `experimental.pallas.ops.gpu.decode_attention.pl` | Object | `` |
| `experimental.pallas.ops.gpu.decode_attention.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.MatmulDimension` | Class | `(...)` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.TuningConfig` | Class | `(tile_m: int, tile_n: int, tile_k: int, max_concurrent_steps: int, epi_tile_n: int \| None, epi_tile_m: int \| None, grid_minor_dim: MatmulDimension, grid_tile_width: int, wg_dimension: MatmulDimension, cluster_dimension: None \| MatmulDimension)` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.backend` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.jtu` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.kernel` | Function | `(a_gmem, b_gmem, c_gmem, out_gmem, config: TuningConfig, pipeline_callback, delay_release)` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.lax` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.main` | Function | `(_) -> None` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.matmul` | Function | `(a, b, c, config: TuningConfig)` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.pl` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_matmul_mgpu.profiler` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.MatmulDimension` | Class | `(...)` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.TuningConfig` | Class | `(tile_m: int, tile_n: int, tile_k: int, max_concurrent_steps: int, epi_tile_n: int \| None, epi_tile_m: int \| None, grid_minor_dim: MatmulDimension, grid_tile_width: int, wg_dimension: MatmulDimension, cluster_dimension: None \| MatmulDimension)` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.backend` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.dtypes` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.jtu` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.lax` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.main` | Function | `(_) -> None` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.mixed_matmul_kernel` | Function | `(a: jax.Array, b: jax.Array, out_dtype: jnp.dtype, config: TuningConfig) -> jax.Array` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.pl` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.profiler` | Object | `` |
| `experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu.reference` | Function | `(a: jax.Array, b: jax.Array, out_dtype: jnp.dtype) -> jax.Array` |
| `experimental.pallas.ops.gpu.layer_norm.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.layer_norm.lax` | Object | `` |
| `experimental.pallas.ops.gpu.layer_norm.layer_norm` | Function | `(x, weight, bias, num_warps: int \| None, num_stages: int \| None, eps: float, backward_pass_impl: str, interpret: bool)` |
| `experimental.pallas.ops.gpu.layer_norm.layer_norm_backward` | Function | `(num_warps: int \| None, num_stages: int \| None, eps: float, backward_pass_impl: str, interpret: bool, res, do)` |
| `experimental.pallas.ops.gpu.layer_norm.layer_norm_backward_kernel_dw_db` | Function | `(x_ref, weight_ref, bias_ref, do_ref, mean_ref, rstd_ref, dw_ref, db_ref, eps: float, block_m: int, block_n: int)` |
| `experimental.pallas.ops.gpu.layer_norm.layer_norm_backward_kernel_dx` | Function | `(x_ref, weight_ref, bias_ref, do_ref, mean_ref, rstd_ref, dx_ref, eps: float, block_size: int)` |
| `experimental.pallas.ops.gpu.layer_norm.layer_norm_forward` | Function | `(x, weight, bias, num_warps: int \| None, num_stages: int \| None, eps: float, backward_pass_impl: str, interpret: bool)` |
| `experimental.pallas.ops.gpu.layer_norm.layer_norm_forward_kernel` | Function | `(x_ref, weight_ref, bias_ref, o_ref, mean_ref, rstd_ref, eps: float, block_size: int)` |
| `experimental.pallas.ops.gpu.layer_norm.layer_norm_reference` | Function | `(x, weight, bias, eps: float)` |
| `experimental.pallas.ops.gpu.layer_norm.pl` | Object | `` |
| `experimental.pallas.ops.gpu.layer_norm.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.paged_attention.DEFAULT_MASK_VALUE` | Object | `` |
| `experimental.pallas.ops.gpu.paged_attention.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.paged_attention.lax` | Object | `` |
| `experimental.pallas.ops.gpu.paged_attention.paged_attention` | Function | `(q: jax.Array, k_pages: jax.Array, v_pages: jax.Array, block_tables: jax.Array, lengths: jax.Array \| None, k_scales_pages: jax.Array \| None, v_scales_pages: jax.Array \| None, block_h: int, pages_per_compute_block: int, k_splits: int, num_warps: int, num_stages: int, interpret: bool, debug: bool, mask_value: float, attn_logits_soft_cap: float \| None) -> jax.Array` |
| `experimental.pallas.ops.gpu.paged_attention.paged_attention_kernel` | Function | `(q_ref, k_pages_ref, k_scales_pages_ref, v_pages_ref, v_scales_pages_ref, block_tables_ref, lengths_ref, o_ref: Any, residual_refs: Any, num_heads: int, pages_per_compute_block: int, mask_value: float, attn_logits_soft_cap: float \| None)` |
| `experimental.pallas.ops.gpu.paged_attention.paged_attention_reference` | Function | `(q: jax.Array, k: jax.Array, v: jax.Array, lengths: jax.Array, mask_value: float, attn_logits_soft_cap: float \| None) -> jax.Array` |
| `experimental.pallas.ops.gpu.paged_attention.paged_attention_unbatched` | Function | `(q: jax.Array, k_pages: jax.Array, v_pages: jax.Array, block_tables: jax.Array, lengths: jax.Array \| None, k_scales_pages: jax.Array \| None, v_scales_pages: jax.Array \| None, block_h: int, pages_per_compute_block: int, k_splits: int, num_warps: int, num_stages: int, interpret: bool, debug: bool, mask_value: float, attn_logits_soft_cap: float \| None) -> jax.Array` |
| `experimental.pallas.ops.gpu.paged_attention.pl` | Object | `` |
| `experimental.pallas.ops.gpu.paged_attention.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.ragged_dot_mgpu.GroupInfo` | Class | `(group_id: jax.Array, block: jax.Array, block_start: jax.Array, actual_start: jax.Array, actual_end: jax.Array, start_within_block: jax.Array, actual_size: jax.Array)` |
| `experimental.pallas.ops.gpu.ragged_dot_mgpu.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.ragged_dot_mgpu.jtu` | Object | `` |
| `experimental.pallas.ops.gpu.ragged_dot_mgpu.lax` | Object | `` |
| `experimental.pallas.ops.gpu.ragged_dot_mgpu.main` | Function | `(unused_argv)` |
| `experimental.pallas.ops.gpu.ragged_dot_mgpu.pl` | Object | `` |
| `experimental.pallas.ops.gpu.ragged_dot_mgpu.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.ragged_dot_mgpu.profiler` | Object | `` |
| `experimental.pallas.ops.gpu.ragged_dot_mgpu.ragged_dot` | Function | `(lhs, rhs, group_sizes, block_m: int, block_n: int, block_k: int, max_concurrent_steps: int, grid_block_n: int, transpose_rhs: bool, load_group_sizes_to_register: bool) -> jax.Array` |
| `experimental.pallas.ops.gpu.ragged_dot_mgpu.random` | Object | `` |
| `experimental.pallas.ops.gpu.reduce_scatter_mgpu.backend` | Object | `` |
| `experimental.pallas.ops.gpu.reduce_scatter_mgpu.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.reduce_scatter_mgpu.jt_multiprocess` | Object | `` |
| `experimental.pallas.ops.gpu.reduce_scatter_mgpu.lax` | Object | `` |
| `experimental.pallas.ops.gpu.reduce_scatter_mgpu.multihost_utils` | Object | `` |
| `experimental.pallas.ops.gpu.reduce_scatter_mgpu.pl` | Object | `` |
| `experimental.pallas.ops.gpu.reduce_scatter_mgpu.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.reduce_scatter_mgpu.profiler` | Object | `` |
| `experimental.pallas.ops.gpu.reduce_scatter_mgpu.reduce_scatter` | Function | `(x: jax.Array, axis_name, scatter_dimension: int \| None, reduction: Literal['add', 'min', 'max', 'and', 'or', 'xor'], num_blocks: int \| None, tile_size: int \| None, vec_size: int \| None) -> jax.Array` |
| `experimental.pallas.ops.gpu.rms_norm.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.rms_norm.lax` | Object | `` |
| `experimental.pallas.ops.gpu.rms_norm.pl` | Object | `` |
| `experimental.pallas.ops.gpu.rms_norm.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.rms_norm.rms_norm` | Function | `(x, weight, bias, num_warps: int \| None, num_stages: int \| None, eps: float, backward_pass_impl: str, interpret: bool)` |
| `experimental.pallas.ops.gpu.rms_norm.rms_norm_backward` | Function | `(num_warps: int \| None, num_stages: int \| None, eps: float, backward_pass_impl: str, interpret: bool, res, do)` |
| `experimental.pallas.ops.gpu.rms_norm.rms_norm_backward_kernel_dw_db` | Function | `(x_ref, weight_ref, bias_ref, do_ref, rstd_ref, dw_ref, db_ref, eps: float, block_m: int, block_n: int)` |
| `experimental.pallas.ops.gpu.rms_norm.rms_norm_backward_kernel_dx` | Function | `(x_ref, weight_ref, bias_ref, do_ref, rstd_ref, dx_ref, eps: float, block_size: int)` |
| `experimental.pallas.ops.gpu.rms_norm.rms_norm_forward` | Function | `(x, weight, bias, num_warps: int \| None, num_stages: int \| None, eps: float, backward_pass_impl: str, interpret: bool)` |
| `experimental.pallas.ops.gpu.rms_norm.rms_norm_forward_kernel` | Function | `(x_ref, weight_ref, bias_ref, o_ref, rstd_ref, eps: float, block_size: int)` |
| `experimental.pallas.ops.gpu.rms_norm.rms_norm_reference` | Function | `(x, weight, bias, eps: float)` |
| `experimental.pallas.ops.gpu.softmax.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.softmax.pl` | Object | `` |
| `experimental.pallas.ops.gpu.softmax.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.softmax.softmax` | Function | `(x: jax.Array, axis: int, num_warps: int, interpret: bool, debug: bool) -> jax.Array` |
| `experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu.jnp` | Object | `` |
| `experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu.jtu` | Object | `` |
| `experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu.lax` | Object | `` |
| `experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu.main` | Function | `(unused_argv)` |
| `experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu.pl` | Object | `` |
| `experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu.plgpu` | Object | `` |
| `experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu.profiler` | Object | `` |
| `experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu.random` | Object | `` |
| `experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu.ref_transposed_ragged_dot` | Function | `(lhs, rhs, group_sizes)` |
| `experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu.transposed_ragged_dot` | Function | `(lhs, rhs, group_sizes, block_m: int, block_n: int, block_k: int, max_concurrent_steps: int, grid_block_n: int) -> jax.Array` |
| `experimental.pallas.ops.tpu.all_gather.P` | Object | `` |
| `experimental.pallas.ops.tpu.all_gather.ag_kernel` | Function | `(x_ref, o_ref, send_sem, recv_sem, axis_name: str, mesh: jax.sharding.Mesh)` |
| `experimental.pallas.ops.tpu.all_gather.all_gather` | Function | `(x, mesh: jax.sharding.Mesh, axis_name: str \| Sequence[str], memory_space: pltpu.MemorySpace)` |
| `experimental.pallas.ops.tpu.all_gather.get_neighbor` | Function | `(idx: jax.Array, mesh: jax.sharding.Mesh, axis_name: str, direction: str) -> tuple[jax.Array, ...]` |
| `experimental.pallas.ops.tpu.all_gather.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.all_gather.lax` | Object | `` |
| `experimental.pallas.ops.tpu.all_gather.pl` | Object | `` |
| `experimental.pallas.ops.tpu.all_gather.pltpu` | Object | `` |
| `experimental.pallas.ops.tpu.all_gather.shard_map` | Object | `` |
| `experimental.pallas.ops.tpu.example_kernel.double` | Function | `(x)` |
| `experimental.pallas.ops.tpu.example_kernel.double_kernel` | Function | `(x_ref, y_ref)` |
| `experimental.pallas.ops.tpu.example_kernel.pl` | Object | `` |
| `experimental.pallas.ops.tpu.flash_attention.BlockSizes` | Class | `(block_q: int, block_k_major: int, block_k: int, block_b: int, block_q_major_dkv: int \| None, block_k_major_dkv: int \| None, block_k_dkv: int \| None, block_q_dkv: int \| None, block_k_major_dq: int \| None, block_k_dq: int \| None, block_q_dq: int \| None)` |
| `experimental.pallas.ops.tpu.flash_attention.DEFAULT_MASK_VALUE` | Object | `` |
| `experimental.pallas.ops.tpu.flash_attention.MIN_BLOCK_SIZE` | Object | `` |
| `experimental.pallas.ops.tpu.flash_attention.NUM_LANES` | Object | `` |
| `experimental.pallas.ops.tpu.flash_attention.NUM_SUBLANES` | Object | `` |
| `experimental.pallas.ops.tpu.flash_attention.SegmentIds` | Class | `(...)` |
| `experimental.pallas.ops.tpu.flash_attention.TRANS_B_DIM_NUMBERS` | Object | `` |
| `experimental.pallas.ops.tpu.flash_attention.below_or_on_diag` | Function | `(r, r_blk_size, c, c_blk_size)` |
| `experimental.pallas.ops.tpu.flash_attention.flash_attention` | Function | `(q, k, v, ab, segment_ids, causal: bool, sm_scale: float, block_sizes: BlockSizes \| None, debug: bool)` |
| `experimental.pallas.ops.tpu.flash_attention.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.flash_attention.lax` | Object | `` |
| `experimental.pallas.ops.tpu.flash_attention.mha_reference` | Function | `(q, k, v, ab, segment_ids: SegmentIds \| None, causal: bool, mask_value: float, sm_scale)` |
| `experimental.pallas.ops.tpu.flash_attention.mha_reference_bwd` | Function | `(q, k, v, ab, segment_ids: SegmentIds \| None, o, l, m, do, causal: bool, mask_value: float, sm_scale: float)` |
| `experimental.pallas.ops.tpu.flash_attention.mha_reference_no_custom_vjp` | Function | `(q, k, v, ab: jax.Array \| None, segment_ids: SegmentIds \| None, causal: bool, mask_value: float, sm_scale: float, save_residuals: bool)` |
| `experimental.pallas.ops.tpu.flash_attention.pl` | Object | `` |
| `experimental.pallas.ops.tpu.flash_attention.pltpu` | Object | `` |
| `experimental.pallas.ops.tpu.matmul.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.matmul.matmul` | Function | `(x: jax.Array, y: jax.Array, block_shape, block_k: int, out_dtype: jnp.dtype \| None, debug: bool) -> jax.Array` |
| `experimental.pallas.ops.tpu.matmul.matmul_kernel` | Function | `(x_tile_ref, y_tile_ref, o_tile_ref, acc_ref)` |
| `experimental.pallas.ops.tpu.matmul.pl` | Object | `` |
| `experimental.pallas.ops.tpu.matmul.pltpu` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.common.assert_is_supported_dtype` | Function | `(dtype: jnp.dtype) -> None` |
| `experimental.pallas.ops.tpu.megablox.common.is_tpu` | Function | `() -> bool` |
| `experimental.pallas.ops.tpu.megablox.common.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.common.select_input_dtype` | Function | `(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.dtype` |
| `experimental.pallas.ops.tpu.megablox.common.supports_bfloat16_matmul` | Function | `() -> bool` |
| `experimental.pallas.ops.tpu.megablox.common.tpu_generation` | Function | `() -> int` |
| `experimental.pallas.ops.tpu.megablox.common.tpu_kind` | Function | `() -> str` |
| `experimental.pallas.ops.tpu.megablox.gmm.GroupMetadata` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.gmm.LutFn` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.gmm.common` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.gmm.gmm` | Function | `(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: jnp.dtype, tiling: tuple[int, int, int] \| LutFn \| None, group_offset: jnp.ndarray \| None, existing_out: jnp.ndarray \| None, transpose_rhs: bool, interpret: bool) -> jnp.ndarray` |
| `experimental.pallas.ops.tpu.megablox.gmm.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.gmm.lax` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.gmm.make_group_metadata` | Function | `(group_sizes: jnp.ndarray, m: int, tm: int, start_group: jnp.ndarray, num_nonzero_groups: int, visit_empty_groups: bool) -> GroupMetadata` |
| `experimental.pallas.ops.tpu.megablox.gmm.partial` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.gmm.pl` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.gmm.pltpu` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.gmm.tgmm` | Function | `(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: jnp.dtype, tiling: tuple[int, int, int] \| LutFn \| None, group_offset: jnp.ndarray \| None, num_actual_groups: int \| None, existing_out: jnp.ndarray \| None, interpret: bool) -> jnp.ndarray` |
| `experimental.pallas.ops.tpu.megablox.ops.backend` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.ops.gmm` | Object | `` |
| `experimental.pallas.ops.tpu.megablox.ops.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.paged_attention` | Function | `(q: jax.Array, k_pages: jax.Array \| quantization_utils.QuantizedTensor, v_pages: jax.Array \| quantization_utils.QuantizedTensor, lengths: jax.Array, page_indices: jax.Array, mask_value: float, attn_logits_soft_cap: float \| None, pages_per_compute_block: int, megacore_mode: str \| None, inline_seq_dim: bool) -> jax.Array` |
| `experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel.DEFAULT_MASK_VALUE` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel.MultiPageAsyncCopyDescriptor` | Class | `(pages_hbm_ref, scales_pages_hbm_ref, vmem_buffer, scales_vmem_buffer, sem, page_indices, page_indices_start_offset, num_pages_to_load, head_index)` |
| `experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel.lax` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel.paged_attention` | Function | `(q: jax.Array, k_pages: jax.Array \| quantization_utils.QuantizedTensor, v_pages: jax.Array \| quantization_utils.QuantizedTensor, lengths: jax.Array, page_indices: jax.Array, mask_value: float, attn_logits_soft_cap: float \| None, pages_per_compute_block: int, megacore_mode: str \| None, inline_seq_dim: bool) -> jax.Array` |
| `experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel.paged_flash_attention_kernel` | Function | `(lengths_ref, page_indices_ref, buffer_index_ref, init_flag_ref, q_ref, k_pages_hbm_ref, k_scales_pages_hbm_ref, v_pages_hbm_ref, v_scales_pages_hbm_ref, o_ref, m_ref, l_ref, k_vmem_buffer, k_scales_vmem_buffer, v_vmem_buffer, v_scales_vmem_buffer, k_sems, v_sems, batch_size: int, pages_per_compute_block: int, pages_per_sequence: int, mask_value: float, attn_logits_soft_cap: float \| None, megacore_mode: str \| None, program_ids)` |
| `experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel.paged_flash_attention_kernel_inline_seq_dim` | Function | `(lengths_ref, page_indices_ref, buffer_index_ref, init_flag_ref, q_ref, k_pages_hbm_ref, k_scales_pages_hbm_ref, v_pages_hbm_ref, v_scales_pages_hbm_ref, o_ref, m_ref, l_ref, k_vmem_buffer, k_scales_vmem_buffer, v_vmem_buffer, v_scales_vmem_buffer, k_sems, v_sems, batch_size: int, pages_per_compute_block: int, pages_per_sequence: int, mask_value: float, attn_logits_soft_cap: float \| None, megacore_mode: str \| None)` |
| `experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel.pl` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel.pltpu` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel.quantization_utils` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.quantization_utils.MAX_INT8` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.quantization_utils.P` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.quantization_utils.QuantizedTensor` | Class | `(...)` |
| `experimental.pallas.ops.tpu.paged_attention.quantization_utils.from_int8` | Function | `(x: jnp.ndarray, h: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray` |
| `experimental.pallas.ops.tpu.paged_attention.quantization_utils.get_quantization_scales` | Function | `(x: jnp.ndarray) -> jnp.ndarray` |
| `experimental.pallas.ops.tpu.paged_attention.quantization_utils.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.quantization_utils.quantize_to_int8` | Function | `(x: jnp.ndarray) -> QuantizedTensor` |
| `experimental.pallas.ops.tpu.paged_attention.quantization_utils.to_int8` | Function | `(x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray` |
| `experimental.pallas.ops.tpu.paged_attention.quantization_utils.unquantize_from_int8` | Function | `(x: QuantizedTensor, dtype: jnp.dtype) -> jnp.ndarray` |
| `experimental.pallas.ops.tpu.paged_attention.util.MASK_VALUE` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.util.grouped_query_attention_reference` | Function | `(queries: jax.Array, k_pages: jax.Array, v_pages: jax.Array, seq_lens: jax.Array, soft_cap: float \| None, debug: bool) -> jax.Array` |
| `experimental.pallas.ops.tpu.paged_attention.util.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.paged_attention.util.quantization_utils` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.dynamic_validate_inputs` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.get_tuned_block_sizes` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.DEFAULT_MASK_VALUE` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.MultiPageAsyncCopyDescriptor` | Class | `(pages_hbm_ref, vmem_buf, sem, page_indices_ref, metadata)` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.dtypes` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.dynamic_validate_inputs` | Function | `(q: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, sm_scale: float \| None, sliding_window: int \| None, soft_cap: float \| None, mask_value: float \| None, k_scale: float \| None, v_scale: float \| None, num_kv_pages_per_block: int \| None, num_queries_per_block: int \| None, vmem_limit_bytes: int \| None)` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.get_dtype_packing` | Function | `(dtype)` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.get_min_heads_per_blk` | Function | `(num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype)` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.get_tuned_block_sizes` | Function | `(q_dtype, kv_dtype, num_q_heads_per_blk, num_kv_heads_per_blk, head_dim, page_size, max_num_batched_tokens, pages_per_seq) -> tuple[int, int]` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.lax` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.pl` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.pltpu` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.ragged_paged_attention` | Function | `(q: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, sm_scale: float, sliding_window: int \| None, soft_cap: float \| None, mask_value: float \| None, k_scale: float \| None, v_scale: float \| None, num_kv_pages_per_block: int \| None, num_queries_per_block: int \| None, vmem_limit_bytes: int \| None)` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.ragged_paged_attention_kernel` | Function | `(kv_lens_ref, page_indices_ref, cu_q_lens_ref, seq_buf_idx_ref, num_seqs_ref, q_ref, kv_pages_hbm_ref, o_ref, kv_bufs, sems, l_ref, m_ref, acc_ref, sm_scale: float, sliding_window: int \| None, soft_cap: float \| None, mask_value: float \| None, k_scale: float \| None, v_scale: float \| None)` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.ref_ragged_paged_attention` | Function | `(queries: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, sm_scale: float, sliding_window: int \| None, soft_cap: float \| None, mask_value: float \| None, k_scale: float \| None, v_scale: float \| None)` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.kernel.static_validate_inputs` | Function | `(q: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, sm_scale: float \| None, sliding_window: int \| None, soft_cap: float \| None, mask_value: float \| None, k_scale: float \| None, v_scale: float \| None, num_kv_pages_per_block: int \| None, num_queries_per_block: int \| None, vmem_limit_bytes: int \| None)` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.ragged_paged_attention` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.ref_ragged_paged_attention` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.static_validate_inputs` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes.MAX_PAGES_PER_SEQ` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes.TUNED_BLOCK_SIZES` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes.get_device_name` | Function | `(num_devices: int \| None)` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes.get_min_page_size` | Function | `(max_model_len, min_page_size)` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes.get_tpu_version` | Function | `() -> int` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes.get_tuned_block_sizes` | Function | `(q_dtype, kv_dtype, num_q_heads_per_blk, num_kv_heads_per_blk, head_dim, page_size, max_num_batched_tokens, pages_per_seq) -> tuple[int, int]` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes.next_power_of_2` | Function | `(x: int)` |
| `experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes.simplify_key` | Function | `(key)` |
| `experimental.pallas.ops.tpu.random.philox.BLOCK_SIZE` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.K_HI_32` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.K_LO_32` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.MUL_A` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.MUL_B` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.Shape` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.mul32_hi_lo` | Function | `(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]` |
| `experimental.pallas.ops.tpu.random.philox.philox_4x32` | Function | `(hi0, lo0, hi1, lo1, k_hi, k_lo, rounds)` |
| `experimental.pallas.ops.tpu.random.philox.philox_4x32_count` | Function | `(key, shape: Shape, offset: typing.ArrayLike, fuse_output: bool)` |
| `experimental.pallas.ops.tpu.random.philox.philox_4x32_kernel` | Function | `(key, shape: Shape, unpadded_shape: Shape, block_size: tuple[int, int], offset: typing.ArrayLike, fuse_output: bool)` |
| `experimental.pallas.ops.tpu.random.philox.philox_fold_in` | Function | `(key, data)` |
| `experimental.pallas.ops.tpu.random.philox.philox_random_bits` | Function | `(key, bit_width: int, shape: Shape)` |
| `experimental.pallas.ops.tpu.random.philox.philox_split` | Function | `(key, shape: Shape)` |
| `experimental.pallas.ops.tpu.random.philox.pl` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.plphilox_prng_impl` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.pltpu` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.prng` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.prng_utils` | Object | `` |
| `experimental.pallas.ops.tpu.random.philox.typing` | Object | `` |
| `experimental.pallas.ops.tpu.random.prng_utils.Shape` | Object | `` |
| `experimental.pallas.ops.tpu.random.prng_utils.blocked_iota` | Function | `(block_shape: Shape, total_shape: Shape)` |
| `experimental.pallas.ops.tpu.random.prng_utils.compute_scalar_offset` | Function | `(iteration_index, total_size: Shape, block_size: Shape)` |
| `experimental.pallas.ops.tpu.random.prng_utils.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.random.prng_utils.lax` | Object | `` |
| `experimental.pallas.ops.tpu.random.prng_utils.round_up` | Object | `` |
| `experimental.pallas.ops.tpu.random.threefry.BLOCK_SIZE` | Object | `` |
| `experimental.pallas.ops.tpu.random.threefry.Shape` | Object | `` |
| `experimental.pallas.ops.tpu.random.threefry.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.random.threefry.pl` | Object | `` |
| `experimental.pallas.ops.tpu.random.threefry.plthreefry_prng_impl` | Object | `` |
| `experimental.pallas.ops.tpu.random.threefry.plthreefry_random_bits` | Function | `(key, bit_width: int, shape: Shape)` |
| `experimental.pallas.ops.tpu.random.threefry.pltpu` | Object | `` |
| `experimental.pallas.ops.tpu.random.threefry.prng` | Object | `` |
| `experimental.pallas.ops.tpu.random.threefry.prng_utils` | Object | `` |
| `experimental.pallas.ops.tpu.random.threefry.threefry_2x32_count` | Function | `(key, shape: Shape, unpadded_shape: Shape, block_size: tuple[int, int])` |
| `experimental.pallas.ops.tpu.splash_attention.BlockSizes` | Class | `(block_q: int, block_kv: int, block_kv_compute: int \| None, block_q_dkv: int \| None, block_kv_dkv: int \| None, block_kv_dkv_compute: int \| None, block_q_dq: int \| None, block_kv_dq: int \| None, use_fused_bwd_kernel: bool, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout)` |
| `experimental.pallas.ops.tpu.splash_attention.CausalMask` | Class | `(shape: tuple[int, int], offset: int, shard_count: int)` |
| `experimental.pallas.ops.tpu.splash_attention.FullMask` | Class | `(_shape: tuple[int, int])` |
| `experimental.pallas.ops.tpu.splash_attention.LocalMask` | Class | `(shape: tuple[int, int], window_size: tuple[int \| None, int \| None], offset: int, shard_count: int)` |
| `experimental.pallas.ops.tpu.splash_attention.Mask` | Class | `(...)` |
| `experimental.pallas.ops.tpu.splash_attention.MultiHeadMask` | Class | `(masks: Sequence[Mask])` |
| `experimental.pallas.ops.tpu.splash_attention.NumpyMask` | Class | `(array: np.ndarray)` |
| `experimental.pallas.ops.tpu.splash_attention.QKVLayout` | Class | `(...)` |
| `experimental.pallas.ops.tpu.splash_attention.SegmentIds` | Class | `(...)` |
| `experimental.pallas.ops.tpu.splash_attention.make_causal_mask` | Function | `(shape: tuple[int, int], offset: int) -> np.ndarray` |
| `experimental.pallas.ops.tpu.splash_attention.make_local_attention_mask` | Function | `(shape: tuple[int, int], window_size: tuple[int \| None, int \| None], offset: int) -> np.ndarray` |
| `experimental.pallas.ops.tpu.splash_attention.make_masked_mha_reference` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.make_masked_mqa_reference` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.make_random_mask` | Function | `(shape: tuple[int, int], sparsity: float, seed: int) -> np.ndarray` |
| `experimental.pallas.ops.tpu.splash_attention.make_splash_mha` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.make_splash_mha_single_device` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.make_splash_mqa` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.make_splash_mqa_single_device` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.BlockSizes` | Class | `(block_q: int, block_kv: int, block_kv_compute: int \| None, block_q_dkv: int \| None, block_kv_dkv: int \| None, block_kv_dkv_compute: int \| None, block_q_dq: int \| None, block_kv_dq: int \| None, use_fused_bwd_kernel: bool, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.DEFAULT_MASK_VALUE` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.MaskFunctionType` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.NN_DIM_NUMBERS` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.NT_DIM_NUMBERS` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.NUM_LANES` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.NUM_SUBLANES` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.QKVLayout` | Class | `(...)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.SegmentIds` | Class | `(...)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.SplashAttentionKernel` | Class | `(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo \| None, dkv_mask_info: mask_info_lib.MaskInfo \| None, kwargs)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.SplashCustomReturnType` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.SplashResidualsType` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.ad_checkpoint` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.attention_reference` | Function | `(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds \| None, sinks: jax.Array \| None, mask_value: float, save_residuals: bool, custom_type: str, attn_logits_soft_cap: float \| None) -> SplashCustomReturnType` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.attention_reference_custom` | Function | `(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds \| None, sinks: jax.Array \| None, mask_value: float, save_residuals: bool, custom_type: str, attn_logits_soft_cap: float \| None)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.flash_attention_kernel` | Function | `(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, sinks_ref, mask_ref, q_sequence_ref, m_scratch_ref, l_scratch_ref, o_scratch_ref, o_ref, logsumexp_ref, mask_value: float, grid_width: int, bq: int, bkv: int, bkv_compute: int, head_dim_v: int, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, attn_logits_soft_cap: float \| None, mask_function: MaskFunctionType \| None)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.from_head_minor` | Function | `(vals: tuple[Any, ...], layout: QKVLayout)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.get_kernel_name` | Function | `(is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str) -> str` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.lax` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.make_attention_reference` | Function | `(mask: mask_lib.Mask \| np.ndarray, is_mqa: bool, backward_impl: str, params: Any) -> Callable` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.make_masked_mha_reference` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.make_masked_mqa_reference` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.make_splash_mha` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.make_splash_mha_single_device` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.make_splash_mqa` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.make_splash_mqa_single_device` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.mask_info_lib` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.mask_lib` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.partial` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.pl` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.pltpu` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.tree_util` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.CausalMask` | Class | `(shape: tuple[int, int], offset: int, shard_count: int)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.ChunkedCausalMask` | Class | `(shape: tuple[int, int], chunk_size: int, shard_count: int)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.FullMask` | Class | `(_shape: tuple[int, int])` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.LocalMask` | Class | `(shape: tuple[int, int], window_size: tuple[int \| None, int \| None], offset: int, shard_count: int)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.LogicalAnd` | Class | `(left: Mask, right: Mask)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.LogicalOr` | Class | `(left: Mask, right: Mask)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.Mask` | Class | `(...)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.MultiHeadMask` | Class | `(masks: Sequence[Mask])` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.NumpyMask` | Class | `(array: np.ndarray)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.make_causal_mask` | Function | `(shape: tuple[int, int], offset: int) -> np.ndarray` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.make_chunk_attention_mask` | Function | `(shape: tuple[int, int], chunk_size: int) -> np.ndarray` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.make_local_attention_mask` | Function | `(shape: tuple[int, int], window_size: tuple[int \| None, int \| None], offset: int) -> np.ndarray` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask.make_random_mask` | Function | `(shape: tuple[int, int], sparsity: float, seed: int) -> np.ndarray` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info.MaskInfo` | Class | `(...)` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info.jax_util` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info.jnp` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info.mask_lib` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info.process_dynamic_mask` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info.process_dynamic_mask_dkv` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info.process_mask` | Object | `` |
| `experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info.process_mask_dkv` | Object | `` |
| `experimental.pallas.pallas_call` | Function | `(kernel: Callable[..., None], out_shape: Any, grid_spec: pallas_core.GridSpec \| None, grid: pallas_core.TupleGrid, in_specs: pallas_core.BlockSpecTree, out_specs: pallas_core.BlockSpecTree, scratch_shapes: pallas_core.ScratchShapeTree, input_output_aliases: Mapping[int, int], debug: bool, interpret: Any, name: str \| None, compiler_params: pallas_core.CompilerParams \| None, cost_estimate: CostEstimate \| None, metadata: dict[str, str] \| None) -> Callable[..., Any]` |
| `experimental.pallas.pallas_call_p` | Object | `` |
| `experimental.pallas.pallas_export_experimental` | Function | `(dynamic_shapes: bool)` |
| `experimental.pallas.program_id` | Function | `(axis: int) -> jax_typing.Array` |
| `experimental.pallas.reciprocal` | Function | `(x, approx, full_range)` |
| `experimental.pallas.run_scoped` | Function | `(f: Callable[..., Any], types: Any, collective_axes: Hashable \| tuple[Hashable, ...], kw_types: Any) -> Any` |
| `experimental.pallas.run_state` | Function | `(f: Callable[..., None]) -> Callable[[T], T]` |
| `experimental.pallas.semaphore` | Class | `(...)` |
| `experimental.pallas.semaphore_read` | Function | `(sem_or_view) -> jax_typing.Array` |
| `experimental.pallas.semaphore_signal` | Function | `(sem_or_view, inc: int \| jax_typing.Array, device_id: DeviceId, device_id_type: DeviceIdType, core_index: int \| jax_typing.Array \| None)` |
| `experimental.pallas.semaphore_wait` | Function | `(sem_or_view, value: int \| jax_typing.Array, decrement: bool)` |
| `experimental.pallas.squeezed` | Object | `` |
| `experimental.pallas.strides_from_shape` | Function | `(shape: tuple[int, ...]) -> tuple[int, ...]` |
| `experimental.pallas.tpu.ANY` | Object | `` |
| `experimental.pallas.tpu.ARBITRARY` | Object | `` |
| `experimental.pallas.tpu.BufferedRef` | Class | `(_spec: pl.BlockSpec, _buffer_type: BufferType, window_ref: ArrayRef \| None, accum_ref: ArrayRef \| None, copy_in_slot: ArrayRef \| None, wait_in_slot: ArrayRef \| None, copy_out_slot: ArrayRef \| None, wait_out_slot: ArrayRef \| None, _copy_in_slot_reg: int \| jax.Array \| None, _wait_in_slot_reg: int \| jax.Array \| None, _copy_out_slot_reg: int \| jax.Array \| None, _wait_out_slot_reg: int \| jax.Array \| None, next_fetch_smem: Sequence[jax.Array] \| None, next_fetch_sreg: Sequence[jax.Array] \| None, sem_recvs: SemaphoreTuple \| None, sem_sends: SemaphoreTuple \| None, swap: ArrayRef \| None, tiling: tpu_info.Tiling \| None)` |
| `experimental.pallas.tpu.BufferedRefBase` | Class | `(...)` |
| `experimental.pallas.tpu.CMEM` | Object | `` |
| `experimental.pallas.tpu.CORE_PARALLEL` | Object | `` |
| `experimental.pallas.tpu.ChipVersion` | Class | `(...)` |
| `experimental.pallas.tpu.CompilerParams` | Class | `(dimension_semantics: Sequence[DimensionSemantics] \| None, allow_input_fusion: Sequence[bool] \| None, vmem_limit_bytes: int \| None, collective_id: int \| None, has_side_effects: bool \| SideEffectType, flags: Mapping[str, Any] \| None, internal_scratch_in_bytes: int \| None, serialization_format: int, kernel_type: CoreType, disable_bounds_checks: bool, disable_semaphore_checks: bool, skip_device_barrier: bool, allow_collective_id_without_custom_barrier: bool, shape_invariant_numerics: bool, use_tc_tiling_on_sc: bool \| None)` |
| `experimental.pallas.tpu.CoreType` | Class | `(...)` |
| `experimental.pallas.tpu.DeviceIdType` | Class | `(...)` |
| `experimental.pallas.tpu.GeneralMemorySpace` | Class | `(...)` |
| `experimental.pallas.tpu.GridDimensionSemantics` | Class | `(...)` |
| `experimental.pallas.tpu.HBM` | Object | `` |
| `experimental.pallas.tpu.HOST` | Object | `` |
| `experimental.pallas.tpu.InterpretParams` | Class | `(detect_races: bool, out_of_bounds_reads: Literal['raise', 'uninitialized'], skip_floating_point_ops: bool, uninitialized_memory: Literal['nan', 'zero'], num_cores_or_threads: int, vector_clock_size: int \| None, logging_mode: LoggingMode \| None, dma_execution_mode: Literal['eager', 'on_wait'], random_seed: int \| None, grid_point_recorder: Callable[[tuple[np.int32, ...], np.int32], None] \| None, allow_hbm_allocation_in_run_scoped: bool)` |
| `experimental.pallas.tpu.KernelType` | Object | `` |
| `experimental.pallas.tpu.LoweringException` | Class | `(...)` |
| `experimental.pallas.tpu.MemorySpace` | Class | `(...)` |
| `experimental.pallas.tpu.PARALLEL` | Object | `` |
| `experimental.pallas.tpu.PrefetchScalarGridSpec` | Class | `(num_scalar_prefetch: int, grid: pallas_core.Grid, in_specs: pallas_core.BlockSpecTree, out_specs: pallas_core.BlockSpecTree, scratch_shapes: pallas_core.ScratchShapeTree)` |
| `experimental.pallas.tpu.SEMAPHORE` | Object | `` |
| `experimental.pallas.tpu.SMEM` | Object | `` |
| `experimental.pallas.tpu.SUBCORE_PARALLEL` | Object | `` |
| `experimental.pallas.tpu.SemaphoreType` | Class | `(...)` |
| `experimental.pallas.tpu.SideEffectType` | Class | `(...)` |
| `experimental.pallas.tpu.Tiling` | Class | `(...)` |
| `experimental.pallas.tpu.TpuInfo` | Class | `(chip_version: ChipVersionBase, generation: int, num_cores: int, num_lanes: int, num_sublanes: int, mxu_column_size: int, vmem_capacity_bytes: int, cmem_capacity_bytes: int, smem_capacity_bytes: int, hbm_capacity_bytes: int, mem_bw_bytes_per_second: int, bf16_ops_per_second: int, int8_ops_per_second: int, fp8_ops_per_second: int, int4_ops_per_second: int, sparse_core: SparseCoreInfo \| None)` |
| `experimental.pallas.tpu.VMEM` | Object | `` |
| `experimental.pallas.tpu.VMEM_SHARED` | Object | `` |
| `experimental.pallas.tpu.async_copy` | Function | `(src_ref, dst_ref, sem, priority: int, add: bool) -> AsyncCopyDescriptor` |
| `experimental.pallas.tpu.async_remote_copy` | Function | `(src_ref, dst_ref, send_sem, recv_sem, device_id, device_id_type: primitives.DeviceIdType) -> AsyncCopyDescriptor` |
| `experimental.pallas.tpu.bitcast` | Function | `(x: jax.Array, ty: DTypeLike) -> jax.Array` |
| `experimental.pallas.tpu.core` | Object | `` |
| `experimental.pallas.tpu.core_barrier` | Function | `(sem, core_axis_name: str)` |
| `experimental.pallas.tpu.create_tensorcore_mesh` | Function | `(axis_name: str, devices: Sequence[jax.Device] \| None, num_cores: int \| None) -> TensorCoreMesh` |
| `experimental.pallas.tpu.delay` | Object | `` |
| `experimental.pallas.tpu.dma_semaphore` | Class | `(...)` |
| `experimental.pallas.tpu.einshape` | Function | `(equation: str, x: jax_typing.Array, assert_is_tile_preserving: bool, sizes: int) -> jax_typing.Array` |
| `experimental.pallas.tpu.emit_pipeline` | Function | `(body, grid: tuple[int \| jax.Array, ...], in_specs, out_specs, tiling: tpu_info.Tiling \| None, should_accumulate_out: bool, core_axis: tuple[int, ...] \| int \| None, core_axis_name: tuple[str, ...] \| str \| None, dimension_semantics: tuple[GridDimensionSemantics, ...] \| None, trace_scopes: bool, no_pipelining: bool, _explicit_indices: bool)` |
| `experimental.pallas.tpu.emit_pipeline_with_allocations` | Function | `(body, grid, in_specs, out_specs, should_accumulate_out)` |
| `experimental.pallas.tpu.force_tpu_interpret_mode` | Function | `(params: InterpretParams)` |
| `experimental.pallas.tpu.get_barrier_semaphore` | Function | `()` |
| `experimental.pallas.tpu.get_pipeline_schedule` | Function | `(schedule) -> Any` |
| `experimental.pallas.tpu.get_tpu_info` | Function | `() -> TpuInfo` |
| `experimental.pallas.tpu.get_tpu_info_for_chip` | Function | `(chip_version: ChipVersion, num_tensor_cores_per_logical_device: int) -> TpuInfo` |
| `experimental.pallas.tpu.is_tpu_device` | Function | `() -> bool` |
| `experimental.pallas.tpu.load` | Function | `(ref: Ref, mask: jax.Array \| None) -> jax.Array` |
| `experimental.pallas.tpu.make_async_copy` | Function | `(src_ref, dst_ref, sem) -> AsyncCopyDescriptor` |
| `experimental.pallas.tpu.make_async_remote_copy` | Function | `(src_ref, dst_ref, send_sem, recv_sem, device_id: MultiDimDeviceId \| IntDeviceId \| None, device_id_type: primitives.DeviceIdType) -> AsyncCopyDescriptor` |
| `experimental.pallas.tpu.make_pipeline_allocations` | Function | `(refs, in_specs, out_specs, tiling: tpu_info.Tiling \| None, should_accumulate_out, needs_swap_ref, grid)` |
| `experimental.pallas.tpu.matmul_acc_lhs` | Function | `(acc_addr: int, lhs: jax.Array, mxu_index: int, load_staged_rhs: int \| None) -> None` |
| `experimental.pallas.tpu.matmul_pop` | Function | `(acc_addr: int, shape: tuple[int, int], dtype: jax.typing.DTypeLike, mxu_index: int)` |
| `experimental.pallas.tpu.matmul_push_rhs` | Function | `(rhs: jax.Array, staging_register: int, mxu_index: int, transpose: bool) -> None` |
| `experimental.pallas.tpu.pack_elementwise` | Function | `(xs, packed_dtype)` |
| `experimental.pallas.tpu.pl_primitives` | Object | `` |
| `experimental.pallas.tpu.prng_random_bits` | Function | `(shape)` |
| `experimental.pallas.tpu.prng_seed` | Function | `(seeds: int \| jax.Array) -> None` |
| `experimental.pallas.tpu.repeat` | Object | `` |
| `experimental.pallas.tpu.reset_tpu_interpret_mode_state` | Function | `()` |
| `experimental.pallas.tpu.roll` | Function | `(x: jax.Array, shift: jax.Array \| int, axis: int, stride: int \| None, stride_axis: int \| None) -> jax.Array` |
| `experimental.pallas.tpu.run_on_first_core` | Function | `(core_axis_name: str)` |
| `experimental.pallas.tpu.sample_block` | Function | `(sampler_fn: SampleFnType, global_key: jax.Array, block_size: Shape, tile_size: Shape, total_size: Shape, block_index: tuple[typing.ArrayLike, ...] \| None, kwargs) -> jax.Array` |
| `experimental.pallas.tpu.semaphore` | Class | `(...)` |
| `experimental.pallas.tpu.semaphore_read` | Function | `(sem_or_view) -> jax_typing.Array` |
| `experimental.pallas.tpu.semaphore_signal` | Function | `(sem_or_view, inc: int \| jax_typing.Array, device_id: DeviceId, device_id_type: DeviceIdType, core_index: int \| jax_typing.Array \| None)` |
| `experimental.pallas.tpu.semaphore_wait` | Function | `(sem_or_view, value: int \| jax_typing.Array, decrement: bool)` |
| `experimental.pallas.tpu.set_tpu_interpret_mode` | Function | `(params: InterpretParams)` |
| `experimental.pallas.tpu.stateful_bernoulli` | Object | `` |
| `experimental.pallas.tpu.stateful_bits` | Object | `` |
| `experimental.pallas.tpu.stateful_normal` | Object | `` |
| `experimental.pallas.tpu.stateful_uniform` | Object | `` |
| `experimental.pallas.tpu.stochastic_round` | Function | `(x, random_bits, target_dtype)` |
| `experimental.pallas.tpu.store` | Function | `(ref: Ref, val: jax.Array, mask: jax.Array \| None) -> None` |
| `experimental.pallas.tpu.sync_copy` | Function | `(src_ref, dst_ref, add: bool) -> None` |
| `experimental.pallas.tpu.to_pallas_key` | Function | `(key: jax.Array) -> jax.Array` |
| `experimental.pallas.tpu.touch` | Function | `(ref: jax.Array \| state.TransformedRef) -> None` |
| `experimental.pallas.tpu.trace_value` | Function | `(label: str, value: jax.Array) -> None` |
| `experimental.pallas.tpu.unpack_elementwise` | Function | `(x, index, packed_dtype, unpacked_dtype)` |
| `experimental.pallas.tpu.with_memory_space_constraint` | Function | `(x: jax.Array, memory_space: Any) -> jax.Array` |
| `experimental.pallas.tpu_sc.BlockSpec` | Class | `(block_shape: Sequence[BlockDim \| int \| None] \| None, index_map: Callable[..., Any] \| None, pipeline_mode: Buffered \| None, indexed_by: int \| None, indexed_dim: int \| None, memory_space: Any \| None)` |
| `experimental.pallas.tpu_sc.MemoryRef` | Class | `(shape: Sequence[int], dtype: jax.typing.DTypeLike, memory_space: tpu_core.MemorySpace, tiling: Tiling \| None)` |
| `experimental.pallas.tpu_sc.PackFormat` | Class | `(...)` |
| `experimental.pallas.tpu_sc.ScalarSubcoreMesh` | Class | `(axis_name: str, num_cores: int)` |
| `experimental.pallas.tpu_sc.VectorSubcoreMesh` | Class | `(core_axis_name: str, subcore_axis_name: str, num_cores: int, num_subcores: int)` |
| `experimental.pallas.tpu_sc.addupdate` | Function | `(ref: Ref, x: jax.Array) -> None` |
| `experimental.pallas.tpu_sc.addupdate_compressed` | Function | `(ref: Ref, x: jax.Array, mask: jax.Array) -> None` |
| `experimental.pallas.tpu_sc.addupdate_scatter` | Function | `(ref: Ref, indices: Sequence[jax.Array], x: jax.Array, mask: jax.Array \| None) -> None` |
| `experimental.pallas.tpu_sc.all_reduce_ffs` | Function | `(x: jax.Array, reduce: int) -> jax.Array` |
| `experimental.pallas.tpu_sc.all_reduce_population_count` | Function | `(x: jax.Array, reduce: int) -> jax.Array` |
| `experimental.pallas.tpu_sc.bitcast` | Function | `(x: jax.Array, dtype: jax.typing.DTypeLike) -> jax.Array` |
| `experimental.pallas.tpu_sc.cummax` | Function | `(x: jax.Array, mask: jax.Array \| None) -> jax.Array` |
| `experimental.pallas.tpu_sc.cumsum` | Function | `(x: jax.Array, mask: jax.Array \| None) -> jax.Array` |
| `experimental.pallas.tpu_sc.fetch_and_add` | Function | `(x_ref: jax.Ref \| state_types.TransformedRef, value: jax.typing.ArrayLike, subcore_id: jax.typing.ArrayLike) -> jax.Array` |
| `experimental.pallas.tpu_sc.get_sparse_core_info` | Function | `() -> tpu_info.SparseCoreInfo` |
| `experimental.pallas.tpu_sc.load_expanded` | Function | `(ref: Ref, mask: jax.Array) -> jax.Array` |
| `experimental.pallas.tpu_sc.load_gather` | Function | `(ref: Ref, indices: Sequence[jax.Array], mask: jax.Array \| None) -> jax.Array` |
| `experimental.pallas.tpu_sc.pack` | Function | `(a: jax.Array, b: jax.Array, format: PackFormat, preferred_element_type: jax.typing.DTypeLike \| None) -> jax.Array` |
| `experimental.pallas.tpu_sc.parallel_loop` | Function | `(lower, upper, step, unroll, carry)` |
| `experimental.pallas.tpu_sc.scan_count` | Function | `(x: jax.Array, mask: jax.Array \| None) -> tuple[jax.Array, jax.Array]` |
| `experimental.pallas.tpu_sc.sort_key_val` | Function | `(keys: jax.Array, values: jax.Array, mask: jax.Array \| None, descending: bool) -> jax.Array` |
| `experimental.pallas.tpu_sc.store_compressed` | Function | `(ref: Ref, x: jax.Array, mask: jax.Array) -> None` |
| `experimental.pallas.tpu_sc.store_scatter` | Function | `(ref: Ref, indices: Sequence[jax.Array], x: jax.Array, mask: jax.Array \| None) -> None` |
| `experimental.pallas.tpu_sc.subcore_barrier` | Function | `()` |
| `experimental.pallas.tpu_sc.unpack` | Function | `(ab: jax.Array, format: PackFormat, preferred_element_type: jax.typing.DTypeLike \| None) -> tuple[jax.Array, jax.Array]` |
| `experimental.pallas.triton.CompilerParams` | Class | `(num_warps: int \| None, num_stages: int \| None)` |
| `experimental.pallas.triton.approx_tanh` | Function | `(x: jax.Array) -> jax.Array` |
| `experimental.pallas.triton.atomic_add` | Function | `(x_ref_or_view, idx, val, mask: Any \| None)` |
| `experimental.pallas.triton.atomic_and` | Function | `(x_ref_or_view, idx, val, mask: Any \| None)` |
| `experimental.pallas.triton.atomic_cas` | Function | `(ref, cmp, val)` |
| `experimental.pallas.triton.atomic_max` | Function | `(x_ref_or_view, idx, val, mask: Any \| None)` |
| `experimental.pallas.triton.atomic_min` | Function | `(x_ref_or_view, idx, val, mask: Any \| None)` |
| `experimental.pallas.triton.atomic_or` | Function | `(x_ref_or_view, idx, val, mask: Any \| None)` |
| `experimental.pallas.triton.atomic_xchg` | Function | `(x_ref_or_view, idx, val, mask: Any \| None)` |
| `experimental.pallas.triton.atomic_xor` | Function | `(x_ref_or_view, idx, val, mask: Any \| None)` |
| `experimental.pallas.triton.debug_barrier` | Function | `() -> None` |
| `experimental.pallas.triton.elementwise_inline_asm` | Function | `(asm: str, args: Sequence[jax.Array], constraints: str, pack: int, result_shape_dtypes: Sequence[jax.ShapeDtypeStruct]) -> Sequence[jax.Array]` |
| `experimental.pallas.triton.load` | Function | `(ref: Ref, mask: jax.Array \| None, other: jax.typing.ArrayLike \| None, cache_modifier: str \| None, eviction_policy: str \| None, volatile: bool) -> jax.Array` |
| `experimental.pallas.triton.max_contiguous` | Function | `(x, values)` |
| `experimental.pallas.triton.store` | Function | `(ref: Ref, val: jax.Array, mask: jax.Array \| None, eviction_policy: str \| None) -> None` |
| `experimental.pallas.when` | Function | `(condition: bool \| jax_typing.ArrayLike) -> Callable[[Callable[[], None]], Callable[[], None]]` |
| `experimental.pallas.with_scoped` | Function | `(types: Any, collective_axes: Hashable \| tuple[Hashable, ...], kw_types: Any)` |
| `experimental.pjit.AUTO` | Class | `(mesh: mesh_lib.Mesh)` |
| `experimental.pjit.pjit` | Object | `` |
| `experimental.primal_tangent_dtype` | Function | `(primal_dtype, tangent_dtype, name: str \| None) -> ExtendedDType` |
| `experimental.profiler.get_profiled_instructions_proto` | Function | `(tensorboard_dir: str) -> bytes` |
| `experimental.random.StatefulPRNG` | Class | `(_base_key: Array, _counter: core.Ref)` |
| `experimental.random.stateful_rng` | Function | `(seed: typing.ArrayLike \| None, impl: random.PRNGSpecDesc \| None) -> StatefulPRNG` |
| `experimental.rnn.Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `experimental.rnn.PRNGKeyArray` | Object | `` |
| `experimental.rnn.Shape` | Object | `` |
| `experimental.rnn.core` | Object | `` |
| `experimental.rnn.custom_vjp` | Class | `(fun: Callable[..., ReturnValue], nondiff_argnums: Sequence[int], nondiff_argnames: Sequence[str])` |
| `experimental.rnn.dispatch` | Object | `` |
| `experimental.rnn.get_num_params_in_lstm` | Function | `(input_size: int, hidden_size: int, num_layers: int, bidirectional: bool) -> int` |
| `experimental.rnn.gpu_rnn` | Object | `` |
| `experimental.rnn.init_lstm_weight` | Function | `(rng: PRNGKeyArray, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool)` |
| `experimental.rnn.jnp` | Object | `` |
| `experimental.rnn.lax` | Object | `` |
| `experimental.rnn.lstm` | Function | `(x: Array, h_0: Array, c_0: Array, weights: Array, seq_lengths: Array, input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool, precision: lax.PrecisionLike) -> tuple[Array, Array, Array]` |
| `experimental.rnn.lstm_bwd` | Function | `(input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool, precision: lax.PrecisionLike, residuals, gradients)` |
| `experimental.rnn.lstm_fwd` | Function | `(x: Array, h_0: Array, c_0: Array, w: Array, seq_lengths: Array, input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool, precision: lax.PrecisionLike)` |
| `experimental.rnn.lstm_ref` | Function | `(x: Array, h_0: Array, c_0: Array, W_ih: dict[int, Array], W_hh: dict[int, Array], b_ih: dict[int, Array], b_hh: dict[int, Array], seq_lengths: Array, input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool) -> tuple[Array, Array, Array]` |
| `experimental.rnn.mlir` | Object | `` |
| `experimental.rnn.rnn_abstract_eval` | Function | `(x_aval, h_0_aval, c_0_aval, w_aval, seq_lengths_aval, input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool, cudnn_allow_tf32: bool)` |
| `experimental.rnn.rnn_bwd_abstract_eval` | Function | `(dy_aval, dhn_aval, dcn_aval, x_aval, h0_aval, c0_aval, w_aval, y_aval, reserve_space_aval, seq_lengths_aval, input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool, cudnn_allow_tf32: bool)` |
| `experimental.rnn.rnn_bwd_p` | Object | `` |
| `experimental.rnn.rnn_fwd_p` | Object | `` |
| `experimental.rnn.sigmoid` | Object | `` |
| `experimental.rnn.swap_lstm_gates` | Function | `(weights, input_size, hidden_size, num_layers, bidirectional)` |
| `experimental.rnn.tanh` | Object | `` |
| `experimental.rnn.unpack_lstm_weights` | Function | `(weights: Array, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool) -> tuple[dict[int, Array], dict[int, Array], dict[int, Array], dict[int, Array]]` |
| `experimental.roofline.RooflineResult` | Class | `(flops: int, unfused_flops: int, ici_bytes: dict[str, int], ici_latency: dict[str, int], hbm_bytes: int, peak_hbm_bytes: int, unfused_hbm_bytes: int)` |
| `experimental.roofline.RooflineRuleContext` | Class | `(name_stack: source_info_util.NameStack, primitive: core.Primitive, avals_in: Sequence[core.AbstractValue], avals_out: Sequence[core.AbstractValue], jaxpr_eqn_ctx: core.JaxprEqnContext, mesh: Mesh \| AbstractMesh \| None, pin_lhs_in_vmem: bool, pin_rhs_in_vmem: bool)` |
| `experimental.roofline.RooflineShape` | Class | `(shape: tuple[int, ...], dtype: ValidRooflineDtype)` |
| `experimental.roofline.register_roofline` | Function | `(prim: core.Primitive)` |
| `experimental.roofline.register_standard_roofline` | Function | `(prim: core.Primitive)` |
| `experimental.roofline.roofline.AbstractMesh` | Class | `(axis_sizes: tuple[int, ...], axis_names: tuple[str, ...], axis_types: AxisType \| tuple[AxisType, ...] \| None, abstract_device)` |
| `experimental.roofline.roofline.Mesh` | Class | `(...)` |
| `experimental.roofline.roofline.NamedSharding` | Class | `(mesh: mesh_lib.Mesh \| mesh_lib.AbstractMesh, spec: PartitionSpec, memory_kind: str \| None, _logical_device_ids)` |
| `experimental.roofline.roofline.RooflineResult` | Class | `(flops: int, unfused_flops: int, ici_bytes: dict[str, int], ici_latency: dict[str, int], hbm_bytes: int, peak_hbm_bytes: int, unfused_hbm_bytes: int)` |
| `experimental.roofline.roofline.RooflineRuleContext` | Class | `(name_stack: source_info_util.NameStack, primitive: core.Primitive, avals_in: Sequence[core.AbstractValue], avals_out: Sequence[core.AbstractValue], jaxpr_eqn_ctx: core.JaxprEqnContext, mesh: Mesh \| AbstractMesh \| None, pin_lhs_in_vmem: bool, pin_rhs_in_vmem: bool)` |
| `experimental.roofline.roofline.RooflineShape` | Class | `(shape: tuple[int, ...], dtype: ValidRooflineDtype)` |
| `experimental.roofline.roofline.ShapeDtypeStructTree` | Object | `` |
| `experimental.roofline.roofline.Specs` | Object | `` |
| `experimental.roofline.roofline.ValidRooflineDtype` | Object | `` |
| `experimental.roofline.roofline.api` | Object | `` |
| `experimental.roofline.roofline.broadcast_prefix` | Function | `(prefix_tree: Any, full_tree: Any, is_leaf: Callable[[Any], bool] \| None) -> list[Any]` |
| `experimental.roofline.roofline.core` | Object | `` |
| `experimental.roofline.roofline.dce_jaxpr` | Function | `(jaxpr: Jaxpr, used_outputs: Sequence[bool], instantiate: bool \| Sequence[bool]) -> tuple[Jaxpr, list[bool]]` |
| `experimental.roofline.roofline.foreach` | Function | `(f, args)` |
| `experimental.roofline.roofline.jnp` | Object | `` |
| `experimental.roofline.roofline.make_jaxpr` | Function | `(fun: Callable, static_argnums: int \| Sequence[int], axis_env: Sequence[tuple[AxisName, int]] \| None, return_shape: bool) -> Callable[..., core.ClosedJaxpr \| tuple[core.ClosedJaxpr, Any]]` |
| `experimental.roofline.roofline.map` | Object | `` |
| `experimental.roofline.roofline.prng` | Object | `` |
| `experimental.roofline.roofline.register_roofline` | Function | `(prim: core.Primitive)` |
| `experimental.roofline.roofline.register_standard_roofline` | Function | `(prim: core.Primitive)` |
| `experimental.roofline.roofline.roofline` | Function | `(f: Callable, mesh: Mesh \| AbstractMesh \| None, in_specs: Specs \| None, out_specs: Specs \| None, pin_lhs_in_vmem: bool, pin_rhs_in_vmem: bool, vjp: bool, print_jaxpr: bool) -> Callable[..., tuple[ShapeDtypeStructTree, RooflineResult]]` |
| `experimental.roofline.roofline.roofline_and_grad` | Function | `(f: Callable, mesh: Mesh \| AbstractMesh, in_specs: Specs, out_specs: Specs, pin_lhs_in_vmem: bool, pin_rhs_in_vmem: bool, print_jaxpr: bool) -> Callable[..., tuple[ShapeDtypeStructTree, RooflineResult, RooflineResult]]` |
| `experimental.roofline.roofline.shard_map` | Function | `(f: F \| None, out_specs: Specs, in_specs: Specs \| None \| InferFromArgs, mesh: Mesh \| AbstractMesh \| None, axis_names: Set[AxisName], check_vma: bool) -> F \| Callable[[G], G]` |
| `experimental.roofline.roofline.shard_map_p` | Object | `` |
| `experimental.roofline.roofline.source_info_util` | Object | `` |
| `experimental.roofline.roofline.traceback_util` | Object | `` |
| `experimental.roofline.roofline.tree_flatten` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> tuple[list[Leaf], PyTreeDef]` |
| `experimental.roofline.roofline.tree_map` | Function | `(f: Callable[..., Any], tree: Any, rest: Any, is_leaf: Callable[[Any], bool] \| None) -> Any` |
| `experimental.roofline.roofline.tree_unflatten` | Function | `(treedef: PyTreeDef, leaves: Iterable[Leaf]) -> Any` |
| `experimental.roofline.roofline.util` | Object | `` |
| `experimental.roofline.roofline_and_grad` | Function | `(f: Callable, mesh: Mesh \| AbstractMesh, in_specs: Specs, out_specs: Specs, pin_lhs_in_vmem: bool, pin_rhs_in_vmem: bool, print_jaxpr: bool) -> Callable[..., tuple[ShapeDtypeStructTree, RooflineResult, RooflineResult]]` |
| `experimental.roofline.rooflines.ad_checkpoint` | Object | `` |
| `experimental.roofline.rooflines.ad_util` | Object | `` |
| `experimental.roofline.rooflines.ann` | Object | `` |
| `experimental.roofline.rooflines.api` | Object | `` |
| `experimental.roofline.rooflines.callback` | Object | `` |
| `experimental.roofline.rooflines.control_flow` | Object | `` |
| `experimental.roofline.rooflines.convolution` | Object | `` |
| `experimental.roofline.rooflines.core` | Object | `` |
| `experimental.roofline.rooflines.debugging` | Object | `` |
| `experimental.roofline.rooflines.dispatch` | Object | `` |
| `experimental.roofline.rooflines.fft` | Object | `` |
| `experimental.roofline.rooflines.lax` | Object | `` |
| `experimental.roofline.rooflines.lax_parallel` | Object | `` |
| `experimental.roofline.rooflines.linalg` | Object | `` |
| `experimental.roofline.rooflines.ops` | Object | `` |
| `experimental.roofline.rooflines.pjit` | Object | `` |
| `experimental.roofline.rooflines.prng` | Object | `` |
| `experimental.roofline.rooflines.random` | Object | `` |
| `experimental.roofline.rooflines.roofline` | Object | `` |
| `experimental.roofline.rooflines.shard_map` | Object | `` |
| `experimental.roofline.rooflines.slicing` | Object | `` |
| `experimental.roofline.rooflines.special` | Object | `` |
| `experimental.roofline.rooflines.util` | Object | `` |
| `experimental.roofline.rooflines.windowed_reductions` | Object | `` |
| `experimental.scheduling_groups.FlatTree` | Class | `(vals, treedef: PyTreeDef, statics)` |
| `experimental.scheduling_groups.ad` | Object | `` |
| `experimental.scheduling_groups.attr_get` | Function | `(x)` |
| `experimental.scheduling_groups.batching` | Object | `` |
| `experimental.scheduling_groups.core` | Object | `` |
| `experimental.scheduling_groups.dce_jaxpr_xla_metadata_rule` | Function | `(used_outputs: list[bool], eqn: pe.JaxprEqn) -> tuple[list[bool], pe.JaxprEqn \| None]` |
| `experimental.scheduling_groups.debug_info` | Function | `(traced_for: str, fun: Callable, args: Sequence[Any], kwargs: dict[str, Any], static_argnums: Sequence[int], static_argnames: Sequence[str], result_paths_thunk: Callable[[], tuple[str, ...]] \| core.InitialResultPaths, sourceinfo: str \| None, signature: inspect.Signature \| None) -> core.DebugInfo` |
| `experimental.scheduling_groups.dispatch` | Object | `` |
| `experimental.scheduling_groups.func_dialect` | Object | `` |
| `experimental.scheduling_groups.ir` | Object | `` |
| `experimental.scheduling_groups.lu` | Object | `` |
| `experimental.scheduling_groups.mlir` | Object | `` |
| `experimental.scheduling_groups.pe` | Object | `` |
| `experimental.scheduling_groups.safe_map` | Function | `(f, args)` |
| `experimental.scheduling_groups.safe_zip` | Function | `(args)` |
| `experimental.scheduling_groups.scheduling_group` | Function | `(name)` |
| `experimental.scheduling_groups.split_list` | Function | `(args: Sequence[T], ns: Sequence[int]) -> list[list[T]]` |
| `experimental.scheduling_groups.tree_flatten` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> tuple[list[Leaf], PyTreeDef]` |
| `experimental.scheduling_groups.tree_unflatten` | Function | `(treedef: PyTreeDef, leaves: Iterable[Leaf]) -> Any` |
| `experimental.scheduling_groups.unzip2` | Function | `(xys: Iterable[tuple[T1, T2]]) -> tuple[tuple[T1, ...], tuple[T2, ...]]` |
| `experimental.scheduling_groups.weakref_lru_cache` | Function | `(f: Callable[P, R] \| None, maxsize: int \| None, trace_context_in_key: bool, explain: Callable \| None)` |
| `experimental.scheduling_groups.xla_metadata_call` | Function | `(f, meta)` |
| `experimental.scheduling_groups.xla_metadata_call_p` | Object | `` |
| `experimental.serialize_executable.deserialize_and_load` | Function | `(serialized, in_tree, out_tree, backend: str \| xc.Client \| None, execution_devices: Sequence[xc.Device] \| None)` |
| `experimental.serialize_executable.serialize` | Function | `(compiled: jax.stages.Compiled)` |
| `experimental.serialize_executable.xc` | Object | `` |
| `experimental.shard_alike.shard_alike` | Function | `(x, y)` |
| `experimental.shard_map.jshmap` | Object | `` |
| `experimental.shard_map.shard_map` | Function | `(f, mesh, in_specs, out_specs, check_rep)` |
| `experimental.shard_map.traceback_util` | Object | `` |
| `experimental.source_mapper.MappingsGenerator` | Class | `()` |
| `experimental.source_mapper.Pass` | Class | `(name: str, compile_fn: CompileFn, generate_dump: GenerateDumpFn)` |
| `experimental.source_mapper.SourceMap` | Class | `(version: int, sources: Sequence[str], sources_content: Sequence[str], names: Sequence[str], mappings: Mappings)` |
| `experimental.source_mapper.SourceMapDump` | Class | `(source_map: sourcemap.SourceMap, generated_code: str, pass_name: str)` |
| `experimental.source_mapper.all_passes` | Function | `() -> Sequence[Pass]` |
| `experimental.source_mapper.canonicalize_filename` | Function | `(file_name: str)` |
| `experimental.source_mapper.common.CompileFn` | Class | `(...)` |
| `experimental.source_mapper.common.GenerateDumpFn` | Class | `(...)` |
| `experimental.source_mapper.common.Pass` | Class | `(name: str, compile_fn: CompileFn, generate_dump: GenerateDumpFn)` |
| `experimental.source_mapper.common.SourceMapDump` | Class | `(source_map: sourcemap.SourceMap, generated_code: str, pass_name: str)` |
| `experimental.source_mapper.common.all_passes` | Function | `() -> Sequence[Pass]` |
| `experimental.source_mapper.common.compile_with_env` | Function | `(f, f_args, f_kwargs, env_flags, compiler_flags)` |
| `experimental.source_mapper.common.filter_passes` | Function | `(regex: str) -> Sequence[Pass]` |
| `experimental.source_mapper.common.flag_env` | Function | `(kwargs)` |
| `experimental.source_mapper.common.register_pass` | Function | `(pass_: Pass)` |
| `experimental.source_mapper.common.sourcemap` | Object | `` |
| `experimental.source_mapper.compile_with_env` | Function | `(f, f_args, f_kwargs, env_flags, compiler_flags)` |
| `experimental.source_mapper.create_mlir_sourcemap` | Function | `(mlir_dump: str) -> sourcemap.SourceMap` |
| `experimental.source_mapper.filter_passes` | Function | `(regex: str) -> Sequence[Pass]` |
| `experimental.source_mapper.generate_map.SourceMapGeneratorFn` | Class | `(...)` |
| `experimental.source_mapper.generate_map.common` | Object | `` |
| `experimental.source_mapper.generate_map.generate_sourcemaps` | Function | `(f, passes: Sequence[common.Pass], pass_kwargs) -> SourceMapGeneratorFn` |
| `experimental.source_mapper.generate_sourcemaps` | Function | `(f, passes: Sequence[common.Pass], pass_kwargs) -> SourceMapGeneratorFn` |
| `experimental.source_mapper.hlo.HloPass` | Class | `(...)` |
| `experimental.source_mapper.hlo.METADATA_REGEX` | Object | `` |
| `experimental.source_mapper.hlo.common` | Object | `` |
| `experimental.source_mapper.hlo.mlir` | Object | `` |
| `experimental.source_mapper.hlo.optimized_generate_dump` | Function | `(args: tuple[Any, str], xla_compiler_flags: dict[str, Any] \| None, _) -> common.SourceMapDump` |
| `experimental.source_mapper.hlo.original_hlo_generate_dump` | Function | `(args: tuple[Any, str], _) -> common.SourceMapDump` |
| `experimental.source_mapper.hlo.parse_hlo_dump` | Function | `(text: str) -> sourcemap.SourceMap` |
| `experimental.source_mapper.hlo.sourcemap` | Object | `` |
| `experimental.source_mapper.hlo.stable_hlo_generate_dump` | Function | `(args: tuple[Any, str], _) -> common.SourceMapDump` |
| `experimental.source_mapper.hlo.trace_and_lower` | Function | `(work_dir, f, f_args, f_kwargs, _)` |
| `experimental.source_mapper.jaxpr.canonicalize_filename` | Function | `(file_name: str)` |
| `experimental.source_mapper.jaxpr.common` | Object | `` |
| `experimental.source_mapper.jaxpr.compile_jaxpr` | Function | `(work_dir, f, f_args, f_kwargs, _)` |
| `experimental.source_mapper.jaxpr.config` | Object | `` |
| `experimental.source_mapper.jaxpr.core` | Object | `` |
| `experimental.source_mapper.jaxpr.make_jaxpr_dump` | Function | `(jaxpr: core.Jaxpr, _) -> common.SourceMapDump` |
| `experimental.source_mapper.jaxpr.source_info_util` | Object | `` |
| `experimental.source_mapper.jaxpr.sourcemap` | Object | `` |
| `experimental.source_mapper.mlir.CALLSITE_REGEX` | Object | `` |
| `experimental.source_mapper.mlir.LOC_REGEX` | Object | `` |
| `experimental.source_mapper.mlir.Location` | Object | `` |
| `experimental.source_mapper.mlir.Redirect` | Object | `` |
| `experimental.source_mapper.mlir.SCOPED_REGEX` | Object | `` |
| `experimental.source_mapper.mlir.SRC_REGEX` | Object | `` |
| `experimental.source_mapper.mlir.create_mlir_sourcemap` | Function | `(mlir_dump: str) -> sourcemap.SourceMap` |
| `experimental.source_mapper.mlir.parse_mlir_locations` | Function | `(mlir_dump: list[str]) -> tuple[dict[int, sourcemap.Segment], list[str]]` |
| `experimental.source_mapper.mlir.sourcemap` | Object | `` |
| `experimental.source_mapper.register_pass` | Function | `(pass_: Pass)` |
| `experimental.sparse.BCOO` | Class | `(args: tuple[Array, Array], shape: Sequence[int], indices_sorted: bool, unique_indices: bool)` |
| `experimental.sparse.BCSR` | Class | `(args: tuple[Array, Array, Array], shape: Sequence[int], indices_sorted: bool, unique_indices: bool)` |
| `experimental.sparse.COO` | Class | `(args: tuple[Array, Array, Array], shape: Shape, rows_sorted: bool, cols_sorted: bool)` |
| `experimental.sparse.CSC` | Class | `(args, shape)` |
| `experimental.sparse.CSR` | Class | `(args, shape)` |
| `experimental.sparse.CuSparseEfficiencyWarning` | Class | `(...)` |
| `experimental.sparse.JAXSparse` | Class | `(args: tuple[Array, ...], shape: Sequence[int])` |
| `experimental.sparse.SparseEfficiencyError` | Class | `(...)` |
| `experimental.sparse.SparseEfficiencyWarning` | Class | `(...)` |
| `experimental.sparse.SparseTracer` | Class | `(trace: SparseTrace, spvalue)` |
| `experimental.sparse.ad.JAXSparse` | Class | `(args: tuple[Array, ...], shape: Sequence[int])` |
| `experimental.sparse.ad.api_boundary` | Function | `(fun: C, repro_api_name: str \| None, repro_user_func: bool) -> C` |
| `experimental.sparse.ad.core` | Object | `` |
| `experimental.sparse.ad.flatten_fun_for_sparse_ad` | Function | `(fun, argnums: int \| tuple[int, ...], args: tuple[Any, ...])` |
| `experimental.sparse.ad.grad` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux, kwargs) -> Callable` |
| `experimental.sparse.ad.is_sparse` | Object | `` |
| `experimental.sparse.ad.jacfwd` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux: bool, kwargs) -> Callable` |
| `experimental.sparse.ad.jacobian` | Object | `` |
| `experimental.sparse.ad.jacrev` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux: bool, kwargs) -> Callable` |
| `experimental.sparse.ad.safe_zip` | Function | `(args)` |
| `experimental.sparse.ad.split_list` | Function | `(args: Sequence[T], ns: Sequence[int]) -> list[list[T]]` |
| `experimental.sparse.ad.tree_util` | Object | `` |
| `experimental.sparse.ad.value_and_grad` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux, kwargs) -> Callable[..., tuple[Any, Any]]` |
| `experimental.sparse.ad.wraps` | Function | `(wrapped: Callable, namestr: str \| None, docstr: str \| None, kwargs) -> Callable[[T], T]` |
| `experimental.sparse.api.Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `experimental.sparse.api.BCOO` | Class | `(args: tuple[Array, Array], shape: Sequence[int], indices_sorted: bool, unique_indices: bool)` |
| `experimental.sparse.api.BCSR` | Class | `(args: tuple[Array, Array, Array], shape: Sequence[int], indices_sorted: bool, unique_indices: bool)` |
| `experimental.sparse.api.COO` | Class | `(args: tuple[Array, Array, Array], shape: Shape, rows_sorted: bool, cols_sorted: bool)` |
| `experimental.sparse.api.CSC` | Class | `(args, shape)` |
| `experimental.sparse.api.CSR` | Class | `(args, shape)` |
| `experimental.sparse.api.DTypeLike` | Object | `` |
| `experimental.sparse.api.JAXSparse` | Class | `(args: tuple[Array, ...], shape: Sequence[int])` |
| `experimental.sparse.api.ad` | Object | `` |
| `experimental.sparse.api.batching` | Object | `` |
| `experimental.sparse.api.core` | Object | `` |
| `experimental.sparse.api.dtypes` | Object | `` |
| `experimental.sparse.api.empty` | Function | `(shape: Sequence[int], dtype: DTypeLike \| None, index_dtype: DTypeLike, sparse_format: str, kwds) -> JAXSparse` |
| `experimental.sparse.api.eye` | Function | `(N: int, M: int \| None, k: int, dtype: DTypeLike \| None, index_dtype: DTypeLike, sparse_format: str, kwds) -> JAXSparse` |
| `experimental.sparse.api.mlir` | Object | `` |
| `experimental.sparse.api.todense` | Function | `(arr: JAXSparse \| Array) -> Array` |
| `experimental.sparse.api.todense_p` | Object | `` |
| `experimental.sparse.api.tree_util` | Object | `` |
| `experimental.sparse.bcoo.Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `experimental.sparse.bcoo.ArrayLike` | Object | `` |
| `experimental.sparse.bcoo.BCOO` | Class | `(args: tuple[Array, Array], shape: Sequence[int], indices_sorted: bool, unique_indices: bool)` |
| `experimental.sparse.bcoo.BCOOProperties` | Class | `(...)` |
| `experimental.sparse.bcoo.Buffer` | Class | `(...)` |
| `experimental.sparse.bcoo.CUSPARSE_DATA_DTYPES` | Object | `` |
| `experimental.sparse.bcoo.CUSPARSE_INDEX_DTYPES` | Object | `` |
| `experimental.sparse.bcoo.CuSparseEfficiencyWarning` | Class | `(...)` |
| `experimental.sparse.bcoo.DTypeLike` | Object | `` |
| `experimental.sparse.bcoo.DotDimensionNumbers` | Object | `` |
| `experimental.sparse.bcoo.GatherDimensionNumbers` | Class | `(...)` |
| `experimental.sparse.bcoo.GatherScatterMode` | Class | `(...)` |
| `experimental.sparse.bcoo.JAXSparse` | Class | `(args: tuple[Array, ...], shape: Sequence[int])` |
| `experimental.sparse.bcoo.Shape` | Object | `` |
| `experimental.sparse.bcoo.SparseEfficiencyError` | Class | `(...)` |
| `experimental.sparse.bcoo.SparseEfficiencyWarning` | Class | `(...)` |
| `experimental.sparse.bcoo.SparseInfo` | Class | `(...)` |
| `experimental.sparse.bcoo.ad` | Object | `` |
| `experimental.sparse.bcoo.api_util` | Object | `` |
| `experimental.sparse.bcoo.batching` | Object | `` |
| `experimental.sparse.bcoo.bcoo_broadcast_in_dim` | Function | `(mat: BCOO, shape: Shape, broadcast_dimensions: Sequence[int], sharding) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_concatenate` | Function | `(operands: Sequence[BCOO], dimension: int) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_conv_general_dilated` | Function | `(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count, batch_group_count, precision, preferred_element_type, out_sharding) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_dot_general` | Function | `(lhs: BCOO \| Array, rhs: BCOO \| Array, dimension_numbers: DotDimensionNumbers, precision: None, preferred_element_type: None, out_sharding) -> BCOO \| Array` |
| `experimental.sparse.bcoo.bcoo_dot_general_p` | Object | `` |
| `experimental.sparse.bcoo.bcoo_dot_general_sampled` | Function | `(A: Array, B: Array, indices: Array, dimension_numbers: DotDimensionNumbers) -> Array` |
| `experimental.sparse.bcoo.bcoo_dot_general_sampled_p` | Object | `` |
| `experimental.sparse.bcoo.bcoo_dynamic_slice` | Function | `(mat: BCOO, start_indices: Sequence[Any], slice_sizes: Sequence[int]) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_eliminate_zeros` | Function | `(mat: BCOO, nse: int \| None) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_extract` | Function | `(sparr: BCOO, arr: ArrayLike, assume_unique: bool \| None) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_extract_p` | Object | `` |
| `experimental.sparse.bcoo.bcoo_fromdense` | Function | `(mat: Array, nse: int \| None, n_batch: int, n_dense: int, index_dtype: DTypeLike) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_fromdense_p` | Object | `` |
| `experimental.sparse.bcoo.bcoo_gather` | Function | `(operand: BCOO, start_indices: Array, dimension_numbers: GatherDimensionNumbers, slice_sizes: Shape, unique_indices: bool, indices_are_sorted: bool, mode: str \| GatherScatterMode \| None, fill_value) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_multiply_dense` | Function | `(sp_mat: BCOO, v: Array) -> Array` |
| `experimental.sparse.bcoo.bcoo_multiply_sparse` | Function | `(lhs: BCOO, rhs: BCOO) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_reduce_sum` | Function | `(mat: BCOO, axes: Sequence[int]) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_reshape` | Function | `(mat: BCOO, new_sizes: Sequence[int], dimensions: Sequence[int] \| None, sharding) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_rev` | Function | `(operand, dimensions)` |
| `experimental.sparse.bcoo.bcoo_slice` | Function | `(mat: BCOO, start_indices: Sequence[int], limit_indices: Sequence[int], strides: Sequence[int] \| None) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_sort_indices` | Function | `(mat: BCOO) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_sort_indices_p` | Object | `` |
| `experimental.sparse.bcoo.bcoo_spdot_general_p` | Object | `` |
| `experimental.sparse.bcoo.bcoo_squeeze` | Function | `(arr: BCOO, dimensions: Sequence[int]) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_sum_duplicates` | Function | `(mat: BCOO, nse: int \| None) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_sum_duplicates_p` | Object | `` |
| `experimental.sparse.bcoo.bcoo_todense` | Function | `(mat: BCOO) -> Array` |
| `experimental.sparse.bcoo.bcoo_todense_p` | Object | `` |
| `experimental.sparse.bcoo.bcoo_transpose` | Function | `(mat: BCOO, permutation: Sequence[int]) -> BCOO` |
| `experimental.sparse.bcoo.bcoo_transpose_p` | Object | `` |
| `experimental.sparse.bcoo.bcoo_update_layout` | Function | `(mat: BCOO, n_batch: int \| None, n_dense: int \| None, on_inefficient: str \| None) -> BCOO` |
| `experimental.sparse.bcoo.canonicalize_axis` | Function | `(axis: SupportsIndex, num_dims: int) -> int` |
| `experimental.sparse.bcoo.config` | Object | `` |
| `experimental.sparse.bcoo.coo_spmm_p` | Object | `` |
| `experimental.sparse.bcoo.coo_spmv_p` | Object | `` |
| `experimental.sparse.bcoo.core` | Object | `` |
| `experimental.sparse.bcoo.dispatch` | Object | `` |
| `experimental.sparse.bcoo.jnp` | Object | `` |
| `experimental.sparse.bcoo.lax` | Object | `` |
| `experimental.sparse.bcoo.mlir` | Object | `` |
| `experimental.sparse.bcoo.nfold_vmap` | Function | `(fun, N, broadcasted, in_axes)` |
| `experimental.sparse.bcoo.pe` | Object | `` |
| `experimental.sparse.bcoo.ranges_like` | Function | `(xs)` |
| `experimental.sparse.bcoo.remaining` | Function | `(original, removed_lists)` |
| `experimental.sparse.bcoo.safe_zip` | Function | `(args)` |
| `experimental.sparse.bcoo.split_list` | Function | `(args: Sequence[T], ns: Sequence[int]) -> list[list[T]]` |
| `experimental.sparse.bcoo.tree_util` | Object | `` |
| `experimental.sparse.bcoo.unzip2` | Function | `(xys: Iterable[tuple[T1, T2]]) -> tuple[tuple[T1, ...], tuple[T2, ...]]` |
| `experimental.sparse.bcoo.vmap` | Function | `(fun: F, in_axes: int \| None \| Sequence[Any], out_axes: Any, axis_name: AxisName \| None, axis_size: int \| None, spmd_axis_name: AxisName \| tuple[AxisName, ...] \| None, sum_match: bool) -> F` |
| `experimental.sparse.bcoo_broadcast_in_dim` | Function | `(mat: BCOO, shape: Shape, broadcast_dimensions: Sequence[int], sharding) -> BCOO` |
| `experimental.sparse.bcoo_concatenate` | Function | `(operands: Sequence[BCOO], dimension: int) -> BCOO` |
| `experimental.sparse.bcoo_conv_general_dilated` | Function | `(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count, batch_group_count, precision, preferred_element_type, out_sharding) -> BCOO` |
| `experimental.sparse.bcoo_dot_general` | Function | `(lhs: BCOO \| Array, rhs: BCOO \| Array, dimension_numbers: DotDimensionNumbers, precision: None, preferred_element_type: None, out_sharding) -> BCOO \| Array` |
| `experimental.sparse.bcoo_dot_general_p` | Object | `` |
| `experimental.sparse.bcoo_dot_general_sampled` | Function | `(A: Array, B: Array, indices: Array, dimension_numbers: DotDimensionNumbers) -> Array` |
| `experimental.sparse.bcoo_dot_general_sampled_p` | Object | `` |
| `experimental.sparse.bcoo_dynamic_slice` | Function | `(mat: BCOO, start_indices: Sequence[Any], slice_sizes: Sequence[int]) -> BCOO` |
| `experimental.sparse.bcoo_extract` | Function | `(sparr: BCOO, arr: ArrayLike, assume_unique: bool \| None) -> BCOO` |
| `experimental.sparse.bcoo_extract_p` | Object | `` |
| `experimental.sparse.bcoo_fromdense` | Function | `(mat: Array, nse: int \| None, n_batch: int, n_dense: int, index_dtype: DTypeLike) -> BCOO` |
| `experimental.sparse.bcoo_fromdense_p` | Object | `` |
| `experimental.sparse.bcoo_gather` | Function | `(operand: BCOO, start_indices: Array, dimension_numbers: GatherDimensionNumbers, slice_sizes: Shape, unique_indices: bool, indices_are_sorted: bool, mode: str \| GatherScatterMode \| None, fill_value) -> BCOO` |
| `experimental.sparse.bcoo_multiply_dense` | Function | `(sp_mat: BCOO, v: Array) -> Array` |
| `experimental.sparse.bcoo_multiply_sparse` | Function | `(lhs: BCOO, rhs: BCOO) -> BCOO` |
| `experimental.sparse.bcoo_reduce_sum` | Function | `(mat: BCOO, axes: Sequence[int]) -> BCOO` |
| `experimental.sparse.bcoo_reshape` | Function | `(mat: BCOO, new_sizes: Sequence[int], dimensions: Sequence[int] \| None, sharding) -> BCOO` |
| `experimental.sparse.bcoo_rev` | Function | `(operand, dimensions)` |
| `experimental.sparse.bcoo_slice` | Function | `(mat: BCOO, start_indices: Sequence[int], limit_indices: Sequence[int], strides: Sequence[int] \| None) -> BCOO` |
| `experimental.sparse.bcoo_sort_indices` | Function | `(mat: BCOO) -> BCOO` |
| `experimental.sparse.bcoo_sort_indices_p` | Object | `` |
| `experimental.sparse.bcoo_spdot_general_p` | Object | `` |
| `experimental.sparse.bcoo_squeeze` | Function | `(arr: BCOO, dimensions: Sequence[int]) -> BCOO` |
| `experimental.sparse.bcoo_sum_duplicates` | Function | `(mat: BCOO, nse: int \| None) -> BCOO` |
| `experimental.sparse.bcoo_sum_duplicates_p` | Object | `` |
| `experimental.sparse.bcoo_todense` | Function | `(mat: BCOO) -> Array` |
| `experimental.sparse.bcoo_todense_p` | Object | `` |
| `experimental.sparse.bcoo_transpose` | Function | `(mat: BCOO, permutation: Sequence[int]) -> BCOO` |
| `experimental.sparse.bcoo_transpose_p` | Object | `` |
| `experimental.sparse.bcoo_update_layout` | Function | `(mat: BCOO, n_batch: int \| None, n_dense: int \| None, on_inefficient: str \| None) -> BCOO` |
| `experimental.sparse.bcsr.Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `experimental.sparse.bcsr.ArrayLike` | Object | `` |
| `experimental.sparse.bcsr.BCSR` | Class | `(args: tuple[Array, Array, Array], shape: Sequence[int], indices_sorted: bool, unique_indices: bool)` |
| `experimental.sparse.bcsr.BCSRProperties` | Class | `(...)` |
| `experimental.sparse.bcsr.CuSparseEfficiencyWarning` | Class | `(...)` |
| `experimental.sparse.bcsr.DTypeLike` | Object | `` |
| `experimental.sparse.bcsr.DotDimensionNumbers` | Object | `` |
| `experimental.sparse.bcsr.JAXSparse` | Class | `(args: tuple[Array, ...], shape: Sequence[int])` |
| `experimental.sparse.bcsr.Shape` | Object | `` |
| `experimental.sparse.bcsr.SparseEfficiencyWarning` | Class | `(...)` |
| `experimental.sparse.bcsr.SparseInfo` | Class | `(...)` |
| `experimental.sparse.bcsr.ad` | Object | `` |
| `experimental.sparse.bcsr.api_util` | Object | `` |
| `experimental.sparse.bcsr.batching` | Object | `` |
| `experimental.sparse.bcsr.bcoo` | Object | `` |
| `experimental.sparse.bcsr.bcsr_broadcast_in_dim` | Function | `(mat: BCSR, shape: Shape, broadcast_dimensions: Sequence[int], sharding) -> BCSR` |
| `experimental.sparse.bcsr.bcsr_concatenate` | Function | `(operands: Sequence[BCSR], dimension: int) -> BCSR` |
| `experimental.sparse.bcsr.bcsr_dot_general` | Function | `(lhs: BCSR \| Array, rhs: Array, dimension_numbers: DotDimensionNumbers, precision: None, preferred_element_type: None, out_sharding) -> Array` |
| `experimental.sparse.bcsr.bcsr_dot_general_p` | Object | `` |
| `experimental.sparse.bcsr.bcsr_eliminate_zeros` | Function | `(mat: BCSR, nse: int \| None) -> BCSR` |
| `experimental.sparse.bcsr.bcsr_extract` | Function | `(indices: ArrayLike, indptr: ArrayLike, mat: ArrayLike) -> Array` |
| `experimental.sparse.bcsr.bcsr_extract_p` | Object | `` |
| `experimental.sparse.bcsr.bcsr_fromdense` | Function | `(mat: ArrayLike, nse: int \| None, n_batch: int, n_dense: int, index_dtype: DTypeLike) -> BCSR` |
| `experimental.sparse.bcsr.bcsr_fromdense_p` | Object | `` |
| `experimental.sparse.bcsr.bcsr_sum_duplicates` | Function | `(mat: BCSR, nse: int \| None) -> BCSR` |
| `experimental.sparse.bcsr.bcsr_todense` | Function | `(mat: BCSR) -> Array` |
| `experimental.sparse.bcsr.bcsr_todense_p` | Object | `` |
| `experimental.sparse.bcsr.config` | Object | `` |
| `experimental.sparse.bcsr.core` | Object | `` |
| `experimental.sparse.bcsr.dispatch` | Object | `` |
| `experimental.sparse.bcsr.hlo` | Object | `` |
| `experimental.sparse.bcsr.jnp` | Object | `` |
| `experimental.sparse.bcsr.lax` | Object | `` |
| `experimental.sparse.bcsr.mlir` | Object | `` |
| `experimental.sparse.bcsr.nfold_vmap` | Function | `(fun, N, broadcasted, in_axes)` |
| `experimental.sparse.bcsr.safe_zip` | Function | `(args)` |
| `experimental.sparse.bcsr.split_list` | Function | `(args: Sequence[T], ns: Sequence[int]) -> list[list[T]]` |
| `experimental.sparse.bcsr.tree_util` | Object | `` |
| `experimental.sparse.bcsr_broadcast_in_dim` | Function | `(mat: BCSR, shape: Shape, broadcast_dimensions: Sequence[int], sharding) -> BCSR` |
| `experimental.sparse.bcsr_concatenate` | Function | `(operands: Sequence[BCSR], dimension: int) -> BCSR` |
| `experimental.sparse.bcsr_dot_general` | Function | `(lhs: BCSR \| Array, rhs: Array, dimension_numbers: DotDimensionNumbers, precision: None, preferred_element_type: None, out_sharding) -> Array` |
| `experimental.sparse.bcsr_dot_general_p` | Object | `` |
| `experimental.sparse.bcsr_extract` | Function | `(indices: ArrayLike, indptr: ArrayLike, mat: ArrayLike) -> Array` |
| `experimental.sparse.bcsr_extract_p` | Object | `` |
| `experimental.sparse.bcsr_fromdense` | Function | `(mat: ArrayLike, nse: int \| None, n_batch: int, n_dense: int, index_dtype: DTypeLike) -> BCSR` |
| `experimental.sparse.bcsr_fromdense_p` | Object | `` |
| `experimental.sparse.bcsr_sum_duplicates` | Function | `(mat: BCSR, nse: int \| None) -> BCSR` |
| `experimental.sparse.bcsr_todense` | Function | `(mat: BCSR) -> Array` |
| `experimental.sparse.bcsr_todense_p` | Object | `` |
| `experimental.sparse.coo.Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `experimental.sparse.coo.ArrayLike` | Object | `` |
| `experimental.sparse.coo.COO` | Class | `(args: tuple[Array, Array, Array], shape: Shape, rows_sorted: bool, cols_sorted: bool)` |
| `experimental.sparse.coo.COOInfo` | Class | `(...)` |
| `experimental.sparse.coo.CuSparseEfficiencyWarning` | Class | `(...)` |
| `experimental.sparse.coo.DTypeLike` | Object | `` |
| `experimental.sparse.coo.Dtype` | Object | `` |
| `experimental.sparse.coo.JAXSparse` | Class | `(args: tuple[Array, ...], shape: Sequence[int])` |
| `experimental.sparse.coo.Shape` | Object | `` |
| `experimental.sparse.coo.ad` | Object | `` |
| `experimental.sparse.coo.coo_fromdense` | Function | `(mat: Array, nse: int \| None, index_dtype: DTypeLike) -> COO` |
| `experimental.sparse.coo.coo_fromdense_p` | Object | `` |
| `experimental.sparse.coo.coo_matmat` | Function | `(mat: COO, B: Array, transpose: bool) -> Array` |
| `experimental.sparse.coo.coo_matmat_p` | Object | `` |
| `experimental.sparse.coo.coo_matvec` | Function | `(mat: COO, v: Array, transpose: bool) -> Array` |
| `experimental.sparse.coo.coo_matvec_p` | Object | `` |
| `experimental.sparse.coo.coo_todense` | Function | `(mat: COO) -> Array` |
| `experimental.sparse.coo.coo_todense_p` | Object | `` |
| `experimental.sparse.coo.core` | Object | `` |
| `experimental.sparse.coo.dispatch` | Object | `` |
| `experimental.sparse.coo.hlo` | Object | `` |
| `experimental.sparse.coo.jnp` | Object | `` |
| `experimental.sparse.coo.lax` | Object | `` |
| `experimental.sparse.coo.mlir` | Object | `` |
| `experimental.sparse.coo.promote_dtypes` | Function | `(args: ArrayLike) -> list[Array]` |
| `experimental.sparse.coo.tree_util` | Object | `` |
| `experimental.sparse.coo_fromdense` | Function | `(mat: Array, nse: int \| None, index_dtype: DTypeLike) -> COO` |
| `experimental.sparse.coo_fromdense_p` | Object | `` |
| `experimental.sparse.coo_matmat` | Function | `(mat: COO, B: Array, transpose: bool) -> Array` |
| `experimental.sparse.coo_matmat_p` | Object | `` |
| `experimental.sparse.coo_matvec` | Function | `(mat: COO, v: Array, transpose: bool) -> Array` |
| `experimental.sparse.coo_matvec_p` | Object | `` |
| `experimental.sparse.coo_todense` | Function | `(mat: COO) -> Array` |
| `experimental.sparse.coo_todense_p` | Object | `` |
| `experimental.sparse.csr.Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `experimental.sparse.csr.COOInfo` | Class | `(...)` |
| `experimental.sparse.csr.CSC` | Class | `(args, shape)` |
| `experimental.sparse.csr.CSR` | Class | `(args, shape)` |
| `experimental.sparse.csr.CuSparseEfficiencyWarning` | Class | `(...)` |
| `experimental.sparse.csr.DTypeLike` | Object | `` |
| `experimental.sparse.csr.JAXSparse` | Class | `(args: tuple[Array, ...], shape: Sequence[int])` |
| `experimental.sparse.csr.Shape` | Object | `` |
| `experimental.sparse.csr.ad` | Object | `` |
| `experimental.sparse.csr.core` | Object | `` |
| `experimental.sparse.csr.csr_fromdense` | Function | `(mat: Array, nse: int \| None, index_dtype: DTypeLike) -> CSR` |
| `experimental.sparse.csr.csr_fromdense_p` | Object | `` |
| `experimental.sparse.csr.csr_matmat` | Function | `(mat: CSR, B: Array, transpose: bool) -> Array` |
| `experimental.sparse.csr.csr_matmat_p` | Object | `` |
| `experimental.sparse.csr.csr_matvec` | Function | `(mat: CSR, v: Array, transpose: bool) -> Array` |
| `experimental.sparse.csr.csr_matvec_p` | Object | `` |
| `experimental.sparse.csr.csr_todense` | Function | `(mat: CSR) -> Array` |
| `experimental.sparse.csr.csr_todense_p` | Object | `` |
| `experimental.sparse.csr.dispatch` | Object | `` |
| `experimental.sparse.csr.jnp` | Object | `` |
| `experimental.sparse.csr.lax` | Object | `` |
| `experimental.sparse.csr.mlir` | Object | `` |
| `experimental.sparse.csr.promote_dtypes` | Function | `(args: ArrayLike) -> list[Array]` |
| `experimental.sparse.csr.tree_util` | Object | `` |
| `experimental.sparse.csr_fromdense` | Function | `(mat: Array, nse: int \| None, index_dtype: DTypeLike) -> CSR` |
| `experimental.sparse.csr_fromdense_p` | Object | `` |
| `experimental.sparse.csr_matmat` | Function | `(mat: CSR, B: Array, transpose: bool) -> Array` |
| `experimental.sparse.csr_matmat_p` | Object | `` |
| `experimental.sparse.csr_matvec` | Function | `(mat: CSR, v: Array, transpose: bool) -> Array` |
| `experimental.sparse.csr_matvec_p` | Object | `` |
| `experimental.sparse.csr_todense` | Function | `(mat: CSR) -> Array` |
| `experimental.sparse.csr_todense_p` | Object | `` |
| `experimental.sparse.empty` | Function | `(shape: Sequence[int], dtype: DTypeLike \| None, index_dtype: DTypeLike, sparse_format: str, kwds) -> JAXSparse` |
| `experimental.sparse.eye` | Function | `(N: int, M: int \| None, k: int, dtype: DTypeLike \| None, index_dtype: DTypeLike, sparse_format: str, kwds) -> JAXSparse` |
| `experimental.sparse.grad` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux, kwargs) -> Callable` |
| `experimental.sparse.jacfwd` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux: bool, kwargs) -> Callable` |
| `experimental.sparse.jacobian` | Object | `` |
| `experimental.sparse.jacrev` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux: bool, kwargs) -> Callable` |
| `experimental.sparse.linalg.ad` | Object | `` |
| `experimental.sparse.linalg.core` | Object | `` |
| `experimental.sparse.linalg.dispatch` | Object | `` |
| `experimental.sparse.linalg.ffi` | Object | `` |
| `experimental.sparse.linalg.jnp` | Object | `` |
| `experimental.sparse.linalg.lobpcg_standard` | Function | `(A: jax.Array \| Callable[[jax.Array], jax.Array], X: jax.Array, m: int, tol: jax.Array \| float \| None)` |
| `experimental.sparse.linalg.mlir` | Object | `` |
| `experimental.sparse.linalg.sparse` | Object | `` |
| `experimental.sparse.linalg.spsolve` | Function | `(data, indices, indptr, b, tol, reorder)` |
| `experimental.sparse.linalg.spsolve_p` | Object | `` |
| `experimental.sparse.nm.Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `experimental.sparse.nm.DTypeLike` | Object | `` |
| `experimental.sparse.nm.DotDimensionNumbers` | Object | `` |
| `experimental.sparse.nm.core` | Object | `` |
| `experimental.sparse.nm.dispatch` | Object | `` |
| `experimental.sparse.nm.jnp` | Object | `` |
| `experimental.sparse.nm.mhlo` | Object | `` |
| `experimental.sparse.nm.mlir` | Object | `` |
| `experimental.sparse.nm.nm_pack` | Function | `(mask: Array, n, m) -> Array` |
| `experimental.sparse.nm.nm_pack_p` | Object | `` |
| `experimental.sparse.nm.nm_spmm` | Function | `(lhs: Array, rhs: Array, metadata: Array, dimension_numbers: DotDimensionNumbers, sparse_operand_idx: int, output_dtype: DTypeLike) -> Array` |
| `experimental.sparse.nm.nm_spmm_p` | Object | `` |
| `experimental.sparse.random.Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `experimental.sparse.random.DTypeLike` | Object | `` |
| `experimental.sparse.random.dtypes` | Object | `` |
| `experimental.sparse.random.jnp` | Object | `` |
| `experimental.sparse.random.random` | Object | `` |
| `experimental.sparse.random.random_bcoo` | Function | `(key: Array, shape: Sequence[int], dtype: DTypeLike, indices_dtype: DTypeLike \| None, nse: float \| int, n_batch: int, n_dense: int, unique_indices: bool, sorted_indices: bool, generator: Callable[..., Array], kwds: Any) -> sparse.BCOO` |
| `experimental.sparse.random.sparse` | Object | `` |
| `experimental.sparse.random.split_list` | Function | `(args: Sequence[T], ns: Sequence[int]) -> list[list[T]]` |
| `experimental.sparse.random.vmap` | Function | `(fun: F, in_axes: int \| None \| Sequence[Any], out_axes: Any, axis_name: AxisName \| None, axis_size: int \| None, spmd_axis_name: AxisName \| tuple[AxisName, ...] \| None, sum_match: bool) -> F` |
| `experimental.sparse.random_bcoo` | Function | `(key: Array, shape: Sequence[int], dtype: DTypeLike, indices_dtype: DTypeLike \| None, nse: float \| int, n_batch: int, n_dense: int, unique_indices: bool, sorted_indices: bool, generator: Callable[..., Array], kwds: Any) -> sparse.BCOO` |
| `experimental.sparse.sparsify` | Function | `(f, use_tracer)` |
| `experimental.sparse.test_util.BatchedDotGeneralProperties` | Class | `(...)` |
| `experimental.sparse.test_util.DTypeLike` | Object | `` |
| `experimental.sparse.test_util.DotDimensionNumbers` | Object | `` |
| `experimental.sparse.test_util.MATMUL_TOL` | Object | `` |
| `experimental.sparse.test_util.SparseLayout` | Class | `(...)` |
| `experimental.sparse.test_util.SparseTestCase` | Class | `(...)` |
| `experimental.sparse.test_util.is_sparse` | Function | `(x)` |
| `experimental.sparse.test_util.iter_bcsr_layouts` | Function | `(shape: Sequence[int], min_n_batch) -> Iterator[SparseLayout]` |
| `experimental.sparse.test_util.iter_sparse_layouts` | Function | `(shape: Sequence[int], min_n_batch) -> Iterator[SparseLayout]` |
| `experimental.sparse.test_util.iter_subsets` | Function | `(s: Sequence) -> Iterable[tuple]` |
| `experimental.sparse.test_util.jnp` | Object | `` |
| `experimental.sparse.test_util.jtu` | Object | `` |
| `experimental.sparse.test_util.lax` | Object | `` |
| `experimental.sparse.test_util.rand_bcoo` | Function | `(rng: np.random.RandomState, rand_method: Callable[..., Any], nse: int \| float, n_batch: int, n_dense: int)` |
| `experimental.sparse.test_util.rand_bcsr` | Function | `(rng: np.random.RandomState, rand_method: Callable[..., Any], nse: int \| float, n_batch: int, n_dense: int)` |
| `experimental.sparse.test_util.rand_sparse` | Function | `(rng, nse, post, rand_method)` |
| `experimental.sparse.test_util.safe_zip` | Function | `(args)` |
| `experimental.sparse.test_util.sparse` | Object | `` |
| `experimental.sparse.test_util.split_list` | Function | `(args: Sequence[T], ns: Sequence[int]) -> list[list[T]]` |
| `experimental.sparse.test_util.tree_util` | Object | `` |
| `experimental.sparse.todense` | Function | `(arr: JAXSparse \| Array) -> Array` |
| `experimental.sparse.todense_p` | Object | `` |
| `experimental.sparse.transform.Array` | Object | `` |
| `experimental.sparse.transform.ArrayOrSparse` | Object | `` |
| `experimental.sparse.transform.BCOO` | Class | `(args: tuple[Array, Array], shape: Sequence[int], indices_sorted: bool, unique_indices: bool)` |
| `experimental.sparse.transform.BCSR` | Class | `(args: tuple[Array, Array, Array], shape: Sequence[int], indices_sorted: bool, unique_indices: bool)` |
| `experimental.sparse.transform.SparseTrace` | Class | `(parent_trace, tag, spenv)` |
| `experimental.sparse.transform.SparseTracer` | Class | `(trace: SparseTrace, spvalue)` |
| `experimental.sparse.transform.SparsifyEnv` | Class | `(bufs)` |
| `experimental.sparse.transform.SparsifyValue` | Class | `(...)` |
| `experimental.sparse.transform.api_util` | Object | `` |
| `experimental.sparse.transform.arrays_to_spvalues` | Function | `(spenv: SparsifyEnv, args: Any) -> Any` |
| `experimental.sparse.transform.bcoo_multiply_dense` | Function | `(sp_mat: BCOO, v: Array) -> Array` |
| `experimental.sparse.transform.bcoo_multiply_sparse` | Function | `(lhs: BCOO, rhs: BCOO) -> BCOO` |
| `experimental.sparse.transform.config` | Object | `` |
| `experimental.sparse.transform.core` | Object | `` |
| `experimental.sparse.transform.eval_sparse` | Function | `(jaxpr: core.Jaxpr, consts: Sequence[Array], spvalues: Sequence[SparsifyValue], spenv: SparsifyEnv) -> Sequence[SparsifyValue]` |
| `experimental.sparse.transform.flatten_fun_nokwargs` | Function | `(f: Callable, store: lu.Store, in_tree: PyTreeDef, args_flat)` |
| `experimental.sparse.transform.jnp` | Object | `` |
| `experimental.sparse.transform.jnp_indexing` | Object | `` |
| `experimental.sparse.transform.lax` | Object | `` |
| `experimental.sparse.transform.lift_jvp` | Function | `(num_consts: int, jvp_jaxpr_fun: lu.WrappedFun) -> lu.WrappedFun` |
| `experimental.sparse.transform.lu` | Object | `` |
| `experimental.sparse.transform.pe` | Object | `` |
| `experimental.sparse.transform.pjit` | Object | `` |
| `experimental.sparse.transform.pytree` | Object | `` |
| `experimental.sparse.transform.safe_map` | Function | `(f, args)` |
| `experimental.sparse.transform.safe_zip` | Function | `(args)` |
| `experimental.sparse.transform.sharding_impls` | Object | `` |
| `experimental.sparse.transform.sparse` | Object | `` |
| `experimental.sparse.transform.sparse_rules_bcoo` | Object | `` |
| `experimental.sparse.transform.sparse_rules_bcsr` | Object | `` |
| `experimental.sparse.transform.sparsify` | Function | `(f, use_tracer)` |
| `experimental.sparse.transform.sparsify_fun` | Function | `(wrapped_fun, args: list[ArrayOrSparse])` |
| `experimental.sparse.transform.sparsify_raw` | Function | `(f)` |
| `experimental.sparse.transform.sparsify_subtrace` | Function | `(f, store, tag, spenv, spvalues, bufs)` |
| `experimental.sparse.transform.split_list` | Function | `(args: Sequence[T], ns: Sequence[int]) -> list[list[T]]` |
| `experimental.sparse.transform.spvalues_to_arrays` | Function | `(spenv: SparsifyEnv, spvalues: Any) -> Any` |
| `experimental.sparse.transform.spvalues_to_avals` | Function | `(spenv: SparsifyEnv, spvalues: Any) -> Any` |
| `experimental.sparse.transform.tree_flatten` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> tuple[list[Leaf], PyTreeDef]` |
| `experimental.sparse.transform.tree_map` | Function | `(f: Callable[..., Any], tree: Any, rest: Any, is_leaf: Callable[[Any], bool] \| None) -> Any` |
| `experimental.sparse.transform.tree_unflatten` | Function | `(treedef: PyTreeDef, leaves: Iterable[Leaf]) -> Any` |
| `experimental.sparse.transform.tree_util` | Object | `` |
| `experimental.sparse.util.Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `experimental.sparse.util.CuSparseEfficiencyWarning` | Class | `(...)` |
| `experimental.sparse.util.DotDimensionNumbers` | Object | `` |
| `experimental.sparse.util.Shape` | Object | `` |
| `experimental.sparse.util.SparseEfficiencyError` | Class | `(...)` |
| `experimental.sparse.util.SparseEfficiencyWarning` | Class | `(...)` |
| `experimental.sparse.util.SparseInfo` | Class | `(...)` |
| `experimental.sparse.util.broadcasting_vmap` | Function | `(fun, in_axes, out_axes)` |
| `experimental.sparse.util.core` | Object | `` |
| `experimental.sparse.util.flatten_axes` | Function | `(name, treedef, axis_tree, kws, tupled_args)` |
| `experimental.sparse.util.jnp` | Object | `` |
| `experimental.sparse.util.lax` | Object | `` |
| `experimental.sparse.util.nfold_vmap` | Function | `(fun, N, broadcasted, in_axes)` |
| `experimental.sparse.util.safe_zip` | Function | `(args)` |
| `experimental.sparse.util.tree_util` | Object | `` |
| `experimental.sparse.util.vmap` | Function | `(fun: F, in_axes: int \| None \| Sequence[Any], out_axes: Any, axis_name: AxisName \| None, axis_size: int \| None, spmd_axis_name: AxisName \| tuple[AxisName, ...] \| None, sum_match: bool) -> F` |
| `experimental.sparse.value_and_grad` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux, kwargs) -> Callable[..., tuple[Any, Any]]` |
| `experimental.topologies.Device` | Object | `` |
| `experimental.topologies.TopologyDescription` | Class | `(devices: list[Device])` |
| `experimental.topologies.get_attached_topology` | Function | `(platform) -> TopologyDescription` |
| `experimental.topologies.get_topology_desc` | Function | `(topology_name: str, platform: str \| None, kwargs) -> TopologyDescription` |
| `experimental.topologies.make_mesh` | Function | `(topo: TopologyDescription, mesh_shape: Sequence[int], axis_names: tuple[str, ...], contiguous_submeshes: bool) -> jax.sharding.Mesh` |
| `experimental.topologies.mesh_utils` | Object | `` |
| `experimental.topologies.xb` | Object | `` |
| `experimental.transfer.TransferConnection` | Class | `(...)` |
| `experimental.transfer.TransferServer` | Class | `(...)` |
| `experimental.transfer.make_error_array` | Function | `(aval, message)` |
| `experimental.transfer.start_transfer_server` | Object | `` |
| `experimental.transfer.use_cpp_class` | Function | `(cpp_cls: type[Any]) -> Callable[[type[T]], type[T]]` |
| `experimental.transfer.use_cpp_method` | Function | `(is_enabled: bool) -> Callable[[T], T]` |
| `experimental.xla_metadata.set_xla_metadata` | Function | `(x, kwargs)` |
| `explain_cache_misses` | Object | `` |
| `export.DisabledSafetyCheck` | Class | `(_impl: str)` |
| `export.Exported` | Class | `(fun_name: str, in_tree: tree_util.PyTreeDef, in_avals: tuple[core.ShapedArray, ...], out_tree: tree_util.PyTreeDef, out_avals: tuple[core.ShapedArray, ...], _has_named_shardings: bool, _in_named_shardings: tuple[NamedSharding \| None, ...], _out_named_shardings: tuple[NamedSharding \| None, ...], in_shardings_hlo: tuple[HloSharding \| None, ...], out_shardings_hlo: tuple[HloSharding \| None, ...], nr_devices: int, platforms: tuple[str, ...], ordered_effects: tuple[effects.Effect, ...], unordered_effects: tuple[effects.Effect, ...], disabled_safety_checks: Sequence[DisabledSafetyCheck], mlir_module_serialized: bytes, calling_convention_version: int, module_kept_var_idx: tuple[int, ...], uses_global_constants: bool, _get_vjp: Callable[[Exported], Exported] \| None)` |
| `export.SymbolicScope` | Class | `(constraints_str: Sequence[str])` |
| `export.default_export_platform` | Function | `() -> str` |
| `export.deserialize` | Function | `(blob: bytearray) -> Exported` |
| `export.export` | Function | `(fun_jit: stages.Wrapped, platforms: Sequence[str] \| None, disabled_checks: Sequence[DisabledSafetyCheck], _override_lowering_rules: Sequence[tuple[Any, Any]] \| None) -> Callable[..., Exported]` |
| `export.is_symbolic_dim` | Function | `(p: DimSize) -> bool` |
| `export.maximum_supported_calling_convention_version` | Object | `` |
| `export.minimum_supported_calling_convention_version` | Object | `` |
| `export.register_namedtuple_serialization` | Function | `(nodetype: type[T], serialized_name: str) -> type[T]` |
| `export.register_pytree_node_serialization` | Function | `(nodetype: type[T], serialized_name: str, serialize_auxdata: _SerializeAuxData, deserialize_auxdata: _DeserializeAuxData, from_children: _BuildFromChildren \| None) -> type[T]` |
| `export.shape_poly_decision` | Object | `` |
| `export.symbolic_args_specs` | Function | `(args, shapes_specs, constraints: Sequence[str], scope: SymbolicScope \| None)` |
| `export.symbolic_shape` | Function | `(shape_spec: str \| None, constraints: Sequence[str], scope: SymbolicScope \| None, like: Sequence[int \| None] \| None) -> Sequence[DimSize]` |
| `extend.backend.backend_xla_version` | Function | `(platform) -> int \| None` |
| `extend.backend.backends` | Function | `() -> dict[str, xla_client.Client]` |
| `extend.backend.clear_backends` | Function | `()` |
| `extend.backend.clear_in_memory_compilation_cache` | Function | `() -> None` |
| `extend.backend.get_backend` | Function | `(platform: None \| str \| xla_client.Client) -> xla_client.Client` |
| `extend.backend.get_compile_options` | Function | `(num_replicas: int, num_partitions: int, device_assignment, use_spmd_partitioning: bool, use_auto_spmd_partitioning: bool, auto_spmd_partitioning_mesh_shape: list[int] \| None, auto_spmd_partitioning_mesh_ids: list[int] \| None, env_options_overrides: dict[str, str] \| None, fdo_profile: bytes \| None, detailed_logging: bool, backend: xc.Client \| None) -> xc.CompileOptions` |
| `extend.backend.get_default_device` | Function | `() -> xc.Device` |
| `extend.backend.ifrt_proxy` | Object | `` |
| `extend.backend.register_backend_cache` | Object | `` |
| `extend.backend.register_backend_factory` | Function | `(name: str, factory: BackendFactory, priority: int, fail_quietly: bool, experimental: bool, make_topology: TopologyFactory \| None, c_api: Any \| None) -> None` |
| `extend.core.ClosedJaxpr` | Class | `(jaxpr: Jaxpr, consts: Sequence)` |
| `extend.core.Jaxpr` | Class | `(constvars: Sequence[Var], invars: Sequence[Var], outvars: Sequence[Atom], eqns: Sequence[JaxprEqn], effects: Effects, debug_info: DebugInfo, is_high: bool)` |
| `extend.core.JaxprEqn` | Class | `(invars, outvars, primitive, params, effs, source_info, ctx)` |
| `extend.core.Literal` | Class | `(val, aval)` |
| `extend.core.Primitive` | Class | `(name: str)` |
| `extend.core.Token` | Class | `(buf)` |
| `extend.core.Var` | Class | `(aval: AbstractValue, initial_qdd, final_qdd)` |
| `extend.core.array_types` | Object | `` |
| `extend.core.jaxpr_as_fun` | Function | `(closed_jaxpr: ClosedJaxpr, args)` |
| `extend.core.mapped_aval` | Function | `(size: AxisSize, axis, aval: AbstractValue) -> AbstractValue` |
| `extend.core.primitives.abs_p` | Object | `` |
| `extend.core.primitives.acos_p` | Object | `` |
| `extend.core.primitives.acosh_p` | Object | `` |
| `extend.core.primitives.add_jaxvals_p` | Object | `` |
| `extend.core.primitives.add_p` | Object | `` |
| `extend.core.primitives.after_all_p` | Object | `` |
| `extend.core.primitives.all_gather_p` | Object | `` |
| `extend.core.primitives.all_to_all_p` | Object | `` |
| `extend.core.primitives.and_p` | Object | `` |
| `extend.core.primitives.approx_top_k_p` | Object | `` |
| `extend.core.primitives.argmax_p` | Object | `` |
| `extend.core.primitives.argmin_p` | Object | `` |
| `extend.core.primitives.asin_p` | Object | `` |
| `extend.core.primitives.asinh_p` | Object | `` |
| `extend.core.primitives.atan2_p` | Object | `` |
| `extend.core.primitives.atan_p` | Object | `` |
| `extend.core.primitives.atanh_p` | Object | `` |
| `extend.core.primitives.axis_index_p` | Object | `` |
| `extend.core.primitives.bessel_i0e_p` | Object | `` |
| `extend.core.primitives.bessel_i1e_p` | Object | `` |
| `extend.core.primitives.bitcast_convert_type_p` | Object | `` |
| `extend.core.primitives.broadcast_in_dim_p` | Object | `` |
| `extend.core.primitives.call_p` | Object | `` |
| `extend.core.primitives.cbrt_p` | Object | `` |
| `extend.core.primitives.ceil_p` | Object | `` |
| `extend.core.primitives.cholesky_p` | Object | `` |
| `extend.core.primitives.clamp_p` | Object | `` |
| `extend.core.primitives.closed_call_p` | Object | `` |
| `extend.core.primitives.clz_p` | Object | `` |
| `extend.core.primitives.complex_p` | Object | `` |
| `extend.core.primitives.concatenate_p` | Object | `` |
| `extend.core.primitives.cond_p` | Object | `` |
| `extend.core.primitives.conj_p` | Object | `` |
| `extend.core.primitives.conv_general_dilated_p` | Object | `` |
| `extend.core.primitives.convert_element_type_p` | Object | `` |
| `extend.core.primitives.copy_p` | Object | `` |
| `extend.core.primitives.cos_p` | Object | `` |
| `extend.core.primitives.cosh_p` | Object | `` |
| `extend.core.primitives.create_token_p` | Object | `` |
| `extend.core.primitives.cumlogsumexp_p` | Object | `` |
| `extend.core.primitives.cummax_p` | Object | `` |
| `extend.core.primitives.cummin_p` | Object | `` |
| `extend.core.primitives.cumprod_p` | Object | `` |
| `extend.core.primitives.cumsum_p` | Object | `` |
| `extend.core.primitives.custom_jvp_call_p` | Object | `` |
| `extend.core.primitives.custom_lin_p` | Object | `` |
| `extend.core.primitives.custom_vjp_call_p` | Object | `` |
| `extend.core.primitives.device_put_p` | Object | `` |
| `extend.core.primitives.digamma_p` | Object | `` |
| `extend.core.primitives.div_p` | Object | `` |
| `extend.core.primitives.dot_general_p` | Object | `` |
| `extend.core.primitives.dynamic_slice_p` | Object | `` |
| `extend.core.primitives.dynamic_update_slice_p` | Object | `` |
| `extend.core.primitives.eig_p` | Object | `` |
| `extend.core.primitives.eigh_p` | Object | `` |
| `extend.core.primitives.empty2_p` | Object | `` |
| `extend.core.primitives.eq_p` | Object | `` |
| `extend.core.primitives.eq_to_p` | Object | `` |
| `extend.core.primitives.erf_inv_p` | Object | `` |
| `extend.core.primitives.erf_p` | Object | `` |
| `extend.core.primitives.erfc_p` | Object | `` |
| `extend.core.primitives.exp2_p` | Object | `` |
| `extend.core.primitives.exp_p` | Object | `` |
| `extend.core.primitives.expm1_p` | Object | `` |
| `extend.core.primitives.fft_p` | Object | `` |
| `extend.core.primitives.floor_p` | Object | `` |
| `extend.core.primitives.gather_p` | Object | `` |
| `extend.core.primitives.ge_p` | Object | `` |
| `extend.core.primitives.gt_p` | Object | `` |
| `extend.core.primitives.hessenberg_p` | Object | `` |
| `extend.core.primitives.householder_product_p` | Object | `` |
| `extend.core.primitives.igamma_grad_a_p` | Object | `` |
| `extend.core.primitives.igamma_p` | Object | `` |
| `extend.core.primitives.igammac_p` | Object | `` |
| `extend.core.primitives.imag_p` | Object | `` |
| `extend.core.primitives.integer_pow_p` | Object | `` |
| `extend.core.primitives.iota_p` | Object | `` |
| `extend.core.primitives.is_finite_p` | Object | `` |
| `extend.core.primitives.jit_p` | Object | `` |
| `extend.core.primitives.le_p` | Object | `` |
| `extend.core.primitives.le_to_p` | Object | `` |
| `extend.core.primitives.lgamma_p` | Object | `` |
| `extend.core.primitives.linear_solve_p` | Object | `` |
| `extend.core.primitives.log1p_p` | Object | `` |
| `extend.core.primitives.log_p` | Object | `` |
| `extend.core.primitives.logistic_p` | Object | `` |
| `extend.core.primitives.lt_p` | Object | `` |
| `extend.core.primitives.lt_to_p` | Object | `` |
| `extend.core.primitives.lu_p` | Object | `` |
| `extend.core.primitives.max_p` | Object | `` |
| `extend.core.primitives.min_p` | Object | `` |
| `extend.core.primitives.mul_p` | Object | `` |
| `extend.core.primitives.name_p` | Object | `` |
| `extend.core.primitives.ne_p` | Object | `` |
| `extend.core.primitives.neg_p` | Object | `` |
| `extend.core.primitives.nextafter_p` | Object | `` |
| `extend.core.primitives.not_p` | Object | `` |
| `extend.core.primitives.or_p` | Object | `` |
| `extend.core.primitives.pad_p` | Object | `` |
| `extend.core.primitives.pmax_p` | Object | `` |
| `extend.core.primitives.pmin_p` | Object | `` |
| `extend.core.primitives.polygamma_p` | Object | `` |
| `extend.core.primitives.population_count_p` | Object | `` |
| `extend.core.primitives.pow_p` | Object | `` |
| `extend.core.primitives.ppermute_p` | Object | `` |
| `extend.core.primitives.psum_p` | Object | `` |
| `extend.core.primitives.qr_p` | Object | `` |
| `extend.core.primitives.ragged_all_to_all_p` | Object | `` |
| `extend.core.primitives.random_bits_p` | Object | `` |
| `extend.core.primitives.random_fold_in_p` | Object | `` |
| `extend.core.primitives.random_gamma_p` | Object | `` |
| `extend.core.primitives.random_seed_p` | Object | `` |
| `extend.core.primitives.random_split_p` | Object | `` |
| `extend.core.primitives.real_p` | Object | `` |
| `extend.core.primitives.reduce_and_p` | Object | `` |
| `extend.core.primitives.reduce_max_p` | Object | `` |
| `extend.core.primitives.reduce_min_p` | Object | `` |
| `extend.core.primitives.reduce_or_p` | Object | `` |
| `extend.core.primitives.reduce_p` | Object | `` |
| `extend.core.primitives.reduce_precision_p` | Object | `` |
| `extend.core.primitives.reduce_prod_p` | Object | `` |
| `extend.core.primitives.reduce_sum_p` | Object | `` |
| `extend.core.primitives.reduce_window_max_p` | Object | `` |
| `extend.core.primitives.reduce_window_min_p` | Object | `` |
| `extend.core.primitives.reduce_window_p` | Object | `` |
| `extend.core.primitives.reduce_window_sum_p` | Object | `` |
| `extend.core.primitives.reduce_xor_p` | Object | `` |
| `extend.core.primitives.regularized_incomplete_beta_p` | Object | `` |
| `extend.core.primitives.rem_p` | Object | `` |
| `extend.core.primitives.remat_p` | Object | `` |
| `extend.core.primitives.reshape_p` | Object | `` |
| `extend.core.primitives.rev_p` | Object | `` |
| `extend.core.primitives.rng_bit_generator_p` | Object | `` |
| `extend.core.primitives.rng_uniform_p` | Object | `` |
| `extend.core.primitives.round_p` | Object | `` |
| `extend.core.primitives.rsqrt_p` | Object | `` |
| `extend.core.primitives.scan_p` | Object | `` |
| `extend.core.primitives.scatter_add_p` | Object | `` |
| `extend.core.primitives.scatter_max_p` | Object | `` |
| `extend.core.primitives.scatter_min_p` | Object | `` |
| `extend.core.primitives.scatter_mul_p` | Object | `` |
| `extend.core.primitives.scatter_p` | Object | `` |
| `extend.core.primitives.schur_p` | Object | `` |
| `extend.core.primitives.select_and_gather_add_p` | Object | `` |
| `extend.core.primitives.select_and_scatter_add_p` | Object | `` |
| `extend.core.primitives.select_and_scatter_p` | Object | `` |
| `extend.core.primitives.select_n_p` | Object | `` |
| `extend.core.primitives.sharding_constraint_p` | Object | `` |
| `extend.core.primitives.shift_left_p` | Object | `` |
| `extend.core.primitives.shift_right_arithmetic_p` | Object | `` |
| `extend.core.primitives.shift_right_logical_p` | Object | `` |
| `extend.core.primitives.sign_p` | Object | `` |
| `extend.core.primitives.sin_p` | Object | `` |
| `extend.core.primitives.sinh_p` | Object | `` |
| `extend.core.primitives.slice_p` | Object | `` |
| `extend.core.primitives.sort_p` | Object | `` |
| `extend.core.primitives.sqrt_p` | Object | `` |
| `extend.core.primitives.square_p` | Object | `` |
| `extend.core.primitives.squeeze_p` | Object | `` |
| `extend.core.primitives.stop_gradient_p` | Object | `` |
| `extend.core.primitives.sub_p` | Object | `` |
| `extend.core.primitives.svd_p` | Object | `` |
| `extend.core.primitives.tan_p` | Object | `` |
| `extend.core.primitives.tanh_p` | Object | `` |
| `extend.core.primitives.threefry2x32_p` | Object | `` |
| `extend.core.primitives.top_k_p` | Object | `` |
| `extend.core.primitives.transpose_p` | Object | `` |
| `extend.core.primitives.triangular_solve_p` | Object | `` |
| `extend.core.primitives.tridiagonal_p` | Object | `` |
| `extend.core.primitives.tridiagonal_solve_p` | Object | `` |
| `extend.core.primitives.while_p` | Object | `` |
| `extend.core.primitives.xla_pmap_p` | Object | `` |
| `extend.core.primitives.xor_p` | Object | `` |
| `extend.core.primitives.zeta_p` | Object | `` |
| `extend.core.take_current_trace` | Object | `` |
| `extend.core.unmapped_aval` | Function | `(size: AxisSize, axis: int \| None, aval: AbstractValue, explicit_mesh_axis) -> AbstractValue` |
| `extend.ifrt_programs.ifrt_programs` | Object | `` |
| `extend.linear_util.StoreException` | Class | `(...)` |
| `extend.linear_util.WrappedFun` | Class | `(f: Callable, f_transformed: Callable, transforms: tuple[tuple[Callable, tuple[Hashable, ...]], ...], stores: tuple[Store \| EqualStore \| None, ...], params: tuple[tuple[str, Hashable], ...], in_type: core.InputType \| None, debug_info: DebugInfo)` |
| `extend.linear_util.cache` | Function | `(call: Callable, explain: Callable[[WrappedFun, bool, dict, tuple, float], None] \| None)` |
| `extend.linear_util.merge_linear_aux` | Function | `(aux1, aux2)` |
| `extend.linear_util.transformation` | Function | `(gen, fun: WrappedFun, gen_static_args) -> WrappedFun` |
| `extend.linear_util.transformation2` | Function | `(gen, fun: WrappedFun, gen_static_args) -> WrappedFun` |
| `extend.linear_util.transformation_with_aux` | Function | `(gen, fun: WrappedFun, gen_static_args) -> WrappedFun` |
| `extend.linear_util.transformation_with_aux2` | Function | `(gen, fun: WrappedFun, gen_static_args, use_eq_store: bool, unk_names: bool) -> tuple[WrappedFun, Callable[[], Any]]` |
| `extend.linear_util.wrap_init` | Function | `(f: Callable, params, debug_info) -> WrappedFun` |
| `extend.mlir.deserialize_portable_artifact` | Object | `` |
| `extend.mlir.hlo_to_stablehlo` | Object | `` |
| `extend.mlir.lower_with_sharding_in_types` | Function | `(ctx, op, aval, sharding_proto)` |
| `extend.mlir.refine_polymorphic_shapes` | Object | `` |
| `extend.mlir.serialize_portable_artifact` | Object | `` |
| `extend.random.define_prng_impl` | Function | `(key_shape: Shape, seed: Callable[[Array], Array], split: Callable[[Array, Shape], Array], random_bits: Callable[[Array, int, Shape], Array], fold_in: Callable[[Array, int], Array], name: str, tag: str) -> Hashable` |
| `extend.random.random_seed` | Function | `(seeds: int \| typing.ArrayLike, impl: PRNGImpl) -> PRNGKeyArray` |
| `extend.random.rbg_prng_impl` | Object | `` |
| `extend.random.seed_with_impl` | Function | `(impl: PRNGImpl, seed: int \| typing.ArrayLike) -> PRNGKeyArray` |
| `extend.random.threefry2x32_p` | Object | `` |
| `extend.random.threefry_2x32` | Function | `(keypair, count)` |
| `extend.random.threefry_prng_impl` | Object | `` |
| `extend.random.unsafe_rbg_prng_impl` | Object | `` |
| `extend.sharding.GSPMDSharding` | Class | `(devices: Sequence[Device] \| xc.DeviceList, op_sharding: xc.OpSharding \| xc.HloSharding, memory_kind: str \| None)` |
| `extend.sharding.get_hlo_sharding_from_serialized_proto` | Function | `(sharding: bytes) -> xla_client.HloSharding` |
| `extend.sharding.get_op_sharding_from_serialized_proto` | Function | `(sharding: bytes) -> xla_client.OpSharding` |
| `extend.sharding.get_serialized_proto_from_hlo_sharding` | Function | `(sharding: xla_client.HloSharding) -> bytes` |
| `extend.sharding.xla_client` | Object | `` |
| `extend.source_info_util.NameStack` | Class | `(stack: tuple[Scope \| Transform, ...])` |
| `extend.source_info_util.SourceInfo` | Class | `(traceback: Traceback \| None, name_stack: NameStack)` |
| `extend.source_info_util.current` | Function | `() -> SourceInfo` |
| `extend.source_info_util.current_name_stack` | Function | `() -> NameStack` |
| `extend.source_info_util.extend_name_stack` | Object | `` |
| `extend.source_info_util.new_name_stack` | Function | `(name: str) -> NameStack` |
| `extend.source_info_util.new_source_info` | Function | `() -> SourceInfo` |
| `extend.source_info_util.register_exclusion` | Function | `(path: str)` |
| `extend.source_info_util.reset_name_stack` | Function | `() -> Generator[None, None, None]` |
| `extend.source_info_util.set_name_stack` | Object | `` |
| `extend.source_info_util.summarize` | Function | `(source_info: SourceInfo, num_frames) -> str` |
| `extend.source_info_util.transform_name_stack` | Object | `` |
| `extend.source_info_util.user_context` | Object | `` |
| `ffi.build_ffi_lowering_function` | Function | `(call_target_name: str, operand_layouts: Sequence[FfiLayoutOptions] \| None, result_layouts: Sequence[FfiLayoutOptions] \| None, backend_config: Mapping[str, ir.Attribute] \| str \| None, skip_ffi_layout_processing: bool, lowering_args: Any) -> Callable[..., ir.Operation]` |
| `ffi.ffi_call` | Function | `(target_name: str, result_shape_dtypes: ResultMetadata \| Sequence[ResultMetadata], has_side_effect: bool, vmap_method: str \| None, input_layouts: Sequence[FfiLayoutOptions] \| None, output_layouts: FfiLayoutOptions \| Sequence[FfiLayoutOptions] \| None, input_output_aliases: dict[int, int] \| None, custom_call_api_version: int, legacy_backend_config: str \| None, vectorized: bool \| None \| DeprecatedArg) -> Callable[..., Array \| Sequence[Array]]` |
| `ffi.ffi_lowering` | Function | `(call_target_name: str, operand_layouts: Sequence[FfiLayoutOptions] \| None, result_layouts: Sequence[FfiLayoutOptions] \| None, backend_config: Mapping[str, ir.Attribute] \| str \| None, skip_ffi_layout_processing: bool, lowering_args: Any) -> mlir.LoweringRule` |
| `ffi.include_dir` | Function | `() -> str` |
| `ffi.pycapsule` | Function | `(funcptr)` |
| `ffi.register_ffi_target` | Function | `(name: str, fn: Any, platform: str, api_version: int, kwargs: Any) -> None` |
| `ffi.register_ffi_target_as_batch_partitionable` | Function | `(name: str) -> None` |
| `ffi.register_ffi_type` | Function | `(name: str, type_registration: TypeRegistration, platform: str) -> None` |
| `ffi.register_ffi_type_id` | Function | `(name: str, obj: Any, platform: str) -> None` |
| `flatten_util.ravel_pytree` | Function | `(pytree: Any) -> tuple[Array, Callable[[Array], Any]]` |
| `float0` | Object | `` |
| `free_ref` | Function | `(ref: Ref)` |
| `freeze` | Function | `(ref: Ref) -> Array` |
| `fwd_and_bwd` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux: bool, jitted: bool) -> tuple[Callable, Callable]` |
| `grad` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux: bool, holomorphic: bool, allow_int: bool, reduce_axes: Sequence[AxisName]) -> Callable` |
| `hessian` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux: bool, holomorphic: bool) -> Callable` |
| `host_count` | Function | `(backend: str \| xla_client.Client \| None) -> int` |
| `host_id` | Function | `(backend: str \| xla_client.Client \| None) -> int` |
| `host_ids` | Function | `(backend: str \| xla_client.Client \| None) -> list[int]` |
| `image.ResizeMethod` | Class | `(...)` |
| `image.resize` | Function | `(image, shape: core.Shape, method: str \| ResizeMethod, antialias: bool, precision)` |
| `image.scale_and_translate` | Function | `(image, shape: core.Shape, spatial_dims: Sequence[int], scale, translation, method: str \| ResizeMethod, antialias: bool, precision)` |
| `interpreters.ad.JVPTrace` | Class | `(parent_trace, tag)` |
| `interpreters.ad.JVPTracer` | Class | `(trace, primal, tangent)` |
| `interpreters.ad.UndefinedPrimal` | Class | `(aval)` |
| `interpreters.ad.Zero` | Class | `(aval: core.AbstractValue)` |
| `interpreters.ad.add_jaxvals` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `interpreters.ad.add_jaxvals_p` | Object | `` |
| `interpreters.ad.add_tangents` | Function | `(x, y)` |
| `interpreters.ad.defbilinear` | Function | `(prim, lhs_rule, rhs_rule)` |
| `interpreters.ad.defjvp` | Function | `(primitive, jvprules)` |
| `interpreters.ad.defjvp2` | Function | `(primitive, jvprules)` |
| `interpreters.ad.deflinear` | Function | `(primitive, transpose_rule)` |
| `interpreters.ad.deflinear2` | Function | `(primitive, transpose_rule)` |
| `interpreters.ad.get_primitive_transpose` | Function | `(p)` |
| `interpreters.ad.instantiate_zeros` | Function | `(tangent)` |
| `interpreters.ad.is_undefined_primal` | Function | `(x)` |
| `interpreters.ad.jvp` | Function | `(fun: lu.WrappedFun, has_aux, instantiate, transform_stack) -> Any` |
| `interpreters.ad.linearize` | Function | `(traceable: lu.WrappedFun, primals, has_aux, is_vjp)` |
| `interpreters.ad.primitive_jvps` | Object | `` |
| `interpreters.ad.primitive_transposes` | Object | `` |
| `interpreters.ad.reducing_transposes` | Object | `` |
| `interpreters.ad.zeros_like_aval` | Function | `(aval: core.AbstractValue) -> Array` |
| `interpreters.batching.NotMapped` | Object | `` |
| `interpreters.batching.axis_primitive_batchers` | Object | `` |
| `interpreters.batching.bdim_at_front` | Function | `(x, bdim, size, mesh_axis)` |
| `interpreters.batching.broadcast` | Function | `(x, sz, axis, mesh_axis)` |
| `interpreters.batching.defbroadcasting` | Function | `(prim)` |
| `interpreters.batching.defreducer` | Function | `(prim)` |
| `interpreters.batching.defvectorized` | Function | `(prim)` |
| `interpreters.batching.fancy_primitive_batchers` | Object | `` |
| `interpreters.batching.not_mapped` | Object | `` |
| `interpreters.batching.primitive_batchers` | Object | `` |
| `interpreters.batching.register_vmappable` | Function | `(data_type: type, spec_type: type, axis_size_type: type, to_elt: Callable, from_elt: Callable, make_iota: Callable \| None)` |
| `interpreters.batching.unregister_vmappable` | Function | `(data_type: type) -> None` |
| `interpreters.mlir.AxisContext` | Object | `` |
| `interpreters.mlir.ConstantHandler` | Class | `(...)` |
| `interpreters.mlir.DEVICE_TO_DEVICE_TYPE` | Object | `` |
| `interpreters.mlir.LoweringParameters` | Class | `(override_lowering_rules: tuple[tuple[core.Primitive, LoweringRule]] \| None, global_constant_computation: bool, for_export: bool, export_ignore_forward_compatibility: bool, hoist_constants_as_args: bool)` |
| `interpreters.mlir.LoweringResult` | Class | `(...)` |
| `interpreters.mlir.LoweringRule` | Class | `(...)` |
| `interpreters.mlir.LoweringRuleContext` | Class | `(module_context: ModuleContext, name_stack: source_info_util.NameStack, traceback: xc.Traceback \| None, primitive: core.Primitive \| None, avals_in: Sequence[core.AbstractValue], avals_out: Any, tokens_in: TokenSet, tokens_out: TokenSet \| None, const_lowering: dict[tuple[int, core.AbstractValue], IrValues], axis_size_env: dict[core.Var, ir.Value] \| None, dim_var_values: Sequence[ir.Value], jaxpr_eqn_ctx: core.JaxprEqnContext \| None, platforms: Sequence[str] \| None)` |
| `interpreters.mlir.Mesh` | Class | `(...)` |
| `interpreters.mlir.MeshAxisName` | Object | `` |
| `interpreters.mlir.ModuleContext` | Class | `(platforms: Sequence[str], backend: xc.Client \| None, axis_context: AxisContext, keepalives: list[Any], channel_iterator: Iterator[int], host_callbacks: list[Any], lowering_parameters: LoweringParameters, context: ir.Context \| None, module: ir.Module \| None, ip: ir.InsertionPoint \| None, symbol_table: ir.SymbolTable \| None, lowering_cache: None \| dict[LoweringCacheKey, Any], cached_primitive_lowerings: None \| dict[Any, func_dialect.FuncOp], traceback_caches: None \| TracebackCaches, shape_poly_state, all_default_mem_kind: bool)` |
| `interpreters.mlir.RECV_FROM_HOST_TYPE` | Object | `` |
| `interpreters.mlir.ReplicaAxisContext` | Class | `(axis_env: AxisEnv)` |
| `interpreters.mlir.SEND_TO_HOST_TYPE` | Object | `` |
| `interpreters.mlir.SPMDAxisContext` | Class | `(mesh: mesh_lib.Mesh, manual_axes: frozenset[MeshAxisName])` |
| `interpreters.mlir.ShapePolyLoweringState` | Class | `(dim_vars: tuple[str, ...], lowering_platforms: tuple[str, ...] \| None)` |
| `interpreters.mlir.ShardingContext` | Class | `(num_devices: int, device_assignment: tuple[xc.Device, ...] \| None, abstract_mesh: mesh_lib.AbstractMesh \| None)` |
| `interpreters.mlir.Token` | Object | `` |
| `interpreters.mlir.TokenSet` | Class | `(args: Any, kwargs: Any)` |
| `interpreters.mlir.Value` | Object | `` |
| `interpreters.mlir.aval_to_ir_type` | Function | `(aval: core.AbstractValue) -> IrTypes` |
| `interpreters.mlir.aval_to_ir_types` | Function | `(aval: core.AbstractValue) -> tuple[ir.Type, ...]` |
| `interpreters.mlir.core_call_lowering` | Function | `(ctx: LoweringRuleContext, args, name, backend, call_jaxpr: core.ClosedJaxpr \| core.Jaxpr)` |
| `interpreters.mlir.dense_int_array` | Object | `` |
| `interpreters.mlir.dense_int_elements` | Function | `(xs) -> ir.DenseElementsAttr` |
| `interpreters.mlir.dtype_to_ir_type` | Function | `(dtype: core.bint \| np.dtype \| np.generic) -> ir.Type` |
| `interpreters.mlir.emit_python_callback` | Function | `(ctx: mlir.LoweringRuleContext, callback, token: Any \| None, operands: Sequence[ir.Value], operand_avals: Sequence[core.ShapedArray], result_avals: Sequence[core.ShapedArray], has_side_effect: bool, returns_token: bool, partitioned: bool, sharding: SdyArrayList \| xc.OpSharding \| None) -> tuple[Sequence[mlir.IrValues], Any, Any]` |
| `interpreters.mlir.flatten_ir_types` | Function | `(xs: Iterable[IrTypes]) -> list[ir.Type]` |
| `interpreters.mlir.flatten_ir_values` | Function | `(xs: Iterable[IrValues]) -> list[ir.Value]` |
| `interpreters.mlir.i32_attr` | Function | `(i)` |
| `interpreters.mlir.i64_attr` | Function | `(i)` |
| `interpreters.mlir.ir` | Object | `` |
| `interpreters.mlir.ir_attribute` | Function | `(val: Any) -> ir.Attribute` |
| `interpreters.mlir.ir_constant` | Function | `(val: Any, const_lowering: dict[tuple[int, core.AbstractValue], IrValues] \| None, aval: core.AbstractValue \| None) -> IrValues` |
| `interpreters.mlir.ir_type_handlers` | Object | `` |
| `interpreters.mlir.jaxpr_subcomp` | Function | `(ctx: ModuleContext, jaxpr: core.Jaxpr, name_stack: source_info_util.NameStack, tokens: TokenSet, consts_for_constvars: Sequence[IrValues], args: IrValues, dim_var_values: Sequence[ir.Value], const_lowering: dict[tuple[int, core.AbstractValue], IrValues], outer_traceback: xc.Traceback \| None) -> tuple[Sequence[IrValues], TokenSet]` |
| `interpreters.mlir.lower_fun` | Function | `(fun: Callable, multiple_results: bool) -> Callable` |
| `interpreters.mlir.lower_jaxpr_to_fun` | Function | `(ctx: ModuleContext, name: str, jaxpr: core.ClosedJaxpr, effects: Sequence[core.Effect], num_const_args: int, main_function: bool, replicated_args: Sequence[bool] \| None, in_avals: Sequence[core.AbstractValue], arg_shardings: Sequence[JSharding \| AUTO \| None] \| None, result_shardings: Sequence[JSharding \| AUTO \| None] \| None, use_sharding_annotations: bool, input_output_aliases: Sequence[int \| None] \| None, xla_donated_args: Sequence[bool] \| None, arg_names: Sequence[str \| None] \| None, result_names: Sequence[str] \| None, arg_memory_kinds: Sequence[str \| None] \| None, result_memory_kinds: Sequence[str \| None] \| None, arg_layouts: Sequence[Layout \| None \| AutoLayout] \| None, result_layouts: Sequence[Layout \| None \| AutoLayout] \| None, propagated_out_mem_kinds: tuple[None \| str, ...] \| None) -> func_dialect.FuncOp` |
| `interpreters.mlir.lower_jaxpr_to_module` | Function | `(module_name: str, jaxpr: core.ClosedJaxpr, num_const_args: int, in_avals: Sequence[core.AbstractValue], ordered_effects: list[core.Effect], platforms: Sequence[str], backend: xc.Client \| None, axis_context: AxisContext, donated_args: Sequence[bool], replicated_args: Sequence[bool] \| None, arg_shardings: Sequence[JSharding \| AUTO \| None] \| None, result_shardings: Sequence[JSharding \| AUTO \| None] \| None, in_layouts: Sequence[Layout \| None \| AutoLayout] \| None, out_layouts: Sequence[Layout \| None \| AutoLayout] \| None, arg_names: Sequence[str] \| None, result_names: Sequence[str] \| None, num_replicas: int, num_partitions: int, all_default_mem_kind: bool, input_output_aliases: None \| tuple[int \| None, ...], propagated_out_mem_kinds: tuple[None \| str, ...] \| None, lowering_parameters: LoweringParameters) -> LoweringResult` |
| `interpreters.mlir.lowerable_effects` | Object | `` |
| `interpreters.mlir.make_ir_context` | Function | `() -> ir.Context` |
| `interpreters.mlir.merge_mlir_modules` | Function | `(dst_module: ir.Module, sym_name: str, src_module: ir.Module, dst_symtab: ir.SymbolTable \| None) -> str` |
| `interpreters.mlir.module_to_bytecode` | Function | `(module: ir.Module) -> bytes` |
| `interpreters.mlir.module_to_string` | Function | `(module: ir.Module, enable_debug_info) -> str` |
| `interpreters.mlir.register_constant_handler` | Function | `(type_: type, handler_fun: ConstantHandler)` |
| `interpreters.mlir.register_lowering` | Function | `(prim: core.Primitive, rule: LoweringRule, platform: str \| None, inline: bool, cacheable: bool) -> None` |
| `interpreters.mlir.shape_tensor` | Function | `(sizes: Sequence[int \| ir.RankedTensorType]) -> IrValues` |
| `interpreters.mlir.token_type` | Object | `` |
| `interpreters.mlir.unflatten_ir_values_like_types` | Function | `(xs: Iterable[ir.Value], ys: Sequence[IrTypes]) -> list[IrValues]` |
| `interpreters.partial_eval.DynamicJaxprTracer` | Class | `(trace: DynamicJaxprTrace, aval: core.AbstractValue \| core.AvalQDD, val: Atom, line_info: source_info_util.SourceInfo \| None, parent: TracingEqn \| None)` |
| `interpreters.partial_eval.JaxprTracer` | Class | `(trace: JaxprTrace, pval: PartialVal, recipe: JaxprTracerRecipe \| None)` |
| `interpreters.partial_eval.PartialVal` | Class | `(...)` |
| `interpreters.partial_eval.Val` | Object | `` |
| `interpreters.partial_eval.custom_partial_eval_rules` | Object | `` |
| `interpreters.partial_eval.dce_jaxpr` | Function | `(jaxpr: Jaxpr, used_outputs: Sequence[bool], instantiate: bool \| Sequence[bool]) -> tuple[Jaxpr, list[bool]]` |
| `interpreters.partial_eval.dce_jaxpr_call_rule` | Function | `(used_outputs: list[bool], eqn: JaxprEqn) -> tuple[list[bool], JaxprEqn \| None]` |
| `interpreters.partial_eval.dce_jaxpr_closed_call_rule` | Function | `(used_outputs: list[bool], eqn: JaxprEqn) -> tuple[list[bool], JaxprEqn \| None]` |
| `interpreters.partial_eval.dce_jaxpr_consts` | Function | `(jaxpr: Jaxpr, used_outputs: Sequence[bool], instantiate: bool \| Sequence[bool]) -> tuple[Jaxpr, list[bool], list[bool]]` |
| `interpreters.partial_eval.dce_rules` | Object | `` |
| `interpreters.partial_eval.partial_eval_jaxpr_custom_rules` | Object | `` |
| `interpreters.partial_eval.trace_to_jaxpr_dynamic` | Function | `(fun: lu.WrappedFun, in_avals: Sequence[AbstractValue \| core.AvalQDD], keep_inputs: list[bool] \| None, lower: bool, auto_dce: bool) -> tuple[Jaxpr, list[AbstractValue], list[Any]]` |
| `interpreters.partial_eval.trace_to_jaxpr_nounits` | Function | `(fun: lu.WrappedFun, pvals: Sequence[PartialVal], instantiate: bool \| Sequence[bool]) -> tuple[Jaxpr, list[PartialVal], list[core.Value]]` |
| `interpreters.pxla.ArrayMapping` | Object | `` |
| `interpreters.pxla.Chunked` | Object | `` |
| `interpreters.pxla.Index` | Object | `` |
| `interpreters.pxla.MapTracer` | Object | `` |
| `interpreters.pxla.MeshAxisName` | Object | `` |
| `interpreters.pxla.MeshComputation` | Object | `` |
| `interpreters.pxla.MeshExecutable` | Object | `` |
| `interpreters.pxla.NoSharding` | Object | `` |
| `interpreters.pxla.PmapExecutable` | Object | `` |
| `interpreters.pxla.Replicated` | Object | `` |
| `interpreters.pxla.ShardedAxis` | Object | `` |
| `interpreters.pxla.ShardingSpec` | Object | `` |
| `interpreters.pxla.Unstacked` | Object | `` |
| `interpreters.pxla.are_hlo_shardings_equal` | Object | `` |
| `interpreters.pxla.array_mapping_to_axis_resources` | Object | `` |
| `interpreters.pxla.global_aval_to_result_handler` | Object | `` |
| `interpreters.pxla.global_avals_to_results_handler` | Object | `` |
| `interpreters.pxla.global_result_handlers` | Object | `` |
| `interpreters.pxla.is_hlo_sharding_replicated` | Object | `` |
| `interpreters.pxla.op_sharding_to_indices` | Object | `` |
| `interpreters.pxla.parallel_callable` | Object | `` |
| `interpreters.pxla.shard_args` | Object | `` |
| `interpreters.pxla.spec_to_indices` | Object | `` |
| `interpreters.pxla.thread_resources` | Object | `` |
| `interpreters.pxla.xla_pmap_p` | Object | `` |
| `interpreters.traceback_util` | Object | `` |
| `interpreters.xla.Backend` | Object | `` |
| `interpreters.xla.apply_primitive` | Function | `(prim, args, params)` |
| `interpreters.xla.canonicalize_dtype_handlers` | Object | `` |
| `jacfwd` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux: bool, holomorphic: bool) -> Callable` |
| `jacobian` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux: bool, holomorphic: bool, allow_int: bool) -> Callable` |
| `jacrev` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux: bool, holomorphic: bool, allow_int: bool) -> Callable` |
| `jax2tf_associative_scan_reductions` | Object | `` |
| `jit` | Function | `(fun: Callable \| NotSpecified, in_shardings: Any, out_shardings: Any, static_argnums: int \| Sequence[int] \| None, static_argnames: str \| Iterable[str] \| None, donate_argnums: int \| Sequence[int] \| None, donate_argnames: str \| Iterable[str] \| None, keep_unused: bool, device: xc.Device \| None, backend: str \| None, inline: bool, compiler_options: dict[str, Any] \| None) -> pjit.JitWrapped \| Callable[[Callable], pjit.JitWrapped]` |
| `jvp` | Function | `(fun: Callable, primals, tangents, has_aux: bool) -> tuple[Any, ...]` |
| `lax.AccuracyMode` | Class | `(...)` |
| `lax.ConvDimensionNumbers` | Class | `(...)` |
| `lax.ConvGeneralDilatedDimensionNumbers` | Object | `` |
| `lax.DotAlgorithm` | Class | `(...)` |
| `lax.DotAlgorithmPreset` | Class | `(...)` |
| `lax.DotDimensionNumbers` | Object | `` |
| `lax.FftType` | Class | `(...)` |
| `lax.GatherDimensionNumbers` | Class | `(...)` |
| `lax.GatherScatterMode` | Class | `(...)` |
| `lax.Precision` | Class | `(...)` |
| `lax.PrecisionLike` | Object | `` |
| `lax.RaggedDotDimensionNumbers` | Class | `(dot_dimension_numbers, lhs_ragged_dimensions, rhs_group_dimensions)` |
| `lax.RandomAlgorithm` | Class | `(...)` |
| `lax.RoundingMethod` | Class | `(...)` |
| `lax.ScatterDimensionNumbers` | Class | `(...)` |
| `lax.Tolerance` | Class | `(atol: float, rtol: float, ulps: int)` |
| `lax.abs` | Function | `(x: ArrayLike) -> Array` |
| `lax.abs_p` | Object | `` |
| `lax.acos` | Function | `(x: ArrayLike) -> Array` |
| `lax.acos_p` | Object | `` |
| `lax.acosh` | Function | `(x: ArrayLike) -> Array` |
| `lax.acosh_p` | Object | `` |
| `lax.add` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.add_p` | Object | `` |
| `lax.after_all` | Function | `(operands)` |
| `lax.after_all_p` | Object | `` |
| `lax.all_gather` | Function | `(x, axis_name, axis_index_groups, axis, tiled, to: str)` |
| `lax.all_gather_done` | Function | `(x)` |
| `lax.all_gather_p` | Object | `` |
| `lax.all_gather_start` | Function | `(x, axis_name, axis, tiled)` |
| `lax.all_to_all` | Function | `(x, axis_name, split_axis, concat_axis, axis_index_groups, tiled)` |
| `lax.all_to_all_p` | Object | `` |
| `lax.and_p` | Object | `` |
| `lax.approx_max_k` | Function | `(operand: Array, k: int, reduction_dimension: int, recall_target: float, reduction_input_size_override: int, aggregate_to_topk: bool) -> tuple[Array, Array]` |
| `lax.approx_min_k` | Function | `(operand: Array, k: int, reduction_dimension: int, recall_target: float, reduction_input_size_override: int, aggregate_to_topk: bool) -> tuple[Array, Array]` |
| `lax.approx_top_k_p` | Object | `` |
| `lax.argmax` | Function | `(operand: ArrayLike, axis: int, index_dtype: DTypeLike) -> Array` |
| `lax.argmax_p` | Object | `` |
| `lax.argmin` | Function | `(operand: ArrayLike, axis: int, index_dtype: DTypeLike) -> Array` |
| `lax.argmin_p` | Object | `` |
| `lax.asin` | Function | `(x: ArrayLike) -> Array` |
| `lax.asin_p` | Object | `` |
| `lax.asinh` | Function | `(x: ArrayLike) -> Array` |
| `lax.asinh_p` | Object | `` |
| `lax.associative_scan` | Function | `(fn: Callable, elems, reverse: bool, axis: int)` |
| `lax.atan` | Function | `(x: ArrayLike) -> Array` |
| `lax.atan2` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.atan2_p` | Object | `` |
| `lax.atan_p` | Object | `` |
| `lax.atanh` | Function | `(x: ArrayLike) -> Array` |
| `lax.atanh_p` | Object | `` |
| `lax.axis_index` | Function | `(axis_name: AxisName) -> Array` |
| `lax.axis_index_p` | Object | `` |
| `lax.axis_size` | Function | `(axis_name: AxisName) -> int` |
| `lax.batch_matmul` | Function | `(lhs: Array, rhs: Array, precision: PrecisionLike) -> Array` |
| `lax.bessel_i0e` | Function | `(x: ArrayLike) -> Array` |
| `lax.bessel_i0e_p` | Object | `` |
| `lax.bessel_i1e` | Function | `(x: ArrayLike) -> Array` |
| `lax.bessel_i1e_p` | Object | `` |
| `lax.betainc` | Function | `(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> Array` |
| `lax.bitcast_convert_type` | Function | `(operand: ArrayLike, new_dtype: DTypeLike) -> Array` |
| `lax.bitcast_convert_type_p` | Object | `` |
| `lax.bitwise_and` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.bitwise_not` | Function | `(x: ArrayLike) -> Array` |
| `lax.bitwise_or` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.bitwise_xor` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.broadcast` | Function | `(operand: ArrayLike, sizes: Sequence[int], out_sharding) -> Array` |
| `lax.broadcast_in_dim` | Function | `(operand: ArrayLike, shape: Shape, broadcast_dimensions: Sequence[int], out_sharding) -> Array` |
| `lax.broadcast_in_dim_p` | Object | `` |
| `lax.broadcast_shapes` | Function | `(shapes)` |
| `lax.broadcast_to_rank` | Function | `(x: ArrayLike, rank: int) -> Array` |
| `lax.broadcasted_iota` | Function | `(dtype: DTypeLike, shape: Shape, dimension: int, out_sharding) -> Array` |
| `lax.cbrt` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.cbrt_p` | Object | `` |
| `lax.ceil` | Function | `(x: ArrayLike) -> Array` |
| `lax.ceil_p` | Object | `` |
| `lax.clamp` | Function | `(min: ArrayLike, x: ArrayLike, max: ArrayLike) -> Array` |
| `lax.clamp_p` | Object | `` |
| `lax.clz` | Function | `(x: ArrayLike) -> Array` |
| `lax.clz_p` | Object | `` |
| `lax.collapse` | Function | `(operand: Array, start_dimension: int, stop_dimension: int \| None) -> Array` |
| `lax.complex` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.complex_p` | Object | `` |
| `lax.composite` | Function | `(decomposition: Callable, name: str, version: int)` |
| `lax.concatenate` | Function | `(operands: Array \| Sequence[ArrayLike], dimension: int) -> Array` |
| `lax.concatenate_p` | Object | `` |
| `lax.cond` | Function | `(pred, true_fun: Callable, false_fun: Callable, operands, operand)` |
| `lax.cond_p` | Object | `` |
| `lax.conj` | Function | `(x: ArrayLike) -> Array` |
| `lax.conj_p` | Object | `` |
| `lax.conv` | Function | `(lhs: Array, rhs: Array, window_strides: Sequence[int], padding: str, precision: lax.PrecisionLike, preferred_element_type: DTypeLike \| None) -> Array` |
| `lax.conv_dimension_numbers` | Function | `(lhs_shape, rhs_shape, dimension_numbers) -> ConvDimensionNumbers` |
| `lax.conv_general_dilated` | Function | `(lhs: Array, rhs: Array, window_strides: Sequence[int], padding: str \| Sequence[tuple[int, int]], lhs_dilation: Sequence[int] \| None, rhs_dilation: Sequence[int] \| None, dimension_numbers: ConvGeneralDilatedDimensionNumbers, feature_group_count: int, batch_group_count: int, precision: lax.PrecisionLike, preferred_element_type: DTypeLike \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `lax.conv_general_dilated_local` | Function | `(lhs: ArrayLike, rhs: ArrayLike, window_strides: Sequence[int], padding: str \| Sequence[tuple[int, int]], filter_shape: Sequence[int], lhs_dilation: Sequence[int] \| None, rhs_dilation: Sequence[int] \| None, dimension_numbers: convolution.ConvGeneralDilatedDimensionNumbers \| None, precision: lax.PrecisionLike) -> Array` |
| `lax.conv_general_dilated_p` | Object | `` |
| `lax.conv_general_dilated_patches` | Function | `(lhs: ArrayLike, filter_shape: Sequence[int], window_strides: Sequence[int], padding: str \| Sequence[tuple[int, int]], lhs_dilation: Sequence[int] \| None, rhs_dilation: Sequence[int] \| None, dimension_numbers: convolution.ConvGeneralDilatedDimensionNumbers \| None, precision: lax.Precision \| None, preferred_element_type: DType \| None) -> Array` |
| `lax.conv_general_permutations` | Function | `(dimension_numbers)` |
| `lax.conv_general_shape_tuple` | Function | `(lhs_shape, rhs_shape, window_strides, padding, dimension_numbers)` |
| `lax.conv_shape_tuple` | Function | `(lhs_shape, rhs_shape, strides, pads, batch_group_count)` |
| `lax.conv_transpose` | Function | `(lhs: Array, rhs: Array, strides: Sequence[int], padding: str \| Sequence[tuple[int, int]], rhs_dilation: Sequence[int] \| None, dimension_numbers: ConvGeneralDilatedDimensionNumbers, transpose_kernel: bool, precision: lax.PrecisionLike, preferred_element_type: DTypeLike \| None, use_consistent_padding: bool) -> Array` |
| `lax.conv_transpose_shape_tuple` | Function | `(lhs_shape, rhs_shape, window_strides, padding, dimension_numbers)` |
| `lax.conv_with_general_padding` | Function | `(lhs: Array, rhs: Array, window_strides: Sequence[int], padding: str \| Sequence[tuple[int, int]], lhs_dilation: Sequence[int] \| None, rhs_dilation: Sequence[int] \| None, precision: lax.PrecisionLike, preferred_element_type: DTypeLike \| None) -> Array` |
| `lax.convert_element_type` | Function | `(operand: ArrayLike, new_dtype: DTypeLike \| dtypes.ExtendedDType) -> Array` |
| `lax.convert_element_type_p` | Object | `` |
| `lax.copy_p` | Object | `` |
| `lax.cos` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.cos_p` | Object | `` |
| `lax.cosh` | Function | `(x: ArrayLike) -> Array` |
| `lax.cosh_p` | Object | `` |
| `lax.create_token` | Function | `(_)` |
| `lax.create_token_p` | Object | `` |
| `lax.cumlogsumexp` | Function | `(operand: Array, axis: int, reverse: bool) -> Array` |
| `lax.cumlogsumexp_p` | Object | `` |
| `lax.cummax` | Function | `(operand: Array, axis: int, reverse: bool) -> Array` |
| `lax.cummax_p` | Object | `` |
| `lax.cummin` | Function | `(operand: Array, axis: int, reverse: bool) -> Array` |
| `lax.cummin_p` | Object | `` |
| `lax.cumprod` | Function | `(operand: Array, axis: int, reverse: bool) -> Array` |
| `lax.cumprod_p` | Object | `` |
| `lax.cumsum` | Function | `(operand: Array, axis: int, reverse: bool) -> Array` |
| `lax.cumsum_p` | Object | `` |
| `lax.custom_linear_solve` | Function | `(matvec: Callable, b: Any, solve: Callable[[Callable, Any], Any], transpose_solve: Callable[[Callable, Any], Any] \| None, symmetric, has_aux)` |
| `lax.custom_root` | Function | `(f: Callable, initial_guess: Any, solve: Callable[[Callable, Any], Any], tangent_solve: Callable[[Callable, Any], Any], has_aux)` |
| `lax.dce_sink` | Function | `(val)` |
| `lax.dce_sink_p` | Object | `` |
| `lax.device_put_p` | Object | `` |
| `lax.digamma` | Function | `(x: ArrayLike) -> Array` |
| `lax.digamma_p` | Object | `` |
| `lax.div` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.div_p` | Object | `` |
| `lax.dot` | Function | `(lhs: ArrayLike, rhs: ArrayLike, args, dimension_numbers: DotDimensionNumbers \| None, precision: PrecisionLike, preferred_element_type: DTypeLike \| None, out_sharding) -> Array` |
| `lax.dot_general` | Function | `(lhs: ArrayLike, rhs: ArrayLike, dimension_numbers: DotDimensionNumbers, precision: PrecisionLike, preferred_element_type: DTypeLike \| None, out_sharding) -> Array` |
| `lax.dot_general_p` | Object | `` |
| `lax.dtype` | Object | `` |
| `lax.dynamic_index_in_dim` | Function | `(operand: Array \| np.ndarray, index: ArrayLike, axis: int, keepdims: bool, allow_negative_indices: bool) -> Array` |
| `lax.dynamic_slice` | Function | `(operand: Array \| np.ndarray, start_indices: Array \| np.ndarray \| Sequence[ArrayLike], slice_sizes: Shape, allow_negative_indices: bool \| Sequence[bool]) -> Array` |
| `lax.dynamic_slice_in_dim` | Function | `(operand: Array \| np.ndarray, start_index: ArrayLike, slice_size: int, axis: int, allow_negative_indices: bool) -> Array` |
| `lax.dynamic_slice_p` | Object | `` |
| `lax.dynamic_update_index_in_dim` | Function | `(operand: Array \| np.ndarray, update: ArrayLike, index: ArrayLike, axis: int, allow_negative_indices: bool) -> Array` |
| `lax.dynamic_update_slice` | Function | `(operand: Array \| np.ndarray, update: ArrayLike, start_indices: Array \| Sequence[ArrayLike], allow_negative_indices: bool \| Sequence[bool]) -> Array` |
| `lax.dynamic_update_slice_in_dim` | Function | `(operand: Array \| np.ndarray, update: ArrayLike, start_index: ArrayLike, axis: int, allow_negative_indices: bool) -> Array` |
| `lax.dynamic_update_slice_p` | Object | `` |
| `lax.empty` | Function | `(shape, dtype, out_sharding)` |
| `lax.eq` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.eq_p` | Object | `` |
| `lax.eq_to_p` | Object | `` |
| `lax.erf` | Function | `(x: ArrayLike) -> Array` |
| `lax.erf_inv` | Function | `(x: ArrayLike) -> Array` |
| `lax.erf_inv_p` | Object | `` |
| `lax.erf_p` | Object | `` |
| `lax.erfc` | Function | `(x: ArrayLike) -> Array` |
| `lax.erfc_p` | Object | `` |
| `lax.exp` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.exp2` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.exp2_p` | Object | `` |
| `lax.exp_p` | Object | `` |
| `lax.expand_dims` | Function | `(array: ArrayLike, dimensions: Sequence[int]) -> Array` |
| `lax.expm1` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.expm1_p` | Object | `` |
| `lax.fft` | Function | `(x, fft_type: FftType \| str, fft_lengths: Sequence[int])` |
| `lax.fft_p` | Object | `` |
| `lax.floor` | Function | `(x: ArrayLike) -> Array` |
| `lax.floor_p` | Object | `` |
| `lax.fori_loop` | Function | `(lower, upper, body_fun, init_val, unroll: int \| bool \| None)` |
| `lax.full` | Function | `(shape: Shape, fill_value: ArrayLike, dtype: DTypeLike \| None, sharding: Sharding \| None) -> Array` |
| `lax.full_like` | Function | `(x: ArrayLike \| DuckTypedArray, fill_value: ArrayLike, dtype: DTypeLike \| None, shape: Shape \| None, sharding: Sharding \| None) -> Array` |
| `lax.gather` | Function | `(operand: ArrayLike, start_indices: ArrayLike, dimension_numbers: GatherDimensionNumbers, slice_sizes: Shape, unique_indices: bool, indices_are_sorted: bool, mode: str \| GatherScatterMode \| None, fill_value) -> Array` |
| `lax.gather_p` | Object | `` |
| `lax.ge` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.ge_p` | Object | `` |
| `lax.gt` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.gt_p` | Object | `` |
| `lax.igamma` | Function | `(a: ArrayLike, x: ArrayLike) -> Array` |
| `lax.igamma_grad_a` | Function | `(a: ArrayLike, x: ArrayLike) -> Array` |
| `lax.igamma_grad_a_p` | Object | `` |
| `lax.igamma_p` | Object | `` |
| `lax.igammac` | Function | `(a: ArrayLike, x: ArrayLike) -> Array` |
| `lax.igammac_p` | Object | `` |
| `lax.imag` | Function | `(x: ArrayLike) -> Array` |
| `lax.imag_p` | Object | `` |
| `lax.index_in_dim` | Function | `(operand: Array \| np.ndarray, index: int, axis: int, keepdims: bool) -> Array` |
| `lax.index_take` | Function | `(src: Array, idxs: Array, axes: Sequence[int]) -> Array` |
| `lax.integer_pow` | Function | `(x: ArrayLike, y: int) -> Array` |
| `lax.integer_pow_p` | Object | `` |
| `lax.iota` | Function | `(dtype: DTypeLike, size: int) -> Array` |
| `lax.iota_p` | Object | `` |
| `lax.is_finite` | Function | `(x: ArrayLike) -> Array` |
| `lax.is_finite_p` | Object | `` |
| `lax.le` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.le_p` | Object | `` |
| `lax.le_to_p` | Object | `` |
| `lax.lgamma` | Function | `(x: ArrayLike) -> Array` |
| `lax.lgamma_p` | Object | `` |
| `lax.linalg.EigImplementation` | Class | `(...)` |
| `lax.linalg.EighImplementation` | Class | `(...)` |
| `lax.linalg.SvdAlgorithm` | Class | `(...)` |
| `lax.linalg.cholesky` | Function | `(x: Array, symmetrize_input: bool) -> Array` |
| `lax.linalg.cholesky_p` | Object | `` |
| `lax.linalg.cholesky_update` | Function | `(r_matrix: ArrayLike, w_vector: ArrayLike) -> Array` |
| `lax.linalg.cholesky_update_p` | Object | `` |
| `lax.linalg.eig` | Function | `(x: ArrayLike, compute_left_eigenvectors: bool, compute_right_eigenvectors: bool, implementation: EigImplementation \| None, use_magma: bool \| None) -> list[Array]` |
| `lax.linalg.eig_p` | Object | `` |
| `lax.linalg.eigh` | Function | `(x: Array, lower: bool, symmetrize_input: bool, sort_eigenvalues: bool, subset_by_index: tuple[int, int] \| None, implementation: EighImplementation \| None) -> tuple[Array, Array]` |
| `lax.linalg.eigh_p` | Object | `` |
| `lax.linalg.hessenberg` | Function | `(a: ArrayLike) -> tuple[Array, Array]` |
| `lax.linalg.hessenberg_p` | Object | `` |
| `lax.linalg.householder_product` | Function | `(a: ArrayLike, taus: ArrayLike) -> Array` |
| `lax.linalg.householder_product_p` | Object | `` |
| `lax.linalg.lu` | Function | `(x: ArrayLike) -> tuple[Array, Array, Array]` |
| `lax.linalg.lu_p` | Object | `` |
| `lax.linalg.lu_pivots_to_permutation` | Function | `(pivots: ArrayLike, permutation_size: int) -> Array` |
| `lax.linalg.lu_pivots_to_permutation_p` | Object | `` |
| `lax.linalg.qdwh` | Function | `(x, is_hermitian: bool, max_iterations: int \| None, eps: float \| None, dynamic_shape: tuple[int, int] \| None)` |
| `lax.linalg.qr` | Function | `(x: ArrayLike, pivoting: bool, full_matrices: bool, use_magma: bool \| None) -> tuple[Array, Array] \| tuple[Array, Array, Array]` |
| `lax.linalg.qr_p` | Object | `` |
| `lax.linalg.schur` | Function | `(x: ArrayLike, compute_schur_vectors: bool, sort_eig_vals: bool, select_callable: Callable[..., Any] \| None) -> tuple[Array, Array]` |
| `lax.linalg.schur_p` | Object | `` |
| `lax.linalg.svd` | Function | `(x: ArrayLike, full_matrices: bool, compute_uv: bool, subset_by_index: tuple[int, int] \| None, algorithm: SvdAlgorithm \| None) -> Array \| tuple[Array, Array, Array]` |
| `lax.linalg.svd_p` | Object | `` |
| `lax.linalg.symmetric_product` | Function | `(a_matrix: ArrayLike, c_matrix: ArrayLike, alpha: float, beta: float, symmetrize_output: bool)` |
| `lax.linalg.symmetric_product_p` | Object | `` |
| `lax.linalg.triangular_solve` | Function | `(a: ArrayLike, b: ArrayLike, left_side: bool, lower: bool, transpose_a: bool, conjugate_a: bool, unit_diagonal: bool) -> Array` |
| `lax.linalg.triangular_solve_p` | Object | `` |
| `lax.linalg.tridiagonal` | Function | `(a: ArrayLike, lower: bool) -> tuple[Array, Array, Array, Array]` |
| `lax.linalg.tridiagonal_p` | Object | `` |
| `lax.linalg.tridiagonal_solve` | Function | `(dl: Array, d: Array, du: Array, b: Array) -> Array` |
| `lax.linalg.tridiagonal_solve_p` | Object | `` |
| `lax.linear_solve_p` | Object | `` |
| `lax.log` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.log1p` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.log1p_p` | Object | `` |
| `lax.log_p` | Object | `` |
| `lax.logistic` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.logistic_p` | Object | `` |
| `lax.lt` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.lt_p` | Object | `` |
| `lax.lt_to_p` | Object | `` |
| `lax.map` | Function | `(f, xs, batch_size: int \| None)` |
| `lax.max` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.max_p` | Object | `` |
| `lax.min` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.min_p` | Object | `` |
| `lax.mul` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.mul_p` | Object | `` |
| `lax.ne` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.ne_p` | Object | `` |
| `lax.neg` | Function | `(x: ArrayLike) -> Array` |
| `lax.neg_p` | Object | `` |
| `lax.nextafter` | Function | `(x1: ArrayLike, x2: ArrayLike) -> Array` |
| `lax.nextafter_p` | Object | `` |
| `lax.not_p` | Object | `` |
| `lax.optimization_barrier` | Function | `(operand)` |
| `lax.optimization_barrier_p` | Object | `` |
| `lax.or_p` | Object | `` |
| `lax.pad` | Function | `(operand: ArrayLike, padding_value: ArrayLike, padding_config: Sequence[tuple[int, int, int]]) -> Array` |
| `lax.pad_p` | Object | `` |
| `lax.padtype_to_pads` | Function | `(in_shape: Sequence[int] \| np.ndarray, window_shape: Sequence[int] \| np.ndarray, window_strides: Sequence[int] \| np.ndarray, padding: str \| PaddingType) -> list[tuple[int, int]]` |
| `lax.pbroadcast` | Function | `(x, axis_name, source)` |
| `lax.pcast` | Function | `(x, axis_name, to: str)` |
| `lax.platform_dependent` | Function | `(args: Any, default: Callable[..., _T] \| None, per_platform: Callable[..., _T])` |
| `lax.pmax` | Function | `(x, axis_name, axis_index_groups)` |
| `lax.pmax_p` | Object | `` |
| `lax.pmean` | Function | `(x, axis_name, axis_index_groups)` |
| `lax.pmin` | Function | `(x, axis_name, axis_index_groups)` |
| `lax.pmin_p` | Object | `` |
| `lax.polygamma` | Function | `(m: ArrayLike, x: ArrayLike) -> Array` |
| `lax.polygamma_p` | Object | `` |
| `lax.population_count` | Function | `(x: ArrayLike) -> Array` |
| `lax.population_count_p` | Object | `` |
| `lax.pow` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.pow_p` | Object | `` |
| `lax.ppermute` | Function | `(x, axis_name, perm)` |
| `lax.ppermute_p` | Object | `` |
| `lax.precv` | Function | `(token, out_shape, axis_name, perm)` |
| `lax.psend` | Function | `(x, axis_name, perm)` |
| `lax.pshuffle` | Function | `(x, axis_name, perm)` |
| `lax.psum` | Function | `(x, axis_name, axis_index_groups)` |
| `lax.psum_p` | Object | `` |
| `lax.psum_scatter` | Function | `(x, axis_name, scatter_dimension, axis_index_groups, tiled)` |
| `lax.pswapaxes` | Function | `(x, axis_name, axis, axis_index_groups)` |
| `lax.pvary` | Object | `` |
| `lax.ragged_all_to_all` | Function | `(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, axis_name, axis_index_groups)` |
| `lax.ragged_all_to_all_p` | Object | `` |
| `lax.ragged_dot` | Function | `(lhs: Array, rhs: Array, group_sizes: Array, precision: PrecisionLike, preferred_element_type: DTypeLike \| None, group_offset: Array \| None) -> Array` |
| `lax.ragged_dot_general` | Function | `(lhs: Array, rhs: Array, group_sizes: Array, ragged_dot_dimension_numbers: RaggedDotDimensionNumbers, precision: PrecisionLike, preferred_element_type: DTypeLike \| None, group_offset: Array \| None) -> Array` |
| `lax.random_gamma_grad` | Function | `(a: ArrayLike, x: ArrayLike, dtype) -> Array` |
| `lax.real` | Function | `(x: ArrayLike) -> Array` |
| `lax.real_p` | Object | `` |
| `lax.reciprocal` | Function | `(x: ArrayLike) -> Array` |
| `lax.reduce` | Function | `(operands: Any, init_values: Any, computation: Callable[[Any, Any], Any], dimensions: Sequence[int], out_sharding: NamedSharding \| P \| None) -> Any` |
| `lax.reduce_and` | Function | `(operand: ArrayLike, axes: Sequence[int]) -> Array` |
| `lax.reduce_and_p` | Object | `` |
| `lax.reduce_max` | Function | `(operand: ArrayLike, axes: Sequence[int]) -> Array` |
| `lax.reduce_max_p` | Object | `` |
| `lax.reduce_min` | Function | `(operand: ArrayLike, axes: Sequence[int]) -> Array` |
| `lax.reduce_min_p` | Object | `` |
| `lax.reduce_or` | Function | `(operand: ArrayLike, axes: Sequence[int]) -> Array` |
| `lax.reduce_or_p` | Object | `` |
| `lax.reduce_p` | Object | `` |
| `lax.reduce_precision` | Function | `(operand: float \| ArrayLike, exponent_bits: int, mantissa_bits: int) -> Array` |
| `lax.reduce_precision_p` | Object | `` |
| `lax.reduce_prod` | Function | `(operand: ArrayLike, axes: Sequence[int]) -> Array` |
| `lax.reduce_prod_p` | Object | `` |
| `lax.reduce_sum` | Function | `(operand: ArrayLike, axes: Sequence[int], out_sharding) -> Array` |
| `lax.reduce_sum_p` | Object | `` |
| `lax.reduce_window` | Function | `(operand: Any, init_value: Any, computation: Callable, window_dimensions: core.Shape, window_strides: Sequence[int] \| None, padding: str \| Sequence[tuple[int, int]], base_dilation: Sequence[int] \| None, window_dilation: Sequence[int] \| None) -> Any` |
| `lax.reduce_window_max_p` | Object | `` |
| `lax.reduce_window_min_p` | Object | `` |
| `lax.reduce_window_p` | Object | `` |
| `lax.reduce_window_shape_tuple` | Function | `(operand_shape, window_dimensions, window_strides, padding, base_dilation, window_dilation)` |
| `lax.reduce_window_sum_p` | Object | `` |
| `lax.reduce_xor` | Function | `(operand: ArrayLike, axes: Sequence[int]) -> Array` |
| `lax.reduce_xor_p` | Object | `` |
| `lax.regularized_incomplete_beta_p` | Object | `` |
| `lax.rem` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.rem_p` | Object | `` |
| `lax.reshape` | Function | `(operand: ArrayLike, new_sizes: Shape, dimensions: Sequence[int] \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `lax.reshape_p` | Object | `` |
| `lax.rev` | Function | `(operand: ArrayLike, dimensions: Sequence[int]) -> Array` |
| `lax.rev_p` | Object | `` |
| `lax.rng_bit_generator` | Function | `(key, shape, dtype, algorithm, out_sharding)` |
| `lax.rng_bit_generator_p` | Object | `` |
| `lax.rng_uniform` | Function | `(a, b, shape)` |
| `lax.rng_uniform_p` | Object | `` |
| `lax.round` | Function | `(x: ArrayLike, rounding_method: RoundingMethod) -> Array` |
| `lax.round_p` | Object | `` |
| `lax.rsqrt` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.rsqrt_p` | Object | `` |
| `lax.scaled_dot` | Function | `(lhs: Array, rhs: Array, lhs_scale: Array \| None, rhs_scale: Array \| None, dimension_numbers: lax.DotDimensionNumbers \| None, preferred_element_type: DTypeLike \| None)` |
| `lax.scan` | Function | `(f: Callable[[Carry, X], tuple[Carry, Y]], init: Carry, xs: X \| None, length: int \| None, reverse: bool, unroll: int \| bool, _split_transpose: bool) -> tuple[Carry, Y]` |
| `lax.scan_p` | Object | `` |
| `lax.scatter` | Function | `(operand: ArrayLike, scatter_indices: ArrayLike, updates: ArrayLike, dimension_numbers: ScatterDimensionNumbers, indices_are_sorted: bool, unique_indices: bool, mode: str \| GatherScatterMode \| None) -> Array` |
| `lax.scatter_add` | Function | `(operand: ArrayLike, scatter_indices: ArrayLike, updates: ArrayLike, dimension_numbers: ScatterDimensionNumbers, indices_are_sorted: bool, unique_indices: bool, mode: str \| GatherScatterMode \| None) -> Array` |
| `lax.scatter_add_p` | Object | `` |
| `lax.scatter_apply` | Function | `(operand: Array, scatter_indices: Array, func: Callable[[Array], Array], dimension_numbers: ScatterDimensionNumbers, update_shape: Shape, indices_are_sorted: bool, unique_indices: bool, mode: str \| GatherScatterMode \| None) -> Array` |
| `lax.scatter_max` | Function | `(operand: ArrayLike, scatter_indices: ArrayLike, updates: ArrayLike, dimension_numbers: ScatterDimensionNumbers, indices_are_sorted: bool, unique_indices: bool, mode: str \| GatherScatterMode \| None) -> Array` |
| `lax.scatter_max_p` | Object | `` |
| `lax.scatter_min` | Function | `(operand: ArrayLike, scatter_indices: ArrayLike, updates: ArrayLike, dimension_numbers: ScatterDimensionNumbers, indices_are_sorted: bool, unique_indices: bool, mode: str \| GatherScatterMode \| None) -> Array` |
| `lax.scatter_min_p` | Object | `` |
| `lax.scatter_mul` | Function | `(operand: ArrayLike, scatter_indices: ArrayLike, updates: ArrayLike, dimension_numbers: ScatterDimensionNumbers, indices_are_sorted: bool, unique_indices: bool, mode: str \| GatherScatterMode \| None) -> Array` |
| `lax.scatter_mul_p` | Object | `` |
| `lax.scatter_p` | Object | `` |
| `lax.scatter_sub` | Function | `(operand: ArrayLike, scatter_indices: ArrayLike, updates: ArrayLike, dimension_numbers: ScatterDimensionNumbers, indices_are_sorted: bool, unique_indices: bool, mode: str \| GatherScatterMode \| None) -> Array` |
| `lax.scatter_sub_p` | Object | `` |
| `lax.select` | Function | `(pred: ArrayLike, on_true: ArrayLike, on_false: ArrayLike) -> Array` |
| `lax.select_and_gather_add_p` | Object | `` |
| `lax.select_and_scatter_add_p` | Object | `` |
| `lax.select_and_scatter_p` | Object | `` |
| `lax.select_n` | Function | `(which: ArrayLike, cases: ArrayLike) -> Array` |
| `lax.select_n_p` | Object | `` |
| `lax.shape_as_value` | Function | `(shape: core.Shape)` |
| `lax.sharding_constraint_p` | Object | `` |
| `lax.shift_left` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.shift_left_p` | Object | `` |
| `lax.shift_right_arithmetic` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.shift_right_arithmetic_p` | Object | `` |
| `lax.shift_right_logical` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.shift_right_logical_p` | Object | `` |
| `lax.sign` | Function | `(x: ArrayLike) -> Array` |
| `lax.sign_p` | Object | `` |
| `lax.sin` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.sin_p` | Object | `` |
| `lax.sinh` | Function | `(x: ArrayLike) -> Array` |
| `lax.sinh_p` | Object | `` |
| `lax.slice` | Function | `(operand: ArrayLike, start_indices: Sequence[int], limit_indices: Sequence[int], strides: Sequence[int] \| None) -> Array` |
| `lax.slice_in_dim` | Function | `(operand: Array \| np.ndarray, start_index: int \| None, limit_index: int \| None, stride: int, axis: int) -> Array` |
| `lax.slice_p` | Object | `` |
| `lax.sort` | Function | `(operand: Array \| Sequence[Array], dimension: int, is_stable: bool, num_keys: int) -> Array \| tuple[Array, ...]` |
| `lax.sort_key_val` | Function | `(keys: Array, values: ArrayLike, dimension: int, is_stable: bool) -> tuple[Array, Array]` |
| `lax.sort_p` | Object | `` |
| `lax.split` | Function | `(operand: ArrayLike, sizes: Sequence[DimSize], axis: int) -> Sequence[Array]` |
| `lax.split_p` | Object | `` |
| `lax.sqrt` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.sqrt_p` | Object | `` |
| `lax.square` | Function | `(x: ArrayLike) -> Array` |
| `lax.square_p` | Object | `` |
| `lax.squeeze` | Function | `(array: ArrayLike, dimensions: Sequence[int]) -> Array` |
| `lax.squeeze_p` | Object | `` |
| `lax.stop_gradient` | Function | `(x: T) -> T` |
| `lax.stop_gradient_p` | Object | `` |
| `lax.sub` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `lax.sub_p` | Object | `` |
| `lax.switch` | Function | `(index, branches: Sequence[Callable], operands: Any, operand: Any)` |
| `lax.tan` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.tan_p` | Object | `` |
| `lax.tanh` | Function | `(x: ArrayLike, accuracy: Tolerance \| AccuracyMode \| None) -> Array` |
| `lax.tanh_p` | Object | `` |
| `lax.tile` | Function | `(operand: ArrayLike, reps: Sequence[int]) -> Array` |
| `lax.tile_p` | Object | `` |
| `lax.top_k` | Function | `(operand: ArrayLike, k: int, axis: int) -> tuple[Array, Array]` |
| `lax.top_k_p` | Object | `` |
| `lax.transpose` | Function | `(operand: ArrayLike, permutation: Sequence[int] \| np.ndarray) -> Array` |
| `lax.transpose_p` | Object | `` |
| `lax.while_loop` | Function | `(cond_fun: Callable[[T], BooleanNumeric], body_fun: Callable[[T], T], init_val: T) -> T` |
| `lax.while_p` | Object | `` |
| `lax.with_sharding_constraint` | Function | `(x, shardings)` |
| `lax.xor_p` | Object | `` |
| `lax.zeta` | Function | `(x: ArrayLike, q: ArrayLike) -> Array` |
| `lax.zeta_p` | Object | `` |
| `legacy_prng_key` | Object | `` |
| `linear_transpose` | Function | `(fun: Callable, primals, reduce_axes) -> Callable` |
| `linearize` | Function | `(fun: Callable, primals, has_aux: bool) -> tuple[Any, Callable] \| tuple[Any, Callable, Any]` |
| `live_arrays` | Function | `(platform)` |
| `local_device_count` | Function | `(backend: str \| xla_client.Client \| None) -> int` |
| `local_devices` | Function | `(process_index: int \| None, backend: str \| xla_client.Client \| None, host_id: int \| None) -> list[xla_client.Device]` |
| `log_compiles` | Object | `` |
| `make_array_from_callback` | Function | `(shape: Shape, sharding: Sharding \| Format, data_callback: Callable[[Index \| None], ArrayLike], dtype: DTypeLike \| None) -> ArrayImpl` |
| `make_array_from_process_local_data` | Function | `(sharding, local_data, global_shape)` |
| `make_array_from_single_device_arrays` | Function | `(shape: Shape, sharding: Sharding, arrays: Sequence[basearray.Array], dtype: DTypeLike \| None) -> ArrayImpl` |
| `make_jaxpr` | Function | `(fun: Callable, static_argnums: int \| Sequence[int], axis_env: Sequence[tuple[AxisName, int]] \| None, return_shape: bool) -> Callable[..., core.ClosedJaxpr \| tuple[core.ClosedJaxpr, Any]]` |
| `make_mesh` | Function | `(axis_shapes: Sequence[int], axis_names: Sequence[str], axis_types: tuple[mesh_lib.AxisType, ...] \| None, devices: Sequence[xc.Device] \| None) -> mesh_lib.Mesh` |
| `make_user_context` | Function | `(default_value)` |
| `memory.Space` | Class | `(...)` |
| `mlir` | Object | `` |
| `monitoring.clear_event_listeners` | Function | `()` |
| `monitoring.record_event` | Function | `(event: str, kwargs: str \| int) -> None` |
| `monitoring.record_event_duration_secs` | Function | `(event: str, duration: float, kwargs: str \| int) -> None` |
| `monitoring.record_event_time_span` | Function | `(event: str, start_time: float, end_time: float, kwargs: str \| int) -> None` |
| `monitoring.record_scalar` | Function | `(event: str, value: float \| int, kwargs: str \| int) -> None` |
| `monitoring.register_event_duration_secs_listener` | Function | `(callback: EventDurationListenerWithMetadata) -> None` |
| `monitoring.register_event_listener` | Function | `(callback: EventListenerWithMetadata) -> None` |
| `monitoring.register_event_time_span_listener` | Function | `(callback: EventTimeSpanListenerWithMetadata) -> None` |
| `monitoring.register_scalar_listener` | Function | `(callback: ScalarListenerWithMetadata) -> None` |
| `monitoring.unregister_event_duration_listener` | Function | `(callback: EventDurationListenerWithMetadata) -> None` |
| `monitoring.unregister_event_listener` | Function | `(callback: EventListenerWithMetadata) -> None` |
| `monitoring.unregister_event_time_span_listener` | Function | `(callback: EventTimeSpanListenerWithMetadata) -> None` |
| `monitoring.unregister_scalar_listener` | Function | `(callback: ScalarListenerWithMetadata) -> None` |
| `named_call` | Function | `(fun: F, name: str \| None) -> F` |
| `named_scope` | Function | `(name: str) -> source_info_util.ExtendNameStackContextManager` |
| `new_ref` | Function | `(init_val: Any, memory_space: Any) -> core.Ref` |
| `nn.Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `nn.ArrayLike` | Object | `` |
| `nn.Axis` | Object | `` |
| `nn.AxisName` | Object | `` |
| `nn.BlockScaleConfig` | Class | `(mode: str, block_size: int, data_type: DTypeLike, scale_type: DTypeLike, global_scale: Array \| None, infer_only: bool)` |
| `nn.DTypeLike` | Object | `` |
| `nn.DotDimensionNumbers` | Object | `` |
| `nn.celu` | Function | `(x: ArrayLike, alpha: ArrayLike) -> Array` |
| `nn.elu` | Function | `(x: ArrayLike, alpha: ArrayLike) -> Array` |
| `nn.gelu` | Function | `(x: ArrayLike, approximate: bool) -> Array` |
| `nn.get_scaled_dot_general_config` | Function | `(mode: Literal['nvfp4', 'mxfp8'], global_scale: Array \| None) -> BlockScaleConfig` |
| `nn.glu` | Function | `(x: ArrayLike, axis: int) -> Array` |
| `nn.hard_sigmoid` | Function | `(x: ArrayLike) -> Array` |
| `nn.hard_silu` | Function | `(x: ArrayLike) -> Array` |
| `nn.hard_swish` | Function | `(x: ArrayLike) -> Array` |
| `nn.hard_tanh` | Function | `(x: ArrayLike) -> Array` |
| `nn.identity` | Function | `(x: ArrayLike) -> Array` |
| `nn.initializers.Initializer` | Class | `(...)` |
| `nn.initializers.constant` | Function | `(value: ArrayLike, dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.delta_orthogonal` | Function | `(scale: RealNumeric, column_axis: int, dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.glorot_normal` | Function | `(in_axis: int \| Sequence[int], out_axis: int \| Sequence[int], batch_axis: int \| Sequence[int], dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.glorot_uniform` | Function | `(in_axis: int \| Sequence[int], out_axis: int \| Sequence[int], batch_axis: int \| Sequence[int], dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.he_normal` | Function | `(in_axis: int \| Sequence[int], out_axis: int \| Sequence[int], batch_axis: int \| Sequence[int], dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.he_uniform` | Function | `(in_axis: int \| Sequence[int], out_axis: int \| Sequence[int], batch_axis: int \| Sequence[int], dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.kaiming_normal` | Object | `` |
| `nn.initializers.kaiming_uniform` | Object | `` |
| `nn.initializers.lecun_normal` | Function | `(in_axis: int \| Sequence[int], out_axis: int \| Sequence[int], batch_axis: int \| Sequence[int], dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.lecun_uniform` | Function | `(in_axis: int \| Sequence[int], out_axis: int \| Sequence[int], batch_axis: int \| Sequence[int], dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.normal` | Function | `(stddev: RealNumeric, dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.ones` | Function | `(key: Array, shape: core.Shape, dtype: DTypeLikeInexact \| None, out_sharding: OutShardingType) -> Array` |
| `nn.initializers.orthogonal` | Function | `(scale: RealNumeric, column_axis: int, dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.truncated_normal` | Function | `(stddev: RealNumeric, dtype: DTypeLikeInexact \| None, lower: RealNumeric, upper: RealNumeric) -> Initializer` |
| `nn.initializers.uniform` | Function | `(scale: RealNumeric, dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.variance_scaling` | Function | `(scale: RealNumeric, mode: Literal['fan_in'] \| Literal['fan_out'] \| Literal['fan_avg'] \| Literal['fan_geo_avg'], distribution: Literal['truncated_normal'] \| Literal['normal'] \| Literal['uniform'], in_axis: int \| Sequence[int], out_axis: int \| Sequence[int], batch_axis: int \| Sequence[int], dtype: DTypeLikeInexact \| None) -> Initializer` |
| `nn.initializers.xavier_normal` | Object | `` |
| `nn.initializers.xavier_uniform` | Object | `` |
| `nn.initializers.zeros` | Function | `(key: Array, shape: core.Shape, dtype: DTypeLikeInexact \| None, out_sharding: OutShardingType) -> Array` |
| `nn.leaky_relu` | Function | `(x: ArrayLike, negative_slope: ArrayLike) -> Array` |
| `nn.log1mexp` | Function | `(x: ArrayLike) -> Array` |
| `nn.log_sigmoid` | Function | `(x: ArrayLike) -> Array` |
| `nn.log_softmax` | Function | `(x: ArrayLike, axis: Axis, where: ArrayLike \| None) -> Array` |
| `nn.logmeanexp` | Function | `(x: ArrayLike, axis: Axis, where: ArrayLike \| None, keepdims: bool) -> Array` |
| `nn.mish` | Function | `(x: ArrayLike) -> Array` |
| `nn.one_hot` | Function | `(x: Any, num_classes: int, dtype: Any, axis: int \| AxisName) -> Array` |
| `nn.relu` | Function | `(x: ArrayLike) -> Array` |
| `nn.relu6` | Function | `(x: ArrayLike) -> Array` |
| `nn.scaled_dot_general` | Function | `(lhs: ArrayLike, rhs: ArrayLike, dimension_numbers: DotDimensionNumbers, preferred_element_type: DTypeLike, configs: List[BlockScaleConfig] \| None, implementation: Literal['cudnn'] \| None) -> Array` |
| `nn.scaled_matmul` | Function | `(lhs: Array, rhs: Array, lhs_scales: Array, rhs_scales: Array, preferred_element_type: DTypeLike) -> Array` |
| `nn.selu` | Function | `(x: ArrayLike) -> Array` |
| `nn.sigmoid` | Function | `(x: ArrayLike) -> Array` |
| `nn.silu` | Function | `(x: ArrayLike) -> Array` |
| `nn.soft_sign` | Function | `(x: ArrayLike) -> Array` |
| `nn.softmax` | Function | `(x: ArrayLike, axis: Axis, where: ArrayLike \| None) -> Array` |
| `nn.softplus` | Function | `(x: ArrayLike) -> Array` |
| `nn.sparse_plus` | Function | `(x: ArrayLike) -> Array` |
| `nn.sparse_sigmoid` | Function | `(x: ArrayLike) -> Array` |
| `nn.squareplus` | Function | `(x: ArrayLike, b: ArrayLike) -> Array` |
| `nn.standardize` | Function | `(x: ArrayLike, axis: Axis, mean: ArrayLike \| None, variance: ArrayLike \| None, epsilon: ArrayLike, where: ArrayLike \| None) -> Array` |
| `nn.swish` | Function | `(x: ArrayLike) -> Array` |
| `nn.tanh` | Function | `(x: ArrayLike) -> Array` |
| `no_execution` | Object | `` |
| `no_tracing` | Object | `` |
| `numpy.Array` | Class | `(shape, dtype, buffer, offset, strides, order)` |
| `numpy.ArrayLike` | Object | `` |
| `numpy.ArrayNamespaceInfo` | Class | `(...)` |
| `numpy.BinaryUfunc` | Class | `(...)` |
| `numpy.ComplexWarning` | Object | `` |
| `numpy.DType` | Object | `` |
| `numpy.DTypeLike` | Object | `` |
| `numpy.DeprecatedArg` | Class | `(...)` |
| `numpy.Device` | Object | `` |
| `numpy.DimSize` | Object | `` |
| `numpy.DuckTypedArray` | Class | `(...)` |
| `numpy.GatherScatterMode` | Class | `(...)` |
| `numpy.NamedSharding` | Class | `(mesh: mesh_lib.Mesh \| mesh_lib.AbstractMesh, spec: PartitionSpec, memory_kind: str \| None, _logical_device_ids)` |
| `numpy.P` | Object | `` |
| `numpy.PadValueLike` | Object | `` |
| `numpy.PrecisionLike` | Object | `` |
| `numpy.Shape` | Object | `` |
| `numpy.StaticScalar` | Object | `` |
| `numpy.SupportsNdim` | Class | `(...)` |
| `numpy.SupportsShape` | Class | `(...)` |
| `numpy.SupportsSize` | Class | `(...)` |
| `numpy.abs` | Function | `(x: ArrayLike) -> Array` |
| `numpy.absolute` | Function | `(x: ArrayLike) -> Array` |
| `numpy.acos` | Function | `(x: ArrayLike) -> Array` |
| `numpy.acosh` | Function | `(x: ArrayLike) -> Array` |
| `numpy.add` | Object | `` |
| `numpy.all` | Function | `(a: ArrayLike, axis: _Axis, out: None, keepdims: builtins.bool, where: ArrayLike \| None) -> Array` |
| `numpy.allclose` | Function | `(a: ArrayLike, b: ArrayLike, rtol: ArrayLike, atol: ArrayLike, equal_nan: builtins.bool) -> Array` |
| `numpy.amax` | Function | `(a: ArrayLike, axis: _Axis, out: None, keepdims: builtins.bool, initial: ArrayLike \| None, where: ArrayLike \| None) -> Array` |
| `numpy.amin` | Function | `(a: ArrayLike, axis: _Axis, out: None, keepdims: builtins.bool, initial: ArrayLike \| None, where: ArrayLike \| None) -> Array` |
| `numpy.angle` | Function | `(z: ArrayLike, deg: builtins.bool) -> Array` |
| `numpy.any` | Function | `(a: ArrayLike, axis: _Axis, out: None, keepdims: builtins.bool, where: ArrayLike \| None) -> Array` |
| `numpy.append` | Function | `(arr: ArrayLike, values: ArrayLike, axis: int \| None) -> Array` |
| `numpy.apply_along_axis` | Function | `(func1d: Callable, axis: int, arr: ArrayLike, args, kwargs) -> Array` |
| `numpy.apply_over_axes` | Function | `(func: Callable, a: ArrayLike, axes: Sequence[int]) -> Array` |
| `numpy.arange` | Function | `(start: ArrayLike \| DimSize, stop: ArrayLike \| DimSize \| None, step: ArrayLike \| None, dtype: DTypeLike \| None, device: _Device \| _Sharding \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.arccos` | Function | `(x: ArrayLike) -> Array` |
| `numpy.arccosh` | Function | `(x: ArrayLike) -> Array` |
| `numpy.arcsin` | Function | `(x: ArrayLike) -> Array` |
| `numpy.arcsinh` | Function | `(x: ArrayLike) -> Array` |
| `numpy.arctan` | Function | `(x: ArrayLike) -> Array` |
| `numpy.arctan2` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.arctanh` | Function | `(x: ArrayLike) -> Array` |
| `numpy.argmax` | Function | `(a: ArrayLike, axis: int \| None, out: None, keepdims: builtins.bool \| None) -> Array` |
| `numpy.argmin` | Function | `(a: ArrayLike, axis: int \| None, out: None, keepdims: builtins.bool \| None) -> Array` |
| `numpy.argpartition` | Function | `(a: ArrayLike, kth: int, axis: int) -> Array` |
| `numpy.argsort` | Function | `(a: ArrayLike, axis: int \| None, stable: builtins.bool, descending: builtins.bool, kind: str \| None, order: None, dtype: DTypeLike \| None) -> Array` |
| `numpy.argwhere` | Function | `(a: ArrayLike, size: int \| None, fill_value: ArrayLike \| None) -> Array` |
| `numpy.around` | Function | `(a: ArrayLike, decimals: int, out: None) -> Array` |
| `numpy.array` | Function | `(object: Any, dtype: DTypeLike \| None, copy: builtins.bool, order: str \| None, ndmin: int, device: _Device \| _Sharding \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.array_equal` | Function | `(a1: ArrayLike, a2: ArrayLike, equal_nan: builtins.bool) -> Array` |
| `numpy.array_equiv` | Function | `(a1: ArrayLike, a2: ArrayLike) -> Array` |
| `numpy.array_repr` | Object | `` |
| `numpy.array_split` | Function | `(ary: ArrayLike, indices_or_sections: int \| Sequence[int] \| ArrayLike, axis: int) -> list[Array]` |
| `numpy.array_str` | Object | `` |
| `numpy.asarray` | Function | `(a: Any, dtype: DTypeLike \| None, order: str \| None, copy: builtins.bool \| None, device: _Device \| _Sharding \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.asin` | Function | `(x: ArrayLike) -> Array` |
| `numpy.asinh` | Function | `(x: ArrayLike) -> Array` |
| `numpy.astype` | Function | `(a: ArrayLike, dtype: DTypeLike \| None, copy: builtins.bool, device: _Device \| _Sharding \| None) -> Array` |
| `numpy.atan` | Function | `(x: ArrayLike) -> Array` |
| `numpy.atan2` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.atanh` | Function | `(x: ArrayLike) -> Array` |
| `numpy.bartlett` | Function | `(M: int) -> Array` |
| `numpy.bfloat16` | Object | `` |
| `numpy.bincount` | Function | `(x: ArrayLike, weights: ArrayLike \| None, minlength: int, length: int \| None) -> Array` |
| `numpy.bitwise_and` | Object | `` |
| `numpy.bitwise_count` | Function | `(x: ArrayLike) -> Array` |
| `numpy.bitwise_invert` | Function | `(x: ArrayLike) -> Array` |
| `numpy.bitwise_left_shift` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.bitwise_not` | Function | `(x: ArrayLike) -> Array` |
| `numpy.bitwise_or` | Object | `` |
| `numpy.bitwise_right_shift` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.bitwise_xor` | Object | `` |
| `numpy.blackman` | Function | `(M: int) -> Array` |
| `numpy.block` | Function | `(arrays: ArrayLike \| Sequence[ArrayLike] \| Sequence[Sequence[ArrayLike]]) -> Array` |
| `numpy.bool` | Object | `` |
| `numpy.bool_` | Object | `` |
| `numpy.broadcast_arrays` | Function | `(args: ArrayLike) -> list[Array]` |
| `numpy.broadcast_to` | Function | `(array: ArrayLike, shape: DimSize \| Shape, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.c_` | Object | `` |
| `numpy.can_cast` | Object | `` |
| `numpy.cbrt` | Function | `(x: ArrayLike) -> Array` |
| `numpy.cdouble` | Object | `` |
| `numpy.ceil` | Function | `(x: ArrayLike) -> Array` |
| `numpy.character` | Object | `` |
| `numpy.choose` | Function | `(a: ArrayLike, choices: Array \| _np.ndarray \| Sequence[ArrayLike], out: None, mode: str) -> Array` |
| `numpy.clip` | Function | `(x: ArrayLike \| None, min: ArrayLike \| None, max: ArrayLike \| None, a: ArrayLike \| DeprecatedArg \| None, a_min: ArrayLike \| DeprecatedArg \| None, a_max: ArrayLike \| DeprecatedArg \| None) -> Array` |
| `numpy.column_stack` | Function | `(tup: _np.ndarray \| Array \| Sequence[ArrayLike]) -> Array` |
| `numpy.complex128` | Object | `` |
| `numpy.complex64` | Object | `` |
| `numpy.complex_` | Object | `` |
| `numpy.complexfloating` | Object | `` |
| `numpy.compress` | Function | `(condition: ArrayLike, a: ArrayLike, axis: int \| None, size: int \| None, fill_value: ArrayLike, out: None) -> Array` |
| `numpy.concat` | Function | `(arrays: Sequence[ArrayLike], axis: int \| None) -> Array` |
| `numpy.concatenate` | Function | `(arrays: _np.ndarray \| Array \| Sequence[ArrayLike], axis: int \| None, dtype: DTypeLike \| None) -> Array` |
| `numpy.conj` | Function | `(x: ArrayLike) -> Array` |
| `numpy.conjugate` | Function | `(x: ArrayLike) -> Array` |
| `numpy.convolve` | Function | `(a: ArrayLike, v: ArrayLike, mode: str, precision: PrecisionLike, preferred_element_type: DTypeLike \| None) -> Array` |
| `numpy.copy` | Function | `(a: ArrayLike, order: str \| None) -> Array` |
| `numpy.copysign` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.corrcoef` | Function | `(x: ArrayLike, y: ArrayLike \| None, rowvar: builtins.bool, dtype: DTypeLike \| None) -> Array` |
| `numpy.correlate` | Function | `(a: ArrayLike, v: ArrayLike, mode: str, precision: PrecisionLike, preferred_element_type: DTypeLike \| None) -> Array` |
| `numpy.cos` | Function | `(x: ArrayLike) -> Array` |
| `numpy.cosh` | Function | `(x: ArrayLike) -> Array` |
| `numpy.count_nonzero` | Function | `(a: ArrayLike, axis: _Axis, keepdims: builtins.bool) -> Array` |
| `numpy.cov` | Function | `(m: ArrayLike, y: ArrayLike \| None, rowvar: builtins.bool, bias: builtins.bool, ddof: int \| None, fweights: ArrayLike \| None, aweights: ArrayLike \| None, dtype: DTypeLike \| None) -> Array` |
| `numpy.cross` | Function | `(a: ArrayLike, b: ArrayLike, axisa: int, axisb: int, axisc: int, axis: int \| None) -> Array` |
| `numpy.csingle` | Object | `` |
| `numpy.cumprod` | Function | `(a: ArrayLike, axis: int \| None, dtype: DTypeLike \| None, out: None) -> Array` |
| `numpy.cumsum` | Function | `(a: ArrayLike, axis: int \| None, dtype: DTypeLike \| None, out: None) -> Array` |
| `numpy.cumulative_prod` | Function | `(x: ArrayLike, axis: int \| None, dtype: DTypeLike \| None, include_initial: builtins.bool) -> Array` |
| `numpy.cumulative_sum` | Function | `(x: ArrayLike, axis: int \| None, dtype: DTypeLike \| None, include_initial: builtins.bool) -> Array` |
| `numpy.deg2rad` | Function | `(x: ArrayLike) -> Array` |
| `numpy.degrees` | Function | `(x: ArrayLike) -> Array` |
| `numpy.delete` | Function | `(arr: ArrayLike, obj: ArrayLike \| slice, axis: int \| None, assume_unique_indices: builtins.bool) -> Array` |
| `numpy.diag` | Function | `(v: ArrayLike, k: int) -> Array` |
| `numpy.diag_indices` | Function | `(n: int, ndim: int) -> tuple[Array, ...]` |
| `numpy.diag_indices_from` | Function | `(arr: ArrayLike) -> tuple[Array, ...]` |
| `numpy.diagflat` | Function | `(v: ArrayLike, k: int) -> Array` |
| `numpy.diagonal` | Function | `(a: ArrayLike, offset: ArrayLike, axis1: int, axis2: int)` |
| `numpy.diff` | Function | `(a: ArrayLike, n: int, axis: int, prepend: ArrayLike \| None, append: ArrayLike \| None) -> Array` |
| `numpy.digitize` | Function | `(x: ArrayLike, bins: ArrayLike, right: builtins.bool, method: str \| None) -> Array` |
| `numpy.divide` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.divmod` | Function | `(x: ArrayLike, y: ArrayLike) -> tuple[Array, Array]` |
| `numpy.dot` | Function | `(a: ArrayLike, b: ArrayLike, precision: PrecisionLike, preferred_element_type: DTypeLike \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.double` | Object | `` |
| `numpy.dsplit` | Function | `(ary: ArrayLike, indices_or_sections: int \| ArrayLike) -> list[Array]` |
| `numpy.dstack` | Function | `(tup: _np.ndarray \| Array \| Sequence[ArrayLike], dtype: DTypeLike \| None) -> Array` |
| `numpy.dtype` | Object | `` |
| `numpy.e` | Object | `` |
| `numpy.ediff1d` | Function | `(ary: ArrayLike, to_end: ArrayLike \| None, to_begin: ArrayLike \| None) -> Array` |
| `numpy.empty` | Function | `(shape: Any, dtype: DTypeLike \| None, device: _Device \| _Sharding \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.empty_like` | Function | `(prototype: ArrayLike \| DuckTypedArray, dtype: DTypeLike \| None, shape: Any, device: _Device \| _Sharding \| None) -> Array` |
| `numpy.equal` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.euler_gamma` | Object | `` |
| `numpy.exp` | Function | `(x: ArrayLike) -> Array` |
| `numpy.exp2` | Function | `(x: ArrayLike) -> Array` |
| `numpy.expand_dims` | Function | `(a: ArrayLike, axis: int \| Sequence[int]) -> Array` |
| `numpy.expm1` | Function | `(x: ArrayLike) -> Array` |
| `numpy.extract` | Function | `(condition: ArrayLike, arr: ArrayLike, size: int \| None, fill_value: ArrayLike) -> Array` |
| `numpy.eye` | Function | `(N: DimSize, M: DimSize \| None, k: int \| ArrayLike, dtype: DTypeLike \| None, device: _Device \| _Sharding \| None) -> Array` |
| `numpy.fabs` | Function | `(x: ArrayLike) -> Array` |
| `numpy.fft.fft` | Function | `(a: ArrayLike, n: int \| None, axis: int, norm: str \| None) -> Array` |
| `numpy.fft.fft2` | Function | `(a: ArrayLike, s: Shape \| None, axes: Sequence[int], norm: str \| None) -> Array` |
| `numpy.fft.fftfreq` | Function | `(n: int, d: ArrayLike, dtype: DTypeLike \| None, device: xla_client.Device \| Sharding \| None) -> Array` |
| `numpy.fft.fftn` | Function | `(a: ArrayLike, s: Shape \| None, axes: Sequence[int] \| None, norm: str \| None) -> Array` |
| `numpy.fft.fftshift` | Function | `(x: ArrayLike, axes: None \| int \| Sequence[int]) -> Array` |
| `numpy.fft.hfft` | Function | `(a: ArrayLike, n: int \| None, axis: int, norm: str \| None) -> Array` |
| `numpy.fft.ifft` | Function | `(a: ArrayLike, n: int \| None, axis: int, norm: str \| None) -> Array` |
| `numpy.fft.ifft2` | Function | `(a: ArrayLike, s: Shape \| None, axes: Sequence[int], norm: str \| None) -> Array` |
| `numpy.fft.ifftn` | Function | `(a: ArrayLike, s: Shape \| None, axes: Sequence[int] \| None, norm: str \| None) -> Array` |
| `numpy.fft.ifftshift` | Function | `(x: ArrayLike, axes: None \| int \| Sequence[int]) -> Array` |
| `numpy.fft.ihfft` | Function | `(a: ArrayLike, n: int \| None, axis: int, norm: str \| None) -> Array` |
| `numpy.fft.irfft` | Function | `(a: ArrayLike, n: int \| None, axis: int, norm: str \| None) -> Array` |
| `numpy.fft.irfft2` | Function | `(a: ArrayLike, s: Shape \| None, axes: Sequence[int], norm: str \| None) -> Array` |
| `numpy.fft.irfftn` | Function | `(a: ArrayLike, s: Shape \| None, axes: Sequence[int] \| None, norm: str \| None) -> Array` |
| `numpy.fft.rfft` | Function | `(a: ArrayLike, n: int \| None, axis: int, norm: str \| None) -> Array` |
| `numpy.fft.rfft2` | Function | `(a: ArrayLike, s: Shape \| None, axes: Sequence[int], norm: str \| None) -> Array` |
| `numpy.fft.rfftfreq` | Function | `(n: int, d: ArrayLike, dtype: DTypeLike \| None, device: xla_client.Device \| Sharding \| None) -> Array` |
| `numpy.fft.rfftn` | Function | `(a: ArrayLike, s: Shape \| None, axes: Sequence[int] \| None, norm: str \| None) -> Array` |
| `numpy.fill_diagonal` | Function | `(a: ArrayLike, val: ArrayLike, wrap: builtins.bool, inplace: builtins.bool) -> Array` |
| `numpy.finfo` | Object | `` |
| `numpy.fix` | Function | `(x: ArrayLike, out: None) -> Array` |
| `numpy.flatnonzero` | Function | `(a: ArrayLike, size: int \| None, fill_value: None \| ArrayLike \| tuple[ArrayLike]) -> Array` |
| `numpy.flexible` | Object | `` |
| `numpy.flip` | Function | `(m: ArrayLike, axis: int \| Sequence[int] \| None) -> Array` |
| `numpy.fliplr` | Function | `(m: ArrayLike) -> Array` |
| `numpy.flipud` | Function | `(m: ArrayLike) -> Array` |
| `numpy.float16` | Object | `` |
| `numpy.float32` | Object | `` |
| `numpy.float4_e2m1fn` | Object | `` |
| `numpy.float64` | Object | `` |
| `numpy.float8_e3m4` | Object | `` |
| `numpy.float8_e4m3` | Object | `` |
| `numpy.float8_e4m3b11fnuz` | Object | `` |
| `numpy.float8_e4m3fn` | Object | `` |
| `numpy.float8_e4m3fnuz` | Object | `` |
| `numpy.float8_e5m2` | Object | `` |
| `numpy.float8_e5m2fnuz` | Object | `` |
| `numpy.float8_e8m0fnu` | Object | `` |
| `numpy.float_` | Object | `` |
| `numpy.float_power` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.floating` | Object | `` |
| `numpy.floor` | Function | `(x: ArrayLike) -> Array` |
| `numpy.floor_divide` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.fmax` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.fmin` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.fmod` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.frexp` | Function | `(x: ArrayLike) -> tuple[Array, Array]` |
| `numpy.from_dlpack` | Function | `(x: Any, device: _Device \| _Sharding \| None, copy: builtins.bool \| None) -> Array` |
| `numpy.frombuffer` | Function | `(buffer: bytes \| Any, dtype: DTypeLike, count: int, offset: int) -> Array` |
| `numpy.fromfile` | Function | `(args, kwargs)` |
| `numpy.fromfunction` | Function | `(function: Callable[..., Array], shape: Any, dtype: DTypeLike, kwargs) -> Array` |
| `numpy.fromiter` | Function | `(args, kwargs)` |
| `numpy.frompyfunc` | Function | `(func: Callable[..., Any], nin: int, nout: int, identity: Any) -> ufunc` |
| `numpy.fromstring` | Function | `(string: str, dtype: DTypeLike, count: int, sep: str) -> Array` |
| `numpy.full` | Function | `(shape: Any, fill_value: ArrayLike, dtype: DTypeLike \| None, device: _Device \| _Sharding \| None) -> Array` |
| `numpy.full_like` | Function | `(a: ArrayLike \| DuckTypedArray, fill_value: ArrayLike, dtype: DTypeLike \| None, shape: Any, device: _Device \| _Sharding \| None) -> Array` |
| `numpy.gcd` | Function | `(x1: ArrayLike, x2: ArrayLike) -> Array` |
| `numpy.generic` | Object | `` |
| `numpy.geomspace` | Function | `(start: ArrayLike, stop: ArrayLike, num: int, endpoint: builtins.bool, dtype: DTypeLike \| None, axis: int) -> Array` |
| `numpy.get_printoptions` | Object | `` |
| `numpy.gradient` | Function | `(f: ArrayLike, varargs: ArrayLike, axis: int \| Sequence[int] \| None, edge_order: int \| None) -> Array \| list[Array]` |
| `numpy.greater` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.greater_equal` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.hamming` | Function | `(M: int) -> Array` |
| `numpy.hanning` | Function | `(M: int) -> Array` |
| `numpy.heaviside` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.histogram` | Function | `(a: ArrayLike, bins: ArrayLike, range: Sequence[ArrayLike] \| None, weights: ArrayLike \| None, density: builtins.bool \| None) -> tuple[Array, Array]` |
| `numpy.histogram2d` | Function | `(x: ArrayLike, y: ArrayLike, bins: ArrayLike \| Sequence[ArrayLike], range: Sequence[None \| Array \| Sequence[ArrayLike]] \| None, weights: ArrayLike \| None, density: builtins.bool \| None) -> tuple[Array, Array, Array]` |
| `numpy.histogram_bin_edges` | Function | `(a: ArrayLike, bins: ArrayLike, range: None \| Array \| Sequence[ArrayLike], weights: ArrayLike \| None) -> Array` |
| `numpy.histogramdd` | Function | `(sample: ArrayLike, bins: ArrayLike \| Sequence[ArrayLike], range: Sequence[None \| Array \| Sequence[ArrayLike]] \| None, weights: ArrayLike \| None, density: builtins.bool \| None) -> tuple[Array, list[Array]]` |
| `numpy.hsplit` | Function | `(ary: ArrayLike, indices_or_sections: int \| ArrayLike) -> list[Array]` |
| `numpy.hstack` | Function | `(tup: _np.ndarray \| Array \| Sequence[ArrayLike], dtype: DTypeLike \| None) -> Array` |
| `numpy.hypot` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.i0` | Function | `(x: ArrayLike) -> Array` |
| `numpy.identity` | Function | `(n: DimSize, dtype: DTypeLike \| None) -> Array` |
| `numpy.iinfo` | Object | `` |
| `numpy.imag` | Function | `(x: ArrayLike) -> Array` |
| `numpy.index_exp` | Object | `` |
| `numpy.inexact` | Object | `` |
| `numpy.inf` | Object | `` |
| `numpy.inner` | Function | `(a: ArrayLike, b: ArrayLike, precision: PrecisionLike, preferred_element_type: DTypeLike \| None) -> Array` |
| `numpy.insert` | Function | `(arr: ArrayLike, obj: ArrayLike \| slice, values: ArrayLike, axis: int \| None) -> Array` |
| `numpy.int16` | Object | `` |
| `numpy.int2` | Object | `` |
| `numpy.int32` | Object | `` |
| `numpy.int4` | Object | `` |
| `numpy.int64` | Object | `` |
| `numpy.int8` | Object | `` |
| `numpy.int_` | Object | `` |
| `numpy.integer` | Object | `` |
| `numpy.interp` | Function | `(x: ArrayLike, xp: ArrayLike, fp: ArrayLike, left: ArrayLike \| str \| None, right: ArrayLike \| str \| None, period: ArrayLike \| None) -> Array` |
| `numpy.intersect1d` | Function | `(ar1: ArrayLike, ar2: ArrayLike, assume_unique: builtins.bool, return_indices: builtins.bool, size: int \| None, fill_value: ArrayLike \| None) -> Array \| tuple[Array, Array, Array]` |
| `numpy.invert` | Function | `(x: ArrayLike) -> Array` |
| `numpy.isclose` | Function | `(a: ArrayLike, b: ArrayLike, rtol: ArrayLike, atol: ArrayLike, equal_nan: builtins.bool) -> Array` |
| `numpy.iscomplex` | Function | `(x: ArrayLike) -> Array` |
| `numpy.iscomplexobj` | Function | `(x: Any) -> builtins.bool` |
| `numpy.isdtype` | Function | `(dtype: DTypeLike, kind: DType \| str \| tuple[DType \| str, ...]) -> builtins.bool` |
| `numpy.isfinite` | Function | `(x: ArrayLike) -> Array` |
| `numpy.isin` | Function | `(element: ArrayLike, test_elements: ArrayLike, assume_unique: builtins.bool, invert: builtins.bool, method: str) -> Array` |
| `numpy.isinf` | Function | `(x: ArrayLike) -> Array` |
| `numpy.isnan` | Function | `(x: ArrayLike) -> Array` |
| `numpy.isneginf` | Function | `(x: ArrayLike) -> Array` |
| `numpy.isposinf` | Function | `(x: ArrayLike) -> Array` |
| `numpy.isreal` | Function | `(x: ArrayLike) -> Array` |
| `numpy.isrealobj` | Function | `(x: Any) -> builtins.bool` |
| `numpy.isscalar` | Function | `(element: Any) -> builtins.bool` |
| `numpy.issubdtype` | Function | `(arg1: DTypeLike, arg2: DTypeLike) -> builtins.bool` |
| `numpy.iterable` | Object | `` |
| `numpy.ix_` | Function | `(args: ArrayLike) -> tuple[Array, ...]` |
| `numpy.kaiser` | Function | `(M: int, beta: ArrayLike) -> Array` |
| `numpy.kron` | Function | `(a: ArrayLike, b: ArrayLike) -> Array` |
| `numpy.lcm` | Function | `(x1: ArrayLike, x2: ArrayLike) -> Array` |
| `numpy.ldexp` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.left_shift` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.less` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.less_equal` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.lexsort` | Function | `(keys: Array \| _np.ndarray \| Sequence[ArrayLike], axis: int) -> Array` |
| `numpy.linalg.cholesky` | Function | `(a: ArrayLike, upper: bool, symmetrize_input: bool) -> Array` |
| `numpy.linalg.cond` | Function | `(x: ArrayLike, p)` |
| `numpy.linalg.cross` | Function | `(x1: ArrayLike, x2: ArrayLike, axis)` |
| `numpy.linalg.det` | Function | `(a: ArrayLike) -> Array` |
| `numpy.linalg.diagonal` | Function | `(x: ArrayLike, offset: int) -> Array` |
| `numpy.linalg.eig` | Function | `(a: ArrayLike) -> EigResult` |
| `numpy.linalg.eigh` | Function | `(a: ArrayLike, UPLO: str \| None, symmetrize_input: bool) -> EighResult` |
| `numpy.linalg.eigvals` | Function | `(a: ArrayLike) -> Array` |
| `numpy.linalg.eigvalsh` | Function | `(a: ArrayLike, UPLO: str \| None, symmetrize_input: bool) -> Array` |
| `numpy.linalg.inv` | Function | `(a: ArrayLike) -> Array` |
| `numpy.linalg.lstsq` | Function | `(a: ArrayLike, b: ArrayLike, rcond: float \| None, numpy_resid: bool) -> tuple[Array, Array, Array, Array]` |
| `numpy.linalg.matmul` | Function | `(x1: ArrayLike, x2: ArrayLike, precision: lax.PrecisionLike, preferred_element_type: DTypeLike \| None) -> Array` |
| `numpy.linalg.matrix_norm` | Function | `(x: ArrayLike, keepdims: bool, ord: str \| int) -> Array` |
| `numpy.linalg.matrix_power` | Function | `(a: ArrayLike, n: int) -> Array` |
| `numpy.linalg.matrix_rank` | Function | `(M: ArrayLike, rtol: ArrayLike \| None, tol: ArrayLike \| None) -> Array` |
| `numpy.linalg.matrix_transpose` | Function | `(x: ArrayLike) -> Array` |
| `numpy.linalg.multi_dot` | Function | `(arrays: Sequence[ArrayLike], precision: lax.PrecisionLike) -> Array` |
| `numpy.linalg.norm` | Function | `(x: ArrayLike, ord: int \| str \| None, axis: None \| tuple[int, ...] \| int, keepdims: bool) -> Array` |
| `numpy.linalg.outer` | Function | `(x1: ArrayLike, x2: ArrayLike) -> Array` |
| `numpy.linalg.pinv` | Function | `(a: ArrayLike, rtol: ArrayLike \| None, hermitian: bool, rcond: ArrayLike \| None) -> Array` |
| `numpy.linalg.qr` | Function | `(a: ArrayLike, mode: str) -> Array \| QRResult` |
| `numpy.linalg.slogdet` | Function | `(a: ArrayLike, method: str \| None) -> SlogdetResult` |
| `numpy.linalg.solve` | Function | `(a: ArrayLike, b: ArrayLike) -> Array` |
| `numpy.linalg.svd` | Function | `(a: ArrayLike, full_matrices: bool, compute_uv: bool, hermitian: bool, subset_by_index: tuple[int, int] \| None) -> Array \| SVDResult` |
| `numpy.linalg.svdvals` | Function | `(x: ArrayLike) -> Array` |
| `numpy.linalg.tensordot` | Function | `(x1: ArrayLike, x2: ArrayLike, axes: int \| tuple[Sequence[int], Sequence[int]], precision: lax.PrecisionLike, preferred_element_type: DTypeLike \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.linalg.tensorinv` | Function | `(a: ArrayLike, ind: int) -> Array` |
| `numpy.linalg.tensorsolve` | Function | `(a: ArrayLike, b: ArrayLike, axes: tuple[int, ...] \| None) -> Array` |
| `numpy.linalg.trace` | Function | `(x: ArrayLike, offset: int, dtype: DTypeLike \| None) -> Array` |
| `numpy.linalg.vecdot` | Function | `(x1: ArrayLike, x2: ArrayLike, axis: int, precision: lax.PrecisionLike, preferred_element_type: DTypeLike \| None) -> Array` |
| `numpy.linalg.vector_norm` | Function | `(x: ArrayLike, axis: int \| tuple[int, ...] \| None, keepdims: bool, ord: int \| str) -> Array` |
| `numpy.load` | Function | `(file: IO[bytes] \| str \| os.PathLike[Any], args: Any, kwargs: Any) -> Array` |
| `numpy.log` | Function | `(x: ArrayLike) -> Array` |
| `numpy.log10` | Function | `(x: ArrayLike) -> Array` |
| `numpy.log1p` | Function | `(x: ArrayLike) -> Array` |
| `numpy.log2` | Function | `(x: ArrayLike) -> Array` |
| `numpy.logaddexp` | Object | `` |
| `numpy.logaddexp2` | Object | `` |
| `numpy.logical_and` | Object | `` |
| `numpy.logical_not` | Function | `(x: ArrayLike) -> Array` |
| `numpy.logical_or` | Object | `` |
| `numpy.logical_xor` | Object | `` |
| `numpy.logspace` | Function | `(start: ArrayLike, stop: ArrayLike, num: int, endpoint: builtins.bool, base: ArrayLike, dtype: DTypeLike \| None, axis: int) -> Array` |
| `numpy.mask_indices` | Function | `(n: int, mask_func: Callable, k: int, size: int \| None) -> tuple[Array, ...]` |
| `numpy.matmul` | Function | `(a: ArrayLike, b: ArrayLike, precision: PrecisionLike, preferred_element_type: DTypeLike \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.matrix_transpose` | Function | `(x: ArrayLike) -> Array` |
| `numpy.matvec` | Function | `(x1: ArrayLike, x2: ArrayLike) -> Array` |
| `numpy.max` | Function | `(a: ArrayLike, axis: _Axis, out: None, keepdims: builtins.bool, initial: ArrayLike \| None, where: ArrayLike \| None) -> Array` |
| `numpy.maximum` | Object | `` |
| `numpy.mean` | Function | `(a: ArrayLike, axis: _Axis, dtype: DTypeLike \| None, out: None, keepdims: builtins.bool, where: ArrayLike \| None) -> Array` |
| `numpy.median` | Function | `(a: ArrayLike, axis: int \| tuple[int, ...] \| None, out: None, overwrite_input: builtins.bool, keepdims: builtins.bool) -> Array` |
| `numpy.meshgrid` | Function | `(xi: ArrayLike, copy: builtins.bool, sparse: builtins.bool, indexing: str) -> list[Array]` |
| `numpy.mgrid` | Object | `` |
| `numpy.min` | Function | `(a: ArrayLike, axis: _Axis, out: None, keepdims: builtins.bool, initial: ArrayLike \| None, where: ArrayLike \| None) -> Array` |
| `numpy.minimum` | Object | `` |
| `numpy.mod` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.modf` | Function | `(x: ArrayLike, out) -> tuple[Array, Array]` |
| `numpy.moveaxis` | Function | `(a: ArrayLike, source: int \| Sequence[int], destination: int \| Sequence[int]) -> Array` |
| `numpy.multiply` | Object | `` |
| `numpy.nan` | Object | `` |
| `numpy.nan_to_num` | Function | `(x: ArrayLike, copy: builtins.bool, nan: ArrayLike, posinf: ArrayLike \| None, neginf: ArrayLike \| None) -> Array` |
| `numpy.nanargmax` | Function | `(a: ArrayLike, axis: int \| None, out: None, keepdims: builtins.bool \| None) -> Array` |
| `numpy.nanargmin` | Function | `(a: ArrayLike, axis: int \| None, out: None, keepdims: builtins.bool \| None) -> Array` |
| `numpy.nancumprod` | Function | `(a: ArrayLike, axis: int \| None, dtype: DTypeLike \| None, out: None) -> Array` |
| `numpy.nancumsum` | Function | `(a: ArrayLike, axis: int \| None, dtype: DTypeLike \| None, out: None) -> Array` |
| `numpy.nanmax` | Function | `(a: ArrayLike, axis: _Axis, out: None, keepdims: builtins.bool, initial: ArrayLike \| None, where: ArrayLike \| None) -> Array` |
| `numpy.nanmean` | Function | `(a: ArrayLike, axis: _Axis, dtype: DTypeLike \| None, out: None, keepdims: builtins.bool, where: ArrayLike \| None) -> Array` |
| `numpy.nanmedian` | Function | `(a: ArrayLike, axis: int \| tuple[int, ...] \| None, out: None, overwrite_input: builtins.bool, keepdims: builtins.bool) -> Array` |
| `numpy.nanmin` | Function | `(a: ArrayLike, axis: _Axis, out: None, keepdims: builtins.bool, initial: ArrayLike \| None, where: ArrayLike \| None) -> Array` |
| `numpy.nanpercentile` | Function | `(a: ArrayLike, q: ArrayLike, axis: int \| tuple[int, ...] \| None, out: None, overwrite_input: builtins.bool, method: str, keepdims: builtins.bool, interpolation: DeprecatedArg \| str) -> Array` |
| `numpy.nanprod` | Function | `(a: ArrayLike, axis: _Axis, dtype: DTypeLike \| None, out: None, keepdims: builtins.bool, initial: ArrayLike \| None, where: ArrayLike \| None) -> Array` |
| `numpy.nanquantile` | Function | `(a: ArrayLike, q: ArrayLike, axis: int \| tuple[int, ...] \| None, out: None, overwrite_input: builtins.bool, method: str, keepdims: builtins.bool, interpolation: DeprecatedArg \| str) -> Array` |
| `numpy.nanstd` | Function | `(a: ArrayLike, axis: _Axis, dtype: DTypeLike \| None, out: None, ddof: int, keepdims: builtins.bool, where: ArrayLike \| None) -> Array` |
| `numpy.nansum` | Function | `(a: ArrayLike, axis: _Axis, dtype: DTypeLike \| None, out: None, keepdims: builtins.bool, initial: ArrayLike \| None, where: ArrayLike \| None) -> Array` |
| `numpy.nanvar` | Function | `(a: ArrayLike, axis: _Axis, dtype: DTypeLike \| None, out: None, ddof: int, keepdims: builtins.bool, where: ArrayLike \| None) -> Array` |
| `numpy.ndarray` | Object | `` |
| `numpy.ndim` | Function | `(a: ArrayLike \| SupportsNdim) -> int` |
| `numpy.negative` | Function | `(x: ArrayLike) -> Array` |
| `numpy.newaxis` | Object | `` |
| `numpy.nextafter` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.nonzero` | Function | `(a: ArrayLike, size: int \| None, fill_value: None \| ArrayLike \| tuple[ArrayLike, ...]) -> tuple[Array, ...]` |
| `numpy.not_equal` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.number` | Object | `` |
| `numpy.object_` | Object | `` |
| `numpy.ogrid` | Object | `` |
| `numpy.ones` | Function | `(shape: Any, dtype: DTypeLike \| None, device: _Device \| _Sharding \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.ones_like` | Function | `(a: ArrayLike \| DuckTypedArray, dtype: DTypeLike \| None, shape: Any, device: _Device \| _Sharding \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.outer` | Function | `(a: ArrayLike, b: Array, out: None) -> Array` |
| `numpy.packbits` | Function | `(a: ArrayLike, axis: int \| None, bitorder: str) -> Array` |
| `numpy.pad` | Function | `(array: ArrayLike, pad_width: PadValueLike[int \| Array \| _np.ndarray], mode: str \| Callable[..., Any], kwargs) -> Array` |
| `numpy.partition` | Function | `(a: ArrayLike, kth: int, axis: int) -> Array` |
| `numpy.percentile` | Function | `(a: ArrayLike, q: ArrayLike, axis: int \| tuple[int, ...] \| None, out: None, overwrite_input: builtins.bool, method: str, keepdims: builtins.bool, interpolation: DeprecatedArg \| str) -> Array` |
| `numpy.permute_dims` | Function | `(x: ArrayLike, axes: tuple[int, ...]) -> Array` |
| `numpy.pi` | Object | `` |
| `numpy.piecewise` | Function | `(x: ArrayLike, condlist: Array \| Sequence[ArrayLike], funclist: Sequence[ArrayLike \| Callable[..., Array]], args, kw) -> Array` |
| `numpy.place` | Function | `(arr: ArrayLike, mask: ArrayLike, vals: ArrayLike, inplace: builtins.bool) -> Array` |
| `numpy.poly` | Function | `(seq_of_zeros: ArrayLike) -> Array` |
| `numpy.polyadd` | Function | `(a1: ArrayLike, a2: ArrayLike) -> Array` |
| `numpy.polyder` | Function | `(p: ArrayLike, m: int) -> Array` |
| `numpy.polydiv` | Function | `(u: ArrayLike, v: ArrayLike, trim_leading_zeros: builtins.bool) -> tuple[Array, Array]` |
| `numpy.polyfit` | Function | `(x: ArrayLike, y: ArrayLike, deg: int, rcond: float \| None, full: builtins.bool, w: ArrayLike \| None, cov: builtins.bool) -> Array \| tuple[Array, ...]` |
| `numpy.polyint` | Function | `(p: ArrayLike, m: int, k: int \| ArrayLike \| None) -> Array` |
| `numpy.polymul` | Function | `(a1: ArrayLike, a2: ArrayLike, trim_leading_zeros: builtins.bool) -> Array` |
| `numpy.polysub` | Function | `(a1: ArrayLike, a2: ArrayLike) -> Array` |
| `numpy.polyval` | Function | `(p: ArrayLike, x: ArrayLike, unroll: int) -> Array` |
| `numpy.positive` | Function | `(x: ArrayLike) -> Array` |
| `numpy.pow` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.power` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.printoptions` | Object | `` |
| `numpy.prod` | Function | `(a: ArrayLike, axis: _Axis, dtype: DTypeLike \| None, out: None, keepdims: builtins.bool, initial: ArrayLike \| None, where: ArrayLike \| None, promote_integers: builtins.bool) -> Array` |
| `numpy.promote_types` | Object | `` |
| `numpy.ptp` | Function | `(a: ArrayLike, axis: _Axis, out: None, keepdims: builtins.bool) -> Array` |
| `numpy.put` | Function | `(a: ArrayLike, ind: ArrayLike, v: ArrayLike, mode: str \| None, inplace: builtins.bool) -> Array` |
| `numpy.put_along_axis` | Function | `(arr: ArrayLike, indices: ArrayLike, values: ArrayLike, axis: int \| None, inplace: bool, mode: str \| None) -> Array` |
| `numpy.quantile` | Function | `(a: ArrayLike, q: ArrayLike, axis: int \| tuple[int, ...] \| None, out: None, overwrite_input: builtins.bool, method: str, keepdims: builtins.bool, interpolation: DeprecatedArg \| str) -> Array` |
| `numpy.r_` | Object | `` |
| `numpy.rad2deg` | Function | `(x: ArrayLike) -> Array` |
| `numpy.radians` | Function | `(x: ArrayLike) -> Array` |
| `numpy.ravel` | Function | `(a: ArrayLike, order: str, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.ravel_multi_index` | Function | `(multi_index: Sequence[ArrayLike], dims: Sequence[int], mode: str, order: str) -> Array` |
| `numpy.real` | Function | `(x: ArrayLike) -> Array` |
| `numpy.reciprocal` | Function | `(x: ArrayLike) -> Array` |
| `numpy.remainder` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.repeat` | Function | `(a: ArrayLike, repeats: ArrayLike, axis: int \| None, total_repeat_length: int \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.reshape` | Function | `(a: ArrayLike, shape: DimSize \| Shape, order: str, copy: bool \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.resize` | Function | `(a: ArrayLike, new_shape: Shape) -> Array` |
| `numpy.result_type` | Function | `(args: Any) -> DType` |
| `numpy.right_shift` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.rint` | Function | `(x: ArrayLike) -> Array` |
| `numpy.roll` | Function | `(a: ArrayLike, shift: ArrayLike \| Sequence[int], axis: int \| Sequence[int] \| None) -> Array` |
| `numpy.rollaxis` | Function | `(a: ArrayLike, axis: int, start: int) -> Array` |
| `numpy.roots` | Function | `(p: ArrayLike, strip_zeros: builtins.bool) -> Array` |
| `numpy.rot90` | Function | `(m: ArrayLike, k: int, axes: tuple[int, int]) -> Array` |
| `numpy.round` | Function | `(a: ArrayLike, decimals: int, out: None) -> Array` |
| `numpy.s_` | Object | `` |
| `numpy.save` | Object | `` |
| `numpy.savez` | Object | `` |
| `numpy.searchsorted` | Function | `(a: ArrayLike, v: ArrayLike, side: str, sorter: ArrayLike \| None, method: str) -> Array` |
| `numpy.select` | Function | `(condlist: Sequence[ArrayLike], choicelist: Sequence[ArrayLike], default: ArrayLike) -> Array` |
| `numpy.set_printoptions` | Object | `` |
| `numpy.setdiff1d` | Function | `(ar1: ArrayLike, ar2: ArrayLike, assume_unique: builtins.bool, size: int \| None, fill_value: ArrayLike \| None) -> Array` |
| `numpy.setxor1d` | Function | `(ar1: ArrayLike, ar2: ArrayLike, assume_unique: builtins.bool, size: int \| None, fill_value: ArrayLike \| None) -> Array` |
| `numpy.shape` | Function | `(a: ArrayLike \| SupportsShape) -> tuple[int, ...]` |
| `numpy.sign` | Function | `(x: ArrayLike) -> Array` |
| `numpy.signbit` | Function | `(x: ArrayLike) -> Array` |
| `numpy.signedinteger` | Object | `` |
| `numpy.sin` | Function | `(x: ArrayLike) -> Array` |
| `numpy.sinc` | Function | `(x: ArrayLike) -> Array` |
| `numpy.single` | Object | `` |
| `numpy.sinh` | Function | `(x: ArrayLike) -> Array` |
| `numpy.size` | Function | `(a: ArrayLike \| SupportsSize, axis: _Axis) -> int` |
| `numpy.sort` | Function | `(a: ArrayLike, axis: int \| None, stable: builtins.bool, descending: builtins.bool, kind: str \| None, order: None) -> Array` |
| `numpy.sort_complex` | Function | `(a: ArrayLike) -> Array` |
| `numpy.spacing` | Function | `(x: ArrayLike) -> Array` |
| `numpy.split` | Function | `(ary: ArrayLike, indices_or_sections: int \| Sequence[int] \| ArrayLike, axis: int) -> list[Array]` |
| `numpy.sqrt` | Function | `(x: ArrayLike) -> Array` |
| `numpy.square` | Function | `(x: ArrayLike) -> Array` |
| `numpy.squeeze` | Function | `(a: ArrayLike, axis: int \| Sequence[int] \| None) -> Array` |
| `numpy.stack` | Function | `(arrays: _np.ndarray \| Array \| Sequence[ArrayLike], axis: int, out: None, dtype: DTypeLike \| None) -> Array` |
| `numpy.std` | Function | `(a: ArrayLike, axis: _Axis, dtype: DTypeLike \| None, out: None, ddof: int, keepdims: builtins.bool, where: ArrayLike \| None, correction: int \| float \| None) -> Array` |
| `numpy.subtract` | Object | `` |
| `numpy.sum` | Function | `(a: ArrayLike, axis: _Axis, dtype: DTypeLike \| None, out: None, keepdims: builtins.bool, initial: ArrayLike \| None, where: ArrayLike \| None, promote_integers: builtins.bool) -> Array` |
| `numpy.swapaxes` | Function | `(a: ArrayLike, axis1: int, axis2: int) -> Array` |
| `numpy.take` | Function | `(a: ArrayLike, indices: ArrayLike, axis: int \| None, out: None, mode: str \| None, unique_indices: builtins.bool, indices_are_sorted: builtins.bool, fill_value: StaticScalar \| None) -> Array` |
| `numpy.take_along_axis` | Function | `(arr: ArrayLike, indices: ArrayLike, axis: int \| None, mode: str \| GatherScatterMode \| None, fill_value: StaticScalar \| None) -> Array` |
| `numpy.tan` | Function | `(x: ArrayLike) -> Array` |
| `numpy.tanh` | Function | `(x: ArrayLike) -> Array` |
| `numpy.tensordot` | Function | `(a: ArrayLike, b: ArrayLike, axes: int \| Sequence[int] \| Sequence[Sequence[int]], precision: PrecisionLike, preferred_element_type: DTypeLike \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.tile` | Function | `(A: ArrayLike, reps: DimSize \| Sequence[DimSize]) -> Array` |
| `numpy.trace` | Function | `(a: ArrayLike, offset: int \| ArrayLike, axis1: int, axis2: int, dtype: DTypeLike \| None, out: None) -> Array` |
| `numpy.transpose` | Function | `(a: ArrayLike, axes: Sequence[int] \| None) -> Array` |
| `numpy.trapezoid` | Function | `(y: ArrayLike, x: ArrayLike \| None, dx: ArrayLike, axis: int) -> Array` |
| `numpy.tri` | Function | `(N: int, M: int \| None, k: int, dtype: DTypeLike \| None) -> Array` |
| `numpy.tril` | Function | `(m: ArrayLike, k: int) -> Array` |
| `numpy.tril_indices` | Function | `(n: int, k: int, m: int \| None) -> tuple[Array, Array]` |
| `numpy.tril_indices_from` | Function | `(arr: ArrayLike \| SupportsShape, k: int) -> tuple[Array, Array]` |
| `numpy.trim_zeros` | Function | `(filt: ArrayLike, trim: str, axis: int \| Sequence[int] \| None) -> Array` |
| `numpy.triu` | Function | `(m: ArrayLike, k: int) -> Array` |
| `numpy.triu_indices` | Function | `(n: int, k: int, m: int \| None) -> tuple[Array, Array]` |
| `numpy.triu_indices_from` | Function | `(arr: ArrayLike \| SupportsShape, k: int) -> tuple[Array, Array]` |
| `numpy.true_divide` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `numpy.trunc` | Function | `(x: ArrayLike) -> Array` |
| `numpy.ufunc` | Class | `(func: Callable[..., Any], nin: int, nout: int, name: str \| None, nargs: int \| None, identity: Any, call: Callable[..., Any] \| None, reduce: Callable[..., Any] \| None, accumulate: Callable[..., Any] \| None, at: Callable[..., Any] \| None, reduceat: Callable[..., Any] \| None)` |
| `numpy.uint` | Object | `` |
| `numpy.uint16` | Object | `` |
| `numpy.uint2` | Object | `` |
| `numpy.uint32` | Object | `` |
| `numpy.uint4` | Object | `` |
| `numpy.uint64` | Object | `` |
| `numpy.uint8` | Object | `` |
| `numpy.union1d` | Function | `(ar1: ArrayLike, ar2: ArrayLike, size: int \| None, fill_value: ArrayLike \| None) -> Array` |
| `numpy.unique` | Function | `(ar: ArrayLike, return_index: builtins.bool, return_inverse: builtins.bool, return_counts: builtins.bool, axis: int \| None, equal_nan: builtins.bool, size: int \| None, fill_value: ArrayLike \| None, sorted: bool)` |
| `numpy.unique_all` | Function | `(x: ArrayLike, size: int \| None, fill_value: ArrayLike \| None) -> _UniqueAllResult` |
| `numpy.unique_counts` | Function | `(x: ArrayLike, size: int \| None, fill_value: ArrayLike \| None) -> _UniqueCountsResult` |
| `numpy.unique_inverse` | Function | `(x: ArrayLike, size: int \| None, fill_value: ArrayLike \| None) -> _UniqueInverseResult` |
| `numpy.unique_values` | Function | `(x: ArrayLike, size: int \| None, fill_value: ArrayLike \| None) -> Array` |
| `numpy.unpackbits` | Function | `(a: ArrayLike, axis: int \| None, count: ArrayLike \| None, bitorder: str) -> Array` |
| `numpy.unravel_index` | Function | `(indices: ArrayLike, shape: Shape) -> tuple[Array, ...]` |
| `numpy.unsignedinteger` | Object | `` |
| `numpy.unstack` | Function | `(x: ArrayLike, axis: int) -> tuple[Array, ...]` |
| `numpy.unwrap` | Function | `(p: ArrayLike, discont: ArrayLike \| None, axis: int, period: ArrayLike) -> Array` |
| `numpy.vander` | Function | `(x: ArrayLike, N: int \| None, increasing: builtins.bool) -> Array` |
| `numpy.var` | Function | `(a: ArrayLike, axis: _Axis, dtype: DTypeLike \| None, out: None, ddof: int, keepdims: builtins.bool, where: ArrayLike \| None, correction: int \| float \| None) -> Array` |
| `numpy.vdot` | Function | `(a: ArrayLike, b: ArrayLike, precision: PrecisionLike, preferred_element_type: DTypeLike \| None) -> Array` |
| `numpy.vecdot` | Function | `(x1: ArrayLike, x2: ArrayLike, axis: int, precision: PrecisionLike, preferred_element_type: DTypeLike \| None) -> Array` |
| `numpy.vecmat` | Function | `(x1: ArrayLike, x2: ArrayLike) -> Array` |
| `numpy.vectorize` | Function | `(pyfunc, excluded, signature) -> Callable` |
| `numpy.vsplit` | Function | `(ary: ArrayLike, indices_or_sections: int \| ArrayLike) -> list[Array]` |
| `numpy.vstack` | Function | `(tup: _np.ndarray \| Array \| Sequence[ArrayLike], dtype: DTypeLike \| None) -> Array` |
| `numpy.zeros` | Function | `(shape: Any, dtype: DTypeLike \| None, device: _Device \| _Sharding \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy.zeros_like` | Function | `(a: ArrayLike \| DuckTypedArray, dtype: DTypeLike \| None, shape: Any, device: _Device \| _Sharding \| None, out_sharding: NamedSharding \| P \| None) -> Array` |
| `numpy_dtype_promotion` | Object | `` |
| `numpy_rank_promotion` | Object | `` |
| `ops.segment_max` | Function | `(data: ArrayLike, segment_ids: ArrayLike, num_segments: int \| None, indices_are_sorted: bool, unique_indices: bool, bucket_size: int \| None, mode: slicing.GatherScatterMode \| str \| None) -> Array` |
| `ops.segment_min` | Function | `(data: ArrayLike, segment_ids: ArrayLike, num_segments: int \| None, indices_are_sorted: bool, unique_indices: bool, bucket_size: int \| None, mode: slicing.GatherScatterMode \| str \| None) -> Array` |
| `ops.segment_prod` | Function | `(data: ArrayLike, segment_ids: ArrayLike, num_segments: int \| None, indices_are_sorted: bool, unique_indices: bool, bucket_size: int \| None, mode: slicing.GatherScatterMode \| str \| None) -> Array` |
| `ops.segment_sum` | Function | `(data: ArrayLike, segment_ids: ArrayLike, num_segments: int \| None, indices_are_sorted: bool, unique_indices: bool, bucket_size: int \| None, mode: slicing.GatherScatterMode \| str \| None) -> Array` |
| `partial_eval` | Object | `` |
| `pmap` | Function | `(fun: Callable, axis_name: AxisName \| None, in_axes: int \| None \| Sequence[Any], out_axes: Any, static_broadcasted_argnums: int \| Iterable[int], devices: Sequence[xc.Device] \| None, backend: str \| None, axis_size: int \| None, donate_argnums: int \| Iterable[int], global_arg_shapes: tuple[tuple[int, ...], ...] \| None) -> Any` |
| `print_environment_info` | Function | `(return_string: bool) -> str \| None` |
| `process_count` | Function | `(backend: str \| xla_client.Client \| None) -> int` |
| `process_index` | Function | `(backend: str \| xla_client.Client \| None) -> int` |
| `process_indices` | Function | `(backend: str \| xla_client.Client \| None) -> list[int]` |
| `profiler.ProfileData` | Object | `` |
| `profiler.ProfileEvent` | Object | `` |
| `profiler.ProfileOptions` | Class | `(...)` |
| `profiler.ProfilePlane` | Object | `` |
| `profiler.StepTraceAnnotation` | Class | `(name: str, kwargs)` |
| `profiler.TraceAnnotation` | Class | `(...)` |
| `profiler.annotate_function` | Function | `(func: Callable, name: str \| None, decorator_kwargs)` |
| `profiler.device_memory_profile` | Function | `(backend: str \| None) -> bytes` |
| `profiler.save_device_memory_profile` | Function | `(filename, backend: str \| None) -> None` |
| `profiler.start_server` | Function | `(port: int) -> _profiler.ProfilerServer` |
| `profiler.start_trace` | Function | `(log_dir: os.PathLike \| str, create_perfetto_link: bool, create_perfetto_trace: bool, profiler_options: ProfileOptions \| None) -> None` |
| `profiler.stop_server` | Function | `()` |
| `profiler.stop_trace` | Function | `()` |
| `profiler.trace` | Function | `(log_dir: os.PathLike \| str, create_perfetto_link, create_perfetto_trace, profiler_options: ProfileOptions \| None)` |
| `pure_callback` | Function | `(callback: Callable[..., Any], result_shape_dtypes: Any, args: Any, sharding: SingleDeviceSharding \| None, vectorized: bool \| None \| DeprecatedArg, vmap_method: str \| None, kwargs: Any)` |
| `pxla` | Object | `` |
| `random.PRNGKey` | Function | `(seed: int \| ArrayLike, impl: PRNGSpecDesc \| None) -> Array` |
| `random.ball` | Function | `(key: ArrayLike, d: int, p: float, shape: Shape, dtype: DTypeLikeFloat \| None)` |
| `random.bernoulli` | Function | `(key: ArrayLike, p: RealArray, shape: Shape \| None, mode: str, out_sharding) -> Array` |
| `random.beta` | Function | `(key: ArrayLike, a: RealArray, b: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.binomial` | Function | `(key: Array, n: RealArray, p: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.bits` | Function | `(key: ArrayLike, shape: Shape, dtype: DTypeLikeUInt \| None, out_sharding) -> Array` |
| `random.categorical` | Function | `(key: ArrayLike, logits: RealArray, axis: int, shape: Shape \| None, replace: bool, mode: str \| None) -> Array` |
| `random.cauchy` | Function | `(key: ArrayLike, shape: Shape, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.chisquare` | Function | `(key: ArrayLike, df: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.choice` | Function | `(key: ArrayLike, a: int \| ArrayLike, shape: Shape, replace: bool, p: RealArray \| None, axis: int, mode: str \| None) -> Array` |
| `random.clone` | Function | `(key)` |
| `random.dirichlet` | Function | `(key: ArrayLike, alpha: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.double_sided_maxwell` | Function | `(key: ArrayLike, loc: RealArray, scale: RealArray, shape: Shape, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.exponential` | Function | `(key: ArrayLike, shape: Shape, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.f` | Function | `(key: ArrayLike, dfnum: RealArray, dfden: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.fold_in` | Function | `(key: ArrayLike, data: IntegerArray) -> Array` |
| `random.gamma` | Function | `(key: ArrayLike, a: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.generalized_normal` | Function | `(key: ArrayLike, p: float, shape: Shape, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.geometric` | Function | `(key: ArrayLike, p: RealArray, shape: Shape \| None, dtype: DTypeLikeInt \| None) -> Array` |
| `random.gumbel` | Function | `(key: ArrayLike, shape: Shape, dtype: DTypeLikeFloat \| None, mode: str \| None, out_sharding) -> Array` |
| `random.key` | Function | `(seed: int \| ArrayLike, impl: PRNGSpecDesc \| None) -> Array` |
| `random.key_data` | Function | `(keys: ArrayLike) -> Array` |
| `random.key_impl` | Function | `(keys: ArrayLike) -> str \| PRNGSpec` |
| `random.laplace` | Function | `(key: ArrayLike, shape: Shape, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.loggamma` | Function | `(key: ArrayLike, a: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.logistic` | Function | `(key: ArrayLike, shape: Shape, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.lognormal` | Function | `(key: ArrayLike, sigma: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.maxwell` | Function | `(key: ArrayLike, shape: Shape, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.multinomial` | Function | `(key: Array, n: RealArray, p: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None, unroll: int \| bool)` |
| `random.multivariate_normal` | Function | `(key: ArrayLike, mean: RealArray, cov: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None, method: str) -> Array` |
| `random.normal` | Function | `(key: ArrayLike, shape: Shape, dtype: DTypeLikeFloat \| None, out_sharding) -> Array` |
| `random.orthogonal` | Function | `(key: ArrayLike, n: int, shape: Shape, dtype: DTypeLikeFloat \| None, m: int \| None) -> Array` |
| `random.pareto` | Function | `(key: ArrayLike, b: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.permutation` | Function | `(key: ArrayLike, x: int \| ArrayLike, axis: int, independent: bool, out_sharding) -> Array` |
| `random.poisson` | Function | `(key: ArrayLike, lam: RealArray, shape: Shape \| None, dtype: DTypeLikeInt \| None) -> Array` |
| `random.rademacher` | Function | `(key: ArrayLike, shape: Shape, dtype: DTypeLikeInt \| None) -> Array` |
| `random.randint` | Function | `(key: ArrayLike, shape: Shape, minval: IntegerArray, maxval: IntegerArray, dtype: DTypeLikeInt \| None, out_sharding) -> Array` |
| `random.random_gamma_p` | Object | `` |
| `random.rayleigh` | Function | `(key: ArrayLike, scale: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.split` | Function | `(key: ArrayLike, num: int \| tuple[int, ...]) -> Array` |
| `random.t` | Function | `(key: ArrayLike, df: RealArray, shape: Shape, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.triangular` | Function | `(key: ArrayLike, left: RealArray, mode: RealArray, right: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.truncated_normal` | Function | `(key: ArrayLike, lower: RealArray, upper: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None, out_sharding) -> Array` |
| `random.uniform` | Function | `(key: ArrayLike, shape: Shape, dtype: DTypeLikeFloat \| None, minval: RealArray, maxval: RealArray, out_sharding) -> Array` |
| `random.wald` | Function | `(key: ArrayLike, mean: RealArray, shape: Shape \| None, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.weibull_min` | Function | `(key: ArrayLike, scale: RealArray, concentration: RealArray, shape: Shape, dtype: DTypeLikeFloat \| None) -> Array` |
| `random.wrap_key_data` | Function | `(key_bits_array: Array, impl: PRNGSpecDesc \| None)` |
| `ref.AbstractRef` | Class | `(inner_aval: core.AbstractValue, memory_space: Any, kind: Any)` |
| `ref.Ref` | Class | `(aval, refs)` |
| `ref.addupdate` | Function | `(ref: core.Ref \| TransformedRef, idx: Indexer \| tuple[Indexer, ...] \| None, x: ArrayLike \| HijaxType) -> None` |
| `ref.empty_ref` | Function | `(ty, memory_space)` |
| `ref.free_ref` | Function | `(ref: Ref)` |
| `ref.freeze` | Function | `(ref: Ref) -> Array` |
| `ref.get` | Function | `(ref: core.Ref \| TransformedRef, idx: Indexer \| tuple[Indexer, ...] \| None) -> Array \| HijaxType` |
| `ref.new_ref` | Function | `(init_val: Any, memory_space: Any) -> core.Ref` |
| `ref.set` | Function | `(ref: core.Ref \| TransformedRef, idx: Indexer \| tuple[Indexer, ...] \| None, value: ArrayLike \| HijaxType) -> None` |
| `ref.swap` | Function | `(ref: core.Ref \| TransformedRef, idx: Indexer \| tuple[Indexer, ...] \| None, value: ArrayLike \| HijaxType, _function_name: str) -> Array \| HijaxType` |
| `remat` | Function | `(fun: Callable, prevent_cse: bool, policy: Callable[..., bool] \| None, static_argnums: int \| tuple[int, ...], concrete: bool \| DeprecatedArg) -> Callable` |
| `remove_size_one_mesh_axis_from_type` | Object | `` |
| `reshard` | Function | `(xs, out_shardings)` |
| `scipy.cluster.vq.vq` | Function | `(obs: ArrayLike, code_book: ArrayLike, check_finite: bool) -> tuple[Array, Array]` |
| `scipy.fft.dct` | Function | `(x: Array, type: int, n: int \| None, axis: int, norm: str \| None) -> Array` |
| `scipy.fft.dctn` | Function | `(x: Array, type: int, s: Sequence[int] \| None, axes: Sequence[int] \| None, norm: str \| None) -> Array` |
| `scipy.fft.idct` | Function | `(x: Array, type: int, n: int \| None, axis: int, norm: str \| None) -> Array` |
| `scipy.fft.idctn` | Function | `(x: Array, type: int, s: Sequence[int] \| None, axes: Sequence[int] \| None, norm: str \| None) -> Array` |
| `scipy.integrate.trapezoid` | Function | `(y: ArrayLike, x: ArrayLike \| None, dx: ArrayLike, axis: int) -> Array` |
| `scipy.interpolate.RegularGridInterpolator` | Class | `(points, values, method, bounds_error, fill_value)` |
| `scipy.linalg.block_diag` | Function | `(arrs: ArrayLike) -> Array` |
| `scipy.linalg.cho_factor` | Function | `(a: ArrayLike, lower: bool, overwrite_a: bool, check_finite: bool) -> tuple[Array, bool]` |
| `scipy.linalg.cho_solve` | Function | `(c_and_lower: tuple[ArrayLike, bool], b: ArrayLike, overwrite_b: bool, check_finite: bool) -> Array` |
| `scipy.linalg.cholesky` | Function | `(a: ArrayLike, lower: bool, overwrite_a: bool, check_finite: bool) -> Array` |
| `scipy.linalg.det` | Function | `(a: ArrayLike, overwrite_a: bool, check_finite: bool) -> Array` |
| `scipy.linalg.eigh` | Function | `(a: ArrayLike, b: ArrayLike \| None, lower: bool, eigvals_only: bool, overwrite_a: bool, overwrite_b: bool, turbo: bool, eigvals: None, type: int, check_finite: bool) -> Array \| tuple[Array, Array]` |
| `scipy.linalg.eigh_tridiagonal` | Function | `(d: ArrayLike, e: ArrayLike, eigvals_only: bool, select: str, select_range: tuple[float, float] \| None, tol: float \| None) -> Array` |
| `scipy.linalg.expm` | Function | `(A: ArrayLike, upper_triangular: bool, max_squarings: int) -> Array` |
| `scipy.linalg.expm_frechet` | Function | `(A: ArrayLike, E: ArrayLike, method: str \| None, compute_expm: bool) -> Array \| tuple[Array, Array]` |
| `scipy.linalg.funm` | Function | `(A: ArrayLike, func: Callable[[Array], Array], disp: bool) -> Array \| tuple[Array, Array]` |
| `scipy.linalg.hankel` | Function | `(c: ArrayLike, r: ArrayLike \| None) -> Array` |
| `scipy.linalg.hessenberg` | Function | `(a: ArrayLike, calc_q: bool, overwrite_a: bool, check_finite: bool) -> Array \| tuple[Array, Array]` |
| `scipy.linalg.hilbert` | Function | `(n: int) -> Array` |
| `scipy.linalg.inv` | Function | `(a: ArrayLike, overwrite_a: bool, check_finite: bool) -> Array` |
| `scipy.linalg.lu` | Function | `(a: ArrayLike, permute_l: bool, overwrite_a: bool, check_finite: bool) -> tuple[Array, Array] \| tuple[Array, Array, Array]` |
| `scipy.linalg.lu_factor` | Function | `(a: ArrayLike, overwrite_a: bool, check_finite: bool) -> tuple[Array, Array]` |
| `scipy.linalg.lu_solve` | Function | `(lu_and_piv: tuple[Array, ArrayLike], b: ArrayLike, trans: int, overwrite_b: bool, check_finite: bool) -> Array` |
| `scipy.linalg.pascal` | Function | `(n: int, kind: str \| None) -> Array` |
| `scipy.linalg.polar` | Function | `(a: ArrayLike, side: str, method: str, eps: float \| None, max_iterations: int \| None) -> tuple[Array, Array]` |
| `scipy.linalg.qr` | Function | `(a: ArrayLike, overwrite_a: bool, lwork: Any, mode: str, pivoting: bool, check_finite: bool) -> tuple[Array] \| tuple[Array, Array] \| tuple[Array, Array, Array]` |
| `scipy.linalg.rsf2csf` | Function | `(T: ArrayLike, Z: ArrayLike, check_finite: bool) -> tuple[Array, Array]` |
| `scipy.linalg.schur` | Function | `(a: ArrayLike, output: str) -> tuple[Array, Array]` |
| `scipy.linalg.solve` | Function | `(a: ArrayLike, b: ArrayLike, lower: bool, overwrite_a: bool, overwrite_b: bool, debug: bool, check_finite: bool, assume_a: str) -> Array` |
| `scipy.linalg.solve_sylvester` | Function | `(A: ArrayLike, B: ArrayLike, C: ArrayLike, method: str, tol: float) -> Array` |
| `scipy.linalg.solve_triangular` | Function | `(a: ArrayLike, b: ArrayLike, trans: int \| str, lower: bool, unit_diagonal: bool, overwrite_b: bool, debug: Any, check_finite: bool) -> Array` |
| `scipy.linalg.sqrtm` | Function | `(A: ArrayLike, blocksize: int) -> Array` |
| `scipy.linalg.svd` | Function | `(a: ArrayLike, full_matrices: bool, compute_uv: bool, overwrite_a: bool, check_finite: bool, lapack_driver: str) -> Array \| tuple[Array, Array, Array]` |
| `scipy.linalg.toeplitz` | Function | `(c: ArrayLike, r: ArrayLike \| None) -> Array` |
| `scipy.ndimage.map_coordinates` | Function | `(input: ArrayLike, coordinates: Sequence[ArrayLike], order: int, mode: str, cval: ArrayLike)` |
| `scipy.optimize.OptimizeResults` | Class | `(...)` |
| `scipy.optimize.minimize` | Function | `(fun: Callable, x0: Array, args: tuple, method: str, tol: float \| None, options: Mapping[str, Any] \| None) -> OptimizeResults` |
| `scipy.signal.convolve` | Function | `(in1: Array, in2: Array, mode: ModeString, method: str, precision: PrecisionLike) -> Array` |
| `scipy.signal.convolve2d` | Function | `(in1: Array, in2: Array, mode: ModeString, boundary: str, fillvalue: float, precision: PrecisionLike) -> Array` |
| `scipy.signal.correlate` | Function | `(in1: Array, in2: Array, mode: ModeString, method: str, precision: PrecisionLike) -> Array` |
| `scipy.signal.correlate2d` | Function | `(in1: Array, in2: Array, mode: ModeString, boundary: str, fillvalue: float, precision: PrecisionLike) -> Array` |
| `scipy.signal.csd` | Function | `(x: Array, y: ArrayLike \| None, fs: ArrayLike, window: str, nperseg: int \| None, noverlap: int \| None, nfft: int \| None, detrend: str, return_onesided: bool, scaling: str, axis: int, average: str) -> tuple[Array, Array]` |
| `scipy.signal.detrend` | Function | `(data: ArrayLike, axis: int, type: str, bp: int, overwrite_data: None) -> Array` |
| `scipy.signal.fftconvolve` | Function | `(in1: ArrayLike, in2: ArrayLike, mode: ModeString, axes: Sequence[int] \| None) -> Array` |
| `scipy.signal.istft` | Function | `(Zxx: Array, fs: ArrayLike, window: str, nperseg: int \| None, noverlap: int \| None, nfft: int \| None, input_onesided: bool, boundary: bool, time_axis: int, freq_axis: int) -> tuple[Array, Array]` |
| `scipy.signal.stft` | Function | `(x: Array, fs: ArrayLike, window: str, nperseg: int, noverlap: int \| None, nfft: int \| None, detrend: bool, return_onesided: bool, boundary: str \| None, padded: bool, axis: int) -> tuple[Array, Array, Array]` |
| `scipy.signal.welch` | Function | `(x: Array, fs: ArrayLike, window: str, nperseg: int \| None, noverlap: int \| None, nfft: int \| None, detrend: str, return_onesided: bool, scaling: str, axis: int, average: str) -> tuple[Array, Array]` |
| `scipy.sparse.linalg.bicgstab` | Function | `(A, b, x0, tol, atol, maxiter, M)` |
| `scipy.sparse.linalg.cg` | Function | `(A, b, x0, tol, atol, maxiter, M)` |
| `scipy.sparse.linalg.gmres` | Function | `(A, b, x0, tol, atol, restart, maxiter, M, solve_method)` |
| `scipy.spatial.transform.Rotation` | Class | `(...)` |
| `scipy.spatial.transform.Slerp` | Class | `(...)` |
| `scipy.special.bernoulli` | Function | `(n: int) -> Array` |
| `scipy.special.bessel_jn` | Function | `(z: ArrayLike, v: int, n_iter: int) -> Array` |
| `scipy.special.beta` | Function | `(a: ArrayLike, b: ArrayLike) -> Array` |
| `scipy.special.betainc` | Function | `(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> Array` |
| `scipy.special.betaln` | Function | `(a: ArrayLike, b: ArrayLike) -> Array` |
| `scipy.special.digamma` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.entr` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.erf` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.erfc` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.erfinv` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.exp1` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.expi` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.expit` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.expn` | Function | `(n: ArrayLike, x: ArrayLike) -> Array` |
| `scipy.special.factorial` | Function | `(n: ArrayLike, exact: bool) -> Array` |
| `scipy.special.fresnel` | Function | `(x: ArrayLike) -> tuple[Array, Array]` |
| `scipy.special.gamma` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.gammainc` | Function | `(a: ArrayLike, x: ArrayLike) -> Array` |
| `scipy.special.gammaincc` | Function | `(a: ArrayLike, x: ArrayLike) -> Array` |
| `scipy.special.gammaln` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.gammasgn` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.hyp1f1` | Function | `(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> Array` |
| `scipy.special.hyp2f1` | Function | `(a: ArrayLike, b: ArrayLike, c: ArrayLike, x: ArrayLike) -> Array` |
| `scipy.special.i0` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.i0e` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.i1` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.i1e` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.kl_div` | Function | `(p: ArrayLike, q: ArrayLike) -> Array` |
| `scipy.special.log_ndtr` | Function | `(x: ArrayLike, series_order: int) -> Array` |
| `scipy.special.log_softmax` | Function | `(x: ArrayLike, axis: int \| tuple[int, ...] \| None) -> Array` |
| `scipy.special.logit` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.logsumexp` | Object | `` |
| `scipy.special.lpmn` | Object | `` |
| `scipy.special.lpmn_values` | Object | `` |
| `scipy.special.multigammaln` | Function | `(a: ArrayLike, d: ArrayLike) -> Array` |
| `scipy.special.ndtr` | Function | `(x: ArrayLike) -> Array` |
| `scipy.special.ndtri` | Function | `(p: ArrayLike) -> Array` |
| `scipy.special.poch` | Function | `(z: ArrayLike, m: ArrayLike) -> Array` |
| `scipy.special.polygamma` | Function | `(n: ArrayLike, x: ArrayLike) -> Array` |
| `scipy.special.rel_entr` | Function | `(p: ArrayLike, q: ArrayLike) -> Array` |
| `scipy.special.sici` | Function | `(x: ArrayLike) -> tuple[Array, Array]` |
| `scipy.special.softmax` | Function | `(x: ArrayLike, axis: int \| tuple[int, ...] \| None) -> Array` |
| `scipy.special.spence` | Function | `(x: Array) -> Array` |
| `scipy.special.sph_harm` | Object | `` |
| `scipy.special.sph_harm_y` | Function | `(n: Array, m: Array, theta: Array, phi: Array, diff_n: int \| None, n_max: int \| None) -> Array` |
| `scipy.special.xlog1py` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `scipy.special.xlogy` | Function | `(x: ArrayLike, y: ArrayLike) -> Array` |
| `scipy.special.zeta` | Function | `(x: ArrayLike, q: ArrayLike \| None) -> Array` |
| `scipy.stats.bernoulli.cdf` | Function | `(k: ArrayLike, p: ArrayLike) -> Array` |
| `scipy.stats.bernoulli.logpmf` | Function | `(k: ArrayLike, p: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.bernoulli.pmf` | Function | `(k: ArrayLike, p: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.bernoulli.ppf` | Function | `(q: ArrayLike, p: ArrayLike) -> Array` |
| `scipy.stats.beta.cdf` | Function | `(x: ArrayLike, a: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.beta.logcdf` | Function | `(x: ArrayLike, a: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.beta.logpdf` | Function | `(x: ArrayLike, a: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.beta.logsf` | Function | `(x: ArrayLike, a: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.beta.pdf` | Function | `(x: ArrayLike, a: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.beta.sf` | Function | `(x: ArrayLike, a: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.betabinom.logpmf` | Function | `(k: ArrayLike, n: ArrayLike, a: ArrayLike, b: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.betabinom.pmf` | Function | `(k: ArrayLike, n: ArrayLike, a: ArrayLike, b: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.binom.logpmf` | Function | `(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.binom.pmf` | Function | `(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.cauchy.cdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.cauchy.isf` | Function | `(q: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.cauchy.logcdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.cauchy.logpdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.cauchy.logsf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.cauchy.pdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.cauchy.ppf` | Function | `(q: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.cauchy.sf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.chi2.cdf` | Function | `(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.chi2.logcdf` | Function | `(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.chi2.logpdf` | Function | `(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.chi2.logsf` | Function | `(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.chi2.pdf` | Function | `(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.chi2.sf` | Function | `(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.dirichlet.logpdf` | Function | `(x: ArrayLike, alpha: ArrayLike) -> Array` |
| `scipy.stats.dirichlet.pdf` | Function | `(x: ArrayLike, alpha: ArrayLike) -> Array` |
| `scipy.stats.expon.cdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.expon.logcdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.expon.logpdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.expon.logsf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.expon.pdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.expon.ppf` | Function | `(q: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.expon.sf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gamma.cdf` | Function | `(x: ArrayLike, a: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gamma.logcdf` | Function | `(x: ArrayLike, a: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gamma.logpdf` | Function | `(x: ArrayLike, a: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gamma.logsf` | Function | `(x: ArrayLike, a: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gamma.pdf` | Function | `(x: ArrayLike, a: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gamma.sf` | Function | `(x: ArrayLike, a: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gaussian_kde` | Class | `(dataset, bw_method: BwMethod, weights)` |
| `scipy.stats.gennorm.cdf` | Function | `(x: ArrayLike, beta: ArrayLike) -> Array` |
| `scipy.stats.gennorm.logpdf` | Function | `(x: ArrayLike, beta: ArrayLike) -> Array` |
| `scipy.stats.gennorm.pdf` | Function | `(x: ArrayLike, beta: ArrayLike) -> Array` |
| `scipy.stats.geom.logpmf` | Function | `(k: ArrayLike, p: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.geom.pmf` | Function | `(k: ArrayLike, p: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.gumbel_l.cdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_l.logcdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_l.logpdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_l.logsf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_l.pdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_l.ppf` | Function | `(p: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_l.sf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_r.cdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_r.logcdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_r.logpdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_r.logsf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_r.pdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_r.ppf` | Function | `(p: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.gumbel_r.sf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.laplace.cdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.laplace.logpdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.laplace.pdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.logistic.cdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.logistic.isf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.logistic.logpdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.logistic.pdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.logistic.ppf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.logistic.sf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.mode` | Function | `(a: ArrayLike, axis: int \| None, nan_policy: str, keepdims: bool) -> ModeResult` |
| `scipy.stats.multinomial.logpmf` | Function | `(x: ArrayLike, n: ArrayLike, p: ArrayLike) -> Array` |
| `scipy.stats.multinomial.pmf` | Function | `(x: ArrayLike, n: ArrayLike, p: ArrayLike) -> Array` |
| `scipy.stats.multivariate_normal.logpdf` | Function | `(x: ArrayLike, mean: ArrayLike, cov: ArrayLike, allow_singular: None) -> ArrayLike` |
| `scipy.stats.multivariate_normal.pdf` | Function | `(x: ArrayLike, mean: ArrayLike, cov: ArrayLike) -> Array` |
| `scipy.stats.nbinom.logpmf` | Function | `(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.nbinom.pmf` | Function | `(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.norm.cdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.norm.isf` | Function | `(q: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.norm.logcdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.norm.logpdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.norm.logsf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.norm.pdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.norm.ppf` | Function | `(q: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.norm.sf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.pareto.cdf` | Function | `(x: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.pareto.logcdf` | Function | `(x: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.pareto.logpdf` | Function | `(x: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.pareto.logsf` | Function | `(x: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.pareto.pdf` | Function | `(x: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.pareto.ppf` | Function | `(q: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.pareto.sf` | Function | `(x: ArrayLike, b: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.poisson.cdf` | Function | `(k: ArrayLike, mu: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.poisson.entropy` | Function | `(mu: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.poisson.logpmf` | Function | `(k: ArrayLike, mu: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.poisson.pmf` | Function | `(k: ArrayLike, mu: ArrayLike, loc: ArrayLike) -> Array` |
| `scipy.stats.rankdata` | Function | `(a: ArrayLike, method: str, axis: int \| None, nan_policy: str) -> Array` |
| `scipy.stats.sem` | Function | `(a: ArrayLike, axis: int \| None, ddof: int, nan_policy: str, keepdims: bool) -> Array` |
| `scipy.stats.t.logpdf` | Function | `(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.t.pdf` | Function | `(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.truncnorm.cdf` | Function | `(x, a, b, loc, scale)` |
| `scipy.stats.truncnorm.logcdf` | Function | `(x, a, b, loc, scale)` |
| `scipy.stats.truncnorm.logpdf` | Function | `(x, a, b, loc, scale)` |
| `scipy.stats.truncnorm.logsf` | Function | `(x, a, b, loc, scale)` |
| `scipy.stats.truncnorm.pdf` | Function | `(x, a, b, loc, scale)` |
| `scipy.stats.truncnorm.sf` | Function | `(x, a, b, loc, scale)` |
| `scipy.stats.uniform.cdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.uniform.logpdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.uniform.pdf` | Function | `(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.uniform.ppf` | Function | `(q: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array` |
| `scipy.stats.vonmises.logpdf` | Function | `(x: ArrayLike, kappa: ArrayLike) -> Array` |
| `scipy.stats.vonmises.pdf` | Function | `(x: ArrayLike, kappa: ArrayLike) -> Array` |
| `scipy.stats.wrapcauchy.logpdf` | Function | `(x: ArrayLike, c: ArrayLike) -> Array` |
| `scipy.stats.wrapcauchy.pdf` | Function | `(x: ArrayLike, c: ArrayLike) -> Array` |
| `set_mesh` | Class | `(mesh: mesh_lib.Mesh)` |
| `shard_map` | Function | `(f: F \| None, out_specs: Specs, in_specs: Specs \| None \| InferFromArgs, mesh: Mesh \| AbstractMesh \| None, axis_names: Set[AxisName], check_vma: bool) -> F \| Callable[[G], G]` |
| `sharding.AbstractDevice` | Class | `(device_kind: str, num_cores: int \| None)` |
| `sharding.AbstractMesh` | Class | `(axis_sizes: tuple[int, ...], axis_names: tuple[str, ...], axis_types: AxisType \| tuple[AxisType, ...] \| None, abstract_device)` |
| `sharding.AxisType` | Class | `(...)` |
| `sharding.Mesh` | Class | `(...)` |
| `sharding.NamedSharding` | Class | `(mesh: mesh_lib.Mesh \| mesh_lib.AbstractMesh, spec: PartitionSpec, memory_kind: str \| None, _logical_device_ids)` |
| `sharding.PartitionSpec` | Object | `` |
| `sharding.PmapSharding` | Object | `` |
| `sharding.Sharding` | Class | `(...)` |
| `sharding.SingleDeviceSharding` | Class | `(device: Device, memory_kind: str \| None)` |
| `sharding.auto_axes` | Function | `(f, axes: str \| tuple[str, ...] \| None, out_sharding)` |
| `sharding.explicit_axes` | Function | `(f, axes: str \| tuple[str, ...] \| None, in_sharding)` |
| `sharding.get_abstract_mesh` | Function | `() -> AbstractMesh` |
| `sharding.get_mesh` | Function | `() -> mesh_lib.Mesh` |
| `sharding.reshard` | Function | `(xs, out_shardings)` |
| `sharding.set_mesh` | Class | `(mesh: mesh_lib.Mesh)` |
| `sharding.use_abstract_mesh` | Class | `(mesh: AbstractMesh)` |
| `smap` | Function | `(f: F \| None, in_axes: int \| None \| InferFromArgs \| tuple[Any, ...], out_axes: Any, axis_name: AxisName) -> F \| Callable[[G], G]` |
| `softmax_custom_jvp` | Object | `` |
| `stages.ArgInfo` | Class | `(_aval: core.AbstractValue, donated: bool)` |
| `stages.Compiled` | Class | `(executable, const_args: list[ArrayLike], args_info, out_tree, no_kwargs, in_types, out_types)` |
| `stages.CompilerOptions` | Object | `` |
| `stages.Lowered` | Class | `(lowering: Lowering, args_info, out_tree: tree_util.PyTreeDef, no_kwargs: bool, in_types, out_types)` |
| `stages.Traced` | Class | `(meta_tys_flat, params, in_tree, out_tree, consts)` |
| `stages.Wrapped` | Class | `(...)` |
| `test_util.check_grads` | Function | `(f, args, order, modes, atol, rtol, eps)` |
| `test_util.check_jvp` | Function | `(f, f_jvp, args, atol, rtol, eps, err_msg)` |
| `test_util.check_vjp` | Function | `(f, f_vjp, args, atol, rtol, eps, err_msg)` |
| `thread_guard` | Object | `` |
| `threefry_partitionable` | Object | `` |
| `tools.jax_to_ir.jax2tf` | Object | `` |
| `tools.jax_to_ir.jax_to_hlo` | Object | `` |
| `tools.jax_to_ir.jax_to_ir` | Function | `(fn, input_shapes, constants, format)` |
| `tools.jax_to_ir.jax_to_tf` | Object | `` |
| `tools.jax_to_ir.jnp` | Object | `` |
| `tools.jax_to_ir.main` | Function | `(argv)` |
| `tools.jax_to_ir.parse_shape_str` | Function | `(s)` |
| `tools.jax_to_ir.set_up_flags` | Function | `()` |
| `tools.jax_to_ir.tf_wrap_with_input_names` | Function | `(f, input_shapes)` |
| `tools.pgo_nsys_converter.args` | Object | `` |
| `tools.pgo_nsys_converter.clean_command` | Object | `` |
| `tools.pgo_nsys_converter.cost_dictionary` | Object | `` |
| `tools.pgo_nsys_converter.m` | Object | `` |
| `tools.pgo_nsys_converter.name` | Object | `` |
| `tools.pgo_nsys_converter.nsys_path` | Object | `` |
| `tools.pgo_nsys_converter.parser` | Object | `` |
| `tools.pgo_nsys_converter.pgle_filename` | Object | `` |
| `tools.pgo_nsys_converter.pgle_folder` | Object | `` |
| `tools.pgo_nsys_converter.proc` | Object | `` |
| `tools.pgo_nsys_converter.profile_folder` | Object | `` |
| `tools.pgo_nsys_converter.query_reports_command` | Object | `` |
| `tools.pgo_nsys_converter.reader` | Object | `` |
| `tools.pgo_nsys_converter.report_name` | Object | `` |
| `tools.pgo_nsys_converter.reports_list` | Object | `` |
| `tools.pgo_nsys_converter.stats_command` | Object | `` |
| `tools.pgo_nsys_converter.thunk_re` | Object | `` |
| `tools.pgo_nsys_converter.time_ns` | Object | `` |
| `transfer_guard` | Function | `(new_val: str) -> Generator[None, None, None]` |
| `transfer_guard_device_to_device` | Object | `` |
| `transfer_guard_device_to_host` | Object | `` |
| `transfer_guard_host_to_device` | Object | `` |
| `tree.all` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> bool` |
| `tree.broadcast` | Function | `(prefix_tree: Any, full_tree: Any, is_leaf: Callable[[Any], bool] \| None) -> Any` |
| `tree.flatten` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> tuple[list[tree_util.Leaf], tree_util.PyTreeDef]` |
| `tree.flatten_with_path` | Function | `(tree: Any, is_leaf: Callable[..., bool] \| None, is_leaf_takes_path: bool) -> tuple[list[tuple[tree_util.KeyPath, Any]], tree_util.PyTreeDef]` |
| `tree.leaves` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> list[tree_util.Leaf]` |
| `tree.leaves_with_path` | Function | `(tree: Any, is_leaf: Callable[..., bool] \| None, is_leaf_takes_path: bool) -> list[tuple[tree_util.KeyPath, Any]]` |
| `tree.map` | Function | `(f: Callable[..., Any], tree: Any, rest: Any, is_leaf: Callable[[Any], bool] \| None) -> Any` |
| `tree.map_with_path` | Function | `(f: Callable[..., Any], tree: Any, rest: Any, is_leaf: Callable[..., bool] \| None, is_leaf_takes_path: bool) -> Any` |
| `tree.reduce` | Function | `(function: Callable[[T, Any], T], tree: Any, initializer: T \| tree_util.Unspecified, is_leaf: Callable[[Any], bool] \| None) -> T` |
| `tree.reduce_associative` | Function | `(operation: Callable[[T, T], T], tree: Any, identity: T \| tree_util.Unspecified, is_leaf: Callable[[Any], bool] \| None) -> T` |
| `tree.static` | Function | `(kwargs)` |
| `tree.structure` | Function | `(tree: Any, is_leaf: None \| Callable[[Any], bool]) -> tree_util.PyTreeDef` |
| `tree.transpose` | Function | `(outer_treedef: tree_util.PyTreeDef, inner_treedef: tree_util.PyTreeDef \| None, pytree_to_transpose: Any) -> Any` |
| `tree.unflatten` | Function | `(treedef: tree_util.PyTreeDef, leaves: Iterable[tree_util.Leaf]) -> Any` |
| `tree_util.DictKey` | Object | `` |
| `tree_util.FlattenedIndexKey` | Object | `` |
| `tree_util.GetAttrKey` | Object | `` |
| `tree_util.KeyEntry` | Object | `` |
| `tree_util.KeyPath` | Object | `` |
| `tree_util.Partial` | Class | `(...)` |
| `tree_util.PyTreeDef` | Object | `` |
| `tree_util.SequenceKey` | Object | `` |
| `tree_util.all_leaves` | Function | `(iterable: Iterable[Any], is_leaf: Callable[[Any], bool] \| None) -> bool` |
| `tree_util.default_registry` | Object | `` |
| `tree_util.keystr` | Function | `(keys: KeyPath, simple: bool, separator: str) -> str` |
| `tree_util.register_dataclass` | Function | `(nodetype: Typ, data_fields: Sequence[str] \| None, meta_fields: Sequence[str] \| None, drop_fields: Sequence[str]) -> Typ` |
| `tree_util.register_pytree_node` | Function | `(nodetype: type[T], flatten_func: Callable[[T], tuple[_Children, _AuxData]], unflatten_func: Callable[[_AuxData, _Children], T], flatten_with_keys_func: Callable[[T], tuple[KeyLeafPairs, _AuxData]] \| None) -> None` |
| `tree_util.register_pytree_node_class` | Function | `(cls: Typ) -> Typ` |
| `tree_util.register_pytree_with_keys` | Function | `(nodetype: type[T], flatten_with_keys: Callable[[T], tuple[Iterable[KeyLeafPair], _AuxData]], unflatten_func: Callable[[_AuxData, Iterable[Any]], T], flatten_func: None \| Callable[[T], tuple[Iterable[Any], _AuxData]])` |
| `tree_util.register_pytree_with_keys_class` | Function | `(cls: Typ) -> Typ` |
| `tree_util.register_static` | Function | `(cls: type[H]) -> type[H]` |
| `tree_util.tree_all` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> bool` |
| `tree_util.tree_broadcast` | Function | `(prefix_tree: Any, full_tree: Any, is_leaf: Callable[[Any], bool] \| None) -> Any` |
| `tree_util.tree_flatten` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> tuple[list[Leaf], PyTreeDef]` |
| `tree_util.tree_flatten_with_path` | Function | `(tree: Any, is_leaf: Callable[..., bool] \| None, is_leaf_takes_path: bool) -> tuple[list[tuple[KeyPath, Any]], PyTreeDef]` |
| `tree_util.tree_leaves` | Function | `(tree: Any, is_leaf: Callable[[Any], bool] \| None) -> list[Leaf]` |
| `tree_util.tree_leaves_with_path` | Function | `(tree: Any, is_leaf: Callable[..., bool] \| None, is_leaf_takes_path: bool) -> list[tuple[KeyPath, Any]]` |
| `tree_util.tree_map` | Function | `(f: Callable[..., Any], tree: Any, rest: Any, is_leaf: Callable[[Any], bool] \| None) -> Any` |
| `tree_util.tree_map_with_path` | Function | `(f: Callable[..., Any], tree: Any, rest: Any, is_leaf: Callable[..., bool] \| None, is_leaf_takes_path: bool) -> Any` |
| `tree_util.tree_reduce` | Function | `(function: Callable[[T, Any], T], tree: Any, initializer: T \| Unspecified, is_leaf: Callable[[Any], bool] \| None) -> T` |
| `tree_util.tree_reduce_associative` | Function | `(operation: Callable[[T, T], T], tree: Any, identity: T \| Unspecified, is_leaf: Callable[[Any], bool] \| None) -> T` |
| `tree_util.tree_structure` | Function | `(tree: Any, is_leaf: None \| Callable[[Any], bool]) -> PyTreeDef` |
| `tree_util.tree_transpose` | Function | `(outer_treedef: PyTreeDef, inner_treedef: PyTreeDef \| None, pytree_to_transpose: Any) -> Any` |
| `tree_util.tree_unflatten` | Function | `(treedef: PyTreeDef, leaves: Iterable[Leaf]) -> Any` |
| `tree_util.treedef_children` | Function | `(treedef: PyTreeDef) -> list[PyTreeDef]` |
| `tree_util.treedef_is_leaf` | Function | `(treedef: PyTreeDef) -> bool` |
| `tree_util.treedef_tuple` | Function | `(treedefs: Iterable[PyTreeDef]) -> PyTreeDef` |
| `typeof` | Function | `(x: Any) -> Any` |
| `typing.ArrayLike` | Object | `` |
| `typing.DTypeLike` | Object | `` |
| `value_and_grad` | Function | `(fun: Callable, argnums: int \| Sequence[int], has_aux: bool, holomorphic: bool, allow_int: bool, reduce_axes: Sequence[AxisName]) -> Callable[..., tuple[Any, Any]]` |
| `vjp` | Function | `(fun: Callable, primals, has_aux: bool, reduce_axes) -> tuple[Any, Callable] \| tuple[Any, Callable, Any]` |
| `vmap` | Function | `(fun: F, in_axes: int \| None \| Sequence[Any], out_axes: Any, axis_name: AxisName \| None, axis_size: int \| None, spmd_axis_name: AxisName \| tuple[AxisName, ...] \| None, sum_match: bool) -> F` |
| `xla` | Object | `` |