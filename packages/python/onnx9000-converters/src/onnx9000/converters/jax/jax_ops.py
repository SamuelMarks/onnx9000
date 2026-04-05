"""Module providing core logic and structural definitions for jax ops."""

from typing import Any

from onnx9000.core.ir import Node
from onnx9000.core.registry import register_op


@register_op("Array", "jax")
def _map_jax_Array(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="Array")


@register_op("config", "jax")
def _map_jax_config(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_config operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="config")


@register_op("enable_checks", "jax")
def _map_jax_enable_checks(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_enable_checks operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="enable_checks")


@register_op("enable_x64", "jax")
def _map_jax_enable_x64(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_enable_x64 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="enable_x64")


@register_op("debug_key_reuse", "jax")
def _map_jax_debug_key_reuse(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_debug_key_reuse operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="debug_key_reuse")


@register_op("check_tracer_leaks", "jax")
def _map_jax_check_tracer_leaks(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_check_tracer_leaks operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="check_tracer_leaks")


@register_op("checking_leaks", "jax")
def _map_jax_checking_leaks(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_checking_leaks operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="checking_leaks")


@register_op("enable_custom_prng", "jax")
def _map_jax_enable_custom_prng(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_enable_custom_prng operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="enable_custom_prng")


@register_op("softmax_custom_jvp", "jax")
def _map_jax_softmax_custom_jvp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_softmax_custom_jvp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="softmax_custom_jvp")


@register_op("enable_custom_vjp_by_custom_transpose", "jax")
def _map_jax_enable_custom_vjp_by_custom_transpose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_enable_custom_vjp_by_custom_transpose operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="enable_custom_vjp_by_custom_transpose",
    )


@register_op("debug_nans", "jax")
def _map_jax_debug_nans(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_debug_nans operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="debug_nans")


@register_op("debug_infs", "jax")
def _map_jax_debug_infs(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_debug_infs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="debug_infs")


@register_op("log_compiles", "jax")
def _map_jax_log_compiles(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_log_compiles operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="log_compiles")


@register_op("no_tracing", "jax")
def _map_jax_no_tracing(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_no_tracing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="no_tracing")


@register_op("no_execution", "jax")
def _map_jax_no_execution(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_no_execution operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="no_execution")


@register_op("explain_cache_misses", "jax")
def _map_jax_explain_cache_misses(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_explain_cache_misses operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="explain_cache_misses")


@register_op("default_device", "jax")
def _map_jax_default_device(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_default_device operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="default_device")


@register_op("default_matmul_precision", "jax")
def _map_jax_default_matmul_precision(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_default_matmul_precision operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="default_matmul_precision")


@register_op("default_prng_impl", "jax")
def _map_jax_default_prng_impl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_default_prng_impl operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="default_prng_impl")


@register_op("numpy_dtype_promotion", "jax")
def _map_jax_numpy_dtype_promotion(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_dtype_promotion operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy_dtype_promotion")


@register_op("numpy_rank_promotion", "jax")
def _map_jax_numpy_rank_promotion(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_rank_promotion operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy_rank_promotion")


@register_op("allow_f16_reductions", "jax")
def _map_jax_allow_f16_reductions(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_allow_f16_reductions operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="allow_f16_reductions")


@register_op("jax2tf_associative_scan_reductions", "jax")
def _map_jax_jax2tf_associative_scan_reductions(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_jax2tf_associative_scan_reductions operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="jax2tf_associative_scan_reductions",
    )


@register_op("legacy_prng_key", "jax")
def _map_jax_legacy_prng_key(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_legacy_prng_key operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="legacy_prng_key")


@register_op("threefry_partitionable", "jax")
def _map_jax_threefry_partitionable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_threefry_partitionable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="threefry_partitionable")


@register_op("array_garbage_collection_guard", "jax")
def _map_jax_array_garbage_collection_guard(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_array_garbage_collection_guard operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="array_garbage_collection_guard"
    )


@register_op("transfer_guard", "jax")
def _map_jax_transfer_guard(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_transfer_guard operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="transfer_guard")


@register_op("transfer_guard_host_to_device", "jax")
def _map_jax_transfer_guard_host_to_device(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_transfer_guard_host_to_device operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="transfer_guard_host_to_device"
    )


@register_op("transfer_guard_device_to_device", "jax")
def _map_jax_transfer_guard_device_to_device(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_transfer_guard_device_to_device operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="transfer_guard_device_to_device"
    )


@register_op("transfer_guard_device_to_host", "jax")
def _map_jax_transfer_guard_device_to_host(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_transfer_guard_device_to_host operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="transfer_guard_device_to_host"
    )


@register_op("make_user_context", "jax")
def _map_jax_make_user_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_make_user_context operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="make_user_context")


@register_op("remove_size_one_mesh_axis_from_type", "jax")
def _map_jax_remove_size_one_mesh_axis_from_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_remove_size_one_mesh_axis_from_type operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="remove_size_one_mesh_axis_from_type",
    )


@register_op("thread_guard", "jax")
def _map_jax_thread_guard(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_thread_guard operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="thread_guard")


@register_op("ensure_compile_time_eval", "jax")
def _map_jax_ensure_compile_time_eval(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ensure_compile_time_eval operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ensure_compile_time_eval")


@register_op("print_environment_info", "jax")
def _map_jax_print_environment_info(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_print_environment_info operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="print_environment_info")


@register_op("Device", "jax")
def _map_jax_Device(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_Device operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="Device")


@register_op("typeof", "jax")
def _map_jax_typeof(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_typeof operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typeof")


@register_op("effects_barrier", "jax")
def _map_jax_effects_barrier(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_effects_barrier operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="effects_barrier")


@register_op("block_until_ready", "jax")
def _map_jax_block_until_ready(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_block_until_ready operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="block_until_ready")


@register_op("checkpoint", "jax")
def _map_jax_checkpoint(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_checkpoint operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="checkpoint")


@register_op("checkpoint_policies", "jax")
def _map_jax_checkpoint_policies(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_checkpoint_policies operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="checkpoint_policies")


@register_op("remat", "jax")
def _map_jax_remat(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_remat operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="remat")


@register_op("clear_caches", "jax")
def _map_jax_clear_caches(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_clear_caches operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="clear_caches")


@register_op("copy_to_host_async", "jax")
def _map_jax_copy_to_host_async(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_copy_to_host_async operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="copy_to_host_async")


@register_op("closure_convert", "jax")
def _map_jax_closure_convert(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_closure_convert operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="closure_convert")


@register_op("custom_gradient", "jax")
def _map_jax_custom_gradient(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_custom_gradient operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="custom_gradient")


@register_op("custom_jvp", "jax")
def _map_jax_custom_jvp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_custom_jvp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="custom_jvp")


@register_op("custom_vjp", "jax")
def _map_jax_custom_vjp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_custom_vjp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="custom_vjp")


@register_op("default_backend", "jax")
def _map_jax_default_backend(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_default_backend operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="default_backend")


@register_op("device_count", "jax")
def _map_jax_device_count(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_device_count operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="device_count")


@register_op("device_get", "jax")
def _map_jax_device_get(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_device_get operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="device_get")


@register_op("device_put", "jax")
def _map_jax_device_put(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_device_put operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="device_put")


@register_op("devices", "jax")
def _map_jax_devices(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_devices operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="devices")


@register_op("disable_jit", "jax")
def _map_jax_disable_jit(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_disable_jit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="disable_jit")


@register_op("eval_shape", "jax")
def _map_jax_eval_shape(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_eval_shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="eval_shape")


@register_op("float0", "jax")
def _map_jax_float0(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_float0 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="float0")


@register_op("fwd_and_bwd", "jax")
def _map_jax_fwd_and_bwd(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_fwd_and_bwd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="fwd_and_bwd")


@register_op("grad", "jax")
def _map_jax_grad(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_grad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="grad")


@register_op("hessian", "jax")
def _map_jax_hessian(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_hessian operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="hessian")


@register_op("host_count", "jax")
def _map_jax_host_count(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_host_count operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="host_count")


@register_op("host_id", "jax")
def _map_jax_host_id(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_host_id operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="host_id")


@register_op("host_ids", "jax")
def _map_jax_host_ids(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_host_ids operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="host_ids")


@register_op("jacobian", "jax")
def _map_jax_jacobian(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_jacobian operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="jacobian")


@register_op("jacfwd", "jax")
def _map_jax_jacfwd(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_jacfwd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="jacfwd")


@register_op("jacrev", "jax")
def _map_jax_jacrev(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_jacrev operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="jacrev")


@register_op("jit", "jax")
def _map_jax_jit(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_jit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="jit")


@register_op("jvp", "jax")
def _map_jax_jvp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_jvp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="jvp")


@register_op("local_device_count", "jax")
def _map_jax_local_device_count(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_local_device_count operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="local_device_count")


@register_op("local_devices", "jax")
def _map_jax_local_devices(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_local_devices operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="local_devices")


@register_op("linearize", "jax")
def _map_jax_linearize(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_linearize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linearize")


@register_op("linear_transpose", "jax")
def _map_jax_linear_transpose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_linear_transpose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linear_transpose")


@register_op("live_arrays", "jax")
def _map_jax_live_arrays(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_live_arrays operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="live_arrays")


@register_op("make_jaxpr", "jax")
def _map_jax_make_jaxpr(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_make_jaxpr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="make_jaxpr")


@register_op("named_call", "jax")
def _map_jax_named_call(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_named_call operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="named_call")


@register_op("named_scope", "jax")
def _map_jax_named_scope(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_named_scope operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="named_scope")


@register_op("pmap", "jax")
def _map_jax_pmap(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_pmap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="pmap")


@register_op("process_count", "jax")
def _map_jax_process_count(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_process_count operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="process_count")


@register_op("process_index", "jax")
def _map_jax_process_index(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_process_index operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="process_index")


@register_op("process_indices", "jax")
def _map_jax_process_indices(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_process_indices operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="process_indices")


@register_op("pure_callback", "jax")
def _map_jax_pure_callback(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_pure_callback operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="pure_callback")


@register_op("ShapeDtypeStruct", "jax")
def _map_jax_ShapeDtypeStruct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ShapeDtypeStruct operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ShapeDtypeStruct")


@register_op("value_and_grad", "jax")
def _map_jax_value_and_grad(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_value_and_grad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="value_and_grad")


@register_op("vjp", "jax")
def _map_jax_vjp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_vjp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="vjp")


@register_op("vmap", "jax")
def _map_jax_vmap(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_vmap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="vmap")


@register_op("ds", "jax")
def _map_jax_ds(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ds operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ds")


@register_op("NamedSharding", "jax")
def _map_jax_NamedSharding(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_NamedSharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="NamedSharding")


@register_op("make_mesh", "jax")
def _map_jax_make_mesh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_make_mesh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="make_mesh")


@register_op("set_mesh", "jax")
def _map_jax_set_mesh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_set_mesh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="set_mesh")


@register_op("P", "jax")
def _map_jax_P(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_P operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="P")


@register_op("reshard", "jax")
def _map_jax_reshard(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_reshard operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="reshard")


@register_op("shard_map", "jax")
def _map_jax_shard_map(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_shard_map operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="shard_map")


@register_op("smap", "jax")
def _map_jax_smap(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_smap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="smap")


@register_op("new_ref", "jax")
def _map_jax_new_ref(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_new_ref operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="new_ref")


@register_op("empty_ref", "jax")
def _map_jax_empty_ref(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_empty_ref operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="empty_ref")


@register_op("free_ref", "jax")
def _map_jax_free_ref(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_free_ref operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="free_ref")


@register_op("freeze", "jax")
def _map_jax_freeze(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_freeze operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="freeze")


@register_op("Ref", "jax")
def _map_jax_Ref(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_Ref operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="Ref")


@register_op("ad", "jax")
def _map_jax_ad(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ad")


@register_op("batching", "jax")
def _map_jax_batching(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_batching operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="batching")


@register_op("mlir", "jax")
def _map_jax_mlir(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_mlir operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="mlir")


@register_op("partial_eval", "jax")
def _map_jax_partial_eval(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_partial_eval operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="partial_eval")


@register_op("pxla", "jax")
def _map_jax_pxla(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_pxla operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="pxla")


@register_op("xla", "jax")
def _map_jax_xla(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_xla operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="xla")


@register_op("make_array_from_single_device_arrays", "jax")
def _map_jax_make_array_from_single_device_arrays(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_make_array_from_single_device_arrays operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="make_array_from_single_device_arrays",
    )


@register_op("make_array_from_callback", "jax")
def _map_jax_make_array_from_callback(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_make_array_from_callback operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="make_array_from_callback")


@register_op("make_array_from_process_local_data", "jax")
def _map_jax_make_array_from_process_local_data(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_make_array_from_process_local_data operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="make_array_from_process_local_data",
    )


@register_op("Shard", "jax")
def _map_jax_Shard(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_Shard operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="Shard")


@register_op("device_put_replicated", "jax")
def _map_jax_device_put_replicated(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_device_put_replicated operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="device_put_replicated")


@register_op("device_put_sharded", "jax")
def _map_jax_device_put_sharded(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_device_put_sharded operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="device_put_sharded")


@register_op("dtypes.bfloat16", "jax")
def _map_jax_dtypes_bfloat16(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_dtypes_bfloat16 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="dtypes.bfloat16")


@register_op("dtypes.itemsize_bits", "jax")
def _map_jax_dtypes_itemsize_bits(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_dtypes_itemsize_bits operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="dtypes.itemsize_bits")


@register_op("dtypes.canonicalize_dtype", "jax")
def _map_jax_dtypes_canonicalize_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_dtypes_canonicalize_dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="dtypes.canonicalize_dtype"
    )


@register_op("dtypes.finfo", "jax")
def _map_jax_dtypes_finfo(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_dtypes_finfo operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="dtypes.finfo")


@register_op("dtypes.float0", "jax")
def _map_jax_dtypes_float0(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_dtypes_float0 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="dtypes.float0")


@register_op("dtypes.iinfo", "jax")
def _map_jax_dtypes_iinfo(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_dtypes_iinfo operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="dtypes.iinfo")


@register_op("dtypes.issubdtype", "jax")
def _map_jax_dtypes_issubdtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_dtypes_issubdtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="dtypes.issubdtype")


@register_op("dtypes.extended", "jax")
def _map_jax_dtypes_extended(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_dtypes_extended operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="dtypes.extended")


@register_op("dtypes.prng_key", "jax")
def _map_jax_dtypes_prng_key(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_dtypes_prng_key operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="dtypes.prng_key")


@register_op("dtypes.result_type", "jax")
def _map_jax_dtypes_result_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_dtypes_result_type operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="dtypes.result_type")


@register_op("dtypes.scalar_type_of", "jax")
def _map_jax_dtypes_scalar_type_of(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_dtypes_scalar_type_of operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="dtypes.scalar_type_of")


@register_op("tree_util.DictKey", "jax")
def _map_jax_tree_util_DictKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_DictKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.DictKey")


@register_op("tree_util.FlattenedIndexKey", "jax")
def _map_jax_tree_util_FlattenedIndexKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_FlattenedIndexKey operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.FlattenedIndexKey"
    )


@register_op("tree_util.GetAttrKey", "jax")
def _map_jax_tree_util_GetAttrKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_GetAttrKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.GetAttrKey")


@register_op("tree_util.KeyEntry", "jax")
def _map_jax_tree_util_KeyEntry(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_KeyEntry operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.KeyEntry")


@register_op("tree_util.KeyPath", "jax")
def _map_jax_tree_util_KeyPath(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_KeyPath operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.KeyPath")


@register_op("tree_util.Partial", "jax")
def _map_jax_tree_util_Partial(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_Partial operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.Partial")


@register_op("tree_util.PyTreeDef", "jax")
def _map_jax_tree_util_PyTreeDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_PyTreeDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.PyTreeDef")


@register_op("tree_util.SequenceKey", "jax")
def _map_jax_tree_util_SequenceKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_SequenceKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.SequenceKey")


@register_op("tree_util.all_leaves", "jax")
def _map_jax_tree_util_all_leaves(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_all_leaves operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.all_leaves")


@register_op("tree_util.default_registry", "jax")
def _map_jax_tree_util_default_registry(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_default_registry operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.default_registry"
    )


@register_op("tree_util.keystr", "jax")
def _map_jax_tree_util_keystr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_keystr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.keystr")


@register_op("tree_util.register_dataclass", "jax")
def _map_jax_tree_util_register_dataclass(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_register_dataclass operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.register_dataclass"
    )


@register_op("tree_util.register_pytree_node_class", "jax")
def _map_jax_tree_util_register_pytree_node_class(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_register_pytree_node_class operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="tree_util.register_pytree_node_class",
    )


@register_op("tree_util.register_pytree_node", "jax")
def _map_jax_tree_util_register_pytree_node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_register_pytree_node operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.register_pytree_node"
    )


@register_op("tree_util.register_pytree_with_keys_class", "jax")
def _map_jax_tree_util_register_pytree_with_keys_class(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_register_pytree_with_keys_class operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="tree_util.register_pytree_with_keys_class",
    )


@register_op("tree_util.register_pytree_with_keys", "jax")
def _map_jax_tree_util_register_pytree_with_keys(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_register_pytree_with_keys operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="tree_util.register_pytree_with_keys",
    )


@register_op("tree_util.register_static", "jax")
def _map_jax_tree_util_register_static(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_register_static operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.register_static"
    )


@register_op("tree_util.tree_all", "jax")
def _map_jax_tree_util_tree_all(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_all operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_all")


@register_op("tree_util.tree_broadcast", "jax")
def _map_jax_tree_util_tree_broadcast(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_broadcast operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_broadcast")


@register_op("tree_util.tree_flatten_with_path", "jax")
def _map_jax_tree_util_tree_flatten_with_path(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_flatten_with_path operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_flatten_with_path"
    )


@register_op("tree_util.tree_flatten", "jax")
def _map_jax_tree_util_tree_flatten(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_flatten operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_flatten")


@register_op("tree_util.tree_leaves_with_path", "jax")
def _map_jax_tree_util_tree_leaves_with_path(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_leaves_with_path operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_leaves_with_path"
    )


@register_op("tree_util.tree_leaves", "jax")
def _map_jax_tree_util_tree_leaves(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_leaves operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_leaves")


@register_op("tree_util.tree_map_with_path", "jax")
def _map_jax_tree_util_tree_map_with_path(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_map_with_path operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_map_with_path"
    )


@register_op("tree_util.tree_map", "jax")
def _map_jax_tree_util_tree_map(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_map operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_map")


@register_op("tree_util.tree_reduce", "jax")
def _map_jax_tree_util_tree_reduce(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_reduce operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_reduce")


@register_op("tree_util.tree_reduce_associative", "jax")
def _map_jax_tree_util_tree_reduce_associative(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_reduce_associative operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_reduce_associative"
    )


@register_op("tree_util.tree_structure", "jax")
def _map_jax_tree_util_tree_structure(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_structure operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_structure")


@register_op("tree_util.tree_transpose", "jax")
def _map_jax_tree_util_tree_transpose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_transpose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_transpose")


@register_op("tree_util.tree_unflatten", "jax")
def _map_jax_tree_util_tree_unflatten(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_tree_unflatten operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.tree_unflatten")


@register_op("tree_util.treedef_children", "jax")
def _map_jax_tree_util_treedef_children(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_treedef_children operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.treedef_children"
    )


@register_op("tree_util.treedef_is_leaf", "jax")
def _map_jax_tree_util_treedef_is_leaf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_treedef_is_leaf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.treedef_is_leaf"
    )


@register_op("tree_util.treedef_tuple", "jax")
def _map_jax_tree_util_treedef_tuple(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_util_treedef_tuple operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree_util.treedef_tuple")


@register_op("dlpack.from_dlpack", "jax")
def _map_jax_dlpack_from_dlpack(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_dlpack_from_dlpack operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="dlpack.from_dlpack")


@register_op("dlpack.is_supported_dtype", "jax")
def _map_jax_dlpack_is_supported_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_dlpack_is_supported_dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="dlpack.is_supported_dtype"
    )


@register_op("custom_transpose.custom_transpose", "jax")
def _map_jax_custom_transpose_custom_transpose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_transpose_custom_transpose operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="custom_transpose.custom_transpose"
    )


@register_op("memory.Space", "jax")
def _map_jax_memory_Space(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_memory_Space operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="memory.Space")


@register_op("ffi.build_ffi_lowering_function", "jax")
def _map_jax_ffi_build_ffi_lowering_function(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ffi_build_ffi_lowering_function operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="ffi.build_ffi_lowering_function"
    )


@register_op("ffi.ffi_call", "jax")
def _map_jax_ffi_ffi_call(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ffi_ffi_call operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ffi.ffi_call")


@register_op("ffi.ffi_lowering", "jax")
def _map_jax_ffi_ffi_lowering(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ffi_ffi_lowering operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ffi.ffi_lowering")


@register_op("ffi.include_dir", "jax")
def _map_jax_ffi_include_dir(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ffi_include_dir operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ffi.include_dir")


@register_op("ffi.pycapsule", "jax")
def _map_jax_ffi_pycapsule(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ffi_pycapsule operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ffi.pycapsule")


@register_op("ffi.register_ffi_target", "jax")
def _map_jax_ffi_register_ffi_target(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ffi_register_ffi_target operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ffi.register_ffi_target")


@register_op("ffi.register_ffi_type_id", "jax")
def _map_jax_ffi_register_ffi_type_id(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ffi_register_ffi_type_id operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ffi.register_ffi_type_id")


@register_op("ffi.register_ffi_type", "jax")
def _map_jax_ffi_register_ffi_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ffi_register_ffi_type operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ffi.register_ffi_type")


@register_op("ffi.register_ffi_target_as_batch_partitionable", "jax")
def _map_jax_ffi_register_ffi_target_as_batch_partitionable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ffi_register_ffi_target_as_batch_partitionable operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="ffi.register_ffi_target_as_batch_partitionable",
    )


@register_op("cloud_tpu_init.cloud_tpu_init", "jax")
def _map_jax_cloud_tpu_init_cloud_tpu_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_cloud_tpu_init_cloud_tpu_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="cloud_tpu_init.cloud_tpu_init"
    )


@register_op("custom_derivatives.closure_convert", "jax")
def _map_jax_custom_derivatives_closure_convert(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_closure_convert operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="custom_derivatives.closure_convert",
    )


@register_op("custom_derivatives.custom_gradient", "jax")
def _map_jax_custom_derivatives_custom_gradient(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_custom_gradient operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="custom_derivatives.custom_gradient",
    )


@register_op("custom_derivatives.custom_jvp", "jax")
def _map_jax_custom_derivatives_custom_jvp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_custom_jvp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="custom_derivatives.custom_jvp"
    )


@register_op("custom_derivatives.custom_jvp_call_p", "jax")
def _map_jax_custom_derivatives_custom_jvp_call_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_custom_jvp_call_p operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="custom_derivatives.custom_jvp_call_p",
    )


@register_op("custom_derivatives.custom_vjp", "jax")
def _map_jax_custom_derivatives_custom_vjp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_custom_vjp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="custom_derivatives.custom_vjp"
    )


@register_op("custom_derivatives.custom_vjp_call_p", "jax")
def _map_jax_custom_derivatives_custom_vjp_call_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_custom_vjp_call_p operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="custom_derivatives.custom_vjp_call_p",
    )


@register_op("custom_derivatives.custom_vjp_primal_tree_values", "jax")
def _map_jax_custom_derivatives_custom_vjp_primal_tree_values(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_custom_vjp_primal_tree_values operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="custom_derivatives.custom_vjp_primal_tree_values",
    )


@register_op("custom_derivatives.CustomVJPPrimal", "jax")
def _map_jax_custom_derivatives_CustomVJPPrimal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_CustomVJPPrimal operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="custom_derivatives.CustomVJPPrimal",
    )


@register_op("custom_derivatives.linear_call", "jax")
def _map_jax_custom_derivatives_linear_call(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_linear_call operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="custom_derivatives.linear_call"
    )


@register_op("custom_derivatives.remat_opt_p", "jax")
def _map_jax_custom_derivatives_remat_opt_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_remat_opt_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="custom_derivatives.remat_opt_p"
    )


@register_op("custom_derivatives.SymbolicZero", "jax")
def _map_jax_custom_derivatives_SymbolicZero(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_SymbolicZero operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="custom_derivatives.SymbolicZero"
    )


@register_op("custom_derivatives.zero_from_primal", "jax")
def _map_jax_custom_derivatives_zero_from_primal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_derivatives_zero_from_primal operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="custom_derivatives.zero_from_primal",
    )


@register_op("custom_batching.custom_vmap", "jax")
def _map_jax_custom_batching_custom_vmap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_batching_custom_vmap operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="custom_batching.custom_vmap"
    )


@register_op("custom_batching.sequential_vmap", "jax")
def _map_jax_custom_batching_sequential_vmap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_custom_batching_sequential_vmap operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="custom_batching.sequential_vmap"
    )


@register_op("typing.ArrayLike", "jax")
def _map_jax_typing_ArrayLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_typing_ArrayLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.ArrayLike")


@register_op("typing.DTypeLike", "jax")
def _map_jax_typing_DTypeLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_typing_DTypeLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.DTypeLike")


@register_op("test_util.check_grads", "jax")
def _map_jax_test_util_check_grads(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_test_util_check_grads operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="test_util.check_grads")


@register_op("test_util.check_jvp", "jax")
def _map_jax_test_util_check_jvp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_test_util_check_jvp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="test_util.check_jvp")


@register_op("test_util.check_vjp", "jax")
def _map_jax_test_util_check_vjp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_test_util_check_vjp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="test_util.check_vjp")


@register_op("ref.Ref", "jax")
def _map_jax_ref_Ref(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ref_Ref operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ref.Ref")


@register_op("ref.empty_ref", "jax")
def _map_jax_ref_empty_ref(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ref_empty_ref operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ref.empty_ref")


@register_op("ref.free_ref", "jax")
def _map_jax_ref_free_ref(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ref_free_ref operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ref.free_ref")


@register_op("ref.freeze", "jax")
def _map_jax_ref_freeze(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ref_freeze operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ref.freeze")


@register_op("ref.new_ref", "jax")
def _map_jax_ref_new_ref(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ref_new_ref operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ref.new_ref")


@register_op("ref.AbstractRef", "jax")
def _map_jax_ref_AbstractRef(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ref_AbstractRef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ref.AbstractRef")


@register_op("ref.get", "jax")
def _map_jax_ref_get(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ref_get operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ref.get")


@register_op("ref.set", "jax")
def _map_jax_ref_set(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ref_set operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ref.set")


@register_op("ref.swap", "jax")
def _map_jax_ref_swap(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ref_swap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ref.swap")


@register_op("ref.addupdate", "jax")
def _map_jax_ref_addupdate(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_ref_addupdate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ref.addupdate")


@register_op("random.PRNGKey", "jax")
def _map_jax_random_PRNGKey(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_PRNGKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.PRNGKey")


@register_op("random.ball", "jax")
def _map_jax_random_ball(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_ball operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.ball")


@register_op("random.bernoulli", "jax")
def _map_jax_random_bernoulli(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_bernoulli operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.bernoulli")


@register_op("random.binomial", "jax")
def _map_jax_random_binomial(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_binomial operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.binomial")


@register_op("random.beta", "jax")
def _map_jax_random_beta(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_beta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.beta")


@register_op("random.bits", "jax")
def _map_jax_random_bits(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_bits operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.bits")


@register_op("random.categorical", "jax")
def _map_jax_random_categorical(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_categorical operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.categorical")


@register_op("random.cauchy", "jax")
def _map_jax_random_cauchy(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_cauchy operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.cauchy")


@register_op("random.chisquare", "jax")
def _map_jax_random_chisquare(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_chisquare operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.chisquare")


@register_op("random.choice", "jax")
def _map_jax_random_choice(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_choice operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.choice")


@register_op("random.clone", "jax")
def _map_jax_random_clone(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_clone operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.clone")


@register_op("random.dirichlet", "jax")
def _map_jax_random_dirichlet(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_dirichlet operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.dirichlet")


@register_op("random.double_sided_maxwell", "jax")
def _map_jax_random_double_sided_maxwell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_double_sided_maxwell operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="random.double_sided_maxwell"
    )


@register_op("random.exponential", "jax")
def _map_jax_random_exponential(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_exponential operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.exponential")


@register_op("random.f", "jax")
def _map_jax_random_f(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_f operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.f")


@register_op("random.fold_in", "jax")
def _map_jax_random_fold_in(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_fold_in operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.fold_in")


@register_op("random.gamma", "jax")
def _map_jax_random_gamma(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_gamma operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.gamma")


@register_op("random.generalized_normal", "jax")
def _map_jax_random_generalized_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_generalized_normal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="random.generalized_normal"
    )


@register_op("random.geometric", "jax")
def _map_jax_random_geometric(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_geometric operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.geometric")


@register_op("random.gumbel", "jax")
def _map_jax_random_gumbel(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_gumbel operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.gumbel")


@register_op("random.key", "jax")
def _map_jax_random_key(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_key operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.key")


@register_op("random.key_data", "jax")
def _map_jax_random_key_data(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_key_data operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.key_data")


@register_op("random.key_impl", "jax")
def _map_jax_random_key_impl(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_key_impl operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.key_impl")


@register_op("random.laplace", "jax")
def _map_jax_random_laplace(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_laplace operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.laplace")


@register_op("random.logistic", "jax")
def _map_jax_random_logistic(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_logistic operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.logistic")


@register_op("random.loggamma", "jax")
def _map_jax_random_loggamma(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_loggamma operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.loggamma")


@register_op("random.lognormal", "jax")
def _map_jax_random_lognormal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_lognormal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.lognormal")


@register_op("random.maxwell", "jax")
def _map_jax_random_maxwell(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_maxwell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.maxwell")


@register_op("random.multinomial", "jax")
def _map_jax_random_multinomial(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_multinomial operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.multinomial")


@register_op("random.multivariate_normal", "jax")
def _map_jax_random_multivariate_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_multivariate_normal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="random.multivariate_normal"
    )


@register_op("random.normal", "jax")
def _map_jax_random_normal(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_normal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.normal")


@register_op("random.orthogonal", "jax")
def _map_jax_random_orthogonal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_orthogonal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.orthogonal")


@register_op("random.pareto", "jax")
def _map_jax_random_pareto(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_pareto operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.pareto")


@register_op("random.permutation", "jax")
def _map_jax_random_permutation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_permutation operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.permutation")


@register_op("random.poisson", "jax")
def _map_jax_random_poisson(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_poisson operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.poisson")


@register_op("random.rademacher", "jax")
def _map_jax_random_rademacher(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_rademacher operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.rademacher")


@register_op("random.randint", "jax")
def _map_jax_random_randint(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_randint operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.randint")


@register_op("random.random_gamma_p", "jax")
def _map_jax_random_random_gamma_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_random_gamma_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.random_gamma_p")


@register_op("random.rayleigh", "jax")
def _map_jax_random_rayleigh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_rayleigh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.rayleigh")


@register_op("random.split", "jax")
def _map_jax_random_split(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_split operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.split")


@register_op("random.t", "jax")
def _map_jax_random_t(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_t operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.t")


@register_op("random.triangular", "jax")
def _map_jax_random_triangular(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_triangular operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.triangular")


@register_op("random.truncated_normal", "jax")
def _map_jax_random_truncated_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_truncated_normal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.truncated_normal")


@register_op("random.uniform", "jax")
def _map_jax_random_uniform(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_uniform operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.uniform")


@register_op("random.wald", "jax")
def _map_jax_random_wald(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_random_wald operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.wald")


@register_op("random.weibull_min", "jax")
def _map_jax_random_weibull_min(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_weibull_min operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.weibull_min")


@register_op("random.wrap_key_data", "jax")
def _map_jax_random_wrap_key_data(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_random_wrap_key_data operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="random.wrap_key_data")


@register_op("stages.Compiled", "jax")
def _map_jax_stages_Compiled(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_stages_Compiled operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="stages.Compiled")


@register_op("stages.CompilerOptions", "jax")
def _map_jax_stages_CompilerOptions(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_stages_CompilerOptions operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="stages.CompilerOptions")


@register_op("stages.Lowered", "jax")
def _map_jax_stages_Lowered(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_stages_Lowered operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="stages.Lowered")


@register_op("stages.Wrapped", "jax")
def _map_jax_stages_Wrapped(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_stages_Wrapped operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="stages.Wrapped")


@register_op("stages.ArgInfo", "jax")
def _map_jax_stages_ArgInfo(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_stages_ArgInfo operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="stages.ArgInfo")


@register_op("stages.Traced", "jax")
def _map_jax_stages_Traced(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_stages_Traced operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="stages.Traced")


@register_op("monitoring.clear_event_listeners", "jax")
def _map_jax_monitoring_clear_event_listeners(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_clear_event_listeners operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="monitoring.clear_event_listeners"
    )


@register_op("monitoring.record_event_duration_secs", "jax")
def _map_jax_monitoring_record_event_duration_secs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_record_event_duration_secs operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="monitoring.record_event_duration_secs",
    )


@register_op("monitoring.record_event_time_span", "jax")
def _map_jax_monitoring_record_event_time_span(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_record_event_time_span operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="monitoring.record_event_time_span"
    )


@register_op("monitoring.record_event", "jax")
def _map_jax_monitoring_record_event(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_record_event operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="monitoring.record_event")


@register_op("monitoring.record_scalar", "jax")
def _map_jax_monitoring_record_scalar(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_record_scalar operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="monitoring.record_scalar")


@register_op("monitoring.register_event_duration_secs_listener", "jax")
def _map_jax_monitoring_register_event_duration_secs_listener(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_register_event_duration_secs_listener operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="monitoring.register_event_duration_secs_listener",
    )


@register_op("monitoring.register_event_listener", "jax")
def _map_jax_monitoring_register_event_listener(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_register_event_listener operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="monitoring.register_event_listener",
    )


@register_op("monitoring.register_event_time_span_listener", "jax")
def _map_jax_monitoring_register_event_time_span_listener(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_register_event_time_span_listener operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="monitoring.register_event_time_span_listener",
    )


@register_op("monitoring.register_scalar_listener", "jax")
def _map_jax_monitoring_register_scalar_listener(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_register_scalar_listener operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="monitoring.register_scalar_listener",
    )


@register_op("monitoring.unregister_event_duration_listener", "jax")
def _map_jax_monitoring_unregister_event_duration_listener(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_unregister_event_duration_listener operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="monitoring.unregister_event_duration_listener",
    )


@register_op("monitoring.unregister_event_listener", "jax")
def _map_jax_monitoring_unregister_event_listener(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_unregister_event_listener operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="monitoring.unregister_event_listener",
    )


@register_op("monitoring.unregister_event_time_span_listener", "jax")
def _map_jax_monitoring_unregister_event_time_span_listener(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_unregister_event_time_span_listener operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="monitoring.unregister_event_time_span_listener",
    )


@register_op("monitoring.unregister_scalar_listener", "jax")
def _map_jax_monitoring_unregister_scalar_listener(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_monitoring_unregister_scalar_listener operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="monitoring.unregister_scalar_listener",
    )


@register_op("debug.callback", "jax")
def _map_jax_debug_callback(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_debug_callback operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="debug.callback")


@register_op("debug.print", "jax")
def _map_jax_debug_print(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_debug_print operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="debug.print")


@register_op("debug.log", "jax")
def _map_jax_debug_log(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_debug_log operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="debug.log")


@register_op("debug.DebugEffect", "jax")
def _map_jax_debug_DebugEffect(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_debug_DebugEffect operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="debug.DebugEffect")


@register_op("debug.OrderedDebugEffect", "jax")
def _map_jax_debug_OrderedDebugEffect(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_debug_OrderedDebugEffect operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="debug.OrderedDebugEffect")


@register_op("debug.visualize_array_sharding", "jax")
def _map_jax_debug_visualize_array_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_debug_visualize_array_sharding operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="debug.visualize_array_sharding"
    )


@register_op("debug.inspect_array_sharding", "jax")
def _map_jax_debug_inspect_array_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_debug_inspect_array_sharding operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="debug.inspect_array_sharding"
    )


@register_op("debug.visualize_sharding", "jax")
def _map_jax_debug_visualize_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_debug_visualize_sharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="debug.visualize_sharding")


@register_op("debug.breakpoint", "jax")
def _map_jax_debug_breakpoint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_debug_breakpoint operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="debug.breakpoint")


@register_op("ad_checkpoint.checkpoint_policies", "jax")
def _map_jax_ad_checkpoint_checkpoint_policies(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ad_checkpoint_checkpoint_policies operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="ad_checkpoint.checkpoint_policies"
    )


@register_op("ad_checkpoint.checkpoint_name", "jax")
def _map_jax_ad_checkpoint_checkpoint_name(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ad_checkpoint_checkpoint_name operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="ad_checkpoint.checkpoint_name"
    )


@register_op("ad_checkpoint.print_saved_residuals", "jax")
def _map_jax_ad_checkpoint_print_saved_residuals(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ad_checkpoint_print_saved_residuals operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="ad_checkpoint.print_saved_residuals",
    )


@register_op("ad_checkpoint.Recompute", "jax")
def _map_jax_ad_checkpoint_Recompute(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ad_checkpoint_Recompute operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ad_checkpoint.Recompute")


@register_op("ad_checkpoint.Saveable", "jax")
def _map_jax_ad_checkpoint_Saveable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ad_checkpoint_Saveable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ad_checkpoint.Saveable")


@register_op("ad_checkpoint.Offloadable", "jax")
def _map_jax_ad_checkpoint_Offloadable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ad_checkpoint_Offloadable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="ad_checkpoint.Offloadable"
    )


@register_op("ad_checkpoint.checkpoint", "jax")
def _map_jax_ad_checkpoint_checkpoint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ad_checkpoint_checkpoint operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ad_checkpoint.checkpoint")


@register_op("ad_checkpoint.remat", "jax")
def _map_jax_ad_checkpoint_remat(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_ad_checkpoint_remat operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ad_checkpoint.remat")


@register_op("flatten_util.ravel_pytree", "jax")
def _map_jax_flatten_util_ravel_pytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_flatten_util_ravel_pytree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="flatten_util.ravel_pytree"
    )


@register_op("collect_profile.jax_profiler", "jax")
def _map_jax_collect_profile_jax_profiler(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_collect_profile_jax_profiler operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="collect_profile.jax_profiler"
    )


@register_op("collect_profile.DEFAULT_NUM_TRACING_ATTEMPTS", "jax")
def _map_jax_collect_profile_DEFAULT_NUM_TRACING_ATTEMPTS(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_collect_profile_DEFAULT_NUM_TRACING_ATTEMPTS operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="collect_profile.DEFAULT_NUM_TRACING_ATTEMPTS",
    )


@register_op("collect_profile.parser", "jax")
def _map_jax_collect_profile_parser(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_collect_profile_parser operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="collect_profile.parser")


@register_op("collect_profile.collect_profile", "jax")
def _map_jax_collect_profile_collect_profile(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_collect_profile_collect_profile operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="collect_profile.collect_profile"
    )


@register_op("collect_profile.main", "jax")
def _map_jax_collect_profile_main(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_collect_profile_main operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="collect_profile.main")


@register_op("tree.all", "jax")
def _map_jax_tree_all(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_tree_all operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.all")


@register_op("tree.broadcast", "jax")
def _map_jax_tree_broadcast(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_tree_broadcast operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.broadcast")


@register_op("tree.flatten_with_path", "jax")
def _map_jax_tree_flatten_with_path(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_flatten_with_path operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.flatten_with_path")


@register_op("tree.flatten", "jax")
def _map_jax_tree_flatten(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_tree_flatten operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.flatten")


@register_op("tree.leaves_with_path", "jax")
def _map_jax_tree_leaves_with_path(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_leaves_with_path operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.leaves_with_path")


@register_op("tree.leaves", "jax")
def _map_jax_tree_leaves(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_tree_leaves operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.leaves")


@register_op("tree.map_with_path", "jax")
def _map_jax_tree_map_with_path(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_map_with_path operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.map_with_path")


@register_op("tree.map", "jax")
def _map_jax_tree_map(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_tree_map operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.map")


@register_op("tree.reduce", "jax")
def _map_jax_tree_reduce(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_tree_reduce operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.reduce")


@register_op("tree.reduce_associative", "jax")
def _map_jax_tree_reduce_associative(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_tree_reduce_associative operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.reduce_associative")


@register_op("tree.static", "jax")
def _map_jax_tree_static(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_tree_static operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.static")


@register_op("tree.structure", "jax")
def _map_jax_tree_structure(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_tree_structure operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.structure")


@register_op("tree.transpose", "jax")
def _map_jax_tree_transpose(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_tree_transpose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.transpose")


@register_op("tree.unflatten", "jax")
def _map_jax_tree_unflatten(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_tree_unflatten operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="tree.unflatten")


@register_op("sharding.Sharding", "jax")
def _map_jax_sharding_Sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_Sharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.Sharding")


@register_op("sharding.NamedSharding", "jax")
def _map_jax_sharding_NamedSharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_NamedSharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.NamedSharding")


@register_op("sharding.SingleDeviceSharding", "jax")
def _map_jax_sharding_SingleDeviceSharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_SingleDeviceSharding operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.SingleDeviceSharding"
    )


@register_op("sharding.set_mesh", "jax")
def _map_jax_sharding_set_mesh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_set_mesh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.set_mesh")


@register_op("sharding.get_mesh", "jax")
def _map_jax_sharding_get_mesh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_get_mesh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.get_mesh")


@register_op("sharding.PartitionSpec", "jax")
def _map_jax_sharding_PartitionSpec(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_PartitionSpec operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.PartitionSpec")


@register_op("sharding.Mesh", "jax")
def _map_jax_sharding_Mesh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_sharding_Mesh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.Mesh")


@register_op("sharding.AbstractDevice", "jax")
def _map_jax_sharding_AbstractDevice(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_AbstractDevice operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.AbstractDevice")


@register_op("sharding.AbstractMesh", "jax")
def _map_jax_sharding_AbstractMesh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_AbstractMesh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.AbstractMesh")


@register_op("sharding.AxisType", "jax")
def _map_jax_sharding_AxisType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_AxisType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.AxisType")


@register_op("sharding.get_abstract_mesh", "jax")
def _map_jax_sharding_get_abstract_mesh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_get_abstract_mesh operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.get_abstract_mesh"
    )


@register_op("sharding.use_abstract_mesh", "jax")
def _map_jax_sharding_use_abstract_mesh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_use_abstract_mesh operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.use_abstract_mesh"
    )


@register_op("sharding.reshard", "jax")
def _map_jax_sharding_reshard(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_reshard operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.reshard")


@register_op("sharding.auto_axes", "jax")
def _map_jax_sharding_auto_axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_auto_axes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.auto_axes")


@register_op("sharding.explicit_axes", "jax")
def _map_jax_sharding_explicit_axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_explicit_axes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.explicit_axes")


@register_op("sharding.PmapSharding", "jax")
def _map_jax_sharding_PmapSharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_sharding_PmapSharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="sharding.PmapSharding")


@register_op("api_util.shaped_abstractify", "jax")
def _map_jax_api_util_shaped_abstractify(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_api_util_shaped_abstractify operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="api_util.shaped_abstractify"
    )


@register_op("api_util.argnums_partial", "jax")
def _map_jax_api_util_argnums_partial(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_api_util_argnums_partial operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="api_util.argnums_partial")


@register_op("api_util.debug_info", "jax")
def _map_jax_api_util_debug_info(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_api_util_debug_info operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="api_util.debug_info")


@register_op("api_util.donation_vector", "jax")
def _map_jax_api_util_donation_vector(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_api_util_donation_vector operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="api_util.donation_vector")


@register_op("api_util.flatten_axes", "jax")
def _map_jax_api_util_flatten_axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_api_util_flatten_axes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="api_util.flatten_axes")


@register_op("api_util.flatten_fun", "jax")
def _map_jax_api_util_flatten_fun(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_api_util_flatten_fun operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="api_util.flatten_fun")


@register_op("api_util.flatten_fun_nokwargs", "jax")
def _map_jax_api_util_flatten_fun_nokwargs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_api_util_flatten_fun_nokwargs operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="api_util.flatten_fun_nokwargs"
    )


@register_op("api_util.rebase_donate_argnums", "jax")
def _map_jax_api_util_rebase_donate_argnums(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_api_util_rebase_donate_argnums operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="api_util.rebase_donate_argnums"
    )


@register_op("api_util.safe_map", "jax")
def _map_jax_api_util_safe_map(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_api_util_safe_map operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="api_util.safe_map")


@register_op("export.DisabledSafetyCheck", "jax")
def _map_jax_export_DisabledSafetyCheck(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_DisabledSafetyCheck operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="export.DisabledSafetyCheck"
    )


@register_op("export.Exported", "jax")
def _map_jax_export_Exported(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_export_Exported operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="export.Exported")


@register_op("export.export", "jax")
def _map_jax_export_export(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_export_export operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="export.export")


@register_op("export.deserialize", "jax")
def _map_jax_export_deserialize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_deserialize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="export.deserialize")


@register_op("export.register_pytree_node_serialization", "jax")
def _map_jax_export_register_pytree_node_serialization(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_register_pytree_node_serialization operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="export.register_pytree_node_serialization",
    )


@register_op("export.register_namedtuple_serialization", "jax")
def _map_jax_export_register_namedtuple_serialization(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_register_namedtuple_serialization operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="export.register_namedtuple_serialization",
    )


@register_op("export.maximum_supported_calling_convention_version", "jax")
def _map_jax_export_maximum_supported_calling_convention_version(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_maximum_supported_calling_convention_version operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="export.maximum_supported_calling_convention_version",
    )


@register_op("export.minimum_supported_calling_convention_version", "jax")
def _map_jax_export_minimum_supported_calling_convention_version(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_minimum_supported_calling_convention_version operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="export.minimum_supported_calling_convention_version",
    )


@register_op("export.default_export_platform", "jax")
def _map_jax_export_default_export_platform(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_default_export_platform operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="export.default_export_platform"
    )


@register_op("export.shape_poly_decision", "jax")
def _map_jax_export_shape_poly_decision(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_shape_poly_decision operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="export.shape_poly_decision"
    )


@register_op("export.SymbolicScope", "jax")
def _map_jax_export_SymbolicScope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_SymbolicScope operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="export.SymbolicScope")


@register_op("export.is_symbolic_dim", "jax")
def _map_jax_export_is_symbolic_dim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_is_symbolic_dim operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="export.is_symbolic_dim")


@register_op("export.symbolic_shape", "jax")
def _map_jax_export_symbolic_shape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_symbolic_shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="export.symbolic_shape")


@register_op("export.symbolic_args_specs", "jax")
def _map_jax_export_symbolic_args_specs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_export_symbolic_args_specs operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="export.symbolic_args_specs"
    )


@register_op("distributed.initialize", "jax")
def _map_jax_distributed_initialize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_distributed_initialize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="distributed.initialize")


@register_op("distributed.is_initialized", "jax")
def _map_jax_distributed_is_initialized(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_distributed_is_initialized operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="distributed.is_initialized"
    )


@register_op("distributed.shutdown", "jax")
def _map_jax_distributed_shutdown(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_distributed_shutdown operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="distributed.shutdown")


@register_op("errors.JAXTypeError", "jax")
def _map_jax_errors_JAXTypeError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_errors_JAXTypeError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.JAXTypeError")


@register_op("errors.JAXIndexError", "jax")
def _map_jax_errors_JAXIndexError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_errors_JAXIndexError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.JAXIndexError")


@register_op("errors.ConcretizationTypeError", "jax")
def _map_jax_errors_ConcretizationTypeError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_errors_ConcretizationTypeError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.ConcretizationTypeError"
    )


@register_op("errors.NonConcreteBooleanIndexError", "jax")
def _map_jax_errors_NonConcreteBooleanIndexError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_errors_NonConcreteBooleanIndexError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.NonConcreteBooleanIndexError",
    )


@register_op("errors.TracerArrayConversionError", "jax")
def _map_jax_errors_TracerArrayConversionError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_errors_TracerArrayConversionError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.TracerArrayConversionError"
    )


@register_op("errors.TracerBoolConversionError", "jax")
def _map_jax_errors_TracerBoolConversionError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_errors_TracerBoolConversionError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.TracerBoolConversionError"
    )


@register_op("errors.TracerIntegerConversionError", "jax")
def _map_jax_errors_TracerIntegerConversionError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_errors_TracerIntegerConversionError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.TracerIntegerConversionError",
    )


@register_op("errors.UnexpectedTracerError", "jax")
def _map_jax_errors_UnexpectedTracerError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_errors_UnexpectedTracerError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.UnexpectedTracerError"
    )


@register_op("errors.KeyReuseError", "jax")
def _map_jax_errors_KeyReuseError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_errors_KeyReuseError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.KeyReuseError")


@register_op("errors.JaxRuntimeError", "jax")
def _map_jax_errors_JaxRuntimeError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_errors_JaxRuntimeError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.JaxRuntimeError")


@register_op("profiler.ProfileData", "jax")
def _map_jax_profiler_ProfileData(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_ProfileData operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.ProfileData")


@register_op("profiler.ProfileEvent", "jax")
def _map_jax_profiler_ProfileEvent(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_ProfileEvent operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.ProfileEvent")


@register_op("profiler.ProfileOptions", "jax")
def _map_jax_profiler_ProfileOptions(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_ProfileOptions operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.ProfileOptions")


@register_op("profiler.ProfilePlane", "jax")
def _map_jax_profiler_ProfilePlane(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_ProfilePlane operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.ProfilePlane")


@register_op("profiler.StepTraceAnnotation", "jax")
def _map_jax_profiler_StepTraceAnnotation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_StepTraceAnnotation operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.StepTraceAnnotation"
    )


@register_op("profiler.TraceAnnotation", "jax")
def _map_jax_profiler_TraceAnnotation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_TraceAnnotation operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.TraceAnnotation")


@register_op("profiler.annotate_function", "jax")
def _map_jax_profiler_annotate_function(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_annotate_function operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.annotate_function"
    )


@register_op("profiler.device_memory_profile", "jax")
def _map_jax_profiler_device_memory_profile(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_device_memory_profile operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.device_memory_profile"
    )


@register_op("profiler.save_device_memory_profile", "jax")
def _map_jax_profiler_save_device_memory_profile(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_save_device_memory_profile operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="profiler.save_device_memory_profile",
    )


@register_op("profiler.start_server", "jax")
def _map_jax_profiler_start_server(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_start_server operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.start_server")


@register_op("profiler.start_trace", "jax")
def _map_jax_profiler_start_trace(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_start_trace operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.start_trace")


@register_op("profiler.stop_server", "jax")
def _map_jax_profiler_stop_server(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_stop_server operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.stop_server")


@register_op("profiler.stop_trace", "jax")
def _map_jax_profiler_stop_trace(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_profiler_stop_trace operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.stop_trace")


@register_op("profiler.trace", "jax")
def _map_jax_profiler_trace(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_profiler_trace operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="profiler.trace")


@register_op("nn.AxisName", "jax")
def _map_jax_nn_AxisName(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_AxisName operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.AxisName")


@register_op("nn.BlockScaleConfig", "jax")
def _map_jax_nn_BlockScaleConfig(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_BlockScaleConfig operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.BlockScaleConfig")


@register_op("nn.DotDimensionNumbers", "jax")
def _map_jax_nn_DotDimensionNumbers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_DotDimensionNumbers operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.DotDimensionNumbers")


@register_op("nn.Array", "jax")
def _map_jax_nn_Array(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.Array")


@register_op("nn.ArrayLike", "jax")
def _map_jax_nn_ArrayLike(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_ArrayLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.ArrayLike")


@register_op("nn.DTypeLike", "jax")
def _map_jax_nn_DTypeLike(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_DTypeLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.DTypeLike")


@register_op("nn.Axis", "jax")
def _map_jax_nn_Axis(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_Axis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.Axis")


@register_op("nn.celu", "jax")
def _map_jax_nn_celu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_celu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.celu")


@register_op("nn.elu", "jax")
def _map_jax_nn_elu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_elu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.elu")


@register_op("nn.gelu", "jax")
def _map_jax_nn_gelu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_gelu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.gelu")


@register_op("nn.get_scaled_dot_general_config", "jax")
def _map_jax_nn_get_scaled_dot_general_config(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_get_scaled_dot_general_config operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.get_scaled_dot_general_config"
    )


@register_op("nn.glu", "jax")
def _map_jax_nn_glu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_glu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.glu")


@register_op("nn.hard_sigmoid", "jax")
def _map_jax_nn_hard_sigmoid(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_hard_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.hard_sigmoid")


@register_op("nn.hard_silu", "jax")
def _map_jax_nn_hard_silu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_hard_silu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.hard_silu")


@register_op("nn.hard_swish", "jax")
def _map_jax_nn_hard_swish(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_hard_swish operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.hard_swish")


@register_op("nn.hard_tanh", "jax")
def _map_jax_nn_hard_tanh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_hard_tanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.hard_tanh")


@register_op("nn.identity", "jax")
def _map_jax_nn_identity(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_identity operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.identity")


@register_op("nn.leaky_relu", "jax")
def _map_jax_nn_leaky_relu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_leaky_relu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.leaky_relu")


@register_op("nn.log_sigmoid", "jax")
def _map_jax_nn_log_sigmoid(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_log_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.log_sigmoid")


@register_op("nn.log_softmax", "jax")
def _map_jax_nn_log_softmax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_log_softmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.log_softmax")


@register_op("nn.logmeanexp", "jax")
def _map_jax_nn_logmeanexp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_logmeanexp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.logmeanexp")


@register_op("nn.mish", "jax")
def _map_jax_nn_mish(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_mish operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.mish")


@register_op("nn.one_hot", "jax")
def _map_jax_nn_one_hot(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_one_hot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.one_hot")


@register_op("nn.relu", "jax")
def _map_jax_nn_relu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_relu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.relu")


@register_op("nn.relu6", "jax")
def _map_jax_nn_relu6(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_relu6 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.relu6")


@register_op("nn.scaled_dot_general", "jax")
def _map_jax_nn_scaled_dot_general(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_scaled_dot_general operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.scaled_dot_general")


@register_op("nn.scaled_matmul", "jax")
def _map_jax_nn_scaled_matmul(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_scaled_matmul operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.scaled_matmul")


@register_op("nn.selu", "jax")
def _map_jax_nn_selu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_selu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.selu")


@register_op("nn.sigmoid", "jax")
def _map_jax_nn_sigmoid(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.sigmoid")


@register_op("nn.silu", "jax")
def _map_jax_nn_silu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_silu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.silu")


@register_op("nn.soft_sign", "jax")
def _map_jax_nn_soft_sign(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_soft_sign operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.soft_sign")


@register_op("nn.softmax", "jax")
def _map_jax_nn_softmax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_softmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.softmax")


@register_op("nn.softplus", "jax")
def _map_jax_nn_softplus(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_softplus operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.softplus")


@register_op("nn.sparse_plus", "jax")
def _map_jax_nn_sparse_plus(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_sparse_plus operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.sparse_plus")


@register_op("nn.sparse_sigmoid", "jax")
def _map_jax_nn_sparse_sigmoid(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_sparse_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.sparse_sigmoid")


@register_op("nn.squareplus", "jax")
def _map_jax_nn_squareplus(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_squareplus operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.squareplus")


@register_op("nn.standardize", "jax")
def _map_jax_nn_standardize(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_standardize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.standardize")


@register_op("nn.swish", "jax")
def _map_jax_nn_swish(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_swish operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.swish")


@register_op("nn.tanh", "jax")
def _map_jax_nn_tanh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_tanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.tanh")


@register_op("nn.log1mexp", "jax")
def _map_jax_nn_log1mexp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_nn_log1mexp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.log1mexp")


@register_op("nn.initializers.constant", "jax")
def _map_jax_nn_initializers_constant(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_constant operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.constant")


@register_op("nn.initializers.Initializer", "jax")
def _map_jax_nn_initializers_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_Initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.Initializer"
    )


@register_op("nn.initializers.delta_orthogonal", "jax")
def _map_jax_nn_initializers_delta_orthogonal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_delta_orthogonal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.delta_orthogonal"
    )


@register_op("nn.initializers.glorot_normal", "jax")
def _map_jax_nn_initializers_glorot_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_glorot_normal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.glorot_normal"
    )


@register_op("nn.initializers.glorot_uniform", "jax")
def _map_jax_nn_initializers_glorot_uniform(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_glorot_uniform operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.glorot_uniform"
    )


@register_op("nn.initializers.he_normal", "jax")
def _map_jax_nn_initializers_he_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_he_normal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.he_normal"
    )


@register_op("nn.initializers.he_uniform", "jax")
def _map_jax_nn_initializers_he_uniform(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_he_uniform operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.he_uniform"
    )


@register_op("nn.initializers.kaiming_normal", "jax")
def _map_jax_nn_initializers_kaiming_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_kaiming_normal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.kaiming_normal"
    )


@register_op("nn.initializers.kaiming_uniform", "jax")
def _map_jax_nn_initializers_kaiming_uniform(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_kaiming_uniform operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.kaiming_uniform"
    )


@register_op("nn.initializers.lecun_normal", "jax")
def _map_jax_nn_initializers_lecun_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_lecun_normal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.lecun_normal"
    )


@register_op("nn.initializers.lecun_uniform", "jax")
def _map_jax_nn_initializers_lecun_uniform(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_lecun_uniform operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.lecun_uniform"
    )


@register_op("nn.initializers.normal", "jax")
def _map_jax_nn_initializers_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_normal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.normal")


@register_op("nn.initializers.ones", "jax")
def _map_jax_nn_initializers_ones(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_ones operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.ones")


@register_op("nn.initializers.orthogonal", "jax")
def _map_jax_nn_initializers_orthogonal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_orthogonal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.orthogonal"
    )


@register_op("nn.initializers.truncated_normal", "jax")
def _map_jax_nn_initializers_truncated_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_truncated_normal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.truncated_normal"
    )


@register_op("nn.initializers.uniform", "jax")
def _map_jax_nn_initializers_uniform(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_uniform operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.uniform")


@register_op("nn.initializers.variance_scaling", "jax")
def _map_jax_nn_initializers_variance_scaling(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_variance_scaling operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.variance_scaling"
    )


@register_op("nn.initializers.xavier_normal", "jax")
def _map_jax_nn_initializers_xavier_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_xavier_normal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.xavier_normal"
    )


@register_op("nn.initializers.xavier_uniform", "jax")
def _map_jax_nn_initializers_xavier_uniform(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_xavier_uniform operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.xavier_uniform"
    )


@register_op("nn.initializers.zeros", "jax")
def _map_jax_nn_initializers_zeros(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_nn_initializers_zeros operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nn.initializers.zeros")


@register_op("example_libraries.stax.lax", "jax")
def _map_jax_example_libraries_stax_lax(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_lax operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.lax"
    )


@register_op("example_libraries.stax.random", "jax")
def _map_jax_example_libraries_stax_random(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_random operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.random"
    )


@register_op("example_libraries.stax.jnp", "jax")
def _map_jax_example_libraries_stax_jnp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_jnp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.jnp"
    )


@register_op("example_libraries.stax.relu", "jax")
def _map_jax_example_libraries_stax_relu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_relu operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.relu"
    )


@register_op("example_libraries.stax.log_softmax", "jax")
def _map_jax_example_libraries_stax_log_softmax(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_log_softmax operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.stax.log_softmax",
    )


@register_op("example_libraries.stax.softmax", "jax")
def _map_jax_example_libraries_stax_softmax(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_softmax operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.softmax"
    )


@register_op("example_libraries.stax.softplus", "jax")
def _map_jax_example_libraries_stax_softplus(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_softplus operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.softplus"
    )


@register_op("example_libraries.stax.sigmoid", "jax")
def _map_jax_example_libraries_stax_sigmoid(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_sigmoid operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.sigmoid"
    )


@register_op("example_libraries.stax.elu", "jax")
def _map_jax_example_libraries_stax_elu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_elu operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.elu"
    )


@register_op("example_libraries.stax.leaky_relu", "jax")
def _map_jax_example_libraries_stax_leaky_relu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_leaky_relu operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.leaky_relu"
    )


@register_op("example_libraries.stax.selu", "jax")
def _map_jax_example_libraries_stax_selu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_selu operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.selu"
    )


@register_op("example_libraries.stax.gelu", "jax")
def _map_jax_example_libraries_stax_gelu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_gelu operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.gelu"
    )


@register_op("example_libraries.stax.standardize", "jax")
def _map_jax_example_libraries_stax_standardize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_standardize operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.stax.standardize",
    )


@register_op("example_libraries.stax.glorot_normal", "jax")
def _map_jax_example_libraries_stax_glorot_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_glorot_normal operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.stax.glorot_normal",
    )


@register_op("example_libraries.stax.normal", "jax")
def _map_jax_example_libraries_stax_normal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_normal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.normal"
    )


@register_op("example_libraries.stax.ones", "jax")
def _map_jax_example_libraries_stax_ones(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_ones operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.ones"
    )


@register_op("example_libraries.stax.zeros", "jax")
def _map_jax_example_libraries_stax_zeros(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_zeros operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.zeros"
    )


@register_op("example_libraries.stax.glorot", "jax")
def _map_jax_example_libraries_stax_glorot(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_glorot operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.glorot"
    )


@register_op("example_libraries.stax.randn", "jax")
def _map_jax_example_libraries_stax_randn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_randn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.randn"
    )


@register_op("example_libraries.stax.logsoftmax", "jax")
def _map_jax_example_libraries_stax_logsoftmax(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_logsoftmax operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.logsoftmax"
    )


@register_op("example_libraries.stax.Dense", "jax")
def _map_jax_example_libraries_stax_Dense(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Dense operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Dense"
    )


@register_op("example_libraries.stax.GeneralConv", "jax")
def _map_jax_example_libraries_stax_GeneralConv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_GeneralConv operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.stax.GeneralConv",
    )


@register_op("example_libraries.stax.Conv", "jax")
def _map_jax_example_libraries_stax_Conv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Conv operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Conv"
    )


@register_op("example_libraries.stax.GeneralConvTranspose", "jax")
def _map_jax_example_libraries_stax_GeneralConvTranspose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_GeneralConvTranspose operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.stax.GeneralConvTranspose",
    )


@register_op("example_libraries.stax.Conv1DTranspose", "jax")
def _map_jax_example_libraries_stax_Conv1DTranspose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Conv1DTranspose operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.stax.Conv1DTranspose",
    )


@register_op("example_libraries.stax.ConvTranspose", "jax")
def _map_jax_example_libraries_stax_ConvTranspose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_ConvTranspose operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.stax.ConvTranspose",
    )


@register_op("example_libraries.stax.BatchNorm", "jax")
def _map_jax_example_libraries_stax_BatchNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_BatchNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.BatchNorm"
    )


@register_op("example_libraries.stax.elementwise", "jax")
def _map_jax_example_libraries_stax_elementwise(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_elementwise operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.stax.elementwise",
    )


@register_op("example_libraries.stax.Tanh", "jax")
def _map_jax_example_libraries_stax_Tanh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Tanh operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Tanh"
    )


@register_op("example_libraries.stax.Relu", "jax")
def _map_jax_example_libraries_stax_Relu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Relu operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Relu"
    )


@register_op("example_libraries.stax.Exp", "jax")
def _map_jax_example_libraries_stax_Exp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Exp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Exp"
    )


@register_op("example_libraries.stax.LogSoftmax", "jax")
def _map_jax_example_libraries_stax_LogSoftmax(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_LogSoftmax operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.LogSoftmax"
    )


@register_op("example_libraries.stax.Softmax", "jax")
def _map_jax_example_libraries_stax_Softmax(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Softmax operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Softmax"
    )


@register_op("example_libraries.stax.Softplus", "jax")
def _map_jax_example_libraries_stax_Softplus(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Softplus operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Softplus"
    )


@register_op("example_libraries.stax.Sigmoid", "jax")
def _map_jax_example_libraries_stax_Sigmoid(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Sigmoid operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Sigmoid"
    )


@register_op("example_libraries.stax.Elu", "jax")
def _map_jax_example_libraries_stax_Elu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Elu operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Elu"
    )


@register_op("example_libraries.stax.LeakyRelu", "jax")
def _map_jax_example_libraries_stax_LeakyRelu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_LeakyRelu operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.LeakyRelu"
    )


@register_op("example_libraries.stax.Selu", "jax")
def _map_jax_example_libraries_stax_Selu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Selu operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Selu"
    )


@register_op("example_libraries.stax.Gelu", "jax")
def _map_jax_example_libraries_stax_Gelu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Gelu operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Gelu"
    )


@register_op("example_libraries.stax.MaxPool", "jax")
def _map_jax_example_libraries_stax_MaxPool(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_MaxPool operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.MaxPool"
    )


@register_op("example_libraries.stax.SumPool", "jax")
def _map_jax_example_libraries_stax_SumPool(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_SumPool operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.SumPool"
    )


@register_op("example_libraries.stax.AvgPool", "jax")
def _map_jax_example_libraries_stax_AvgPool(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_AvgPool operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.AvgPool"
    )


@register_op("example_libraries.stax.Flatten", "jax")
def _map_jax_example_libraries_stax_Flatten(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Flatten operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Flatten"
    )


@register_op("example_libraries.stax.Identity", "jax")
def _map_jax_example_libraries_stax_Identity(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Identity operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Identity"
    )


@register_op("example_libraries.stax.FanOut", "jax")
def _map_jax_example_libraries_stax_FanOut(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_FanOut operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.FanOut"
    )


@register_op("example_libraries.stax.FanInSum", "jax")
def _map_jax_example_libraries_stax_FanInSum(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_FanInSum operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.FanInSum"
    )


@register_op("example_libraries.stax.FanInConcat", "jax")
def _map_jax_example_libraries_stax_FanInConcat(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_FanInConcat operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.stax.FanInConcat",
    )


@register_op("example_libraries.stax.Dropout", "jax")
def _map_jax_example_libraries_stax_Dropout(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_Dropout operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.Dropout"
    )


@register_op("example_libraries.stax.serial", "jax")
def _map_jax_example_libraries_stax_serial(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_serial operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.serial"
    )


@register_op("example_libraries.stax.parallel", "jax")
def _map_jax_example_libraries_stax_parallel(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_parallel operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.stax.parallel"
    )


@register_op("example_libraries.stax.shape_dependent", "jax")
def _map_jax_example_libraries_stax_shape_dependent(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_stax_shape_dependent operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.stax.shape_dependent",
    )


@register_op("example_libraries.optimizers.jnp", "jax")
def _map_jax_example_libraries_optimizers_jnp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_jnp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.optimizers.jnp"
    )


@register_op("example_libraries.optimizers.safe_zip", "jax")
def _map_jax_example_libraries_optimizers_safe_zip(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_safe_zip operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.safe_zip",
    )


@register_op("example_libraries.optimizers.safe_map", "jax")
def _map_jax_example_libraries_optimizers_safe_map(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_safe_map operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.safe_map",
    )


@register_op("example_libraries.optimizers.unzip2", "jax")
def _map_jax_example_libraries_optimizers_unzip2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_unzip2 operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.unzip2",
    )


@register_op("example_libraries.optimizers.map", "jax")
def _map_jax_example_libraries_optimizers_map(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_map operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.optimizers.map"
    )


@register_op("example_libraries.optimizers.zip", "jax")
def _map_jax_example_libraries_optimizers_zip(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_zip operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.optimizers.zip"
    )


@register_op("example_libraries.optimizers.OptimizerState", "jax")
def _map_jax_example_libraries_optimizers_OptimizerState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_OptimizerState operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.OptimizerState",
    )


@register_op("example_libraries.optimizers.Array", "jax")
def _map_jax_example_libraries_optimizers_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_Array operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.Array",
    )


@register_op("example_libraries.optimizers.Params", "jax")
def _map_jax_example_libraries_optimizers_Params(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_Params operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.Params",
    )


@register_op("example_libraries.optimizers.State", "jax")
def _map_jax_example_libraries_optimizers_State(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_State operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.State",
    )


@register_op("example_libraries.optimizers.Updates", "jax")
def _map_jax_example_libraries_optimizers_Updates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_Updates operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.Updates",
    )


@register_op("example_libraries.optimizers.InitFn", "jax")
def _map_jax_example_libraries_optimizers_InitFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_InitFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.InitFn",
    )


@register_op("example_libraries.optimizers.Step", "jax")
def _map_jax_example_libraries_optimizers_Step(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_Step operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.optimizers.Step"
    )


@register_op("example_libraries.optimizers.UpdateFn", "jax")
def _map_jax_example_libraries_optimizers_UpdateFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_UpdateFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.UpdateFn",
    )


@register_op("example_libraries.optimizers.ParamsFn", "jax")
def _map_jax_example_libraries_optimizers_ParamsFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_ParamsFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.ParamsFn",
    )


@register_op("example_libraries.optimizers.Optimizer", "jax")
def _map_jax_example_libraries_optimizers_Optimizer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_Optimizer operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.Optimizer",
    )


@register_op("example_libraries.optimizers.Schedule", "jax")
def _map_jax_example_libraries_optimizers_Schedule(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_Schedule operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.Schedule",
    )


@register_op("example_libraries.optimizers.optimizer", "jax")
def _map_jax_example_libraries_optimizers_optimizer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_optimizer operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.optimizer",
    )


@register_op("example_libraries.optimizers.sgd", "jax")
def _map_jax_example_libraries_optimizers_sgd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_sgd operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.optimizers.sgd"
    )


@register_op("example_libraries.optimizers.momentum", "jax")
def _map_jax_example_libraries_optimizers_momentum(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_momentum operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.momentum",
    )


@register_op("example_libraries.optimizers.nesterov", "jax")
def _map_jax_example_libraries_optimizers_nesterov(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_nesterov operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.nesterov",
    )


@register_op("example_libraries.optimizers.adagrad", "jax")
def _map_jax_example_libraries_optimizers_adagrad(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_adagrad operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.adagrad",
    )


@register_op("example_libraries.optimizers.rmsprop", "jax")
def _map_jax_example_libraries_optimizers_rmsprop(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_rmsprop operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.rmsprop",
    )


@register_op("example_libraries.optimizers.rmsprop_momentum", "jax")
def _map_jax_example_libraries_optimizers_rmsprop_momentum(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_rmsprop_momentum operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.rmsprop_momentum",
    )


@register_op("example_libraries.optimizers.adam", "jax")
def _map_jax_example_libraries_optimizers_adam(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_adam operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.optimizers.adam"
    )


@register_op("example_libraries.optimizers.adamax", "jax")
def _map_jax_example_libraries_optimizers_adamax(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_adamax operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.adamax",
    )


@register_op("example_libraries.optimizers.sm3", "jax")
def _map_jax_example_libraries_optimizers_sm3(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_sm3 operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="example_libraries.optimizers.sm3"
    )


@register_op("example_libraries.optimizers.constant", "jax")
def _map_jax_example_libraries_optimizers_constant(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_constant operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.constant",
    )


@register_op("example_libraries.optimizers.exponential_decay", "jax")
def _map_jax_example_libraries_optimizers_exponential_decay(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_exponential_decay operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.exponential_decay",
    )


@register_op("example_libraries.optimizers.inverse_time_decay", "jax")
def _map_jax_example_libraries_optimizers_inverse_time_decay(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_inverse_time_decay operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.inverse_time_decay",
    )


@register_op("example_libraries.optimizers.polynomial_decay", "jax")
def _map_jax_example_libraries_optimizers_polynomial_decay(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_polynomial_decay operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.polynomial_decay",
    )


@register_op("example_libraries.optimizers.piecewise_constant", "jax")
def _map_jax_example_libraries_optimizers_piecewise_constant(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_piecewise_constant operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.piecewise_constant",
    )


@register_op("example_libraries.optimizers.make_schedule", "jax")
def _map_jax_example_libraries_optimizers_make_schedule(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_make_schedule operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.make_schedule",
    )


@register_op("example_libraries.optimizers.l2_norm", "jax")
def _map_jax_example_libraries_optimizers_l2_norm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_l2_norm operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.l2_norm",
    )


@register_op("example_libraries.optimizers.clip_grads", "jax")
def _map_jax_example_libraries_optimizers_clip_grads(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_clip_grads operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.clip_grads",
    )


@register_op("example_libraries.optimizers.JoinPoint", "jax")
def _map_jax_example_libraries_optimizers_JoinPoint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_JoinPoint operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.JoinPoint",
    )


@register_op("example_libraries.optimizers.unpack_optimizer_state", "jax")
def _map_jax_example_libraries_optimizers_unpack_optimizer_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_unpack_optimizer_state operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.unpack_optimizer_state",
    )


@register_op("example_libraries.optimizers.pack_optimizer_state", "jax")
def _map_jax_example_libraries_optimizers_pack_optimizer_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_example_libraries_optimizers_pack_optimizer_state operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="example_libraries.optimizers.pack_optimizer_state",
    )


@register_op("image.resize", "jax")
def _map_jax_image_resize(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_image_resize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="image.resize")


@register_op("image.ResizeMethod", "jax")
def _map_jax_image_ResizeMethod(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_image_ResizeMethod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="image.ResizeMethod")


@register_op("image.scale_and_translate", "jax")
def _map_jax_image_scale_and_translate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_image_scale_and_translate operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="image.scale_and_translate"
    )


@register_op("interpreters.traceback_util", "jax")
def _map_jax_interpreters_traceback_util(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_traceback_util operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.traceback_util"
    )


@register_op("interpreters.xla.canonicalize_dtype_handlers", "jax")
def _map_jax_interpreters_xla_canonicalize_dtype_handlers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_xla_canonicalize_dtype_handlers operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.xla.canonicalize_dtype_handlers",
    )


@register_op("interpreters.xla.apply_primitive", "jax")
def _map_jax_interpreters_xla_apply_primitive(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_xla_apply_primitive operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.xla.apply_primitive"
    )


@register_op("interpreters.xla.Backend", "jax")
def _map_jax_interpreters_xla_Backend(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_xla_Backend operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.xla.Backend")


@register_op("interpreters.pxla.Index", "jax")
def _map_jax_interpreters_pxla_Index(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_Index operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.Index")


@register_op("interpreters.pxla.MapTracer", "jax")
def _map_jax_interpreters_pxla_MapTracer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_MapTracer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.MapTracer"
    )


@register_op("interpreters.pxla.MeshAxisName", "jax")
def _map_jax_interpreters_pxla_MeshAxisName(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_MeshAxisName operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.MeshAxisName"
    )


@register_op("interpreters.pxla.MeshComputation", "jax")
def _map_jax_interpreters_pxla_MeshComputation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_MeshComputation operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.MeshComputation"
    )


@register_op("interpreters.pxla.MeshExecutable", "jax")
def _map_jax_interpreters_pxla_MeshExecutable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_MeshExecutable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.MeshExecutable"
    )


@register_op("interpreters.pxla.PmapExecutable", "jax")
def _map_jax_interpreters_pxla_PmapExecutable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_PmapExecutable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.PmapExecutable"
    )


@register_op("interpreters.pxla.global_aval_to_result_handler", "jax")
def _map_jax_interpreters_pxla_global_aval_to_result_handler(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_global_aval_to_result_handler operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.pxla.global_aval_to_result_handler",
    )


@register_op("interpreters.pxla.global_avals_to_results_handler", "jax")
def _map_jax_interpreters_pxla_global_avals_to_results_handler(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_global_avals_to_results_handler operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.pxla.global_avals_to_results_handler",
    )


@register_op("interpreters.pxla.global_result_handlers", "jax")
def _map_jax_interpreters_pxla_global_result_handlers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_global_result_handlers operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.pxla.global_result_handlers",
    )


@register_op("interpreters.pxla.parallel_callable", "jax")
def _map_jax_interpreters_pxla_parallel_callable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_parallel_callable operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.pxla.parallel_callable",
    )


@register_op("interpreters.pxla.shard_args", "jax")
def _map_jax_interpreters_pxla_shard_args(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_shard_args operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.shard_args"
    )


@register_op("interpreters.pxla.xla_pmap_p", "jax")
def _map_jax_interpreters_pxla_xla_pmap_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_xla_pmap_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.xla_pmap_p"
    )


@register_op("interpreters.pxla.thread_resources", "jax")
def _map_jax_interpreters_pxla_thread_resources(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_thread_resources operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.pxla.thread_resources",
    )


@register_op("interpreters.pxla.are_hlo_shardings_equal", "jax")
def _map_jax_interpreters_pxla_are_hlo_shardings_equal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_are_hlo_shardings_equal operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.pxla.are_hlo_shardings_equal",
    )


@register_op("interpreters.pxla.is_hlo_sharding_replicated", "jax")
def _map_jax_interpreters_pxla_is_hlo_sharding_replicated(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_is_hlo_sharding_replicated operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.pxla.is_hlo_sharding_replicated",
    )


@register_op("interpreters.pxla.op_sharding_to_indices", "jax")
def _map_jax_interpreters_pxla_op_sharding_to_indices(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_op_sharding_to_indices operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.pxla.op_sharding_to_indices",
    )


@register_op("interpreters.pxla.ArrayMapping", "jax")
def _map_jax_interpreters_pxla_ArrayMapping(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_ArrayMapping operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.ArrayMapping"
    )


@register_op("interpreters.pxla.array_mapping_to_axis_resources", "jax")
def _map_jax_interpreters_pxla_array_mapping_to_axis_resources(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_array_mapping_to_axis_resources operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.pxla.array_mapping_to_axis_resources",
    )


@register_op("interpreters.pxla.Chunked", "jax")
def _map_jax_interpreters_pxla_Chunked(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_Chunked operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.Chunked"
    )


@register_op("interpreters.pxla.NoSharding", "jax")
def _map_jax_interpreters_pxla_NoSharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_NoSharding operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.NoSharding"
    )


@register_op("interpreters.pxla.Replicated", "jax")
def _map_jax_interpreters_pxla_Replicated(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_Replicated operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.Replicated"
    )


@register_op("interpreters.pxla.ShardedAxis", "jax")
def _map_jax_interpreters_pxla_ShardedAxis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_ShardedAxis operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.ShardedAxis"
    )


@register_op("interpreters.pxla.ShardingSpec", "jax")
def _map_jax_interpreters_pxla_ShardingSpec(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_ShardingSpec operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.ShardingSpec"
    )


@register_op("interpreters.pxla.Unstacked", "jax")
def _map_jax_interpreters_pxla_Unstacked(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_Unstacked operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.Unstacked"
    )


@register_op("interpreters.pxla.spec_to_indices", "jax")
def _map_jax_interpreters_pxla_spec_to_indices(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_pxla_spec_to_indices operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.pxla.spec_to_indices"
    )


@register_op("interpreters.batching.axis_primitive_batchers", "jax")
def _map_jax_interpreters_batching_axis_primitive_batchers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_axis_primitive_batchers operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.batching.axis_primitive_batchers",
    )


@register_op("interpreters.batching.bdim_at_front", "jax")
def _map_jax_interpreters_batching_bdim_at_front(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_bdim_at_front operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.batching.bdim_at_front",
    )


@register_op("interpreters.batching.broadcast", "jax")
def _map_jax_interpreters_batching_broadcast(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_broadcast operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.batching.broadcast"
    )


@register_op("interpreters.batching.defbroadcasting", "jax")
def _map_jax_interpreters_batching_defbroadcasting(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_defbroadcasting operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.batching.defbroadcasting",
    )


@register_op("interpreters.batching.defreducer", "jax")
def _map_jax_interpreters_batching_defreducer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_defreducer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.batching.defreducer"
    )


@register_op("interpreters.batching.defvectorized", "jax")
def _map_jax_interpreters_batching_defvectorized(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_defvectorized operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.batching.defvectorized",
    )


@register_op("interpreters.batching.fancy_primitive_batchers", "jax")
def _map_jax_interpreters_batching_fancy_primitive_batchers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_fancy_primitive_batchers operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.batching.fancy_primitive_batchers",
    )


@register_op("interpreters.batching.not_mapped", "jax")
def _map_jax_interpreters_batching_not_mapped(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_not_mapped operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.batching.not_mapped"
    )


@register_op("interpreters.batching.primitive_batchers", "jax")
def _map_jax_interpreters_batching_primitive_batchers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_primitive_batchers operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.batching.primitive_batchers",
    )


@register_op("interpreters.batching.register_vmappable", "jax")
def _map_jax_interpreters_batching_register_vmappable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_register_vmappable operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.batching.register_vmappable",
    )


@register_op("interpreters.batching.unregister_vmappable", "jax")
def _map_jax_interpreters_batching_unregister_vmappable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_unregister_vmappable operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.batching.unregister_vmappable",
    )


@register_op("interpreters.batching.NotMapped", "jax")
def _map_jax_interpreters_batching_NotMapped(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_batching_NotMapped operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.batching.NotMapped"
    )


@register_op("interpreters.partial_eval.DynamicJaxprTracer", "jax")
def _map_jax_interpreters_partial_eval_DynamicJaxprTracer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_DynamicJaxprTracer operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.DynamicJaxprTracer",
    )


@register_op("interpreters.partial_eval.JaxprTracer", "jax")
def _map_jax_interpreters_partial_eval_JaxprTracer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_JaxprTracer operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.JaxprTracer",
    )


@register_op("interpreters.partial_eval.PartialVal", "jax")
def _map_jax_interpreters_partial_eval_PartialVal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_PartialVal operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.PartialVal",
    )


@register_op("interpreters.partial_eval.Val", "jax")
def _map_jax_interpreters_partial_eval_Val(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_Val operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.partial_eval.Val"
    )


@register_op("interpreters.partial_eval.custom_partial_eval_rules", "jax")
def _map_jax_interpreters_partial_eval_custom_partial_eval_rules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_custom_partial_eval_rules operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.custom_partial_eval_rules",
    )


@register_op("interpreters.partial_eval.dce_jaxpr", "jax")
def _map_jax_interpreters_partial_eval_dce_jaxpr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_dce_jaxpr operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.dce_jaxpr",
    )


@register_op("interpreters.partial_eval.dce_jaxpr_call_rule", "jax")
def _map_jax_interpreters_partial_eval_dce_jaxpr_call_rule(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_dce_jaxpr_call_rule operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.dce_jaxpr_call_rule",
    )


@register_op("interpreters.partial_eval.dce_jaxpr_closed_call_rule", "jax")
def _map_jax_interpreters_partial_eval_dce_jaxpr_closed_call_rule(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_dce_jaxpr_closed_call_rule operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.dce_jaxpr_closed_call_rule",
    )


@register_op("interpreters.partial_eval.dce_jaxpr_consts", "jax")
def _map_jax_interpreters_partial_eval_dce_jaxpr_consts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_dce_jaxpr_consts operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.dce_jaxpr_consts",
    )


@register_op("interpreters.partial_eval.dce_rules", "jax")
def _map_jax_interpreters_partial_eval_dce_rules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_dce_rules operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.dce_rules",
    )


@register_op("interpreters.partial_eval.partial_eval_jaxpr_custom_rules", "jax")
def _map_jax_interpreters_partial_eval_partial_eval_jaxpr_custom_rules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_partial_eval_jaxpr_custom_rules operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.partial_eval_jaxpr_custom_rules",
    )


@register_op("interpreters.partial_eval.trace_to_jaxpr_dynamic", "jax")
def _map_jax_interpreters_partial_eval_trace_to_jaxpr_dynamic(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_trace_to_jaxpr_dynamic operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.trace_to_jaxpr_dynamic",
    )


@register_op("interpreters.partial_eval.trace_to_jaxpr_nounits", "jax")
def _map_jax_interpreters_partial_eval_trace_to_jaxpr_nounits(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_partial_eval_trace_to_jaxpr_nounits operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.partial_eval.trace_to_jaxpr_nounits",
    )


@register_op("interpreters.mlir.AxisContext", "jax")
def _map_jax_interpreters_mlir_AxisContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_AxisContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.AxisContext"
    )


@register_op("interpreters.mlir.ConstantHandler", "jax")
def _map_jax_interpreters_mlir_ConstantHandler(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_ConstantHandler operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.ConstantHandler"
    )


@register_op("interpreters.mlir.DEVICE_TO_DEVICE_TYPE", "jax")
def _map_jax_interpreters_mlir_DEVICE_TO_DEVICE_TYPE(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_DEVICE_TO_DEVICE_TYPE operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.DEVICE_TO_DEVICE_TYPE",
    )


@register_op("interpreters.mlir.LoweringParameters", "jax")
def _map_jax_interpreters_mlir_LoweringParameters(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_LoweringParameters operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.LoweringParameters",
    )


@register_op("interpreters.mlir.LoweringResult", "jax")
def _map_jax_interpreters_mlir_LoweringResult(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_LoweringResult operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.LoweringResult"
    )


@register_op("interpreters.mlir.LoweringRule", "jax")
def _map_jax_interpreters_mlir_LoweringRule(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_LoweringRule operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.LoweringRule"
    )


@register_op("interpreters.mlir.LoweringRuleContext", "jax")
def _map_jax_interpreters_mlir_LoweringRuleContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_LoweringRuleContext operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.LoweringRuleContext",
    )


@register_op("interpreters.mlir.ModuleContext", "jax")
def _map_jax_interpreters_mlir_ModuleContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_ModuleContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.ModuleContext"
    )


@register_op("interpreters.mlir.RECV_FROM_HOST_TYPE", "jax")
def _map_jax_interpreters_mlir_RECV_FROM_HOST_TYPE(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_RECV_FROM_HOST_TYPE operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.RECV_FROM_HOST_TYPE",
    )


@register_op("interpreters.mlir.SEND_TO_HOST_TYPE", "jax")
def _map_jax_interpreters_mlir_SEND_TO_HOST_TYPE(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_SEND_TO_HOST_TYPE operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.SEND_TO_HOST_TYPE",
    )


@register_op("interpreters.mlir.ShapePolyLoweringState", "jax")
def _map_jax_interpreters_mlir_ShapePolyLoweringState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_ShapePolyLoweringState operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.ShapePolyLoweringState",
    )


@register_op("interpreters.mlir.Token", "jax")
def _map_jax_interpreters_mlir_Token(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_Token operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.Token")


@register_op("interpreters.mlir.TokenSet", "jax")
def _map_jax_interpreters_mlir_TokenSet(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_TokenSet operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.TokenSet"
    )


@register_op("interpreters.mlir.Value", "jax")
def _map_jax_interpreters_mlir_Value(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_Value operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.Value")


@register_op("interpreters.mlir.aval_to_ir_type", "jax")
def _map_jax_interpreters_mlir_aval_to_ir_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_aval_to_ir_type operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.aval_to_ir_type"
    )


@register_op("interpreters.mlir.aval_to_ir_types", "jax")
def _map_jax_interpreters_mlir_aval_to_ir_types(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_aval_to_ir_types operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.aval_to_ir_types",
    )


@register_op("interpreters.mlir.core_call_lowering", "jax")
def _map_jax_interpreters_mlir_core_call_lowering(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_core_call_lowering operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.core_call_lowering",
    )


@register_op("interpreters.mlir.dense_int_array", "jax")
def _map_jax_interpreters_mlir_dense_int_array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_dense_int_array operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.dense_int_array"
    )


@register_op("interpreters.mlir.dense_int_elements", "jax")
def _map_jax_interpreters_mlir_dense_int_elements(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_dense_int_elements operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.dense_int_elements",
    )


@register_op("interpreters.mlir.dtype_to_ir_type", "jax")
def _map_jax_interpreters_mlir_dtype_to_ir_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_dtype_to_ir_type operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.dtype_to_ir_type",
    )


@register_op("interpreters.mlir.flatten_ir_types", "jax")
def _map_jax_interpreters_mlir_flatten_ir_types(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_flatten_ir_types operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.flatten_ir_types",
    )


@register_op("interpreters.mlir.flatten_ir_values", "jax")
def _map_jax_interpreters_mlir_flatten_ir_values(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_flatten_ir_values operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.flatten_ir_values",
    )


@register_op("interpreters.mlir.unflatten_ir_values_like_types", "jax")
def _map_jax_interpreters_mlir_unflatten_ir_values_like_types(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_unflatten_ir_values_like_types operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.unflatten_ir_values_like_types",
    )


@register_op("interpreters.mlir.i32_attr", "jax")
def _map_jax_interpreters_mlir_i32_attr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_i32_attr operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.i32_attr"
    )


@register_op("interpreters.mlir.i64_attr", "jax")
def _map_jax_interpreters_mlir_i64_attr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_i64_attr operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.i64_attr"
    )


@register_op("interpreters.mlir.ir", "jax")
def _map_jax_interpreters_mlir_ir(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_ir operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.ir")


@register_op("interpreters.mlir.ir_attribute", "jax")
def _map_jax_interpreters_mlir_ir_attribute(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_ir_attribute operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.ir_attribute"
    )


@register_op("interpreters.mlir.ir_constant", "jax")
def _map_jax_interpreters_mlir_ir_constant(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_ir_constant operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.ir_constant"
    )


@register_op("interpreters.mlir.ir_type_handlers", "jax")
def _map_jax_interpreters_mlir_ir_type_handlers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_ir_type_handlers operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.ir_type_handlers",
    )


@register_op("interpreters.mlir.jaxpr_subcomp", "jax")
def _map_jax_interpreters_mlir_jaxpr_subcomp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_jaxpr_subcomp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.jaxpr_subcomp"
    )


@register_op("interpreters.mlir.lower_fun", "jax")
def _map_jax_interpreters_mlir_lower_fun(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_lower_fun operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.lower_fun"
    )


@register_op("interpreters.mlir.lower_jaxpr_to_fun", "jax")
def _map_jax_interpreters_mlir_lower_jaxpr_to_fun(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_lower_jaxpr_to_fun operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.lower_jaxpr_to_fun",
    )


@register_op("interpreters.mlir.lower_jaxpr_to_module", "jax")
def _map_jax_interpreters_mlir_lower_jaxpr_to_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_lower_jaxpr_to_module operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.lower_jaxpr_to_module",
    )


@register_op("interpreters.mlir.make_ir_context", "jax")
def _map_jax_interpreters_mlir_make_ir_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_make_ir_context operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.make_ir_context"
    )


@register_op("interpreters.mlir.merge_mlir_modules", "jax")
def _map_jax_interpreters_mlir_merge_mlir_modules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_merge_mlir_modules operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.merge_mlir_modules",
    )


@register_op("interpreters.mlir.module_to_bytecode", "jax")
def _map_jax_interpreters_mlir_module_to_bytecode(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_module_to_bytecode operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.module_to_bytecode",
    )


@register_op("interpreters.mlir.module_to_string", "jax")
def _map_jax_interpreters_mlir_module_to_string(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_module_to_string operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.module_to_string",
    )


@register_op("interpreters.mlir.register_constant_handler", "jax")
def _map_jax_interpreters_mlir_register_constant_handler(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_register_constant_handler operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.register_constant_handler",
    )


@register_op("interpreters.mlir.register_lowering", "jax")
def _map_jax_interpreters_mlir_register_lowering(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_register_lowering operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.register_lowering",
    )


@register_op("interpreters.mlir.shape_tensor", "jax")
def _map_jax_interpreters_mlir_shape_tensor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_shape_tensor operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.shape_tensor"
    )


@register_op("interpreters.mlir.token_type", "jax")
def _map_jax_interpreters_mlir_token_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_token_type operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.token_type"
    )


@register_op("interpreters.mlir.Mesh", "jax")
def _map_jax_interpreters_mlir_Mesh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_Mesh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.Mesh")


@register_op("interpreters.mlir.MeshAxisName", "jax")
def _map_jax_interpreters_mlir_MeshAxisName(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_MeshAxisName operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.MeshAxisName"
    )


@register_op("interpreters.mlir.ReplicaAxisContext", "jax")
def _map_jax_interpreters_mlir_ReplicaAxisContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_ReplicaAxisContext operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.ReplicaAxisContext",
    )


@register_op("interpreters.mlir.SPMDAxisContext", "jax")
def _map_jax_interpreters_mlir_SPMDAxisContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_SPMDAxisContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.SPMDAxisContext"
    )


@register_op("interpreters.mlir.ShardingContext", "jax")
def _map_jax_interpreters_mlir_ShardingContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_ShardingContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.mlir.ShardingContext"
    )


@register_op("interpreters.mlir.lowerable_effects", "jax")
def _map_jax_interpreters_mlir_lowerable_effects(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_lowerable_effects operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.lowerable_effects",
    )


@register_op("interpreters.mlir.emit_python_callback", "jax")
def _map_jax_interpreters_mlir_emit_python_callback(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_mlir_emit_python_callback operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.mlir.emit_python_callback",
    )


@register_op("interpreters.ad.JVPTrace", "jax")
def _map_jax_interpreters_ad_JVPTrace(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_JVPTrace operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.JVPTrace")


@register_op("interpreters.ad.JVPTracer", "jax")
def _map_jax_interpreters_ad_JVPTracer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_JVPTracer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.JVPTracer"
    )


@register_op("interpreters.ad.UndefinedPrimal", "jax")
def _map_jax_interpreters_ad_UndefinedPrimal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_UndefinedPrimal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.UndefinedPrimal"
    )


@register_op("interpreters.ad.Zero", "jax")
def _map_jax_interpreters_ad_Zero(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_Zero operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.Zero")


@register_op("interpreters.ad.add_jaxvals", "jax")
def _map_jax_interpreters_ad_add_jaxvals(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_add_jaxvals operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.add_jaxvals"
    )


@register_op("interpreters.ad.add_jaxvals_p", "jax")
def _map_jax_interpreters_ad_add_jaxvals_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_add_jaxvals_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.add_jaxvals_p"
    )


@register_op("interpreters.ad.add_tangents", "jax")
def _map_jax_interpreters_ad_add_tangents(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_add_tangents operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.add_tangents"
    )


@register_op("interpreters.ad.defbilinear", "jax")
def _map_jax_interpreters_ad_defbilinear(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_defbilinear operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.defbilinear"
    )


@register_op("interpreters.ad.defjvp", "jax")
def _map_jax_interpreters_ad_defjvp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_defjvp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.defjvp")


@register_op("interpreters.ad.defjvp2", "jax")
def _map_jax_interpreters_ad_defjvp2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_defjvp2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.defjvp2")


@register_op("interpreters.ad.deflinear", "jax")
def _map_jax_interpreters_ad_deflinear(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_deflinear operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.deflinear"
    )


@register_op("interpreters.ad.deflinear2", "jax")
def _map_jax_interpreters_ad_deflinear2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_deflinear2 operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.deflinear2"
    )


@register_op("interpreters.ad.get_primitive_transpose", "jax")
def _map_jax_interpreters_ad_get_primitive_transpose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_get_primitive_transpose operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.ad.get_primitive_transpose",
    )


@register_op("interpreters.ad.instantiate_zeros", "jax")
def _map_jax_interpreters_ad_instantiate_zeros(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_instantiate_zeros operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.instantiate_zeros"
    )


@register_op("interpreters.ad.is_undefined_primal", "jax")
def _map_jax_interpreters_ad_is_undefined_primal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_is_undefined_primal operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.ad.is_undefined_primal",
    )


@register_op("interpreters.ad.jvp", "jax")
def _map_jax_interpreters_ad_jvp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_jvp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.jvp")


@register_op("interpreters.ad.linearize", "jax")
def _map_jax_interpreters_ad_linearize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_linearize operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.linearize"
    )


@register_op("interpreters.ad.primitive_jvps", "jax")
def _map_jax_interpreters_ad_primitive_jvps(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_primitive_jvps operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.primitive_jvps"
    )


@register_op("interpreters.ad.primitive_transposes", "jax")
def _map_jax_interpreters_ad_primitive_transposes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_primitive_transposes operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.ad.primitive_transposes",
    )


@register_op("interpreters.ad.zeros_like_aval", "jax")
def _map_jax_interpreters_ad_zeros_like_aval(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_zeros_like_aval operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="interpreters.ad.zeros_like_aval"
    )


@register_op("interpreters.ad.reducing_transposes", "jax")
def _map_jax_interpreters_ad_reducing_transposes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_interpreters_ad_reducing_transposes operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="interpreters.ad.reducing_transposes",
    )


@register_op("extend.linear_util.StoreException", "jax")
def _map_jax_extend_linear_util_StoreException(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_linear_util_StoreException operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.linear_util.StoreException"
    )


@register_op("extend.linear_util.WrappedFun", "jax")
def _map_jax_extend_linear_util_WrappedFun(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_linear_util_WrappedFun operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.linear_util.WrappedFun"
    )


@register_op("extend.linear_util.cache", "jax")
def _map_jax_extend_linear_util_cache(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_linear_util_cache operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="extend.linear_util.cache")


@register_op("extend.linear_util.merge_linear_aux", "jax")
def _map_jax_extend_linear_util_merge_linear_aux(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_linear_util_merge_linear_aux operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.linear_util.merge_linear_aux",
    )


@register_op("extend.linear_util.transformation", "jax")
def _map_jax_extend_linear_util_transformation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_linear_util_transformation operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.linear_util.transformation"
    )


@register_op("extend.linear_util.transformation_with_aux", "jax")
def _map_jax_extend_linear_util_transformation_with_aux(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_linear_util_transformation_with_aux operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.linear_util.transformation_with_aux",
    )


@register_op("extend.linear_util.transformation2", "jax")
def _map_jax_extend_linear_util_transformation2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_linear_util_transformation2 operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.linear_util.transformation2",
    )


@register_op("extend.linear_util.transformation_with_aux2", "jax")
def _map_jax_extend_linear_util_transformation_with_aux2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_linear_util_transformation_with_aux2 operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.linear_util.transformation_with_aux2",
    )


@register_op("extend.linear_util.wrap_init", "jax")
def _map_jax_extend_linear_util_wrap_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_linear_util_wrap_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.linear_util.wrap_init"
    )


@register_op("extend.source_info_util.NameStack", "jax")
def _map_jax_extend_source_info_util_NameStack(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_NameStack operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.source_info_util.NameStack"
    )


@register_op("extend.source_info_util.SourceInfo", "jax")
def _map_jax_extend_source_info_util_SourceInfo(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_SourceInfo operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.source_info_util.SourceInfo",
    )


@register_op("extend.source_info_util.current", "jax")
def _map_jax_extend_source_info_util_current(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_current operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.source_info_util.current"
    )


@register_op("extend.source_info_util.current_name_stack", "jax")
def _map_jax_extend_source_info_util_current_name_stack(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_current_name_stack operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.source_info_util.current_name_stack",
    )


@register_op("extend.source_info_util.extend_name_stack", "jax")
def _map_jax_extend_source_info_util_extend_name_stack(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_extend_name_stack operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.source_info_util.extend_name_stack",
    )


@register_op("extend.source_info_util.new_name_stack", "jax")
def _map_jax_extend_source_info_util_new_name_stack(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_new_name_stack operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.source_info_util.new_name_stack",
    )


@register_op("extend.source_info_util.new_source_info", "jax")
def _map_jax_extend_source_info_util_new_source_info(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_new_source_info operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.source_info_util.new_source_info",
    )


@register_op("extend.source_info_util.register_exclusion", "jax")
def _map_jax_extend_source_info_util_register_exclusion(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_register_exclusion operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.source_info_util.register_exclusion",
    )


@register_op("extend.source_info_util.reset_name_stack", "jax")
def _map_jax_extend_source_info_util_reset_name_stack(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_reset_name_stack operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.source_info_util.reset_name_stack",
    )


@register_op("extend.source_info_util.set_name_stack", "jax")
def _map_jax_extend_source_info_util_set_name_stack(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_set_name_stack operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.source_info_util.set_name_stack",
    )


@register_op("extend.source_info_util.summarize", "jax")
def _map_jax_extend_source_info_util_summarize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_summarize operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.source_info_util.summarize"
    )


@register_op("extend.source_info_util.transform_name_stack", "jax")
def _map_jax_extend_source_info_util_transform_name_stack(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_transform_name_stack operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.source_info_util.transform_name_stack",
    )


@register_op("extend.source_info_util.user_context", "jax")
def _map_jax_extend_source_info_util_user_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_source_info_util_user_context operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.source_info_util.user_context",
    )


@register_op("extend.random.define_prng_impl", "jax")
def _map_jax_extend_random_define_prng_impl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_random_define_prng_impl operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.random.define_prng_impl"
    )


@register_op("extend.random.random_seed", "jax")
def _map_jax_extend_random_random_seed(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_random_random_seed operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.random.random_seed"
    )


@register_op("extend.random.seed_with_impl", "jax")
def _map_jax_extend_random_seed_with_impl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_random_seed_with_impl operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.random.seed_with_impl"
    )


@register_op("extend.random.threefry2x32_p", "jax")
def _map_jax_extend_random_threefry2x32_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_random_threefry2x32_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.random.threefry2x32_p"
    )


@register_op("extend.random.threefry_2x32", "jax")
def _map_jax_extend_random_threefry_2x32(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_random_threefry_2x32 operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.random.threefry_2x32"
    )


@register_op("extend.random.threefry_prng_impl", "jax")
def _map_jax_extend_random_threefry_prng_impl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_random_threefry_prng_impl operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.random.threefry_prng_impl"
    )


@register_op("extend.random.rbg_prng_impl", "jax")
def _map_jax_extend_random_rbg_prng_impl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_random_rbg_prng_impl operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.random.rbg_prng_impl"
    )


@register_op("extend.random.unsafe_rbg_prng_impl", "jax")
def _map_jax_extend_random_unsafe_rbg_prng_impl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_random_unsafe_rbg_prng_impl operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.random.unsafe_rbg_prng_impl",
    )


@register_op("extend.sharding.xla_client", "jax")
def _map_jax_extend_sharding_xla_client(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_sharding_xla_client operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.sharding.xla_client"
    )


@register_op("extend.sharding.GSPMDSharding", "jax")
def _map_jax_extend_sharding_GSPMDSharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_sharding_GSPMDSharding operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.sharding.GSPMDSharding"
    )


@register_op("extend.sharding.get_op_sharding_from_serialized_proto", "jax")
def _map_jax_extend_sharding_get_op_sharding_from_serialized_proto(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_sharding_get_op_sharding_from_serialized_proto operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.sharding.get_op_sharding_from_serialized_proto",
    )


@register_op("extend.sharding.get_hlo_sharding_from_serialized_proto", "jax")
def _map_jax_extend_sharding_get_hlo_sharding_from_serialized_proto(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_sharding_get_hlo_sharding_from_serialized_proto operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.sharding.get_hlo_sharding_from_serialized_proto",
    )


@register_op("extend.sharding.get_serialized_proto_from_hlo_sharding", "jax")
def _map_jax_extend_sharding_get_serialized_proto_from_hlo_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_sharding_get_serialized_proto_from_hlo_sharding operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.sharding.get_serialized_proto_from_hlo_sharding",
    )


@register_op("extend.ifrt_programs.ifrt_programs", "jax")
def _map_jax_extend_ifrt_programs_ifrt_programs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_ifrt_programs_ifrt_programs operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.ifrt_programs.ifrt_programs",
    )


@register_op("extend.mlir.lower_with_sharding_in_types", "jax")
def _map_jax_extend_mlir_lower_with_sharding_in_types(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_mlir_lower_with_sharding_in_types operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.mlir.lower_with_sharding_in_types",
    )


@register_op("extend.mlir.deserialize_portable_artifact", "jax")
def _map_jax_extend_mlir_deserialize_portable_artifact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_mlir_deserialize_portable_artifact operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.mlir.deserialize_portable_artifact",
    )


@register_op("extend.mlir.serialize_portable_artifact", "jax")
def _map_jax_extend_mlir_serialize_portable_artifact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_mlir_serialize_portable_artifact operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.mlir.serialize_portable_artifact",
    )


@register_op("extend.mlir.refine_polymorphic_shapes", "jax")
def _map_jax_extend_mlir_refine_polymorphic_shapes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_mlir_refine_polymorphic_shapes operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="extend.mlir.refine_polymorphic_shapes",
    )


@register_op("extend.mlir.hlo_to_stablehlo", "jax")
def _map_jax_extend_mlir_hlo_to_stablehlo(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_extend_mlir_hlo_to_stablehlo operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="extend.mlir.hlo_to_stablehlo"
    )


@register_op("scipy.signal.fftconvolve", "jax")
def _map_jax_scipy_signal_fftconvolve(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_signal_fftconvolve operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.signal.fftconvolve")


@register_op("scipy.signal.convolve", "jax")
def _map_jax_scipy_signal_convolve(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_signal_convolve operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.signal.convolve")


@register_op("scipy.signal.convolve2d", "jax")
def _map_jax_scipy_signal_convolve2d(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_signal_convolve2d operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.signal.convolve2d")


@register_op("scipy.signal.correlate", "jax")
def _map_jax_scipy_signal_correlate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_signal_correlate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.signal.correlate")


@register_op("scipy.signal.correlate2d", "jax")
def _map_jax_scipy_signal_correlate2d(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_signal_correlate2d operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.signal.correlate2d")


@register_op("scipy.signal.detrend", "jax")
def _map_jax_scipy_signal_detrend(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_signal_detrend operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.signal.detrend")


@register_op("scipy.signal.csd", "jax")
def _map_jax_scipy_signal_csd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_signal_csd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.signal.csd")


@register_op("scipy.signal.istft", "jax")
def _map_jax_scipy_signal_istft(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_signal_istft operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.signal.istft")


@register_op("scipy.signal.stft", "jax")
def _map_jax_scipy_signal_stft(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_signal_stft operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.signal.stft")


@register_op("scipy.signal.welch", "jax")
def _map_jax_scipy_signal_welch(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_signal_welch operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.signal.welch")


@register_op("scipy.integrate.trapezoid", "jax")
def _map_jax_scipy_integrate_trapezoid(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_integrate_trapezoid operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.integrate.trapezoid"
    )


@register_op("scipy.ndimage.map_coordinates", "jax")
def _map_jax_scipy_ndimage_map_coordinates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_ndimage_map_coordinates operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.ndimage.map_coordinates"
    )


@register_op("scipy.fft.dct", "jax")
def _map_jax_scipy_fft_dct(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_scipy_fft_dct operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.fft.dct")


@register_op("scipy.fft.dctn", "jax")
def _map_jax_scipy_fft_dctn(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_scipy_fft_dctn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.fft.dctn")


@register_op("scipy.fft.idct", "jax")
def _map_jax_scipy_fft_idct(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_scipy_fft_idct operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.fft.idct")


@register_op("scipy.fft.idctn", "jax")
def _map_jax_scipy_fft_idctn(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_scipy_fft_idctn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.fft.idctn")


@register_op("scipy.linalg.block_diag", "jax")
def _map_jax_scipy_linalg_block_diag(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_block_diag operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.block_diag")


@register_op("scipy.linalg.cholesky", "jax")
def _map_jax_scipy_linalg_cholesky(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_cholesky operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.cholesky")


@register_op("scipy.linalg.cho_factor", "jax")
def _map_jax_scipy_linalg_cho_factor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_cho_factor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.cho_factor")


@register_op("scipy.linalg.cho_solve", "jax")
def _map_jax_scipy_linalg_cho_solve(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_cho_solve operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.cho_solve")


@register_op("scipy.linalg.det", "jax")
def _map_jax_scipy_linalg_det(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_det operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.det")


@register_op("scipy.linalg.eigh", "jax")
def _map_jax_scipy_linalg_eigh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_eigh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.eigh")


@register_op("scipy.linalg.eigh_tridiagonal", "jax")
def _map_jax_scipy_linalg_eigh_tridiagonal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_eigh_tridiagonal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.eigh_tridiagonal"
    )


@register_op("scipy.linalg.expm", "jax")
def _map_jax_scipy_linalg_expm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_expm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.expm")


@register_op("scipy.linalg.expm_frechet", "jax")
def _map_jax_scipy_linalg_expm_frechet(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_expm_frechet operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.expm_frechet"
    )


@register_op("scipy.linalg.hessenberg", "jax")
def _map_jax_scipy_linalg_hessenberg(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_hessenberg operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.hessenberg")


@register_op("scipy.linalg.hankel", "jax")
def _map_jax_scipy_linalg_hankel(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_hankel operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.hankel")


@register_op("scipy.linalg.hilbert", "jax")
def _map_jax_scipy_linalg_hilbert(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_hilbert operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.hilbert")


@register_op("scipy.linalg.inv", "jax")
def _map_jax_scipy_linalg_inv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_inv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.inv")


@register_op("scipy.linalg.lu", "jax")
def _map_jax_scipy_linalg_lu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_scipy_linalg_lu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.lu")


@register_op("scipy.linalg.lu_factor", "jax")
def _map_jax_scipy_linalg_lu_factor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_lu_factor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.lu_factor")


@register_op("scipy.linalg.lu_solve", "jax")
def _map_jax_scipy_linalg_lu_solve(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_lu_solve operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.lu_solve")


@register_op("scipy.linalg.pascal", "jax")
def _map_jax_scipy_linalg_pascal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_pascal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.pascal")


@register_op("scipy.linalg.polar", "jax")
def _map_jax_scipy_linalg_polar(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_polar operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.polar")


@register_op("scipy.linalg.qr", "jax")
def _map_jax_scipy_linalg_qr(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_scipy_linalg_qr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.qr")


@register_op("scipy.linalg.rsf2csf", "jax")
def _map_jax_scipy_linalg_rsf2csf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_rsf2csf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.rsf2csf")


@register_op("scipy.linalg.schur", "jax")
def _map_jax_scipy_linalg_schur(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_schur operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.schur")


@register_op("scipy.linalg.sqrtm", "jax")
def _map_jax_scipy_linalg_sqrtm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_sqrtm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.sqrtm")


@register_op("scipy.linalg.solve", "jax")
def _map_jax_scipy_linalg_solve(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_solve operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.solve")


@register_op("scipy.linalg.solve_sylvester", "jax")
def _map_jax_scipy_linalg_solve_sylvester(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_solve_sylvester operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.solve_sylvester"
    )


@register_op("scipy.linalg.solve_triangular", "jax")
def _map_jax_scipy_linalg_solve_triangular(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_solve_triangular operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.solve_triangular"
    )


@register_op("scipy.linalg.svd", "jax")
def _map_jax_scipy_linalg_svd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_svd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.svd")


@register_op("scipy.linalg.toeplitz", "jax")
def _map_jax_scipy_linalg_toeplitz(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_toeplitz operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.toeplitz")


@register_op("scipy.linalg.funm", "jax")
def _map_jax_scipy_linalg_funm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_linalg_funm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.linalg.funm")


@register_op("scipy.special.bernoulli", "jax")
def _map_jax_scipy_special_bernoulli(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_bernoulli operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.bernoulli")


@register_op("scipy.special.bessel_jn", "jax")
def _map_jax_scipy_special_bessel_jn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_bessel_jn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.bessel_jn")


@register_op("scipy.special.beta", "jax")
def _map_jax_scipy_special_beta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_beta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.beta")


@register_op("scipy.special.betainc", "jax")
def _map_jax_scipy_special_betainc(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_betainc operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.betainc")


@register_op("scipy.special.betaln", "jax")
def _map_jax_scipy_special_betaln(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_betaln operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.betaln")


@register_op("scipy.special.digamma", "jax")
def _map_jax_scipy_special_digamma(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_digamma operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.digamma")


@register_op("scipy.special.entr", "jax")
def _map_jax_scipy_special_entr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_entr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.entr")


@register_op("scipy.special.erf", "jax")
def _map_jax_scipy_special_erf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_erf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.erf")


@register_op("scipy.special.erfc", "jax")
def _map_jax_scipy_special_erfc(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_erfc operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.erfc")


@register_op("scipy.special.erfinv", "jax")
def _map_jax_scipy_special_erfinv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_erfinv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.erfinv")


@register_op("scipy.special.exp1", "jax")
def _map_jax_scipy_special_exp1(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_exp1 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.exp1")


@register_op("scipy.special.expi", "jax")
def _map_jax_scipy_special_expi(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_expi operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.expi")


@register_op("scipy.special.expit", "jax")
def _map_jax_scipy_special_expit(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_expit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.expit")


@register_op("scipy.special.expn", "jax")
def _map_jax_scipy_special_expn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_expn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.expn")


@register_op("scipy.special.factorial", "jax")
def _map_jax_scipy_special_factorial(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_factorial operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.factorial")


@register_op("scipy.special.gamma", "jax")
def _map_jax_scipy_special_gamma(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_gamma operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.gamma")


@register_op("scipy.special.gammainc", "jax")
def _map_jax_scipy_special_gammainc(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_gammainc operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.gammainc")


@register_op("scipy.special.gammaincc", "jax")
def _map_jax_scipy_special_gammaincc(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_gammaincc operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.gammaincc")


@register_op("scipy.special.gammaln", "jax")
def _map_jax_scipy_special_gammaln(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_gammaln operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.gammaln")


@register_op("scipy.special.gammasgn", "jax")
def _map_jax_scipy_special_gammasgn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_gammasgn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.gammasgn")


@register_op("scipy.special.hyp1f1", "jax")
def _map_jax_scipy_special_hyp1f1(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_hyp1f1 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.hyp1f1")


@register_op("scipy.special.hyp2f1", "jax")
def _map_jax_scipy_special_hyp2f1(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_hyp2f1 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.hyp2f1")


@register_op("scipy.special.i0", "jax")
def _map_jax_scipy_special_i0(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_i0 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.i0")


@register_op("scipy.special.i0e", "jax")
def _map_jax_scipy_special_i0e(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_i0e operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.i0e")


@register_op("scipy.special.i1", "jax")
def _map_jax_scipy_special_i1(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_i1 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.i1")


@register_op("scipy.special.i1e", "jax")
def _map_jax_scipy_special_i1e(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_i1e operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.i1e")


@register_op("scipy.special.kl_div", "jax")
def _map_jax_scipy_special_kl_div(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_kl_div operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.kl_div")


@register_op("scipy.special.log_ndtr", "jax")
def _map_jax_scipy_special_log_ndtr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_log_ndtr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.log_ndtr")


@register_op("scipy.special.log_softmax", "jax")
def _map_jax_scipy_special_log_softmax(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_log_softmax operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.log_softmax"
    )


@register_op("scipy.special.logit", "jax")
def _map_jax_scipy_special_logit(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_logit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.logit")


@register_op("scipy.special.logsumexp", "jax")
def _map_jax_scipy_special_logsumexp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_logsumexp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.logsumexp")


@register_op("scipy.special.multigammaln", "jax")
def _map_jax_scipy_special_multigammaln(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_multigammaln operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.multigammaln"
    )


@register_op("scipy.special.ndtr", "jax")
def _map_jax_scipy_special_ndtr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_ndtr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.ndtr")


@register_op("scipy.special.ndtri", "jax")
def _map_jax_scipy_special_ndtri(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_ndtri operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.ndtri")


@register_op("scipy.special.poch", "jax")
def _map_jax_scipy_special_poch(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_poch operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.poch")


@register_op("scipy.special.polygamma", "jax")
def _map_jax_scipy_special_polygamma(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_polygamma operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.polygamma")


@register_op("scipy.special.rel_entr", "jax")
def _map_jax_scipy_special_rel_entr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_rel_entr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.rel_entr")


@register_op("scipy.special.sici", "jax")
def _map_jax_scipy_special_sici(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_sici operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.sici")


@register_op("scipy.special.softmax", "jax")
def _map_jax_scipy_special_softmax(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_softmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.softmax")


@register_op("scipy.special.spence", "jax")
def _map_jax_scipy_special_spence(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_spence operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.spence")


@register_op("scipy.special.sph_harm_y", "jax")
def _map_jax_scipy_special_sph_harm_y(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_sph_harm_y operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.sph_harm_y")


@register_op("scipy.special.xlog1py", "jax")
def _map_jax_scipy_special_xlog1py(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_xlog1py operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.xlog1py")


@register_op("scipy.special.xlogy", "jax")
def _map_jax_scipy_special_xlogy(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_xlogy operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.xlogy")


@register_op("scipy.special.zeta", "jax")
def _map_jax_scipy_special_zeta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_zeta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.zeta")


@register_op("scipy.special.fresnel", "jax")
def _map_jax_scipy_special_fresnel(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_fresnel operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.fresnel")


@register_op("scipy.special.lpmn", "jax")
def _map_jax_scipy_special_lpmn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_lpmn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.lpmn")


@register_op("scipy.special.lpmn_values", "jax")
def _map_jax_scipy_special_lpmn_values(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_lpmn_values operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.lpmn_values"
    )


@register_op("scipy.special.sph_harm", "jax")
def _map_jax_scipy_special_sph_harm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_special_sph_harm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.special.sph_harm")


@register_op("scipy.spatial.transform.Rotation", "jax")
def _map_jax_scipy_spatial_transform_Rotation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_spatial_transform_Rotation operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.spatial.transform.Rotation"
    )


@register_op("scipy.spatial.transform.Slerp", "jax")
def _map_jax_scipy_spatial_transform_Slerp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_spatial_transform_Slerp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.spatial.transform.Slerp"
    )


@register_op("scipy.optimize.minimize", "jax")
def _map_jax_scipy_optimize_minimize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_optimize_minimize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.optimize.minimize")


@register_op("scipy.optimize.OptimizeResults", "jax")
def _map_jax_scipy_optimize_OptimizeResults(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_optimize_OptimizeResults operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.optimize.OptimizeResults"
    )


@register_op("scipy.interpolate.RegularGridInterpolator", "jax")
def _map_jax_scipy_interpolate_RegularGridInterpolator(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_interpolate_RegularGridInterpolator operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="scipy.interpolate.RegularGridInterpolator",
    )


@register_op("scipy.stats.gaussian_kde", "jax")
def _map_jax_scipy_stats_gaussian_kde(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gaussian_kde operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gaussian_kde")


@register_op("scipy.stats.mode", "jax")
def _map_jax_scipy_stats_mode(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_mode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.mode")


@register_op("scipy.stats.rankdata", "jax")
def _map_jax_scipy_stats_rankdata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_rankdata operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.rankdata")


@register_op("scipy.stats.sem", "jax")
def _map_jax_scipy_stats_sem(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_scipy_stats_sem operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.sem")


@register_op("scipy.stats.vonmises.logpdf", "jax")
def _map_jax_scipy_stats_vonmises_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_vonmises_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.vonmises.logpdf"
    )


@register_op("scipy.stats.vonmises.pdf", "jax")
def _map_jax_scipy_stats_vonmises_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_vonmises_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.vonmises.pdf")


@register_op("scipy.stats.beta.cdf", "jax")
def _map_jax_scipy_stats_beta_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_beta_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.beta.cdf")


@register_op("scipy.stats.beta.logcdf", "jax")
def _map_jax_scipy_stats_beta_logcdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_beta_logcdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.beta.logcdf")


@register_op("scipy.stats.beta.logpdf", "jax")
def _map_jax_scipy_stats_beta_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_beta_logpdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.beta.logpdf")


@register_op("scipy.stats.beta.logsf", "jax")
def _map_jax_scipy_stats_beta_logsf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_beta_logsf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.beta.logsf")


@register_op("scipy.stats.beta.pdf", "jax")
def _map_jax_scipy_stats_beta_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_beta_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.beta.pdf")


@register_op("scipy.stats.beta.sf", "jax")
def _map_jax_scipy_stats_beta_sf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_beta_sf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.beta.sf")


@register_op("scipy.stats.gamma.cdf", "jax")
def _map_jax_scipy_stats_gamma_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gamma_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gamma.cdf")


@register_op("scipy.stats.gamma.logcdf", "jax")
def _map_jax_scipy_stats_gamma_logcdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gamma_logcdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gamma.logcdf")


@register_op("scipy.stats.gamma.logpdf", "jax")
def _map_jax_scipy_stats_gamma_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gamma_logpdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gamma.logpdf")


@register_op("scipy.stats.gamma.logsf", "jax")
def _map_jax_scipy_stats_gamma_logsf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gamma_logsf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gamma.logsf")


@register_op("scipy.stats.gamma.pdf", "jax")
def _map_jax_scipy_stats_gamma_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gamma_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gamma.pdf")


@register_op("scipy.stats.gamma.sf", "jax")
def _map_jax_scipy_stats_gamma_sf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gamma_sf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gamma.sf")


@register_op("scipy.stats.t.logpdf", "jax")
def _map_jax_scipy_stats_t_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_t_logpdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.t.logpdf")


@register_op("scipy.stats.t.pdf", "jax")
def _map_jax_scipy_stats_t_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_t_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.t.pdf")


@register_op("scipy.stats.nbinom.logpmf", "jax")
def _map_jax_scipy_stats_nbinom_logpmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_nbinom_logpmf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.nbinom.logpmf"
    )


@register_op("scipy.stats.nbinom.pmf", "jax")
def _map_jax_scipy_stats_nbinom_pmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_nbinom_pmf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.nbinom.pmf")


@register_op("scipy.stats.betabinom.logpmf", "jax")
def _map_jax_scipy_stats_betabinom_logpmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_betabinom_logpmf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.betabinom.logpmf"
    )


@register_op("scipy.stats.betabinom.pmf", "jax")
def _map_jax_scipy_stats_betabinom_pmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_betabinom_pmf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.betabinom.pmf"
    )


@register_op("scipy.stats.truncnorm.cdf", "jax")
def _map_jax_scipy_stats_truncnorm_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_truncnorm_cdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.truncnorm.cdf"
    )


@register_op("scipy.stats.truncnorm.logcdf", "jax")
def _map_jax_scipy_stats_truncnorm_logcdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_truncnorm_logcdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.truncnorm.logcdf"
    )


@register_op("scipy.stats.truncnorm.logpdf", "jax")
def _map_jax_scipy_stats_truncnorm_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_truncnorm_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.truncnorm.logpdf"
    )


@register_op("scipy.stats.truncnorm.pdf", "jax")
def _map_jax_scipy_stats_truncnorm_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_truncnorm_pdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.truncnorm.pdf"
    )


@register_op("scipy.stats.truncnorm.logsf", "jax")
def _map_jax_scipy_stats_truncnorm_logsf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_truncnorm_logsf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.truncnorm.logsf"
    )


@register_op("scipy.stats.truncnorm.sf", "jax")
def _map_jax_scipy_stats_truncnorm_sf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_truncnorm_sf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.truncnorm.sf")


@register_op("scipy.stats.multivariate_normal.logpdf", "jax")
def _map_jax_scipy_stats_multivariate_normal_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_multivariate_normal_logpdf operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="scipy.stats.multivariate_normal.logpdf",
    )


@register_op("scipy.stats.multivariate_normal.pdf", "jax")
def _map_jax_scipy_stats_multivariate_normal_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_multivariate_normal_pdf operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="scipy.stats.multivariate_normal.pdf",
    )


@register_op("scipy.stats.gumbel_l.logpdf", "jax")
def _map_jax_scipy_stats_gumbel_l_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_l_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_l.logpdf"
    )


@register_op("scipy.stats.gumbel_l.pdf", "jax")
def _map_jax_scipy_stats_gumbel_l_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_l_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_l.pdf")


@register_op("scipy.stats.gumbel_l.logcdf", "jax")
def _map_jax_scipy_stats_gumbel_l_logcdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_l_logcdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_l.logcdf"
    )


@register_op("scipy.stats.gumbel_l.cdf", "jax")
def _map_jax_scipy_stats_gumbel_l_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_l_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_l.cdf")


@register_op("scipy.stats.gumbel_l.ppf", "jax")
def _map_jax_scipy_stats_gumbel_l_ppf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_l_ppf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_l.ppf")


@register_op("scipy.stats.gumbel_l.sf", "jax")
def _map_jax_scipy_stats_gumbel_l_sf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_l_sf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_l.sf")


@register_op("scipy.stats.gumbel_l.logsf", "jax")
def _map_jax_scipy_stats_gumbel_l_logsf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_l_logsf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_l.logsf"
    )


@register_op("scipy.stats.expon.cdf", "jax")
def _map_jax_scipy_stats_expon_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_expon_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.expon.cdf")


@register_op("scipy.stats.expon.logcdf", "jax")
def _map_jax_scipy_stats_expon_logcdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_expon_logcdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.expon.logcdf")


@register_op("scipy.stats.expon.logpdf", "jax")
def _map_jax_scipy_stats_expon_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_expon_logpdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.expon.logpdf")


@register_op("scipy.stats.expon.logsf", "jax")
def _map_jax_scipy_stats_expon_logsf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_expon_logsf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.expon.logsf")


@register_op("scipy.stats.expon.pdf", "jax")
def _map_jax_scipy_stats_expon_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_expon_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.expon.pdf")


@register_op("scipy.stats.expon.ppf", "jax")
def _map_jax_scipy_stats_expon_ppf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_expon_ppf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.expon.ppf")


@register_op("scipy.stats.expon.sf", "jax")
def _map_jax_scipy_stats_expon_sf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_expon_sf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.expon.sf")


@register_op("scipy.stats.poisson.logpmf", "jax")
def _map_jax_scipy_stats_poisson_logpmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_poisson_logpmf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.poisson.logpmf"
    )


@register_op("scipy.stats.poisson.pmf", "jax")
def _map_jax_scipy_stats_poisson_pmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_poisson_pmf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.poisson.pmf")


@register_op("scipy.stats.poisson.cdf", "jax")
def _map_jax_scipy_stats_poisson_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_poisson_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.poisson.cdf")


@register_op("scipy.stats.poisson.entropy", "jax")
def _map_jax_scipy_stats_poisson_entropy(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_poisson_entropy operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.poisson.entropy"
    )


@register_op("scipy.stats.gennorm.cdf", "jax")
def _map_jax_scipy_stats_gennorm_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gennorm_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gennorm.cdf")


@register_op("scipy.stats.gennorm.logpdf", "jax")
def _map_jax_scipy_stats_gennorm_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gennorm_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gennorm.logpdf"
    )


@register_op("scipy.stats.gennorm.pdf", "jax")
def _map_jax_scipy_stats_gennorm_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gennorm_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gennorm.pdf")


@register_op("scipy.stats.chi2.cdf", "jax")
def _map_jax_scipy_stats_chi2_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_chi2_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.chi2.cdf")


@register_op("scipy.stats.chi2.logcdf", "jax")
def _map_jax_scipy_stats_chi2_logcdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_chi2_logcdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.chi2.logcdf")


@register_op("scipy.stats.chi2.logpdf", "jax")
def _map_jax_scipy_stats_chi2_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_chi2_logpdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.chi2.logpdf")


@register_op("scipy.stats.chi2.logsf", "jax")
def _map_jax_scipy_stats_chi2_logsf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_chi2_logsf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.chi2.logsf")


@register_op("scipy.stats.chi2.pdf", "jax")
def _map_jax_scipy_stats_chi2_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_chi2_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.chi2.pdf")


@register_op("scipy.stats.chi2.sf", "jax")
def _map_jax_scipy_stats_chi2_sf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_chi2_sf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.chi2.sf")


@register_op("scipy.stats.laplace.cdf", "jax")
def _map_jax_scipy_stats_laplace_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_laplace_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.laplace.cdf")


@register_op("scipy.stats.laplace.logpdf", "jax")
def _map_jax_scipy_stats_laplace_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_laplace_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.laplace.logpdf"
    )


@register_op("scipy.stats.laplace.pdf", "jax")
def _map_jax_scipy_stats_laplace_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_laplace_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.laplace.pdf")


@register_op("scipy.stats.gumbel_r.logpdf", "jax")
def _map_jax_scipy_stats_gumbel_r_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_r_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_r.logpdf"
    )


@register_op("scipy.stats.gumbel_r.pdf", "jax")
def _map_jax_scipy_stats_gumbel_r_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_r_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_r.pdf")


@register_op("scipy.stats.gumbel_r.logcdf", "jax")
def _map_jax_scipy_stats_gumbel_r_logcdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_r_logcdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_r.logcdf"
    )


@register_op("scipy.stats.gumbel_r.cdf", "jax")
def _map_jax_scipy_stats_gumbel_r_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_r_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_r.cdf")


@register_op("scipy.stats.gumbel_r.ppf", "jax")
def _map_jax_scipy_stats_gumbel_r_ppf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_r_ppf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_r.ppf")


@register_op("scipy.stats.gumbel_r.sf", "jax")
def _map_jax_scipy_stats_gumbel_r_sf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_r_sf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_r.sf")


@register_op("scipy.stats.gumbel_r.logsf", "jax")
def _map_jax_scipy_stats_gumbel_r_logsf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_gumbel_r_logsf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.gumbel_r.logsf"
    )


@register_op("scipy.stats.geom.logpmf", "jax")
def _map_jax_scipy_stats_geom_logpmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_geom_logpmf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.geom.logpmf")


@register_op("scipy.stats.geom.pmf", "jax")
def _map_jax_scipy_stats_geom_pmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_geom_pmf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.geom.pmf")


@register_op("scipy.stats.bernoulli.logpmf", "jax")
def _map_jax_scipy_stats_bernoulli_logpmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_bernoulli_logpmf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.bernoulli.logpmf"
    )


@register_op("scipy.stats.bernoulli.pmf", "jax")
def _map_jax_scipy_stats_bernoulli_pmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_bernoulli_pmf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.bernoulli.pmf"
    )


@register_op("scipy.stats.bernoulli.cdf", "jax")
def _map_jax_scipy_stats_bernoulli_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_bernoulli_cdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.bernoulli.cdf"
    )


@register_op("scipy.stats.bernoulli.ppf", "jax")
def _map_jax_scipy_stats_bernoulli_ppf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_bernoulli_ppf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.bernoulli.ppf"
    )


@register_op("scipy.stats.binom.logpmf", "jax")
def _map_jax_scipy_stats_binom_logpmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_binom_logpmf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.binom.logpmf")


@register_op("scipy.stats.binom.pmf", "jax")
def _map_jax_scipy_stats_binom_pmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_binom_pmf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.binom.pmf")


@register_op("scipy.stats.wrapcauchy.logpdf", "jax")
def _map_jax_scipy_stats_wrapcauchy_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_wrapcauchy_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.wrapcauchy.logpdf"
    )


@register_op("scipy.stats.wrapcauchy.pdf", "jax")
def _map_jax_scipy_stats_wrapcauchy_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_wrapcauchy_pdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.wrapcauchy.pdf"
    )


@register_op("scipy.stats.dirichlet.logpdf", "jax")
def _map_jax_scipy_stats_dirichlet_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_dirichlet_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.dirichlet.logpdf"
    )


@register_op("scipy.stats.dirichlet.pdf", "jax")
def _map_jax_scipy_stats_dirichlet_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_dirichlet_pdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.dirichlet.pdf"
    )


@register_op("scipy.stats.multinomial.logpmf", "jax")
def _map_jax_scipy_stats_multinomial_logpmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_multinomial_logpmf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.multinomial.logpmf"
    )


@register_op("scipy.stats.multinomial.pmf", "jax")
def _map_jax_scipy_stats_multinomial_pmf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_multinomial_pmf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.multinomial.pmf"
    )


@register_op("scipy.stats.uniform.logpdf", "jax")
def _map_jax_scipy_stats_uniform_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_uniform_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.uniform.logpdf"
    )


@register_op("scipy.stats.uniform.pdf", "jax")
def _map_jax_scipy_stats_uniform_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_uniform_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.uniform.pdf")


@register_op("scipy.stats.uniform.cdf", "jax")
def _map_jax_scipy_stats_uniform_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_uniform_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.uniform.cdf")


@register_op("scipy.stats.uniform.ppf", "jax")
def _map_jax_scipy_stats_uniform_ppf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_uniform_ppf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.uniform.ppf")


@register_op("scipy.stats.logistic.cdf", "jax")
def _map_jax_scipy_stats_logistic_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_logistic_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.logistic.cdf")


@register_op("scipy.stats.logistic.isf", "jax")
def _map_jax_scipy_stats_logistic_isf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_logistic_isf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.logistic.isf")


@register_op("scipy.stats.logistic.logpdf", "jax")
def _map_jax_scipy_stats_logistic_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_logistic_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.logistic.logpdf"
    )


@register_op("scipy.stats.logistic.pdf", "jax")
def _map_jax_scipy_stats_logistic_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_logistic_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.logistic.pdf")


@register_op("scipy.stats.logistic.ppf", "jax")
def _map_jax_scipy_stats_logistic_ppf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_logistic_ppf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.logistic.ppf")


@register_op("scipy.stats.logistic.sf", "jax")
def _map_jax_scipy_stats_logistic_sf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_logistic_sf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.logistic.sf")


@register_op("scipy.stats.pareto.logcdf", "jax")
def _map_jax_scipy_stats_pareto_logcdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_pareto_logcdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.pareto.logcdf"
    )


@register_op("scipy.stats.pareto.logpdf", "jax")
def _map_jax_scipy_stats_pareto_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_pareto_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.pareto.logpdf"
    )


@register_op("scipy.stats.pareto.logsf", "jax")
def _map_jax_scipy_stats_pareto_logsf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_pareto_logsf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.pareto.logsf")


@register_op("scipy.stats.pareto.cdf", "jax")
def _map_jax_scipy_stats_pareto_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_pareto_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.pareto.cdf")


@register_op("scipy.stats.pareto.pdf", "jax")
def _map_jax_scipy_stats_pareto_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_pareto_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.pareto.pdf")


@register_op("scipy.stats.pareto.ppf", "jax")
def _map_jax_scipy_stats_pareto_ppf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_pareto_ppf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.pareto.ppf")


@register_op("scipy.stats.pareto.sf", "jax")
def _map_jax_scipy_stats_pareto_sf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_pareto_sf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.pareto.sf")


@register_op("scipy.stats.norm.cdf", "jax")
def _map_jax_scipy_stats_norm_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_norm_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.norm.cdf")


@register_op("scipy.stats.norm.logcdf", "jax")
def _map_jax_scipy_stats_norm_logcdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_norm_logcdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.norm.logcdf")


@register_op("scipy.stats.norm.logpdf", "jax")
def _map_jax_scipy_stats_norm_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_norm_logpdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.norm.logpdf")


@register_op("scipy.stats.norm.logsf", "jax")
def _map_jax_scipy_stats_norm_logsf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_norm_logsf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.norm.logsf")


@register_op("scipy.stats.norm.pdf", "jax")
def _map_jax_scipy_stats_norm_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_norm_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.norm.pdf")


@register_op("scipy.stats.norm.ppf", "jax")
def _map_jax_scipy_stats_norm_ppf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_norm_ppf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.norm.ppf")


@register_op("scipy.stats.norm.sf", "jax")
def _map_jax_scipy_stats_norm_sf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_norm_sf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.norm.sf")


@register_op("scipy.stats.norm.isf", "jax")
def _map_jax_scipy_stats_norm_isf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_norm_isf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.norm.isf")


@register_op("scipy.stats.cauchy.cdf", "jax")
def _map_jax_scipy_stats_cauchy_cdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_cauchy_cdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.cauchy.cdf")


@register_op("scipy.stats.cauchy.isf", "jax")
def _map_jax_scipy_stats_cauchy_isf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_cauchy_isf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.cauchy.isf")


@register_op("scipy.stats.cauchy.logcdf", "jax")
def _map_jax_scipy_stats_cauchy_logcdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_cauchy_logcdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.cauchy.logcdf"
    )


@register_op("scipy.stats.cauchy.logpdf", "jax")
def _map_jax_scipy_stats_cauchy_logpdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_cauchy_logpdf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.cauchy.logpdf"
    )


@register_op("scipy.stats.cauchy.logsf", "jax")
def _map_jax_scipy_stats_cauchy_logsf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_cauchy_logsf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.cauchy.logsf")


@register_op("scipy.stats.cauchy.pdf", "jax")
def _map_jax_scipy_stats_cauchy_pdf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_cauchy_pdf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.cauchy.pdf")


@register_op("scipy.stats.cauchy.ppf", "jax")
def _map_jax_scipy_stats_cauchy_ppf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_cauchy_ppf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.cauchy.ppf")


@register_op("scipy.stats.cauchy.sf", "jax")
def _map_jax_scipy_stats_cauchy_sf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_stats_cauchy_sf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.stats.cauchy.sf")


@register_op("scipy.cluster.vq.vq", "jax")
def _map_jax_scipy_cluster_vq_vq(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_cluster_vq_vq operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.cluster.vq.vq")


@register_op("scipy.sparse.linalg.cg", "jax")
def _map_jax_scipy_sparse_linalg_cg(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_sparse_linalg_cg operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.sparse.linalg.cg")


@register_op("scipy.sparse.linalg.gmres", "jax")
def _map_jax_scipy_sparse_linalg_gmres(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_sparse_linalg_gmres operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.sparse.linalg.gmres"
    )


@register_op("scipy.sparse.linalg.bicgstab", "jax")
def _map_jax_scipy_sparse_linalg_bicgstab(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_scipy_sparse_linalg_bicgstab operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="scipy.sparse.linalg.bicgstab"
    )


@register_op("lax.DotDimensionNumbers", "jax")
def _map_jax_lax_DotDimensionNumbers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_DotDimensionNumbers operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.DotDimensionNumbers")


@register_op("lax.RaggedDotDimensionNumbers", "jax")
def _map_jax_lax_RaggedDotDimensionNumbers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_RaggedDotDimensionNumbers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.RaggedDotDimensionNumbers"
    )


@register_op("lax.AccuracyMode", "jax")
def _map_jax_lax_AccuracyMode(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_AccuracyMode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.AccuracyMode")


@register_op("lax.Tolerance", "jax")
def _map_jax_lax_Tolerance(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_Tolerance operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.Tolerance")


@register_op("lax.Precision", "jax")
def _map_jax_lax_Precision(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_Precision operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.Precision")


@register_op("lax.PrecisionLike", "jax")
def _map_jax_lax_PrecisionLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_PrecisionLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.PrecisionLike")


@register_op("lax.DotAlgorithm", "jax")
def _map_jax_lax_DotAlgorithm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_DotAlgorithm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.DotAlgorithm")


@register_op("lax.DotAlgorithmPreset", "jax")
def _map_jax_lax_DotAlgorithmPreset(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_DotAlgorithmPreset operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.DotAlgorithmPreset")


@register_op("lax.RandomAlgorithm", "jax")
def _map_jax_lax_RandomAlgorithm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_RandomAlgorithm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.RandomAlgorithm")


@register_op("lax.RoundingMethod", "jax")
def _map_jax_lax_RoundingMethod(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_RoundingMethod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.RoundingMethod")


@register_op("lax.abs", "jax")
def _map_jax_lax_abs(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_abs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.abs")


@register_op("lax.abs_p", "jax")
def _map_jax_lax_abs_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_abs_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.abs_p")


@register_op("lax.acos", "jax")
def _map_jax_lax_acos(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_acos operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.acos")


@register_op("lax.acos_p", "jax")
def _map_jax_lax_acos_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_acos_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.acos_p")


@register_op("lax.acosh", "jax")
def _map_jax_lax_acosh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_acosh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.acosh")


@register_op("lax.acosh_p", "jax")
def _map_jax_lax_acosh_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_acosh_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.acosh_p")


@register_op("lax.add", "jax")
def _map_jax_lax_add(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_add operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.add")


@register_op("lax.add_p", "jax")
def _map_jax_lax_add_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_add_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.add_p")


@register_op("lax.after_all", "jax")
def _map_jax_lax_after_all(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_after_all operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.after_all")


@register_op("lax.after_all_p", "jax")
def _map_jax_lax_after_all_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_after_all_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.after_all_p")


@register_op("lax.and_p", "jax")
def _map_jax_lax_and_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_and_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.and_p")


@register_op("lax.argmax", "jax")
def _map_jax_lax_argmax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_argmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.argmax")


@register_op("lax.argmax_p", "jax")
def _map_jax_lax_argmax_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_argmax_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.argmax_p")


@register_op("lax.argmin", "jax")
def _map_jax_lax_argmin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_argmin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.argmin")


@register_op("lax.argmin_p", "jax")
def _map_jax_lax_argmin_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_argmin_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.argmin_p")


@register_op("lax.asin", "jax")
def _map_jax_lax_asin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_asin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.asin")


@register_op("lax.asin_p", "jax")
def _map_jax_lax_asin_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_asin_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.asin_p")


@register_op("lax.asinh", "jax")
def _map_jax_lax_asinh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_asinh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.asinh")


@register_op("lax.asinh_p", "jax")
def _map_jax_lax_asinh_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_asinh_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.asinh_p")


@register_op("lax.atan", "jax")
def _map_jax_lax_atan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_atan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.atan")


@register_op("lax.atan_p", "jax")
def _map_jax_lax_atan_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_atan_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.atan_p")


@register_op("lax.atan2", "jax")
def _map_jax_lax_atan2(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_atan2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.atan2")


@register_op("lax.atan2_p", "jax")
def _map_jax_lax_atan2_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_atan2_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.atan2_p")


@register_op("lax.atanh", "jax")
def _map_jax_lax_atanh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_atanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.atanh")


@register_op("lax.atanh_p", "jax")
def _map_jax_lax_atanh_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_atanh_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.atanh_p")


@register_op("lax.batch_matmul", "jax")
def _map_jax_lax_batch_matmul(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_batch_matmul operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.batch_matmul")


@register_op("lax.bitcast_convert_type", "jax")
def _map_jax_lax_bitcast_convert_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_bitcast_convert_type operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.bitcast_convert_type")


@register_op("lax.bitcast_convert_type_p", "jax")
def _map_jax_lax_bitcast_convert_type_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_bitcast_convert_type_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.bitcast_convert_type_p"
    )


@register_op("lax.bitwise_and", "jax")
def _map_jax_lax_bitwise_and(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_bitwise_and operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.bitwise_and")


@register_op("lax.bitwise_not", "jax")
def _map_jax_lax_bitwise_not(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_bitwise_not operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.bitwise_not")


@register_op("lax.bitwise_or", "jax")
def _map_jax_lax_bitwise_or(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_bitwise_or operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.bitwise_or")


@register_op("lax.bitwise_xor", "jax")
def _map_jax_lax_bitwise_xor(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_bitwise_xor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.bitwise_xor")


@register_op("lax.broadcast", "jax")
def _map_jax_lax_broadcast(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_broadcast operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.broadcast")


@register_op("lax.broadcast_in_dim", "jax")
def _map_jax_lax_broadcast_in_dim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_broadcast_in_dim operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.broadcast_in_dim")


@register_op("lax.broadcast_in_dim_p", "jax")
def _map_jax_lax_broadcast_in_dim_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_broadcast_in_dim_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.broadcast_in_dim_p")


@register_op("lax.broadcast_shapes", "jax")
def _map_jax_lax_broadcast_shapes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_broadcast_shapes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.broadcast_shapes")


@register_op("lax.broadcast_to_rank", "jax")
def _map_jax_lax_broadcast_to_rank(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_broadcast_to_rank operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.broadcast_to_rank")


@register_op("lax.broadcasted_iota", "jax")
def _map_jax_lax_broadcasted_iota(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_broadcasted_iota operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.broadcasted_iota")


@register_op("lax.cbrt", "jax")
def _map_jax_lax_cbrt(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cbrt operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cbrt")


@register_op("lax.cbrt_p", "jax")
def _map_jax_lax_cbrt_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cbrt_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cbrt_p")


@register_op("lax.ceil", "jax")
def _map_jax_lax_ceil(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_ceil operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ceil")


@register_op("lax.ceil_p", "jax")
def _map_jax_lax_ceil_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_ceil_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ceil_p")


@register_op("lax.clamp", "jax")
def _map_jax_lax_clamp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_clamp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.clamp")


@register_op("lax.clamp_p", "jax")
def _map_jax_lax_clamp_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_clamp_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.clamp_p")


@register_op("lax.clz", "jax")
def _map_jax_lax_clz(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_clz operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.clz")


@register_op("lax.clz_p", "jax")
def _map_jax_lax_clz_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_clz_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.clz_p")


@register_op("lax.collapse", "jax")
def _map_jax_lax_collapse(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_collapse operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.collapse")


@register_op("lax.complex", "jax")
def _map_jax_lax_complex(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_complex operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.complex")


@register_op("lax.complex_p", "jax")
def _map_jax_lax_complex_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_complex_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.complex_p")


@register_op("lax.composite", "jax")
def _map_jax_lax_composite(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_composite operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.composite")


@register_op("lax.concatenate", "jax")
def _map_jax_lax_concatenate(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_concatenate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.concatenate")


@register_op("lax.concatenate_p", "jax")
def _map_jax_lax_concatenate_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_concatenate_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.concatenate_p")


@register_op("lax.conj", "jax")
def _map_jax_lax_conj(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_conj operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conj")


@register_op("lax.conj_p", "jax")
def _map_jax_lax_conj_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_conj_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conj_p")


@register_op("lax.convert_element_type", "jax")
def _map_jax_lax_convert_element_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_convert_element_type operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.convert_element_type")


@register_op("lax.convert_element_type_p", "jax")
def _map_jax_lax_convert_element_type_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_convert_element_type_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.convert_element_type_p"
    )


@register_op("lax.copy_p", "jax")
def _map_jax_lax_copy_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_copy_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.copy_p")


@register_op("lax.cos", "jax")
def _map_jax_lax_cos(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cos operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cos")


@register_op("lax.dce_sink_p", "jax")
def _map_jax_lax_dce_sink_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_dce_sink_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dce_sink_p")


@register_op("lax.dce_sink", "jax")
def _map_jax_lax_dce_sink(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_dce_sink operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dce_sink")


@register_op("lax.cos_p", "jax")
def _map_jax_lax_cos_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cos_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cos_p")


@register_op("lax.cosh", "jax")
def _map_jax_lax_cosh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cosh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cosh")


@register_op("lax.cosh_p", "jax")
def _map_jax_lax_cosh_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cosh_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cosh_p")


@register_op("lax.create_token", "jax")
def _map_jax_lax_create_token(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_create_token operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.create_token")


@register_op("lax.create_token_p", "jax")
def _map_jax_lax_create_token_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_create_token_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.create_token_p")


@register_op("lax.div", "jax")
def _map_jax_lax_div(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_div operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.div")


@register_op("lax.div_p", "jax")
def _map_jax_lax_div_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_div_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.div_p")


@register_op("lax.dot", "jax")
def _map_jax_lax_dot(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_dot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dot")


@register_op("lax.dot_general", "jax")
def _map_jax_lax_dot_general(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_dot_general operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dot_general")


@register_op("lax.dot_general_p", "jax")
def _map_jax_lax_dot_general_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_dot_general_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dot_general_p")


@register_op("lax.dtype", "jax")
def _map_jax_lax_dtype(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dtype")


@register_op("lax.eq", "jax")
def _map_jax_lax_eq(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_eq operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.eq")


@register_op("lax.eq_p", "jax")
def _map_jax_lax_eq_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_eq_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.eq_p")


@register_op("lax.eq_to_p", "jax")
def _map_jax_lax_eq_to_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_eq_to_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.eq_to_p")


@register_op("lax.exp", "jax")
def _map_jax_lax_exp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_exp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.exp")


@register_op("lax.exp_p", "jax")
def _map_jax_lax_exp_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_exp_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.exp_p")


@register_op("lax.exp2", "jax")
def _map_jax_lax_exp2(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_exp2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.exp2")


@register_op("lax.exp2_p", "jax")
def _map_jax_lax_exp2_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_exp2_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.exp2_p")


@register_op("lax.expand_dims", "jax")
def _map_jax_lax_expand_dims(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_expand_dims operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.expand_dims")


@register_op("lax.expm1", "jax")
def _map_jax_lax_expm1(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_expm1 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.expm1")


@register_op("lax.expm1_p", "jax")
def _map_jax_lax_expm1_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_expm1_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.expm1_p")


@register_op("lax.floor", "jax")
def _map_jax_lax_floor(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_floor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.floor")


@register_op("lax.floor_p", "jax")
def _map_jax_lax_floor_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_floor_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.floor_p")


@register_op("lax.full", "jax")
def _map_jax_lax_full(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_full operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.full")


@register_op("lax.full_like", "jax")
def _map_jax_lax_full_like(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_full_like operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.full_like")


@register_op("lax.ge", "jax")
def _map_jax_lax_ge(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_ge operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ge")


@register_op("lax.ge_p", "jax")
def _map_jax_lax_ge_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_ge_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ge_p")


@register_op("lax.gt", "jax")
def _map_jax_lax_gt(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_gt operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.gt")


@register_op("lax.gt_p", "jax")
def _map_jax_lax_gt_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_gt_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.gt_p")


@register_op("lax.imag", "jax")
def _map_jax_lax_imag(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_imag operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.imag")


@register_op("lax.imag_p", "jax")
def _map_jax_lax_imag_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_imag_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.imag_p")


@register_op("lax.integer_pow", "jax")
def _map_jax_lax_integer_pow(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_integer_pow operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.integer_pow")


@register_op("lax.integer_pow_p", "jax")
def _map_jax_lax_integer_pow_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_integer_pow_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.integer_pow_p")


@register_op("lax.iota", "jax")
def _map_jax_lax_iota(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_iota operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.iota")


@register_op("lax.iota_p", "jax")
def _map_jax_lax_iota_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_iota_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.iota_p")


@register_op("lax.is_finite", "jax")
def _map_jax_lax_is_finite(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_is_finite operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.is_finite")


@register_op("lax.is_finite_p", "jax")
def _map_jax_lax_is_finite_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_is_finite_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.is_finite_p")


@register_op("lax.le", "jax")
def _map_jax_lax_le(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_le operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.le")


@register_op("lax.le_p", "jax")
def _map_jax_lax_le_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_le_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.le_p")


@register_op("lax.le_to_p", "jax")
def _map_jax_lax_le_to_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_le_to_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.le_to_p")


@register_op("lax.log", "jax")
def _map_jax_lax_log(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_log operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.log")


@register_op("lax.log1p", "jax")
def _map_jax_lax_log1p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_log1p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.log1p")


@register_op("lax.log1p_p", "jax")
def _map_jax_lax_log1p_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_log1p_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.log1p_p")


@register_op("lax.log_p", "jax")
def _map_jax_lax_log_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_log_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.log_p")


@register_op("lax.logistic", "jax")
def _map_jax_lax_logistic(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_logistic operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.logistic")


@register_op("lax.logistic_p", "jax")
def _map_jax_lax_logistic_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_logistic_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.logistic_p")


@register_op("lax.lt", "jax")
def _map_jax_lax_lt(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_lt operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.lt")


@register_op("lax.lt_p", "jax")
def _map_jax_lax_lt_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_lt_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.lt_p")


@register_op("lax.lt_to_p", "jax")
def _map_jax_lax_lt_to_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_lt_to_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.lt_to_p")


@register_op("lax.max", "jax")
def _map_jax_lax_max(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_max operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.max")


@register_op("lax.max_p", "jax")
def _map_jax_lax_max_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_max_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.max_p")


@register_op("lax.min", "jax")
def _map_jax_lax_min(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_min operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.min")


@register_op("lax.min_p", "jax")
def _map_jax_lax_min_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_min_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.min_p")


@register_op("lax.mul", "jax")
def _map_jax_lax_mul(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_mul operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.mul")


@register_op("lax.mul_p", "jax")
def _map_jax_lax_mul_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_mul_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.mul_p")


@register_op("lax.ne", "jax")
def _map_jax_lax_ne(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_ne operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ne")


@register_op("lax.ne_p", "jax")
def _map_jax_lax_ne_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_ne_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ne_p")


@register_op("lax.neg", "jax")
def _map_jax_lax_neg(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_neg operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.neg")


@register_op("lax.neg_p", "jax")
def _map_jax_lax_neg_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_neg_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.neg_p")


@register_op("lax.nextafter", "jax")
def _map_jax_lax_nextafter(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_nextafter operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.nextafter")


@register_op("lax.nextafter_p", "jax")
def _map_jax_lax_nextafter_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_nextafter_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.nextafter_p")


@register_op("lax.not_p", "jax")
def _map_jax_lax_not_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_not_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.not_p")


@register_op("lax.optimization_barrier", "jax")
def _map_jax_lax_optimization_barrier(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_optimization_barrier operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.optimization_barrier")


@register_op("lax.optimization_barrier_p", "jax")
def _map_jax_lax_optimization_barrier_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_optimization_barrier_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.optimization_barrier_p"
    )


@register_op("lax.or_p", "jax")
def _map_jax_lax_or_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_or_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.or_p")


@register_op("lax.pad", "jax")
def _map_jax_lax_pad(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pad")


@register_op("lax.pad_p", "jax")
def _map_jax_lax_pad_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pad_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pad_p")


@register_op("lax.padtype_to_pads", "jax")
def _map_jax_lax_padtype_to_pads(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_padtype_to_pads operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.padtype_to_pads")


@register_op("lax.population_count", "jax")
def _map_jax_lax_population_count(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_population_count operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.population_count")


@register_op("lax.population_count_p", "jax")
def _map_jax_lax_population_count_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_population_count_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.population_count_p")


@register_op("lax.pow", "jax")
def _map_jax_lax_pow(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pow operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pow")


@register_op("lax.pow_p", "jax")
def _map_jax_lax_pow_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pow_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pow_p")


@register_op("lax.ragged_dot", "jax")
def _map_jax_lax_ragged_dot(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_ragged_dot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ragged_dot")


@register_op("lax.ragged_dot_general", "jax")
def _map_jax_lax_ragged_dot_general(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_ragged_dot_general operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ragged_dot_general")


@register_op("lax.real", "jax")
def _map_jax_lax_real(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_real operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.real")


@register_op("lax.real_p", "jax")
def _map_jax_lax_real_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_real_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.real_p")


@register_op("lax.reciprocal", "jax")
def _map_jax_lax_reciprocal(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reciprocal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reciprocal")


@register_op("lax.reduce", "jax")
def _map_jax_lax_reduce(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reduce operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce")


@register_op("lax.reduce_and", "jax")
def _map_jax_lax_reduce_and(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reduce_and operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_and")


@register_op("lax.reduce_and_p", "jax")
def _map_jax_lax_reduce_and_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_and_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_and_p")


@register_op("lax.reduce_max", "jax")
def _map_jax_lax_reduce_max(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reduce_max operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_max")


@register_op("lax.reduce_max_p", "jax")
def _map_jax_lax_reduce_max_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_max_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_max_p")


@register_op("lax.reduce_min", "jax")
def _map_jax_lax_reduce_min(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reduce_min operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_min")


@register_op("lax.reduce_min_p", "jax")
def _map_jax_lax_reduce_min_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_min_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_min_p")


@register_op("lax.reduce_or", "jax")
def _map_jax_lax_reduce_or(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reduce_or operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_or")


@register_op("lax.reduce_or_p", "jax")
def _map_jax_lax_reduce_or_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reduce_or_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_or_p")


@register_op("lax.reduce_p", "jax")
def _map_jax_lax_reduce_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reduce_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_p")


@register_op("lax.reduce_precision", "jax")
def _map_jax_lax_reduce_precision(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_precision operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_precision")


@register_op("lax.reduce_precision_p", "jax")
def _map_jax_lax_reduce_precision_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_precision_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_precision_p")


@register_op("lax.reduce_prod", "jax")
def _map_jax_lax_reduce_prod(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reduce_prod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_prod")


@register_op("lax.reduce_prod_p", "jax")
def _map_jax_lax_reduce_prod_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_prod_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_prod_p")


@register_op("lax.reduce_sum", "jax")
def _map_jax_lax_reduce_sum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reduce_sum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_sum")


@register_op("lax.reduce_sum_p", "jax")
def _map_jax_lax_reduce_sum_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_sum_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_sum_p")


@register_op("lax.reduce_xor", "jax")
def _map_jax_lax_reduce_xor(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reduce_xor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_xor")


@register_op("lax.reduce_xor_p", "jax")
def _map_jax_lax_reduce_xor_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_xor_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_xor_p")


@register_op("lax.rem", "jax")
def _map_jax_lax_rem(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_rem operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.rem")


@register_op("lax.rem_p", "jax")
def _map_jax_lax_rem_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_rem_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.rem_p")


@register_op("lax.reshape", "jax")
def _map_jax_lax_reshape(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reshape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reshape")


@register_op("lax.reshape_p", "jax")
def _map_jax_lax_reshape_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_reshape_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reshape_p")


@register_op("lax.rev", "jax")
def _map_jax_lax_rev(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_rev operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.rev")


@register_op("lax.rev_p", "jax")
def _map_jax_lax_rev_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_rev_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.rev_p")


@register_op("lax.rng_bit_generator", "jax")
def _map_jax_lax_rng_bit_generator(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_rng_bit_generator operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.rng_bit_generator")


@register_op("lax.rng_bit_generator_p", "jax")
def _map_jax_lax_rng_bit_generator_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_rng_bit_generator_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.rng_bit_generator_p")


@register_op("lax.rng_uniform", "jax")
def _map_jax_lax_rng_uniform(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_rng_uniform operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.rng_uniform")


@register_op("lax.rng_uniform_p", "jax")
def _map_jax_lax_rng_uniform_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_rng_uniform_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.rng_uniform_p")


@register_op("lax.round", "jax")
def _map_jax_lax_round(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_round operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.round")


@register_op("lax.round_p", "jax")
def _map_jax_lax_round_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_round_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.round_p")


@register_op("lax.rsqrt", "jax")
def _map_jax_lax_rsqrt(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_rsqrt operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.rsqrt")


@register_op("lax.rsqrt_p", "jax")
def _map_jax_lax_rsqrt_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_rsqrt_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.rsqrt_p")


@register_op("lax.select", "jax")
def _map_jax_lax_select(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_select operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.select")


@register_op("lax.select_n", "jax")
def _map_jax_lax_select_n(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_select_n operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.select_n")


@register_op("lax.select_n_p", "jax")
def _map_jax_lax_select_n_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_select_n_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.select_n_p")


@register_op("lax.shape_as_value", "jax")
def _map_jax_lax_shape_as_value(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_shape_as_value operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.shape_as_value")


@register_op("lax.shift_left", "jax")
def _map_jax_lax_shift_left(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_shift_left operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.shift_left")


@register_op("lax.shift_left_p", "jax")
def _map_jax_lax_shift_left_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_shift_left_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.shift_left_p")


@register_op("lax.shift_right_arithmetic", "jax")
def _map_jax_lax_shift_right_arithmetic(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_shift_right_arithmetic operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.shift_right_arithmetic"
    )


@register_op("lax.shift_right_arithmetic_p", "jax")
def _map_jax_lax_shift_right_arithmetic_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_shift_right_arithmetic_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.shift_right_arithmetic_p"
    )


@register_op("lax.shift_right_logical", "jax")
def _map_jax_lax_shift_right_logical(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_shift_right_logical operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.shift_right_logical")


@register_op("lax.shift_right_logical_p", "jax")
def _map_jax_lax_shift_right_logical_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_shift_right_logical_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.shift_right_logical_p"
    )


@register_op("lax.sign", "jax")
def _map_jax_lax_sign(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sign operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sign")


@register_op("lax.sign_p", "jax")
def _map_jax_lax_sign_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sign_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sign_p")


@register_op("lax.sin", "jax")
def _map_jax_lax_sin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sin")


@register_op("lax.sin_p", "jax")
def _map_jax_lax_sin_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sin_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sin_p")


@register_op("lax.sinh", "jax")
def _map_jax_lax_sinh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sinh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sinh")


@register_op("lax.sinh_p", "jax")
def _map_jax_lax_sinh_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sinh_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sinh_p")


@register_op("lax.sort", "jax")
def _map_jax_lax_sort(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sort operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sort")


@register_op("lax.sort_key_val", "jax")
def _map_jax_lax_sort_key_val(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_sort_key_val operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sort_key_val")


@register_op("lax.sort_p", "jax")
def _map_jax_lax_sort_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sort_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sort_p")


@register_op("lax.split", "jax")
def _map_jax_lax_split(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_split operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.split")


@register_op("lax.split_p", "jax")
def _map_jax_lax_split_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_split_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.split_p")


@register_op("lax.sqrt", "jax")
def _map_jax_lax_sqrt(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sqrt operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sqrt")


@register_op("lax.sqrt_p", "jax")
def _map_jax_lax_sqrt_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sqrt_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sqrt_p")


@register_op("lax.square", "jax")
def _map_jax_lax_square(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_square operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.square")


@register_op("lax.square_p", "jax")
def _map_jax_lax_square_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_square_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.square_p")


@register_op("lax.squeeze", "jax")
def _map_jax_lax_squeeze(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_squeeze operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.squeeze")


@register_op("lax.squeeze_p", "jax")
def _map_jax_lax_squeeze_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_squeeze_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.squeeze_p")


@register_op("lax.stop_gradient", "jax")
def _map_jax_lax_stop_gradient(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_stop_gradient operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.stop_gradient")


@register_op("lax.sub", "jax")
def _map_jax_lax_sub(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sub operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sub")


@register_op("lax.sub_p", "jax")
def _map_jax_lax_sub_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_sub_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sub_p")


@register_op("lax.tan", "jax")
def _map_jax_lax_tan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_tan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.tan")


@register_op("lax.tan_p", "jax")
def _map_jax_lax_tan_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_tan_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.tan_p")


@register_op("lax.tanh", "jax")
def _map_jax_lax_tanh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_tanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.tanh")


@register_op("lax.tanh_p", "jax")
def _map_jax_lax_tanh_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_tanh_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.tanh_p")


@register_op("lax.tile", "jax")
def _map_jax_lax_tile(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_tile operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.tile")


@register_op("lax.tile_p", "jax")
def _map_jax_lax_tile_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_tile_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.tile_p")


@register_op("lax.top_k", "jax")
def _map_jax_lax_top_k(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_top_k operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.top_k")


@register_op("lax.top_k_p", "jax")
def _map_jax_lax_top_k_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_top_k_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.top_k_p")


@register_op("lax.transpose", "jax")
def _map_jax_lax_transpose(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_transpose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.transpose")


@register_op("lax.transpose_p", "jax")
def _map_jax_lax_transpose_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_transpose_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.transpose_p")


@register_op("lax.xor_p", "jax")
def _map_jax_lax_xor_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_xor_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.xor_p")


@register_op("lax.empty", "jax")
def _map_jax_lax_empty(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_empty operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.empty")


@register_op("lax.bessel_i0e", "jax")
def _map_jax_lax_bessel_i0e(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_bessel_i0e operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.bessel_i0e")


@register_op("lax.bessel_i0e_p", "jax")
def _map_jax_lax_bessel_i0e_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_bessel_i0e_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.bessel_i0e_p")


@register_op("lax.bessel_i1e", "jax")
def _map_jax_lax_bessel_i1e(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_bessel_i1e operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.bessel_i1e")


@register_op("lax.bessel_i1e_p", "jax")
def _map_jax_lax_bessel_i1e_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_bessel_i1e_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.bessel_i1e_p")


@register_op("lax.betainc", "jax")
def _map_jax_lax_betainc(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_betainc operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.betainc")


@register_op("lax.digamma", "jax")
def _map_jax_lax_digamma(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_digamma operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.digamma")


@register_op("lax.digamma_p", "jax")
def _map_jax_lax_digamma_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_digamma_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.digamma_p")


@register_op("lax.erf", "jax")
def _map_jax_lax_erf(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_erf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.erf")


@register_op("lax.erfc", "jax")
def _map_jax_lax_erfc(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_erfc operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.erfc")


@register_op("lax.erfc_p", "jax")
def _map_jax_lax_erfc_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_erfc_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.erfc_p")


@register_op("lax.erf_inv", "jax")
def _map_jax_lax_erf_inv(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_erf_inv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.erf_inv")


@register_op("lax.erf_inv_p", "jax")
def _map_jax_lax_erf_inv_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_erf_inv_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.erf_inv_p")


@register_op("lax.erf_p", "jax")
def _map_jax_lax_erf_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_erf_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.erf_p")


@register_op("lax.igamma", "jax")
def _map_jax_lax_igamma(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_igamma operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.igamma")


@register_op("lax.igammac", "jax")
def _map_jax_lax_igammac(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_igammac operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.igammac")


@register_op("lax.igammac_p", "jax")
def _map_jax_lax_igammac_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_igammac_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.igammac_p")


@register_op("lax.igamma_grad_a", "jax")
def _map_jax_lax_igamma_grad_a(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_igamma_grad_a operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.igamma_grad_a")


@register_op("lax.igamma_grad_a_p", "jax")
def _map_jax_lax_igamma_grad_a_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_igamma_grad_a_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.igamma_grad_a_p")


@register_op("lax.igamma_p", "jax")
def _map_jax_lax_igamma_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_igamma_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.igamma_p")


@register_op("lax.lgamma", "jax")
def _map_jax_lax_lgamma(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_lgamma operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.lgamma")


@register_op("lax.lgamma_p", "jax")
def _map_jax_lax_lgamma_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_lgamma_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.lgamma_p")


@register_op("lax.polygamma", "jax")
def _map_jax_lax_polygamma(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_polygamma operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.polygamma")


@register_op("lax.polygamma_p", "jax")
def _map_jax_lax_polygamma_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_polygamma_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.polygamma_p")


@register_op("lax.random_gamma_grad", "jax")
def _map_jax_lax_random_gamma_grad(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_random_gamma_grad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.random_gamma_grad")


@register_op("lax.regularized_incomplete_beta_p", "jax")
def _map_jax_lax_regularized_incomplete_beta_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_regularized_incomplete_beta_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.regularized_incomplete_beta_p"
    )


@register_op("lax.zeta", "jax")
def _map_jax_lax_zeta(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_zeta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.zeta")


@register_op("lax.zeta_p", "jax")
def _map_jax_lax_zeta_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_zeta_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.zeta_p")


@register_op("lax.GatherDimensionNumbers", "jax")
def _map_jax_lax_GatherDimensionNumbers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_GatherDimensionNumbers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.GatherDimensionNumbers"
    )


@register_op("lax.GatherScatterMode", "jax")
def _map_jax_lax_GatherScatterMode(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_GatherScatterMode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.GatherScatterMode")


@register_op("lax.ScatterDimensionNumbers", "jax")
def _map_jax_lax_ScatterDimensionNumbers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_ScatterDimensionNumbers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ScatterDimensionNumbers"
    )


@register_op("lax.dynamic_index_in_dim", "jax")
def _map_jax_lax_dynamic_index_in_dim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_dynamic_index_in_dim operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dynamic_index_in_dim")


@register_op("lax.dynamic_slice", "jax")
def _map_jax_lax_dynamic_slice(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_dynamic_slice operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dynamic_slice")


@register_op("lax.dynamic_slice_in_dim", "jax")
def _map_jax_lax_dynamic_slice_in_dim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_dynamic_slice_in_dim operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dynamic_slice_in_dim")


@register_op("lax.dynamic_slice_p", "jax")
def _map_jax_lax_dynamic_slice_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_dynamic_slice_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dynamic_slice_p")


@register_op("lax.dynamic_update_index_in_dim", "jax")
def _map_jax_lax_dynamic_update_index_in_dim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_dynamic_update_index_in_dim operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dynamic_update_index_in_dim"
    )


@register_op("lax.dynamic_update_slice", "jax")
def _map_jax_lax_dynamic_update_slice(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_dynamic_update_slice operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dynamic_update_slice")


@register_op("lax.dynamic_update_slice_in_dim", "jax")
def _map_jax_lax_dynamic_update_slice_in_dim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_dynamic_update_slice_in_dim operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dynamic_update_slice_in_dim"
    )


@register_op("lax.dynamic_update_slice_p", "jax")
def _map_jax_lax_dynamic_update_slice_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_dynamic_update_slice_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.dynamic_update_slice_p"
    )


@register_op("lax.gather", "jax")
def _map_jax_lax_gather(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_gather operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.gather")


@register_op("lax.gather_p", "jax")
def _map_jax_lax_gather_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_gather_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.gather_p")


@register_op("lax.index_in_dim", "jax")
def _map_jax_lax_index_in_dim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_index_in_dim operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.index_in_dim")


@register_op("lax.index_take", "jax")
def _map_jax_lax_index_take(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_index_take operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.index_take")


@register_op("lax.scatter", "jax")
def _map_jax_lax_scatter(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_scatter operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter")


@register_op("lax.scatter_apply", "jax")
def _map_jax_lax_scatter_apply(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_scatter_apply operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_apply")


@register_op("lax.scatter_add", "jax")
def _map_jax_lax_scatter_add(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_scatter_add operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_add")


@register_op("lax.scatter_add_p", "jax")
def _map_jax_lax_scatter_add_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_scatter_add_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_add_p")


@register_op("lax.scatter_max", "jax")
def _map_jax_lax_scatter_max(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_scatter_max operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_max")


@register_op("lax.scatter_max_p", "jax")
def _map_jax_lax_scatter_max_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_scatter_max_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_max_p")


@register_op("lax.scatter_min", "jax")
def _map_jax_lax_scatter_min(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_scatter_min operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_min")


@register_op("lax.scatter_min_p", "jax")
def _map_jax_lax_scatter_min_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_scatter_min_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_min_p")


@register_op("lax.scatter_mul", "jax")
def _map_jax_lax_scatter_mul(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_scatter_mul operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_mul")


@register_op("lax.scatter_mul_p", "jax")
def _map_jax_lax_scatter_mul_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_scatter_mul_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_mul_p")


@register_op("lax.scatter_p", "jax")
def _map_jax_lax_scatter_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_scatter_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_p")


@register_op("lax.scatter_sub", "jax")
def _map_jax_lax_scatter_sub(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_scatter_sub operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_sub")


@register_op("lax.scatter_sub_p", "jax")
def _map_jax_lax_scatter_sub_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_scatter_sub_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scatter_sub_p")


@register_op("lax.slice", "jax")
def _map_jax_lax_slice(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_slice operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.slice")


@register_op("lax.slice_in_dim", "jax")
def _map_jax_lax_slice_in_dim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_slice_in_dim operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.slice_in_dim")


@register_op("lax.slice_p", "jax")
def _map_jax_lax_slice_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_slice_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.slice_p")


@register_op("lax.ConvDimensionNumbers", "jax")
def _map_jax_lax_ConvDimensionNumbers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_ConvDimensionNumbers operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ConvDimensionNumbers")


@register_op("lax.ConvGeneralDilatedDimensionNumbers", "jax")
def _map_jax_lax_ConvGeneralDilatedDimensionNumbers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_ConvGeneralDilatedDimensionNumbers operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="lax.ConvGeneralDilatedDimensionNumbers",
    )


@register_op("lax.conv", "jax")
def _map_jax_lax_conv(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_conv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv")


@register_op("lax.conv_dimension_numbers", "jax")
def _map_jax_lax_conv_dimension_numbers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_conv_dimension_numbers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv_dimension_numbers"
    )


@register_op("lax.conv_general_dilated", "jax")
def _map_jax_lax_conv_general_dilated(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_conv_general_dilated operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv_general_dilated")


@register_op("lax.conv_general_dilated_p", "jax")
def _map_jax_lax_conv_general_dilated_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_conv_general_dilated_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv_general_dilated_p"
    )


@register_op("lax.conv_general_permutations", "jax")
def _map_jax_lax_conv_general_permutations(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_conv_general_permutations operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv_general_permutations"
    )


@register_op("lax.conv_general_shape_tuple", "jax")
def _map_jax_lax_conv_general_shape_tuple(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_conv_general_shape_tuple operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv_general_shape_tuple"
    )


@register_op("lax.conv_shape_tuple", "jax")
def _map_jax_lax_conv_shape_tuple(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_conv_shape_tuple operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv_shape_tuple")


@register_op("lax.conv_transpose", "jax")
def _map_jax_lax_conv_transpose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_conv_transpose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv_transpose")


@register_op("lax.conv_transpose_shape_tuple", "jax")
def _map_jax_lax_conv_transpose_shape_tuple(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_conv_transpose_shape_tuple operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv_transpose_shape_tuple"
    )


@register_op("lax.conv_with_general_padding", "jax")
def _map_jax_lax_conv_with_general_padding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_conv_with_general_padding operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv_with_general_padding"
    )


@register_op("lax.reduce_window", "jax")
def _map_jax_lax_reduce_window(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_window operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_window")


@register_op("lax.reduce_window_max_p", "jax")
def _map_jax_lax_reduce_window_max_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_window_max_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_window_max_p")


@register_op("lax.reduce_window_min_p", "jax")
def _map_jax_lax_reduce_window_min_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_window_min_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_window_min_p")


@register_op("lax.reduce_window_p", "jax")
def _map_jax_lax_reduce_window_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_window_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_window_p")


@register_op("lax.reduce_window_shape_tuple", "jax")
def _map_jax_lax_reduce_window_shape_tuple(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_window_shape_tuple operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_window_shape_tuple"
    )


@register_op("lax.reduce_window_sum_p", "jax")
def _map_jax_lax_reduce_window_sum_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_reduce_window_sum_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.reduce_window_sum_p")


@register_op("lax.select_and_gather_add_p", "jax")
def _map_jax_lax_select_and_gather_add_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_select_and_gather_add_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.select_and_gather_add_p"
    )


@register_op("lax.select_and_scatter_p", "jax")
def _map_jax_lax_select_and_scatter_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_select_and_scatter_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.select_and_scatter_p")


@register_op("lax.select_and_scatter_add_p", "jax")
def _map_jax_lax_select_and_scatter_add_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_select_and_scatter_add_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.select_and_scatter_add_p"
    )


@register_op("lax.associative_scan", "jax")
def _map_jax_lax_associative_scan(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_associative_scan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.associative_scan")


@register_op("lax.cond", "jax")
def _map_jax_lax_cond(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cond operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cond")


@register_op("lax.cond_p", "jax")
def _map_jax_lax_cond_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cond_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cond_p")


@register_op("lax.cumlogsumexp", "jax")
def _map_jax_lax_cumlogsumexp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_cumlogsumexp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cumlogsumexp")


@register_op("lax.cumlogsumexp_p", "jax")
def _map_jax_lax_cumlogsumexp_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_cumlogsumexp_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cumlogsumexp_p")


@register_op("lax.cummax", "jax")
def _map_jax_lax_cummax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cummax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cummax")


@register_op("lax.cummax_p", "jax")
def _map_jax_lax_cummax_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cummax_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cummax_p")


@register_op("lax.cummin", "jax")
def _map_jax_lax_cummin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cummin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cummin")


@register_op("lax.cummin_p", "jax")
def _map_jax_lax_cummin_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cummin_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cummin_p")


@register_op("lax.cumprod", "jax")
def _map_jax_lax_cumprod(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cumprod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cumprod")


@register_op("lax.cumprod_p", "jax")
def _map_jax_lax_cumprod_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cumprod_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cumprod_p")


@register_op("lax.cumsum", "jax")
def _map_jax_lax_cumsum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cumsum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cumsum")


@register_op("lax.cumsum_p", "jax")
def _map_jax_lax_cumsum_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_cumsum_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.cumsum_p")


@register_op("lax.custom_linear_solve", "jax")
def _map_jax_lax_custom_linear_solve(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_custom_linear_solve operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.custom_linear_solve")


@register_op("lax.custom_root", "jax")
def _map_jax_lax_custom_root(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_custom_root operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.custom_root")


@register_op("lax.fori_loop", "jax")
def _map_jax_lax_fori_loop(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_fori_loop operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.fori_loop")


@register_op("lax.linear_solve_p", "jax")
def _map_jax_lax_linear_solve_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linear_solve_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linear_solve_p")


@register_op("lax.map", "jax")
def _map_jax_lax_map(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_map operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.map")


@register_op("lax.scan", "jax")
def _map_jax_lax_scan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_scan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scan")


@register_op("lax.scan_p", "jax")
def _map_jax_lax_scan_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_scan_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scan_p")


@register_op("lax.switch", "jax")
def _map_jax_lax_switch(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_switch operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.switch")


@register_op("lax.while_loop", "jax")
def _map_jax_lax_while_loop(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_while_loop operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.while_loop")


@register_op("lax.while_p", "jax")
def _map_jax_lax_while_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_while_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.while_p")


@register_op("lax.platform_dependent", "jax")
def _map_jax_lax_platform_dependent(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_platform_dependent operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.platform_dependent")


@register_op("lax.fft", "jax")
def _map_jax_lax_fft(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_fft operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.fft")


@register_op("lax.fft_p", "jax")
def _map_jax_lax_fft_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_fft_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.fft_p")


@register_op("lax.FftType", "jax")
def _map_jax_lax_FftType(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_FftType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.FftType")


@register_op("lax.all_gather", "jax")
def _map_jax_lax_all_gather(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_all_gather operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.all_gather")


@register_op("lax.all_gather_start", "jax")
def _map_jax_lax_all_gather_start(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_all_gather_start operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.all_gather_start")


@register_op("lax.all_gather_done", "jax")
def _map_jax_lax_all_gather_done(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_all_gather_done operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.all_gather_done")


@register_op("lax.pcast", "jax")
def _map_jax_lax_pcast(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pcast operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pcast")


@register_op("lax.all_gather_p", "jax")
def _map_jax_lax_all_gather_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_all_gather_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.all_gather_p")


@register_op("lax.all_to_all", "jax")
def _map_jax_lax_all_to_all(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_all_to_all operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.all_to_all")


@register_op("lax.all_to_all_p", "jax")
def _map_jax_lax_all_to_all_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_all_to_all_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.all_to_all_p")


@register_op("lax.axis_index", "jax")
def _map_jax_lax_axis_index(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_axis_index operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.axis_index")


@register_op("lax.axis_index_p", "jax")
def _map_jax_lax_axis_index_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_axis_index_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.axis_index_p")


@register_op("lax.axis_size", "jax")
def _map_jax_lax_axis_size(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_axis_size operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.axis_size")


@register_op("lax.pbroadcast", "jax")
def _map_jax_lax_pbroadcast(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pbroadcast operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pbroadcast")


@register_op("lax.pmax", "jax")
def _map_jax_lax_pmax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pmax")


@register_op("lax.pmax_p", "jax")
def _map_jax_lax_pmax_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pmax_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pmax_p")


@register_op("lax.pmean", "jax")
def _map_jax_lax_pmean(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pmean operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pmean")


@register_op("lax.pmin", "jax")
def _map_jax_lax_pmin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pmin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pmin")


@register_op("lax.pmin_p", "jax")
def _map_jax_lax_pmin_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pmin_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pmin_p")


@register_op("lax.ppermute", "jax")
def _map_jax_lax_ppermute(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_ppermute operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ppermute")


@register_op("lax.ppermute_p", "jax")
def _map_jax_lax_ppermute_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_ppermute_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ppermute_p")


@register_op("lax.psend", "jax")
def _map_jax_lax_psend(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_psend operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.psend")


@register_op("lax.precv", "jax")
def _map_jax_lax_precv(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_precv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.precv")


@register_op("lax.pshuffle", "jax")
def _map_jax_lax_pshuffle(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pshuffle operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pshuffle")


@register_op("lax.psum", "jax")
def _map_jax_lax_psum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_psum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.psum")


@register_op("lax.psum_p", "jax")
def _map_jax_lax_psum_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_psum_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.psum_p")


@register_op("lax.psum_scatter", "jax")
def _map_jax_lax_psum_scatter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_psum_scatter operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.psum_scatter")


@register_op("lax.pswapaxes", "jax")
def _map_jax_lax_pswapaxes(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pswapaxes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pswapaxes")


@register_op("lax.ragged_all_to_all", "jax")
def _map_jax_lax_ragged_all_to_all(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_ragged_all_to_all operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ragged_all_to_all")


@register_op("lax.ragged_all_to_all_p", "jax")
def _map_jax_lax_ragged_all_to_all_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_ragged_all_to_all_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.ragged_all_to_all_p")


@register_op("lax.conv_general_dilated_local", "jax")
def _map_jax_lax_conv_general_dilated_local(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_conv_general_dilated_local operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv_general_dilated_local"
    )


@register_op("lax.conv_general_dilated_patches", "jax")
def _map_jax_lax_conv_general_dilated_patches(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_conv_general_dilated_patches operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.conv_general_dilated_patches"
    )


@register_op("lax.approx_max_k", "jax")
def _map_jax_lax_approx_max_k(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_approx_max_k operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.approx_max_k")


@register_op("lax.approx_min_k", "jax")
def _map_jax_lax_approx_min_k(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_approx_min_k operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.approx_min_k")


@register_op("lax.approx_top_k_p", "jax")
def _map_jax_lax_approx_top_k_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_approx_top_k_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.approx_top_k_p")


@register_op("lax.stop_gradient_p", "jax")
def _map_jax_lax_stop_gradient_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_stop_gradient_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.stop_gradient_p")


@register_op("lax.with_sharding_constraint", "jax")
def _map_jax_lax_with_sharding_constraint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_with_sharding_constraint operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.with_sharding_constraint"
    )


@register_op("lax.sharding_constraint_p", "jax")
def _map_jax_lax_sharding_constraint_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_sharding_constraint_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.sharding_constraint_p"
    )


@register_op("lax.device_put_p", "jax")
def _map_jax_lax_device_put_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_device_put_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.device_put_p")


@register_op("lax.scaled_dot", "jax")
def _map_jax_lax_scaled_dot(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_scaled_dot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.scaled_dot")


@register_op("lax.pvary", "jax")
def _map_jax_lax_pvary(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_pvary operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.pvary")


@register_op("lax.linalg.cholesky", "jax")
def _map_jax_lax_linalg_cholesky(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_cholesky operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.cholesky")


@register_op("lax.linalg.cholesky_p", "jax")
def _map_jax_lax_linalg_cholesky_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_cholesky_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.cholesky_p")


@register_op("lax.linalg.cholesky_update", "jax")
def _map_jax_lax_linalg_cholesky_update(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_cholesky_update operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.cholesky_update"
    )


@register_op("lax.linalg.cholesky_update_p", "jax")
def _map_jax_lax_linalg_cholesky_update_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_cholesky_update_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.cholesky_update_p"
    )


@register_op("lax.linalg.EigImplementation", "jax")
def _map_jax_lax_linalg_EigImplementation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_EigImplementation operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.EigImplementation"
    )


@register_op("lax.linalg.eig", "jax")
def _map_jax_lax_linalg_eig(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_linalg_eig operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.eig")


@register_op("lax.linalg.eig_p", "jax")
def _map_jax_lax_linalg_eig_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_eig_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.eig_p")


@register_op("lax.linalg.eigh", "jax")
def _map_jax_lax_linalg_eigh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_linalg_eigh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.eigh")


@register_op("lax.linalg.EighImplementation", "jax")
def _map_jax_lax_linalg_EighImplementation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_EighImplementation operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.EighImplementation"
    )


@register_op("lax.linalg.eigh_p", "jax")
def _map_jax_lax_linalg_eigh_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_eigh_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.eigh_p")


@register_op("lax.linalg.hessenberg", "jax")
def _map_jax_lax_linalg_hessenberg(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_hessenberg operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.hessenberg")


@register_op("lax.linalg.hessenberg_p", "jax")
def _map_jax_lax_linalg_hessenberg_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_hessenberg_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.hessenberg_p")


@register_op("lax.linalg.householder_product", "jax")
def _map_jax_lax_linalg_householder_product(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_householder_product operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.householder_product"
    )


@register_op("lax.linalg.householder_product_p", "jax")
def _map_jax_lax_linalg_householder_product_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_householder_product_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.householder_product_p"
    )


@register_op("lax.linalg.lu", "jax")
def _map_jax_lax_linalg_lu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_linalg_lu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.lu")


@register_op("lax.linalg.lu_p", "jax")
def _map_jax_lax_linalg_lu_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_linalg_lu_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.lu_p")


@register_op("lax.linalg.lu_pivots_to_permutation", "jax")
def _map_jax_lax_linalg_lu_pivots_to_permutation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_lu_pivots_to_permutation operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="lax.linalg.lu_pivots_to_permutation",
    )


@register_op("lax.linalg.lu_pivots_to_permutation_p", "jax")
def _map_jax_lax_linalg_lu_pivots_to_permutation_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_lu_pivots_to_permutation_p operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="lax.linalg.lu_pivots_to_permutation_p",
    )


@register_op("lax.linalg.qr", "jax")
def _map_jax_lax_linalg_qr(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_linalg_qr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.qr")


@register_op("lax.linalg.qr_p", "jax")
def _map_jax_lax_linalg_qr_p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_linalg_qr_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.qr_p")


@register_op("lax.linalg.schur", "jax")
def _map_jax_lax_linalg_schur(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_schur operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.schur")


@register_op("lax.linalg.schur_p", "jax")
def _map_jax_lax_linalg_schur_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_schur_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.schur_p")


@register_op("lax.linalg.svd", "jax")
def _map_jax_lax_linalg_svd(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_linalg_svd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.svd")


@register_op("lax.linalg.svd_p", "jax")
def _map_jax_lax_linalg_svd_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_svd_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.svd_p")


@register_op("lax.linalg.SvdAlgorithm", "jax")
def _map_jax_lax_linalg_SvdAlgorithm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_SvdAlgorithm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.SvdAlgorithm")


@register_op("lax.linalg.symmetric_product", "jax")
def _map_jax_lax_linalg_symmetric_product(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_symmetric_product operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.symmetric_product"
    )


@register_op("lax.linalg.symmetric_product_p", "jax")
def _map_jax_lax_linalg_symmetric_product_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_symmetric_product_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.symmetric_product_p"
    )


@register_op("lax.linalg.triangular_solve", "jax")
def _map_jax_lax_linalg_triangular_solve(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_triangular_solve operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.triangular_solve"
    )


@register_op("lax.linalg.triangular_solve_p", "jax")
def _map_jax_lax_linalg_triangular_solve_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_triangular_solve_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.triangular_solve_p"
    )


@register_op("lax.linalg.tridiagonal", "jax")
def _map_jax_lax_linalg_tridiagonal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_tridiagonal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.tridiagonal")


@register_op("lax.linalg.tridiagonal_p", "jax")
def _map_jax_lax_linalg_tridiagonal_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_tridiagonal_p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.tridiagonal_p")


@register_op("lax.linalg.tridiagonal_solve", "jax")
def _map_jax_lax_linalg_tridiagonal_solve(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_tridiagonal_solve operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.tridiagonal_solve"
    )


@register_op("lax.linalg.tridiagonal_solve_p", "jax")
def _map_jax_lax_linalg_tridiagonal_solve_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_lax_linalg_tridiagonal_solve_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.tridiagonal_solve_p"
    )


@register_op("lax.linalg.qdwh", "jax")
def _map_jax_lax_linalg_qdwh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_lax_linalg_qdwh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="lax.linalg.qdwh")


@register_op("numpy.PrecisionLike", "jax")
def _map_jax_numpy_PrecisionLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_PrecisionLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.PrecisionLike")


@register_op("numpy.GatherScatterMode", "jax")
def _map_jax_numpy_GatherScatterMode(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_GatherScatterMode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.GatherScatterMode")


@register_op("numpy.Device", "jax")
def _map_jax_numpy_Device(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_Device operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.Device")


@register_op("numpy.ArrayNamespaceInfo", "jax")
def _map_jax_numpy_ArrayNamespaceInfo(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_ArrayNamespaceInfo operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ArrayNamespaceInfo")


@register_op("numpy.Array", "jax")
def _map_jax_numpy_Array(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.Array")


@register_op("numpy.ArrayLike", "jax")
def _map_jax_numpy_ArrayLike(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ArrayLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ArrayLike")


@register_op("numpy.DType", "jax")
def _map_jax_numpy_DType(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_DType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.DType")


@register_op("numpy.DTypeLike", "jax")
def _map_jax_numpy_DTypeLike(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_DTypeLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.DTypeLike")


@register_op("numpy.DeprecatedArg", "jax")
def _map_jax_numpy_DeprecatedArg(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_DeprecatedArg operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.DeprecatedArg")


@register_op("numpy.DimSize", "jax")
def _map_jax_numpy_DimSize(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_DimSize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.DimSize")


@register_op("numpy.DuckTypedArray", "jax")
def _map_jax_numpy_DuckTypedArray(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_DuckTypedArray operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.DuckTypedArray")


@register_op("numpy.Shape", "jax")
def _map_jax_numpy_Shape(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_Shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.Shape")


@register_op("numpy.StaticScalar", "jax")
def _map_jax_numpy_StaticScalar(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_StaticScalar operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.StaticScalar")


@register_op("numpy.SupportsNdim", "jax")
def _map_jax_numpy_SupportsNdim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_SupportsNdim operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.SupportsNdim")


@register_op("numpy.SupportsShape", "jax")
def _map_jax_numpy_SupportsShape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_SupportsShape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.SupportsShape")


@register_op("numpy.SupportsSize", "jax")
def _map_jax_numpy_SupportsSize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_SupportsSize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.SupportsSize")


@register_op("numpy.NamedSharding", "jax")
def _map_jax_numpy_NamedSharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_NamedSharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.NamedSharding")


@register_op("numpy.P", "jax")
def _map_jax_numpy_P(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_P operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.P")


@register_op("numpy.ComplexWarning", "jax")
def _map_jax_numpy_ComplexWarning(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_ComplexWarning operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ComplexWarning")


@register_op("numpy.ufunc", "jax")
def _map_jax_numpy_ufunc(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ufunc operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ufunc")


@register_op("numpy.BinaryUfunc", "jax")
def _map_jax_numpy_BinaryUfunc(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_BinaryUfunc operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.BinaryUfunc")


@register_op("numpy.abs", "jax")
def _map_jax_numpy_abs(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_abs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.abs")


@register_op("numpy.absolute", "jax")
def _map_jax_numpy_absolute(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_absolute operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.absolute")


@register_op("numpy.acos", "jax")
def _map_jax_numpy_acos(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_acos operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.acos")


@register_op("numpy.acosh", "jax")
def _map_jax_numpy_acosh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_acosh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.acosh")


@register_op("numpy.add", "jax")
def _map_jax_numpy_add(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_add operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.add")


@register_op("numpy.amax", "jax")
def _map_jax_numpy_amax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_amax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.amax")


@register_op("numpy.amin", "jax")
def _map_jax_numpy_amin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_amin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.amin")


@register_op("numpy.all", "jax")
def _map_jax_numpy_all(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_all operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.all")


@register_op("numpy.allclose", "jax")
def _map_jax_numpy_allclose(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_allclose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.allclose")


@register_op("numpy.angle", "jax")
def _map_jax_numpy_angle(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_angle operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.angle")


@register_op("numpy.any", "jax")
def _map_jax_numpy_any(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_any operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.any")


@register_op("numpy.append", "jax")
def _map_jax_numpy_append(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_append operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.append")


@register_op("numpy.apply_along_axis", "jax")
def _map_jax_numpy_apply_along_axis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_apply_along_axis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.apply_along_axis")


@register_op("numpy.apply_over_axes", "jax")
def _map_jax_numpy_apply_over_axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_apply_over_axes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.apply_over_axes")


@register_op("numpy.arange", "jax")
def _map_jax_numpy_arange(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_arange operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.arange")


@register_op("numpy.arccos", "jax")
def _map_jax_numpy_arccos(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_arccos operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.arccos")


@register_op("numpy.arccosh", "jax")
def _map_jax_numpy_arccosh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_arccosh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.arccosh")


@register_op("numpy.arcsin", "jax")
def _map_jax_numpy_arcsin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_arcsin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.arcsin")


@register_op("numpy.arcsinh", "jax")
def _map_jax_numpy_arcsinh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_arcsinh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.arcsinh")


@register_op("numpy.arctan", "jax")
def _map_jax_numpy_arctan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_arctan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.arctan")


@register_op("numpy.arctan2", "jax")
def _map_jax_numpy_arctan2(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_arctan2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.arctan2")


@register_op("numpy.arctanh", "jax")
def _map_jax_numpy_arctanh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_arctanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.arctanh")


@register_op("numpy.argmax", "jax")
def _map_jax_numpy_argmax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_argmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.argmax")


@register_op("numpy.argmin", "jax")
def _map_jax_numpy_argmin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_argmin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.argmin")


@register_op("numpy.argpartition", "jax")
def _map_jax_numpy_argpartition(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_argpartition operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.argpartition")


@register_op("numpy.argsort", "jax")
def _map_jax_numpy_argsort(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_argsort operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.argsort")


@register_op("numpy.argwhere", "jax")
def _map_jax_numpy_argwhere(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_argwhere operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.argwhere")


@register_op("numpy.around", "jax")
def _map_jax_numpy_around(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_around operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.around")


@register_op("numpy.array", "jax")
def _map_jax_numpy_array(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.array")


@register_op("numpy.array_equal", "jax")
def _map_jax_numpy_array_equal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_array_equal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.array_equal")


@register_op("numpy.array_equiv", "jax")
def _map_jax_numpy_array_equiv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_array_equiv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.array_equiv")


@register_op("numpy.array_repr", "jax")
def _map_jax_numpy_array_repr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_array_repr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.array_repr")


@register_op("numpy.array_split", "jax")
def _map_jax_numpy_array_split(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_array_split operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.array_split")


@register_op("numpy.array_str", "jax")
def _map_jax_numpy_array_str(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_array_str operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.array_str")


@register_op("numpy.asarray", "jax")
def _map_jax_numpy_asarray(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_asarray operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.asarray")


@register_op("numpy.asin", "jax")
def _map_jax_numpy_asin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_asin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.asin")


@register_op("numpy.asinh", "jax")
def _map_jax_numpy_asinh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_asinh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.asinh")


@register_op("numpy.astype", "jax")
def _map_jax_numpy_astype(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_astype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.astype")


@register_op("numpy.atan", "jax")
def _map_jax_numpy_atan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_atan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.atan")


@register_op("numpy.atan2", "jax")
def _map_jax_numpy_atan2(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_atan2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.atan2")


@register_op("numpy.atanh", "jax")
def _map_jax_numpy_atanh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_atanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.atanh")


@register_op("numpy.bartlett", "jax")
def _map_jax_numpy_bartlett(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_bartlett operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bartlett")


@register_op("numpy.bfloat16", "jax")
def _map_jax_numpy_bfloat16(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_bfloat16 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bfloat16")


@register_op("numpy.bincount", "jax")
def _map_jax_numpy_bincount(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_bincount operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bincount")


@register_op("numpy.bitwise_and", "jax")
def _map_jax_numpy_bitwise_and(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_bitwise_and operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bitwise_and")


@register_op("numpy.bitwise_count", "jax")
def _map_jax_numpy_bitwise_count(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_bitwise_count operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bitwise_count")


@register_op("numpy.bitwise_invert", "jax")
def _map_jax_numpy_bitwise_invert(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_bitwise_invert operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bitwise_invert")


@register_op("numpy.bitwise_left_shift", "jax")
def _map_jax_numpy_bitwise_left_shift(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_bitwise_left_shift operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bitwise_left_shift")


@register_op("numpy.bitwise_not", "jax")
def _map_jax_numpy_bitwise_not(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_bitwise_not operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bitwise_not")


@register_op("numpy.bitwise_or", "jax")
def _map_jax_numpy_bitwise_or(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_bitwise_or operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bitwise_or")


@register_op("numpy.bitwise_right_shift", "jax")
def _map_jax_numpy_bitwise_right_shift(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_bitwise_right_shift operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bitwise_right_shift"
    )


@register_op("numpy.bitwise_xor", "jax")
def _map_jax_numpy_bitwise_xor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_bitwise_xor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bitwise_xor")


@register_op("numpy.blackman", "jax")
def _map_jax_numpy_blackman(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_blackman operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.blackman")


@register_op("numpy.block", "jax")
def _map_jax_numpy_block(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_block operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.block")


@register_op("numpy.bool", "jax")
def _map_jax_numpy_bool(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_bool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bool")


@register_op("numpy.bool_", "jax")
def _map_jax_numpy_bool_(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_bool_ operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.bool_")


@register_op("numpy.broadcast_arrays", "jax")
def _map_jax_numpy_broadcast_arrays(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_broadcast_arrays operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.broadcast_arrays")


@register_op("numpy.broadcast_to", "jax")
def _map_jax_numpy_broadcast_to(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_broadcast_to operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.broadcast_to")


@register_op("numpy.c_", "jax")
def _map_jax_numpy_c_(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_c_ operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.c_")


@register_op("numpy.can_cast", "jax")
def _map_jax_numpy_can_cast(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_can_cast operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.can_cast")


@register_op("numpy.cbrt", "jax")
def _map_jax_numpy_cbrt(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_cbrt operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.cbrt")


@register_op("numpy.cdouble", "jax")
def _map_jax_numpy_cdouble(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_cdouble operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.cdouble")


@register_op("numpy.ceil", "jax")
def _map_jax_numpy_ceil(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ceil operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ceil")


@register_op("numpy.character", "jax")
def _map_jax_numpy_character(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_character operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.character")


@register_op("numpy.choose", "jax")
def _map_jax_numpy_choose(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_choose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.choose")


@register_op("numpy.clip", "jax")
def _map_jax_numpy_clip(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_clip operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.clip")


@register_op("numpy.column_stack", "jax")
def _map_jax_numpy_column_stack(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_column_stack operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.column_stack")


@register_op("numpy.complex128", "jax")
def _map_jax_numpy_complex128(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_complex128 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.complex128")


@register_op("numpy.complex64", "jax")
def _map_jax_numpy_complex64(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_complex64 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.complex64")


@register_op("numpy.complex_", "jax")
def _map_jax_numpy_complex_(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_complex_ operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.complex_")


@register_op("numpy.complexfloating", "jax")
def _map_jax_numpy_complexfloating(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_complexfloating operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.complexfloating")


@register_op("numpy.compress", "jax")
def _map_jax_numpy_compress(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_compress operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.compress")


@register_op("numpy.concat", "jax")
def _map_jax_numpy_concat(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_concat operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.concat")


@register_op("numpy.concatenate", "jax")
def _map_jax_numpy_concatenate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_concatenate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.concatenate")


@register_op("numpy.conjugate", "jax")
def _map_jax_numpy_conjugate(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_conjugate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.conjugate")


@register_op("numpy.conj", "jax")
def _map_jax_numpy_conj(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_conj operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.conj")


@register_op("numpy.convolve", "jax")
def _map_jax_numpy_convolve(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_convolve operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.convolve")


@register_op("numpy.copy", "jax")
def _map_jax_numpy_copy(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_copy operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.copy")


@register_op("numpy.copysign", "jax")
def _map_jax_numpy_copysign(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_copysign operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.copysign")


@register_op("numpy.corrcoef", "jax")
def _map_jax_numpy_corrcoef(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_corrcoef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.corrcoef")


@register_op("numpy.correlate", "jax")
def _map_jax_numpy_correlate(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_correlate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.correlate")


@register_op("numpy.cos", "jax")
def _map_jax_numpy_cos(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_cos operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.cos")


@register_op("numpy.cosh", "jax")
def _map_jax_numpy_cosh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_cosh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.cosh")


@register_op("numpy.count_nonzero", "jax")
def _map_jax_numpy_count_nonzero(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_count_nonzero operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.count_nonzero")


@register_op("numpy.cov", "jax")
def _map_jax_numpy_cov(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_cov operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.cov")


@register_op("numpy.cross", "jax")
def _map_jax_numpy_cross(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_cross operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.cross")


@register_op("numpy.csingle", "jax")
def _map_jax_numpy_csingle(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_csingle operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.csingle")


@register_op("numpy.cumprod", "jax")
def _map_jax_numpy_cumprod(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_cumprod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.cumprod")


@register_op("numpy.cumsum", "jax")
def _map_jax_numpy_cumsum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_cumsum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.cumsum")


@register_op("numpy.cumulative_prod", "jax")
def _map_jax_numpy_cumulative_prod(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_cumulative_prod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.cumulative_prod")


@register_op("numpy.cumulative_sum", "jax")
def _map_jax_numpy_cumulative_sum(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_cumulative_sum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.cumulative_sum")


@register_op("numpy.deg2rad", "jax")
def _map_jax_numpy_deg2rad(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_deg2rad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.deg2rad")


@register_op("numpy.degrees", "jax")
def _map_jax_numpy_degrees(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_degrees operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.degrees")


@register_op("numpy.delete", "jax")
def _map_jax_numpy_delete(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_delete operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.delete")


@register_op("numpy.diag", "jax")
def _map_jax_numpy_diag(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_diag operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.diag")


@register_op("numpy.diag_indices", "jax")
def _map_jax_numpy_diag_indices(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_diag_indices operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.diag_indices")


@register_op("numpy.diag_indices_from", "jax")
def _map_jax_numpy_diag_indices_from(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_diag_indices_from operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.diag_indices_from")


@register_op("numpy.diagflat", "jax")
def _map_jax_numpy_diagflat(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_diagflat operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.diagflat")


@register_op("numpy.diagonal", "jax")
def _map_jax_numpy_diagonal(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_diagonal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.diagonal")


@register_op("numpy.diff", "jax")
def _map_jax_numpy_diff(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_diff operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.diff")


@register_op("numpy.digitize", "jax")
def _map_jax_numpy_digitize(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_digitize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.digitize")


@register_op("numpy.divide", "jax")
def _map_jax_numpy_divide(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_divide operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.divide")


@register_op("numpy.divmod", "jax")
def _map_jax_numpy_divmod(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_divmod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.divmod")


@register_op("numpy.dot", "jax")
def _map_jax_numpy_dot(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_dot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.dot")


@register_op("numpy.double", "jax")
def _map_jax_numpy_double(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_double operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.double")


@register_op("numpy.dsplit", "jax")
def _map_jax_numpy_dsplit(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_dsplit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.dsplit")


@register_op("numpy.dstack", "jax")
def _map_jax_numpy_dstack(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_dstack operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.dstack")


@register_op("numpy.dtype", "jax")
def _map_jax_numpy_dtype(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.dtype")


@register_op("numpy.e", "jax")
def _map_jax_numpy_e(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_e operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.e")


@register_op("numpy.ediff1d", "jax")
def _map_jax_numpy_ediff1d(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ediff1d operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ediff1d")


@register_op("numpy.empty", "jax")
def _map_jax_numpy_empty(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_empty operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.empty")


@register_op("numpy.empty_like", "jax")
def _map_jax_numpy_empty_like(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_empty_like operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.empty_like")


@register_op("numpy.equal", "jax")
def _map_jax_numpy_equal(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_equal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.equal")


@register_op("numpy.euler_gamma", "jax")
def _map_jax_numpy_euler_gamma(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_euler_gamma operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.euler_gamma")


@register_op("numpy.exp", "jax")
def _map_jax_numpy_exp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_exp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.exp")


@register_op("numpy.exp2", "jax")
def _map_jax_numpy_exp2(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_exp2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.exp2")


@register_op("numpy.expand_dims", "jax")
def _map_jax_numpy_expand_dims(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_expand_dims operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.expand_dims")


@register_op("numpy.expm1", "jax")
def _map_jax_numpy_expm1(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_expm1 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.expm1")


@register_op("numpy.extract", "jax")
def _map_jax_numpy_extract(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_extract operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.extract")


@register_op("numpy.eye", "jax")
def _map_jax_numpy_eye(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_eye operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.eye")


@register_op("numpy.fabs", "jax")
def _map_jax_numpy_fabs(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fabs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fabs")


@register_op("numpy.finfo", "jax")
def _map_jax_numpy_finfo(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_finfo operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.finfo")


@register_op("numpy.fix", "jax")
def _map_jax_numpy_fix(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fix operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fix")


@register_op("numpy.flatnonzero", "jax")
def _map_jax_numpy_flatnonzero(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_flatnonzero operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.flatnonzero")


@register_op("numpy.flexible", "jax")
def _map_jax_numpy_flexible(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_flexible operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.flexible")


@register_op("numpy.flip", "jax")
def _map_jax_numpy_flip(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_flip operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.flip")


@register_op("numpy.fliplr", "jax")
def _map_jax_numpy_fliplr(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fliplr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fliplr")


@register_op("numpy.flipud", "jax")
def _map_jax_numpy_flipud(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_flipud operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.flipud")


@register_op("numpy.float16", "jax")
def _map_jax_numpy_float16(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_float16 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float16")


@register_op("numpy.float32", "jax")
def _map_jax_numpy_float32(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_float32 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float32")


@register_op("numpy.float4_e2m1fn", "jax")
def _map_jax_numpy_float4_e2m1fn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_float4_e2m1fn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float4_e2m1fn")


@register_op("numpy.float64", "jax")
def _map_jax_numpy_float64(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_float64 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float64")


@register_op("numpy.float8_e3m4", "jax")
def _map_jax_numpy_float8_e3m4(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_float8_e3m4 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float8_e3m4")


@register_op("numpy.float8_e4m3", "jax")
def _map_jax_numpy_float8_e4m3(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_float8_e4m3 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float8_e4m3")


@register_op("numpy.float8_e4m3b11fnuz", "jax")
def _map_jax_numpy_float8_e4m3b11fnuz(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_float8_e4m3b11fnuz operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float8_e4m3b11fnuz")


@register_op("numpy.float8_e4m3fn", "jax")
def _map_jax_numpy_float8_e4m3fn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_float8_e4m3fn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float8_e4m3fn")


@register_op("numpy.float8_e4m3fnuz", "jax")
def _map_jax_numpy_float8_e4m3fnuz(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_float8_e4m3fnuz operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float8_e4m3fnuz")


@register_op("numpy.float8_e5m2", "jax")
def _map_jax_numpy_float8_e5m2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_float8_e5m2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float8_e5m2")


@register_op("numpy.float8_e5m2fnuz", "jax")
def _map_jax_numpy_float8_e5m2fnuz(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_float8_e5m2fnuz operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float8_e5m2fnuz")


@register_op("numpy.float8_e8m0fnu", "jax")
def _map_jax_numpy_float8_e8m0fnu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_float8_e8m0fnu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float8_e8m0fnu")


@register_op("numpy.float_", "jax")
def _map_jax_numpy_float_(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_float_ operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float_")


@register_op("numpy.float_power", "jax")
def _map_jax_numpy_float_power(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_float_power operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.float_power")


@register_op("numpy.floating", "jax")
def _map_jax_numpy_floating(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_floating operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.floating")


@register_op("numpy.floor", "jax")
def _map_jax_numpy_floor(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_floor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.floor")


@register_op("numpy.floor_divide", "jax")
def _map_jax_numpy_floor_divide(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_floor_divide operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.floor_divide")


@register_op("numpy.fmax", "jax")
def _map_jax_numpy_fmax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fmax")


@register_op("numpy.fmin", "jax")
def _map_jax_numpy_fmin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fmin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fmin")


@register_op("numpy.fmod", "jax")
def _map_jax_numpy_fmod(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fmod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fmod")


@register_op("numpy.frexp", "jax")
def _map_jax_numpy_frexp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_frexp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.frexp")


@register_op("numpy.from_dlpack", "jax")
def _map_jax_numpy_from_dlpack(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_from_dlpack operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.from_dlpack")


@register_op("numpy.frombuffer", "jax")
def _map_jax_numpy_frombuffer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_frombuffer operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.frombuffer")


@register_op("numpy.fromfile", "jax")
def _map_jax_numpy_fromfile(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fromfile operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fromfile")


@register_op("numpy.fromfunction", "jax")
def _map_jax_numpy_fromfunction(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_fromfunction operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fromfunction")


@register_op("numpy.fromiter", "jax")
def _map_jax_numpy_fromiter(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fromiter operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fromiter")


@register_op("numpy.frompyfunc", "jax")
def _map_jax_numpy_frompyfunc(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_frompyfunc operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.frompyfunc")


@register_op("numpy.fromstring", "jax")
def _map_jax_numpy_fromstring(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_fromstring operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fromstring")


@register_op("numpy.full", "jax")
def _map_jax_numpy_full(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_full operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.full")


@register_op("numpy.full_like", "jax")
def _map_jax_numpy_full_like(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_full_like operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.full_like")


@register_op("numpy.gcd", "jax")
def _map_jax_numpy_gcd(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_gcd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.gcd")


@register_op("numpy.generic", "jax")
def _map_jax_numpy_generic(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_generic operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.generic")


@register_op("numpy.geomspace", "jax")
def _map_jax_numpy_geomspace(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_geomspace operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.geomspace")


@register_op("numpy.get_printoptions", "jax")
def _map_jax_numpy_get_printoptions(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_get_printoptions operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.get_printoptions")


@register_op("numpy.gradient", "jax")
def _map_jax_numpy_gradient(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_gradient operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.gradient")


@register_op("numpy.greater", "jax")
def _map_jax_numpy_greater(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_greater operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.greater")


@register_op("numpy.greater_equal", "jax")
def _map_jax_numpy_greater_equal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_greater_equal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.greater_equal")


@register_op("numpy.hamming", "jax")
def _map_jax_numpy_hamming(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_hamming operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.hamming")


@register_op("numpy.hanning", "jax")
def _map_jax_numpy_hanning(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_hanning operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.hanning")


@register_op("numpy.heaviside", "jax")
def _map_jax_numpy_heaviside(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_heaviside operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.heaviside")


@register_op("numpy.histogram", "jax")
def _map_jax_numpy_histogram(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_histogram operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.histogram")


@register_op("numpy.histogram2d", "jax")
def _map_jax_numpy_histogram2d(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_histogram2d operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.histogram2d")


@register_op("numpy.histogram_bin_edges", "jax")
def _map_jax_numpy_histogram_bin_edges(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_histogram_bin_edges operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.histogram_bin_edges"
    )


@register_op("numpy.histogramdd", "jax")
def _map_jax_numpy_histogramdd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_histogramdd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.histogramdd")


@register_op("numpy.hsplit", "jax")
def _map_jax_numpy_hsplit(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_hsplit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.hsplit")


@register_op("numpy.hstack", "jax")
def _map_jax_numpy_hstack(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_hstack operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.hstack")


@register_op("numpy.hypot", "jax")
def _map_jax_numpy_hypot(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_hypot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.hypot")


@register_op("numpy.i0", "jax")
def _map_jax_numpy_i0(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_i0 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.i0")


@register_op("numpy.identity", "jax")
def _map_jax_numpy_identity(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_identity operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.identity")


@register_op("numpy.iinfo", "jax")
def _map_jax_numpy_iinfo(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_iinfo operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.iinfo")


@register_op("numpy.imag", "jax")
def _map_jax_numpy_imag(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_imag operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.imag")


@register_op("numpy.index_exp", "jax")
def _map_jax_numpy_index_exp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_index_exp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.index_exp")


@register_op("numpy.inexact", "jax")
def _map_jax_numpy_inexact(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_inexact operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.inexact")


@register_op("numpy.inf", "jax")
def _map_jax_numpy_inf(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_inf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.inf")


@register_op("numpy.inner", "jax")
def _map_jax_numpy_inner(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_inner operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.inner")


@register_op("numpy.insert", "jax")
def _map_jax_numpy_insert(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_insert operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.insert")


@register_op("numpy.int16", "jax")
def _map_jax_numpy_int16(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_int16 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.int16")


@register_op("numpy.int2", "jax")
def _map_jax_numpy_int2(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_int2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.int2")


@register_op("numpy.int32", "jax")
def _map_jax_numpy_int32(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_int32 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.int32")


@register_op("numpy.int4", "jax")
def _map_jax_numpy_int4(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_int4 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.int4")


@register_op("numpy.int64", "jax")
def _map_jax_numpy_int64(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_int64 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.int64")


@register_op("numpy.int8", "jax")
def _map_jax_numpy_int8(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_int8 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.int8")


@register_op("numpy.int_", "jax")
def _map_jax_numpy_int_(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_int_ operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.int_")


@register_op("numpy.integer", "jax")
def _map_jax_numpy_integer(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_integer operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.integer")


@register_op("numpy.interp", "jax")
def _map_jax_numpy_interp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_interp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.interp")


@register_op("numpy.intersect1d", "jax")
def _map_jax_numpy_intersect1d(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_intersect1d operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.intersect1d")


@register_op("numpy.invert", "jax")
def _map_jax_numpy_invert(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_invert operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.invert")


@register_op("numpy.isclose", "jax")
def _map_jax_numpy_isclose(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_isclose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.isclose")


@register_op("numpy.iscomplex", "jax")
def _map_jax_numpy_iscomplex(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_iscomplex operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.iscomplex")


@register_op("numpy.iscomplexobj", "jax")
def _map_jax_numpy_iscomplexobj(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_iscomplexobj operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.iscomplexobj")


@register_op("numpy.isdtype", "jax")
def _map_jax_numpy_isdtype(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_isdtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.isdtype")


@register_op("numpy.isfinite", "jax")
def _map_jax_numpy_isfinite(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_isfinite operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.isfinite")


@register_op("numpy.isin", "jax")
def _map_jax_numpy_isin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_isin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.isin")


@register_op("numpy.isinf", "jax")
def _map_jax_numpy_isinf(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_isinf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.isinf")


@register_op("numpy.isnan", "jax")
def _map_jax_numpy_isnan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_isnan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.isnan")


@register_op("numpy.isneginf", "jax")
def _map_jax_numpy_isneginf(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_isneginf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.isneginf")


@register_op("numpy.isposinf", "jax")
def _map_jax_numpy_isposinf(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_isposinf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.isposinf")


@register_op("numpy.isreal", "jax")
def _map_jax_numpy_isreal(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_isreal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.isreal")


@register_op("numpy.isrealobj", "jax")
def _map_jax_numpy_isrealobj(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_isrealobj operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.isrealobj")


@register_op("numpy.isscalar", "jax")
def _map_jax_numpy_isscalar(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_isscalar operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.isscalar")


@register_op("numpy.issubdtype", "jax")
def _map_jax_numpy_issubdtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_issubdtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.issubdtype")


@register_op("numpy.iterable", "jax")
def _map_jax_numpy_iterable(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_iterable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.iterable")


@register_op("numpy.ix_", "jax")
def _map_jax_numpy_ix_(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ix_ operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ix_")


@register_op("numpy.kaiser", "jax")
def _map_jax_numpy_kaiser(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_kaiser operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.kaiser")


@register_op("numpy.kron", "jax")
def _map_jax_numpy_kron(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_kron operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.kron")


@register_op("numpy.lcm", "jax")
def _map_jax_numpy_lcm(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_lcm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.lcm")


@register_op("numpy.ldexp", "jax")
def _map_jax_numpy_ldexp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ldexp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ldexp")


@register_op("numpy.left_shift", "jax")
def _map_jax_numpy_left_shift(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_left_shift operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.left_shift")


@register_op("numpy.less", "jax")
def _map_jax_numpy_less(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_less operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.less")


@register_op("numpy.less_equal", "jax")
def _map_jax_numpy_less_equal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_less_equal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.less_equal")


@register_op("numpy.lexsort", "jax")
def _map_jax_numpy_lexsort(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_lexsort operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.lexsort")


@register_op("numpy.load", "jax")
def _map_jax_numpy_load(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_load operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.load")


@register_op("numpy.log", "jax")
def _map_jax_numpy_log(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_log operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.log")


@register_op("numpy.log10", "jax")
def _map_jax_numpy_log10(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_log10 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.log10")


@register_op("numpy.log1p", "jax")
def _map_jax_numpy_log1p(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_log1p operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.log1p")


@register_op("numpy.log2", "jax")
def _map_jax_numpy_log2(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_log2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.log2")


@register_op("numpy.logaddexp", "jax")
def _map_jax_numpy_logaddexp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_logaddexp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.logaddexp")


@register_op("numpy.logaddexp2", "jax")
def _map_jax_numpy_logaddexp2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_logaddexp2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.logaddexp2")


@register_op("numpy.logical_and", "jax")
def _map_jax_numpy_logical_and(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_logical_and operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.logical_and")


@register_op("numpy.logical_not", "jax")
def _map_jax_numpy_logical_not(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_logical_not operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.logical_not")


@register_op("numpy.logical_or", "jax")
def _map_jax_numpy_logical_or(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_logical_or operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.logical_or")


@register_op("numpy.logical_xor", "jax")
def _map_jax_numpy_logical_xor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_logical_xor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.logical_xor")


@register_op("numpy.logspace", "jax")
def _map_jax_numpy_logspace(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_logspace operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.logspace")


@register_op("numpy.mask_indices", "jax")
def _map_jax_numpy_mask_indices(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_mask_indices operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.mask_indices")


@register_op("numpy.matmul", "jax")
def _map_jax_numpy_matmul(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_matmul operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.matmul")


@register_op("numpy.matrix_transpose", "jax")
def _map_jax_numpy_matrix_transpose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_matrix_transpose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.matrix_transpose")


@register_op("numpy.matvec", "jax")
def _map_jax_numpy_matvec(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_matvec operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.matvec")


@register_op("numpy.max", "jax")
def _map_jax_numpy_max(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_max operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.max")


@register_op("numpy.maximum", "jax")
def _map_jax_numpy_maximum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_maximum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.maximum")


@register_op("numpy.mean", "jax")
def _map_jax_numpy_mean(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_mean operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.mean")


@register_op("numpy.median", "jax")
def _map_jax_numpy_median(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_median operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.median")


@register_op("numpy.meshgrid", "jax")
def _map_jax_numpy_meshgrid(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_meshgrid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.meshgrid")


@register_op("numpy.mgrid", "jax")
def _map_jax_numpy_mgrid(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_mgrid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.mgrid")


@register_op("numpy.min", "jax")
def _map_jax_numpy_min(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_min operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.min")


@register_op("numpy.minimum", "jax")
def _map_jax_numpy_minimum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_minimum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.minimum")


@register_op("numpy.mod", "jax")
def _map_jax_numpy_mod(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_mod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.mod")


@register_op("numpy.modf", "jax")
def _map_jax_numpy_modf(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_modf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.modf")


@register_op("numpy.moveaxis", "jax")
def _map_jax_numpy_moveaxis(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_moveaxis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.moveaxis")


@register_op("numpy.multiply", "jax")
def _map_jax_numpy_multiply(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_multiply operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.multiply")


@register_op("numpy.nan", "jax")
def _map_jax_numpy_nan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nan")


@register_op("numpy.nan_to_num", "jax")
def _map_jax_numpy_nan_to_num(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_nan_to_num operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nan_to_num")


@register_op("numpy.nanargmax", "jax")
def _map_jax_numpy_nanargmax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nanargmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nanargmax")


@register_op("numpy.nanargmin", "jax")
def _map_jax_numpy_nanargmin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nanargmin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nanargmin")


@register_op("numpy.nancumprod", "jax")
def _map_jax_numpy_nancumprod(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_nancumprod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nancumprod")


@register_op("numpy.nancumsum", "jax")
def _map_jax_numpy_nancumsum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nancumsum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nancumsum")


@register_op("numpy.nanmax", "jax")
def _map_jax_numpy_nanmax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nanmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nanmax")


@register_op("numpy.nanmean", "jax")
def _map_jax_numpy_nanmean(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nanmean operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nanmean")


@register_op("numpy.nanmedian", "jax")
def _map_jax_numpy_nanmedian(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nanmedian operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nanmedian")


@register_op("numpy.nanmin", "jax")
def _map_jax_numpy_nanmin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nanmin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nanmin")


@register_op("numpy.nanpercentile", "jax")
def _map_jax_numpy_nanpercentile(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_nanpercentile operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nanpercentile")


@register_op("numpy.nanprod", "jax")
def _map_jax_numpy_nanprod(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nanprod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nanprod")


@register_op("numpy.nanquantile", "jax")
def _map_jax_numpy_nanquantile(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_nanquantile operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nanquantile")


@register_op("numpy.nanstd", "jax")
def _map_jax_numpy_nanstd(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nanstd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nanstd")


@register_op("numpy.nansum", "jax")
def _map_jax_numpy_nansum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nansum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nansum")


@register_op("numpy.nanvar", "jax")
def _map_jax_numpy_nanvar(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nanvar operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nanvar")


@register_op("numpy.ndarray", "jax")
def _map_jax_numpy_ndarray(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ndarray operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ndarray")


@register_op("numpy.ndim", "jax")
def _map_jax_numpy_ndim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ndim operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ndim")


@register_op("numpy.negative", "jax")
def _map_jax_numpy_negative(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_negative operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.negative")


@register_op("numpy.newaxis", "jax")
def _map_jax_numpy_newaxis(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_newaxis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.newaxis")


@register_op("numpy.nextafter", "jax")
def _map_jax_numpy_nextafter(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nextafter operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nextafter")


@register_op("numpy.nonzero", "jax")
def _map_jax_numpy_nonzero(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_nonzero operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.nonzero")


@register_op("numpy.not_equal", "jax")
def _map_jax_numpy_not_equal(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_not_equal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.not_equal")


@register_op("numpy.number", "jax")
def _map_jax_numpy_number(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_number operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.number")


@register_op("numpy.object_", "jax")
def _map_jax_numpy_object_(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_object_ operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.object_")


@register_op("numpy.ogrid", "jax")
def _map_jax_numpy_ogrid(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ogrid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ogrid")


@register_op("numpy.ones", "jax")
def _map_jax_numpy_ones(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ones operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ones")


@register_op("numpy.ones_like", "jax")
def _map_jax_numpy_ones_like(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ones_like operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ones_like")


@register_op("numpy.outer", "jax")
def _map_jax_numpy_outer(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_outer operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.outer")


@register_op("numpy.packbits", "jax")
def _map_jax_numpy_packbits(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_packbits operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.packbits")


@register_op("numpy.PadValueLike", "jax")
def _map_jax_numpy_PadValueLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_PadValueLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.PadValueLike")


@register_op("numpy.pad", "jax")
def _map_jax_numpy_pad(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_pad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.pad")


@register_op("numpy.partition", "jax")
def _map_jax_numpy_partition(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_partition operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.partition")


@register_op("numpy.percentile", "jax")
def _map_jax_numpy_percentile(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_percentile operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.percentile")


@register_op("numpy.permute_dims", "jax")
def _map_jax_numpy_permute_dims(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_permute_dims operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.permute_dims")


@register_op("numpy.pi", "jax")
def _map_jax_numpy_pi(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_pi operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.pi")


@register_op("numpy.piecewise", "jax")
def _map_jax_numpy_piecewise(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_piecewise operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.piecewise")


@register_op("numpy.place", "jax")
def _map_jax_numpy_place(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_place operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.place")


@register_op("numpy.poly", "jax")
def _map_jax_numpy_poly(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_poly operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.poly")


@register_op("numpy.polyadd", "jax")
def _map_jax_numpy_polyadd(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_polyadd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.polyadd")


@register_op("numpy.polyder", "jax")
def _map_jax_numpy_polyder(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_polyder operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.polyder")


@register_op("numpy.polydiv", "jax")
def _map_jax_numpy_polydiv(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_polydiv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.polydiv")


@register_op("numpy.polyfit", "jax")
def _map_jax_numpy_polyfit(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_polyfit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.polyfit")


@register_op("numpy.polyint", "jax")
def _map_jax_numpy_polyint(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_polyint operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.polyint")


@register_op("numpy.polymul", "jax")
def _map_jax_numpy_polymul(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_polymul operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.polymul")


@register_op("numpy.polysub", "jax")
def _map_jax_numpy_polysub(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_polysub operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.polysub")


@register_op("numpy.polyval", "jax")
def _map_jax_numpy_polyval(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_polyval operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.polyval")


@register_op("numpy.positive", "jax")
def _map_jax_numpy_positive(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_positive operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.positive")


@register_op("numpy.pow", "jax")
def _map_jax_numpy_pow(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_pow operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.pow")


@register_op("numpy.power", "jax")
def _map_jax_numpy_power(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_power operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.power")


@register_op("numpy.printoptions", "jax")
def _map_jax_numpy_printoptions(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_printoptions operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.printoptions")


@register_op("numpy.prod", "jax")
def _map_jax_numpy_prod(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_prod operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.prod")


@register_op("numpy.promote_types", "jax")
def _map_jax_numpy_promote_types(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_promote_types operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.promote_types")


@register_op("numpy.ptp", "jax")
def _map_jax_numpy_ptp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ptp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ptp")


@register_op("numpy.put", "jax")
def _map_jax_numpy_put(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_put operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.put")


@register_op("numpy.put_along_axis", "jax")
def _map_jax_numpy_put_along_axis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_put_along_axis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.put_along_axis")


@register_op("numpy.quantile", "jax")
def _map_jax_numpy_quantile(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_quantile operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.quantile")


@register_op("numpy.r_", "jax")
def _map_jax_numpy_r_(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_r_ operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.r_")


@register_op("numpy.rad2deg", "jax")
def _map_jax_numpy_rad2deg(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_rad2deg operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.rad2deg")


@register_op("numpy.radians", "jax")
def _map_jax_numpy_radians(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_radians operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.radians")


@register_op("numpy.ravel", "jax")
def _map_jax_numpy_ravel(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_ravel operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ravel")


@register_op("numpy.ravel_multi_index", "jax")
def _map_jax_numpy_ravel_multi_index(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_ravel_multi_index operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.ravel_multi_index")


@register_op("numpy.real", "jax")
def _map_jax_numpy_real(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_real operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.real")


@register_op("numpy.reciprocal", "jax")
def _map_jax_numpy_reciprocal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_reciprocal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.reciprocal")


@register_op("numpy.remainder", "jax")
def _map_jax_numpy_remainder(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_remainder operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.remainder")


@register_op("numpy.repeat", "jax")
def _map_jax_numpy_repeat(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_repeat operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.repeat")


@register_op("numpy.reshape", "jax")
def _map_jax_numpy_reshape(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_reshape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.reshape")


@register_op("numpy.resize", "jax")
def _map_jax_numpy_resize(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_resize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.resize")


@register_op("numpy.result_type", "jax")
def _map_jax_numpy_result_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_result_type operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.result_type")


@register_op("numpy.right_shift", "jax")
def _map_jax_numpy_right_shift(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_right_shift operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.right_shift")


@register_op("numpy.rint", "jax")
def _map_jax_numpy_rint(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_rint operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.rint")


@register_op("numpy.roll", "jax")
def _map_jax_numpy_roll(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_roll operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.roll")


@register_op("numpy.rollaxis", "jax")
def _map_jax_numpy_rollaxis(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_rollaxis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.rollaxis")


@register_op("numpy.roots", "jax")
def _map_jax_numpy_roots(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_roots operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.roots")


@register_op("numpy.rot90", "jax")
def _map_jax_numpy_rot90(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_rot90 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.rot90")


@register_op("numpy.round", "jax")
def _map_jax_numpy_round(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_round operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.round")


@register_op("numpy.s_", "jax")
def _map_jax_numpy_s_(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_s_ operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.s_")


@register_op("numpy.save", "jax")
def _map_jax_numpy_save(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_save operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.save")


@register_op("numpy.savez", "jax")
def _map_jax_numpy_savez(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_savez operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.savez")


@register_op("numpy.searchsorted", "jax")
def _map_jax_numpy_searchsorted(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_searchsorted operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.searchsorted")


@register_op("numpy.select", "jax")
def _map_jax_numpy_select(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_select operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.select")


@register_op("numpy.set_printoptions", "jax")
def _map_jax_numpy_set_printoptions(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_set_printoptions operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.set_printoptions")


@register_op("numpy.setdiff1d", "jax")
def _map_jax_numpy_setdiff1d(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_setdiff1d operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.setdiff1d")


@register_op("numpy.setxor1d", "jax")
def _map_jax_numpy_setxor1d(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_setxor1d operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.setxor1d")


@register_op("numpy.shape", "jax")
def _map_jax_numpy_shape(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.shape")


@register_op("numpy.sign", "jax")
def _map_jax_numpy_sign(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_sign operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.sign")


@register_op("numpy.signbit", "jax")
def _map_jax_numpy_signbit(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_signbit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.signbit")


@register_op("numpy.signedinteger", "jax")
def _map_jax_numpy_signedinteger(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_signedinteger operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.signedinteger")


@register_op("numpy.sin", "jax")
def _map_jax_numpy_sin(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_sin operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.sin")


@register_op("numpy.sinc", "jax")
def _map_jax_numpy_sinc(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_sinc operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.sinc")


@register_op("numpy.single", "jax")
def _map_jax_numpy_single(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_single operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.single")


@register_op("numpy.sinh", "jax")
def _map_jax_numpy_sinh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_sinh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.sinh")


@register_op("numpy.size", "jax")
def _map_jax_numpy_size(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_size operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.size")


@register_op("numpy.sort", "jax")
def _map_jax_numpy_sort(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_sort operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.sort")


@register_op("numpy.sort_complex", "jax")
def _map_jax_numpy_sort_complex(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_sort_complex operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.sort_complex")


@register_op("numpy.spacing", "jax")
def _map_jax_numpy_spacing(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_spacing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.spacing")


@register_op("numpy.split", "jax")
def _map_jax_numpy_split(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_split operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.split")


@register_op("numpy.sqrt", "jax")
def _map_jax_numpy_sqrt(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_sqrt operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.sqrt")


@register_op("numpy.square", "jax")
def _map_jax_numpy_square(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_square operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.square")


@register_op("numpy.squeeze", "jax")
def _map_jax_numpy_squeeze(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_squeeze operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.squeeze")


@register_op("numpy.stack", "jax")
def _map_jax_numpy_stack(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_stack operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.stack")


@register_op("numpy.std", "jax")
def _map_jax_numpy_std(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_std operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.std")


@register_op("numpy.subtract", "jax")
def _map_jax_numpy_subtract(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_subtract operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.subtract")


@register_op("numpy.sum", "jax")
def _map_jax_numpy_sum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_sum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.sum")


@register_op("numpy.swapaxes", "jax")
def _map_jax_numpy_swapaxes(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_swapaxes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.swapaxes")


@register_op("numpy.take", "jax")
def _map_jax_numpy_take(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_take operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.take")


@register_op("numpy.take_along_axis", "jax")
def _map_jax_numpy_take_along_axis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_take_along_axis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.take_along_axis")


@register_op("numpy.tan", "jax")
def _map_jax_numpy_tan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_tan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.tan")


@register_op("numpy.tanh", "jax")
def _map_jax_numpy_tanh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_tanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.tanh")


@register_op("numpy.tensordot", "jax")
def _map_jax_numpy_tensordot(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_tensordot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.tensordot")


@register_op("numpy.tile", "jax")
def _map_jax_numpy_tile(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_tile operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.tile")


@register_op("numpy.trace", "jax")
def _map_jax_numpy_trace(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_trace operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.trace")


@register_op("numpy.transpose", "jax")
def _map_jax_numpy_transpose(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_transpose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.transpose")


@register_op("numpy.trapezoid", "jax")
def _map_jax_numpy_trapezoid(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_trapezoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.trapezoid")


@register_op("numpy.tri", "jax")
def _map_jax_numpy_tri(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_tri operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.tri")


@register_op("numpy.tril", "jax")
def _map_jax_numpy_tril(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_tril operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.tril")


@register_op("numpy.tril_indices", "jax")
def _map_jax_numpy_tril_indices(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_tril_indices operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.tril_indices")


@register_op("numpy.tril_indices_from", "jax")
def _map_jax_numpy_tril_indices_from(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_tril_indices_from operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.tril_indices_from")


@register_op("numpy.fill_diagonal", "jax")
def _map_jax_numpy_fill_diagonal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_fill_diagonal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fill_diagonal")


@register_op("numpy.trim_zeros", "jax")
def _map_jax_numpy_trim_zeros(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_trim_zeros operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.trim_zeros")


@register_op("numpy.triu", "jax")
def _map_jax_numpy_triu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_triu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.triu")


@register_op("numpy.triu_indices", "jax")
def _map_jax_numpy_triu_indices(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_triu_indices operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.triu_indices")


@register_op("numpy.triu_indices_from", "jax")
def _map_jax_numpy_triu_indices_from(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_triu_indices_from operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.triu_indices_from")


@register_op("numpy.true_divide", "jax")
def _map_jax_numpy_true_divide(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_true_divide operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.true_divide")


@register_op("numpy.trunc", "jax")
def _map_jax_numpy_trunc(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_trunc operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.trunc")


@register_op("numpy.uint", "jax")
def _map_jax_numpy_uint(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_uint operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.uint")


@register_op("numpy.uint16", "jax")
def _map_jax_numpy_uint16(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_uint16 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.uint16")


@register_op("numpy.uint2", "jax")
def _map_jax_numpy_uint2(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_uint2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.uint2")


@register_op("numpy.uint32", "jax")
def _map_jax_numpy_uint32(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_uint32 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.uint32")


@register_op("numpy.uint4", "jax")
def _map_jax_numpy_uint4(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_uint4 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.uint4")


@register_op("numpy.uint64", "jax")
def _map_jax_numpy_uint64(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_uint64 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.uint64")


@register_op("numpy.uint8", "jax")
def _map_jax_numpy_uint8(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_uint8 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.uint8")


@register_op("numpy.union1d", "jax")
def _map_jax_numpy_union1d(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_union1d operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.union1d")


@register_op("numpy.unique", "jax")
def _map_jax_numpy_unique(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_unique operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.unique")


@register_op("numpy.unique_all", "jax")
def _map_jax_numpy_unique_all(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_unique_all operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.unique_all")


@register_op("numpy.unique_counts", "jax")
def _map_jax_numpy_unique_counts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_unique_counts operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.unique_counts")


@register_op("numpy.unique_inverse", "jax")
def _map_jax_numpy_unique_inverse(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_unique_inverse operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.unique_inverse")


@register_op("numpy.unique_values", "jax")
def _map_jax_numpy_unique_values(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_unique_values operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.unique_values")


@register_op("numpy.unpackbits", "jax")
def _map_jax_numpy_unpackbits(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_unpackbits operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.unpackbits")


@register_op("numpy.unravel_index", "jax")
def _map_jax_numpy_unravel_index(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_unravel_index operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.unravel_index")


@register_op("numpy.unsignedinteger", "jax")
def _map_jax_numpy_unsignedinteger(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_unsignedinteger operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.unsignedinteger")


@register_op("numpy.unstack", "jax")
def _map_jax_numpy_unstack(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_unstack operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.unstack")


@register_op("numpy.unwrap", "jax")
def _map_jax_numpy_unwrap(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_unwrap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.unwrap")


@register_op("numpy.vander", "jax")
def _map_jax_numpy_vander(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_vander operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.vander")


@register_op("numpy.var", "jax")
def _map_jax_numpy_var(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_var operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.var")


@register_op("numpy.vdot", "jax")
def _map_jax_numpy_vdot(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_vdot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.vdot")


@register_op("numpy.vecdot", "jax")
def _map_jax_numpy_vecdot(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_vecdot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.vecdot")


@register_op("numpy.vecmat", "jax")
def _map_jax_numpy_vecmat(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_vecmat operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.vecmat")


@register_op("numpy.vsplit", "jax")
def _map_jax_numpy_vsplit(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_vsplit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.vsplit")


@register_op("numpy.vstack", "jax")
def _map_jax_numpy_vstack(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_vstack operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.vstack")


@register_op("numpy.zeros", "jax")
def _map_jax_numpy_zeros(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_zeros operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.zeros")


@register_op("numpy.zeros_like", "jax")
def _map_jax_numpy_zeros_like(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_zeros_like operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.zeros_like")


@register_op("numpy.vectorize", "jax")
def _map_jax_numpy_vectorize(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_vectorize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.vectorize")


@register_op("numpy.fft.ifft", "jax")
def _map_jax_numpy_fft_ifft(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_ifft operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.ifft")


@register_op("numpy.fft.ifft2", "jax")
def _map_jax_numpy_fft_ifft2(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_ifft2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.ifft2")


@register_op("numpy.fft.ifftn", "jax")
def _map_jax_numpy_fft_ifftn(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_ifftn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.ifftn")


@register_op("numpy.fft.ifftshift", "jax")
def _map_jax_numpy_fft_ifftshift(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_fft_ifftshift operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.ifftshift")


@register_op("numpy.fft.ihfft", "jax")
def _map_jax_numpy_fft_ihfft(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_ihfft operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.ihfft")


@register_op("numpy.fft.irfft", "jax")
def _map_jax_numpy_fft_irfft(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_irfft operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.irfft")


@register_op("numpy.fft.irfft2", "jax")
def _map_jax_numpy_fft_irfft2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_fft_irfft2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.irfft2")


@register_op("numpy.fft.irfftn", "jax")
def _map_jax_numpy_fft_irfftn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_fft_irfftn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.irfftn")


@register_op("numpy.fft.fft", "jax")
def _map_jax_numpy_fft_fft(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_fft operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.fft")


@register_op("numpy.fft.fft2", "jax")
def _map_jax_numpy_fft_fft2(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_fft2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.fft2")


@register_op("numpy.fft.fftfreq", "jax")
def _map_jax_numpy_fft_fftfreq(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_fft_fftfreq operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.fftfreq")


@register_op("numpy.fft.fftn", "jax")
def _map_jax_numpy_fft_fftn(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_fftn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.fftn")


@register_op("numpy.fft.fftshift", "jax")
def _map_jax_numpy_fft_fftshift(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_fft_fftshift operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.fftshift")


@register_op("numpy.fft.hfft", "jax")
def _map_jax_numpy_fft_hfft(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_hfft operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.hfft")


@register_op("numpy.fft.rfft", "jax")
def _map_jax_numpy_fft_rfft(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_rfft operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.rfft")


@register_op("numpy.fft.rfft2", "jax")
def _map_jax_numpy_fft_rfft2(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_rfft2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.rfft2")


@register_op("numpy.fft.rfftfreq", "jax")
def _map_jax_numpy_fft_rfftfreq(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_fft_rfftfreq operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.rfftfreq")


@register_op("numpy.fft.rfftn", "jax")
def _map_jax_numpy_fft_rfftn(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_fft_rfftn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.fft.rfftn")


@register_op("numpy.linalg.cholesky", "jax")
def _map_jax_numpy_linalg_cholesky(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_cholesky operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.cholesky")


@register_op("numpy.linalg.cond", "jax")
def _map_jax_numpy_linalg_cond(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_cond operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.cond")


@register_op("numpy.linalg.cross", "jax")
def _map_jax_numpy_linalg_cross(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_cross operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.cross")


@register_op("numpy.linalg.det", "jax")
def _map_jax_numpy_linalg_det(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_det operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.det")


@register_op("numpy.linalg.diagonal", "jax")
def _map_jax_numpy_linalg_diagonal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_diagonal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.diagonal")


@register_op("numpy.linalg.eig", "jax")
def _map_jax_numpy_linalg_eig(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_eig operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.eig")


@register_op("numpy.linalg.eigh", "jax")
def _map_jax_numpy_linalg_eigh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_eigh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.eigh")


@register_op("numpy.linalg.eigvals", "jax")
def _map_jax_numpy_linalg_eigvals(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_eigvals operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.eigvals")


@register_op("numpy.linalg.eigvalsh", "jax")
def _map_jax_numpy_linalg_eigvalsh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_eigvalsh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.eigvalsh")


@register_op("numpy.linalg.inv", "jax")
def _map_jax_numpy_linalg_inv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_inv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.inv")


@register_op("numpy.linalg.lstsq", "jax")
def _map_jax_numpy_linalg_lstsq(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_lstsq operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.lstsq")


@register_op("numpy.linalg.matmul", "jax")
def _map_jax_numpy_linalg_matmul(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_matmul operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.matmul")


@register_op("numpy.linalg.matrix_norm", "jax")
def _map_jax_numpy_linalg_matrix_norm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_matrix_norm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.matrix_norm")


@register_op("numpy.linalg.matrix_power", "jax")
def _map_jax_numpy_linalg_matrix_power(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_matrix_power operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.matrix_power"
    )


@register_op("numpy.linalg.matrix_rank", "jax")
def _map_jax_numpy_linalg_matrix_rank(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_matrix_rank operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.matrix_rank")


@register_op("numpy.linalg.matrix_transpose", "jax")
def _map_jax_numpy_linalg_matrix_transpose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_matrix_transpose operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.matrix_transpose"
    )


@register_op("numpy.linalg.multi_dot", "jax")
def _map_jax_numpy_linalg_multi_dot(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_multi_dot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.multi_dot")


@register_op("numpy.linalg.norm", "jax")
def _map_jax_numpy_linalg_norm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_norm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.norm")


@register_op("numpy.linalg.outer", "jax")
def _map_jax_numpy_linalg_outer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_outer operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.outer")


@register_op("numpy.linalg.pinv", "jax")
def _map_jax_numpy_linalg_pinv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_pinv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.pinv")


@register_op("numpy.linalg.qr", "jax")
def _map_jax_numpy_linalg_qr(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_numpy_linalg_qr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.qr")


@register_op("numpy.linalg.slogdet", "jax")
def _map_jax_numpy_linalg_slogdet(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_slogdet operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.slogdet")


@register_op("numpy.linalg.solve", "jax")
def _map_jax_numpy_linalg_solve(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_solve operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.solve")


@register_op("numpy.linalg.svd", "jax")
def _map_jax_numpy_linalg_svd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_svd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.svd")


@register_op("numpy.linalg.svdvals", "jax")
def _map_jax_numpy_linalg_svdvals(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_svdvals operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.svdvals")


@register_op("numpy.linalg.tensordot", "jax")
def _map_jax_numpy_linalg_tensordot(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_tensordot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.tensordot")


@register_op("numpy.linalg.tensorinv", "jax")
def _map_jax_numpy_linalg_tensorinv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_tensorinv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.tensorinv")


@register_op("numpy.linalg.tensorsolve", "jax")
def _map_jax_numpy_linalg_tensorsolve(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_tensorsolve operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.tensorsolve")


@register_op("numpy.linalg.trace", "jax")
def _map_jax_numpy_linalg_trace(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_trace operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.trace")


@register_op("numpy.linalg.vector_norm", "jax")
def _map_jax_numpy_linalg_vector_norm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_vector_norm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.vector_norm")


@register_op("numpy.linalg.vecdot", "jax")
def _map_jax_numpy_linalg_vecdot(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_numpy_linalg_vecdot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="numpy.linalg.vecdot")


@register_op("add", "jax")
def _map_jax_add_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_add_prim operation."""
    return Node(
        op_type="Add",
        inputs=inputs,
        outputs=outputs,
        name=f"add_{outputs[0]}" if outputs else "add",
    )


@register_op("mul", "jax")
def _map_jax_mul_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_mul_prim operation."""
    return Node(
        op_type="Mul",
        inputs=inputs,
        outputs=outputs,
        name=f"mul_{outputs[0]}" if outputs else "mul",
    )


@register_op("dot_general", "jax")
def _map_jax_dot_general_prim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_dot_general_prim operation."""
    return Node(
        op_type="MatMul",
        inputs=inputs,
        outputs=outputs,
        name=f"dot_general_{outputs[0]}" if outputs else "dot_general",
    )


@register_op("broadcast_in_dim", "jax")
def _map_jax_broadcast_in_dim_prim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_broadcast_in_dim_prim operation."""
    return Node(
        op_type="Expand",
        inputs=inputs,
        outputs=outputs,
        attributes={"broadcast_dimensions": params.get("broadcast_dimensions", [])},
        name=f"broadcast_in_dim_{outputs[0]}" if outputs else "broadcast_in_dim",
    )


@register_op("xla_pmap", "jax")
def _map_jax_xla_pmap_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_xla_pmap_prim operation."""
    return Node(
        op_type="XlaPmap",
        inputs=inputs,
        outputs=outputs,
        attributes={"axis_name": params.get("axis_name", "")},
        name=f"xla_pmap_{outputs[0]}" if outputs else "xla_pmap",
    )


@register_op("grad_core", "jax")
def _map_jax_grad_core_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_grad_core_prim operation."""
    return Node(
        op_type="Grad",
        inputs=inputs,
        outputs=outputs,
        name=f"grad_core_{outputs[0]}" if outputs else "grad_core",
    )


@register_op("sub", "jax")
def _map_jax_sub_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax sub prim."""
    return Node(
        op_type="Sub",
        inputs=inputs,
        outputs=outputs,
        name=f"sub_{outputs[0]}" if outputs else "sub",
    )


@register_op("div", "jax")
def _map_jax_div_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax div prim."""
    return Node(
        op_type="Div",
        inputs=inputs,
        outputs=outputs,
        name=f"div_{outputs[0]}" if outputs else "div",
    )


@register_op("conv_general_dilated", "jax")
def _map_jax_conv_general_dilated(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    # Need to handle RHS and LHS dilation and dimension mapping properly,
    # mapping this strictly to IR.ConvND according to the plan.
    """map jax conv general dilated."""
    attributes = {}
    if "dimension_numbers" in params:
        # In a real implementation we'd translate these to ONNX formats
        attributes["dimension_numbers"] = str(params["dimension_numbers"])
    if "window_strides" in params:
        attributes["strides"] = list(params["window_strides"])
    if "padding" in params:
        attributes["pads"] = list(params["padding"])
    if "lhs_dilation" in params:
        attributes["lhs_dilation"] = list(params["lhs_dilation"])
    if "rhs_dilation" in params:
        attributes["dilations"] = list(params["rhs_dilation"])
    if "feature_group_count" in params:
        attributes["group"] = params["feature_group_count"]

    return Node(
        op_type="Conv",
        inputs=inputs,
        outputs=outputs,
        attributes=attributes,
        name=f"conv_general_dilated_{outputs[0]}" if outputs else "conv_general_dilated",
    )


@register_op("reduce_sum", "jax")
def _map_jax_reduce_sum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax reduce sum."""
    return Node(
        op_type="ReduceSum",
        inputs=inputs,
        outputs=outputs,
        attributes={"axes": params.get("axes", [])},
        name=f"reduce_sum_{outputs[0]}" if outputs else "reduce_sum",
    )


@register_op("reduce_max", "jax")
def _map_jax_reduce_max(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax reduce max."""
    return Node(
        op_type="ReduceMax",
        inputs=inputs,
        outputs=outputs,
        attributes={"axes": params.get("axes", [])},
        name=f"reduce_max_{outputs[0]}" if outputs else "reduce_max",
    )


@register_op("reduce_min", "jax")
def _map_jax_reduce_min(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax reduce min."""
    return Node(
        op_type="ReduceMin",
        inputs=inputs,
        outputs=outputs,
        attributes={"axes": params.get("axes", [])},
        name=f"reduce_min_{outputs[0]}" if outputs else "reduce_min",
    )


@register_op("reduce_prod", "jax")
def _map_jax_reduce_prod(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax reduce prod."""
    return Node(
        op_type="ReduceProd",
        inputs=inputs,
        outputs=outputs,
        attributes={"axes": params.get("axes", [])},
        name=f"reduce_prod_{outputs[0]}" if outputs else "reduce_prod",
    )


@register_op("reduce_window_max", "jax")
def _map_jax_reduce_window_max(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """map jax reduce window max."""
    return Node(
        op_type="MaxPool",
        inputs=inputs,
        outputs=outputs,
        attributes={
            "kernel_shape": params.get("window_dimensions", []),
            "strides": params.get("window_strides", []),
            "pads": params.get("padding", []),
        },
        name=f"reduce_window_max_{outputs[0]}" if outputs else "reduce_window_max",
    )


@register_op("reduce_window_sum", "jax")
def _map_jax_reduce_window_sum(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """map jax reduce window sum."""
    return Node(
        op_type="AveragePool",
        inputs=inputs,
        outputs=outputs,
        attributes={
            "kernel_shape": params.get("window_dimensions", []),
            "strides": params.get("window_strides", []),
            "pads": params.get("padding", []),
        },
        name=f"reduce_window_sum_{outputs[0]}" if outputs else "reduce_window_sum",
    )


@register_op("pad", "jax")
def _map_jax_pad(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax pad."""
    return Node(
        op_type="Pad",
        inputs=inputs,
        outputs=outputs,
        attributes={"padding_config": str(params.get("padding_config", []))},
        name=f"pad_{outputs[0]}" if outputs else "pad",
    )


@register_op("slice", "jax")
def _map_jax_slice(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax slice."""
    return Node(
        op_type="Slice",
        inputs=inputs,
        outputs=outputs,
        attributes={
            "start_indices": params.get("start_indices", []),
            "limit_indices": params.get("limit_indices", []),
            "strides": params.get("strides", []),
        },
        name=f"slice_{outputs[0]}" if outputs else "slice",
    )


@register_op("dynamic_slice", "jax")
def _map_jax_dynamic_slice(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax dynamic slice."""
    return Node(
        op_type="Slice",
        inputs=inputs,
        outputs=outputs,
        attributes={"slice_sizes": params.get("slice_sizes", [])},
        name=f"dynamic_slice_{outputs[0]}" if outputs else "dynamic_slice",
    )


@register_op("gather", "jax")
def _map_jax_gather(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax gather."""
    return Node(
        op_type="GatherElements",
        inputs=inputs,
        outputs=outputs,
        attributes={"dimension_numbers": str(params.get("dimension_numbers", {}))},
        name=f"gather_{outputs[0]}" if outputs else "gather",
    )


@register_op("scatter", "jax")
def _map_jax_scatter(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax scatter."""
    return Node(
        op_type="ScatterND",
        inputs=inputs,
        outputs=outputs,
        attributes={"dimension_numbers": str(params.get("dimension_numbers", {}))},
        name=f"scatter_{outputs[0]}" if outputs else "scatter",
    )


@register_op("cond", "jax")
def _map_jax_cond(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax cond."""
    return Node(
        op_type="If",
        inputs=inputs,
        outputs=outputs,
        attributes={
            "true_branch": str(params.get("true_jaxpr", {})),
            "false_branch": str(params.get("false_jaxpr", {})),
        },
        name=f"cond_{outputs[0]}" if outputs else "cond",
    )


@register_op("scan", "jax")
def _map_jax_scan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax scan."""
    return Node(
        op_type="Scan",
        inputs=inputs,
        outputs=outputs,
        attributes={"body": str(params.get("jaxpr", {}))},
        name=f"scan_{outputs[0]}" if outputs else "scan",
    )


@register_op("while_loop", "jax")
def _map_jax_while_loop(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax while loop."""
    return Node(
        op_type="Loop",
        inputs=inputs,
        outputs=outputs,
        attributes={
            "cond": str(params.get("cond_jaxpr", {})),
            "body": str(params.get("body_jaxpr", {})),
        },
        name=f"while_loop_{outputs[0]}" if outputs else "while_loop",
    )
