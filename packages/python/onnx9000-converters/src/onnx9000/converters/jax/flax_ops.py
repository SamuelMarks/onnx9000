"""Module providing core logic and structural definitions for flax ops."""

from typing import Any

from onnx9000.core.ir import Node
from onnx9000.core.registry import register_op


@register_op("config", "flax")
def _map_flax_config(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_config operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="config")


@register_op("configurations.Config", "flax")
def _map_flax_configurations_Config(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_Config operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.Config")


@register_op("configurations.config", "flax")
def _map_flax_configurations_config(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_config operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.config")


@register_op("configurations.FlagHolder", "flax")
def _map_flax_configurations_FlagHolder(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_FlagHolder operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.FlagHolder"
    )


@register_op("configurations.bool_flag", "flax")
def _map_flax_configurations_bool_flag(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_bool_flag operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.bool_flag")


@register_op("configurations.int_flag", "flax")
def _map_flax_configurations_int_flag(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_int_flag operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.int_flag")


@register_op("configurations.static_bool_env", "flax")
def _map_flax_configurations_static_bool_env(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_static_bool_env operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.static_bool_env"
    )


@register_op("configurations.static_int_env", "flax")
def _map_flax_configurations_static_int_env(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_static_int_env operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.static_int_env"
    )


@register_op("configurations.flax_filter_frames", "flax")
def _map_flax_configurations_flax_filter_frames(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_filter_frames operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.flax_filter_frames"
    )


@register_op("configurations.flax_profile", "flax")
def _map_flax_configurations_flax_profile(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_profile operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.flax_profile"
    )


@register_op("configurations.flax_use_orbax_checkpointing", "flax")
def _map_flax_configurations_flax_use_orbax_checkpointing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_use_orbax_checkpointing operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="configurations.flax_use_orbax_checkpointing",
    )


@register_op("configurations.flax_preserve_adopted_names", "flax")
def _map_flax_configurations_flax_preserve_adopted_names(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_preserve_adopted_names operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="configurations.flax_preserve_adopted_names",
    )


@register_op("configurations.flax_return_frozendict", "flax")
def _map_flax_configurations_flax_return_frozendict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_return_frozendict operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="configurations.flax_return_frozendict",
    )


@register_op("configurations.flax_fix_rng", "flax")
def _map_flax_configurations_flax_fix_rng(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_fix_rng operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.flax_fix_rng"
    )


@register_op("configurations.flax_use_flaxlib", "flax")
def _map_flax_configurations_flax_use_flaxlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_use_flaxlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.flax_use_flaxlib"
    )


@register_op("configurations.flax_array_ref", "flax")
def _map_flax_configurations_flax_array_ref(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_array_ref operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.flax_array_ref"
    )


@register_op("configurations.flax_pytree_module", "flax")
def _map_flax_configurations_flax_pytree_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_pytree_module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.flax_pytree_module"
    )


@register_op("configurations.flax_max_repr_depth", "flax")
def _map_flax_configurations_flax_max_repr_depth(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_max_repr_depth operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="configurations.flax_max_repr_depth",
    )


@register_op("configurations.flax_always_shard_variable", "flax")
def _map_flax_configurations_flax_always_shard_variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_always_shard_variable operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="configurations.flax_always_shard_variable",
    )


@register_op("configurations.flax_hijax_variable", "flax")
def _map_flax_configurations_flax_hijax_variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_flax_hijax_variable operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="configurations.flax_hijax_variable",
    )


@register_op("configurations.nnx_graph_mode", "flax")
def _map_flax_configurations_nnx_graph_mode(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_nnx_graph_mode operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.nnx_graph_mode"
    )


@register_op("configurations.nnx_graph_updates", "flax")
def _map_flax_configurations_nnx_graph_updates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_configurations_nnx_graph_updates operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="configurations.nnx_graph_updates"
    )


@register_op("ids.UUIDManager", "flax")
def _map_flax_ids_UUIDManager(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_ids_UUIDManager operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ids.UUIDManager")


@register_op("ids.uuid", "flax")
def _map_flax_ids_uuid(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_ids_uuid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ids.uuid")


@register_op("ids.FlaxId", "flax")
def _map_flax_ids_FlaxId(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_ids_FlaxId operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="ids.FlaxId")


@register_op("struct.serialization", "flax")
def _map_flax_struct_serialization(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_struct_serialization operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="struct.serialization")


@register_op("struct.field", "flax")
def _map_flax_struct_field(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_struct_field operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="struct.field")


@register_op("struct.dataclass", "flax")
def _map_flax_struct_dataclass(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_struct_dataclass operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="struct.dataclass")


@register_op("struct.TNode", "flax")
def _map_flax_struct_TNode(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_struct_TNode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="struct.TNode")


@register_op("struct.PyTreeNode", "flax")
def _map_flax_struct_PyTreeNode(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_struct_PyTreeNode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="struct.PyTreeNode")


@register_op("jax_utils.replicate", "flax")
def _map_flax_jax_utils_replicate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_jax_utils_replicate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="jax_utils.replicate")


@register_op("jax_utils.unreplicate", "flax")
def _map_flax_jax_utils_unreplicate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_jax_utils_unreplicate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="jax_utils.unreplicate")


@register_op("jax_utils.pmean", "flax")
def _map_flax_jax_utils_pmean(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_jax_utils_pmean operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="jax_utils.pmean")


@register_op("jax_utils.partial_eval_by_shape", "flax")
def _map_flax_jax_utils_partial_eval_by_shape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_jax_utils_partial_eval_by_shape operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="jax_utils.partial_eval_by_shape"
    )


@register_op("jax_utils.prefetch_to_device", "flax")
def _map_flax_jax_utils_prefetch_to_device(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_jax_utils_prefetch_to_device operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="jax_utils.prefetch_to_device"
    )


@register_op("jax_utils.scan_in_dim", "flax")
def _map_flax_jax_utils_scan_in_dim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_jax_utils_scan_in_dim operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="jax_utils.scan_in_dim")


@register_op("jax_utils.pad_shard_unpad", "flax")
def _map_flax_jax_utils_pad_shard_unpad(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_jax_utils_pad_shard_unpad operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="jax_utils.pad_shard_unpad"
    )


@register_op("typing.FrozenDict", "flax")
def _map_flax_typing_FrozenDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_FrozenDict operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.FrozenDict")


@register_op("typing.Array", "flax")
def _map_flax_typing_Array(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Array")


@register_op("typing.PRNGKey", "flax")
def _map_flax_typing_PRNGKey(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_PRNGKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.PRNGKey")


@register_op("typing.RNGSequences", "flax")
def _map_flax_typing_RNGSequences(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_RNGSequences operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.RNGSequences")


@register_op("typing.Dtype", "flax")
def _map_flax_typing_Dtype(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Dtype")


@register_op("typing.Shape", "flax")
def _map_flax_typing_Shape(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_Shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Shape")


@register_op("typing.K", "flax")
def _map_flax_typing_K(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_K operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.K")


@register_op("typing.Key", "flax")
def _map_flax_typing_Key(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_Key operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Key")


@register_op("typing.is_key_like", "flax")
def _map_flax_typing_is_key_like(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_is_key_like operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.is_key_like")


@register_op("typing.Path", "flax")
def _map_flax_typing_Path(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_Path operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Path")


@register_op("typing.PathParts", "flax")
def _map_flax_typing_PathParts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_PathParts operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.PathParts")


@register_op("typing.Leaf", "flax")
def _map_flax_typing_Leaf(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_Leaf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Leaf")


@register_op("typing.PrecisionLike", "flax")
def _map_flax_typing_PrecisionLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_PrecisionLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.PrecisionLike")


@register_op("typing.DotGeneralT", "flax")
def _map_flax_typing_DotGeneralT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_DotGeneralT operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.DotGeneralT")


@register_op("typing.ConvGeneralDilatedT", "flax")
def _map_flax_typing_ConvGeneralDilatedT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_ConvGeneralDilatedT operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="typing.ConvGeneralDilatedT"
    )


@register_op("typing.EinsumT", "flax")
def _map_flax_typing_EinsumT(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_EinsumT operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.EinsumT")


@register_op("typing.PaddingLike", "flax")
def _map_flax_typing_PaddingLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_PaddingLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.PaddingLike")


@register_op("typing.LaxPadding", "flax")
def _map_flax_typing_LaxPadding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_LaxPadding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.LaxPadding")


@register_op("typing.Initializer", "flax")
def _map_flax_typing_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_Initializer operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Initializer")


@register_op("typing.Collection", "flax")
def _map_flax_typing_Collection(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_Collection operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Collection")


@register_op("typing.MutableCollection", "flax")
def _map_flax_typing_MutableCollection(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_MutableCollection operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.MutableCollection")


@register_op("typing.VariableDict", "flax")
def _map_flax_typing_VariableDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_VariableDict operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.VariableDict")


@register_op("typing.FrozenVariableDict", "flax")
def _map_flax_typing_FrozenVariableDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_FrozenVariableDict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="typing.FrozenVariableDict"
    )


@register_op("typing.MutableVariableDict", "flax")
def _map_flax_typing_MutableVariableDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_MutableVariableDict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="typing.MutableVariableDict"
    )


@register_op("typing.PRNGFoldable", "flax")
def _map_flax_typing_PRNGFoldable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_PRNGFoldable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.PRNGFoldable")


@register_op("typing.T", "flax")
def _map_flax_typing_T(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_T operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.T")


@register_op("typing.In", "flax")
def _map_flax_typing_In(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_In operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.In")


@register_op("typing.Out", "flax")
def _map_flax_typing_Out(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_Out operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Out")


@register_op("typing.Axis", "flax")
def _map_flax_typing_Axis(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_Axis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Axis")


@register_op("typing.InOutAxis", "flax")
def _map_flax_typing_InOutAxis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_InOutAxis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.InOutAxis")


@register_op("typing.ScanAxis", "flax")
def _map_flax_typing_ScanAxis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_ScanAxis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.ScanAxis")


@register_op("typing.InOutScanAxis", "flax")
def _map_flax_typing_InOutScanAxis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_InOutScanAxis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.InOutScanAxis")


@register_op("typing.Axes", "flax")
def _map_flax_typing_Axes(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_Axes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Axes")


@register_op("typing.LogicalNames", "flax")
def _map_flax_typing_LogicalNames(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_LogicalNames operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.LogicalNames")


@register_op("typing.AxisName", "flax")
def _map_flax_typing_AxisName(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_AxisName operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.AxisName")


@register_op("typing.LogicalRules", "flax")
def _map_flax_typing_LogicalRules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_LogicalRules operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.LogicalRules")


@register_op("typing.ArrayPytree", "flax")
def _map_flax_typing_ArrayPytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_ArrayPytree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.ArrayPytree")


@register_op("typing.LogicalPartitionSpec", "flax")
def _map_flax_typing_LogicalPartitionSpec(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_LogicalPartitionSpec operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="typing.LogicalPartitionSpec"
    )


@register_op("typing.LogicalPartitionSpecPytree", "flax")
def _map_flax_typing_LogicalPartitionSpecPytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_LogicalPartitionSpecPytree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="typing.LogicalPartitionSpecPytree"
    )


@register_op("typing.PartitionSpecPytree", "flax")
def _map_flax_typing_PartitionSpecPytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_PartitionSpecPytree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="typing.PartitionSpecPytree"
    )


@register_op("typing.Sharding", "flax")
def _map_flax_typing_Sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_Sharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Sharding")


@register_op("typing.A", "flax")
def _map_flax_typing_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.A")


@register_op("typing.HA", "flax")
def _map_flax_typing_HA(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_HA operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.HA")


@register_op("typing.HB", "flax")
def _map_flax_typing_HB(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_HB operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.HB")


@register_op("typing.PytreeDeque", "flax")
def _map_flax_typing_PytreeDeque(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_PytreeDeque operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.PytreeDeque")


@register_op("typing.Missing", "flax")
def _map_flax_typing_Missing(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_Missing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.Missing")


@register_op("typing.MISSING", "flax")
def _map_flax_typing_MISSING(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_MISSING operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.MISSING")


@register_op("typing.ShapeDtype", "flax")
def _map_flax_typing_ShapeDtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_ShapeDtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.ShapeDtype")


@register_op("typing.has_shape_dtype", "flax")
def _map_flax_typing_has_shape_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_has_shape_dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.has_shape_dtype")


@register_op("typing.SizeBytes", "flax")
def _map_flax_typing_SizeBytes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_SizeBytes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.SizeBytes")


@register_op("typing.TupleArg", "flax")
def _map_flax_typing_TupleArg(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_TupleArg operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.TupleArg")


@register_op("typing.PromoteDtypeFn", "flax")
def _map_flax_typing_PromoteDtypeFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_PromoteDtypeFn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.PromoteDtypeFn")


@register_op("typing.HashableMapping", "flax")
def _map_flax_typing_HashableMapping(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_HashableMapping operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.HashableMapping")


@register_op("typing.F", "flax")
def _map_flax_typing_F(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_typing_F operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.F")


@register_op("typing.BaseConfigContext", "flax")
def _map_flax_typing_BaseConfigContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_typing_BaseConfigContext operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="typing.BaseConfigContext")


@register_op("traceback_util.config", "flax")
def _map_flax_traceback_util_config(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traceback_util_config operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="traceback_util.config")


@register_op("traceback_util.api_boundary", "flax")
def _map_flax_traceback_util_api_boundary(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traceback_util_api_boundary operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traceback_util.api_boundary"
    )


@register_op("traceback_util.register_exclusion", "flax")
def _map_flax_traceback_util_register_exclusion(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traceback_util_register_exclusion operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traceback_util.register_exclusion"
    )


@register_op("traceback_util.hide_flax_in_tracebacks", "flax")
def _map_flax_traceback_util_hide_flax_in_tracebacks(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traceback_util_hide_flax_in_tracebacks operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="traceback_util.hide_flax_in_tracebacks",
    )


@register_op("traceback_util.show_flax_in_tracebacks", "flax")
def _map_flax_traceback_util_show_flax_in_tracebacks(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traceback_util_show_flax_in_tracebacks operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="traceback_util.show_flax_in_tracebacks",
    )


@register_op("cursor.FrozenDict", "flax")
def _map_flax_cursor_FrozenDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_cursor_FrozenDict operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="cursor.FrozenDict")


@register_op("cursor.CursorFindError", "flax")
def _map_flax_cursor_CursorFindError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_cursor_CursorFindError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="cursor.CursorFindError")


@register_op("cursor.TraverseTreeError", "flax")
def _map_flax_cursor_TraverseTreeError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_cursor_TraverseTreeError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="cursor.TraverseTreeError")


@register_op("cursor.A", "flax")
def _map_flax_cursor_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_cursor_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="cursor.A")


@register_op("cursor.Key", "flax")
def _map_flax_cursor_Key(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_cursor_Key operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="cursor.Key")


@register_op("cursor.Indexable", "flax")
def _map_flax_cursor_Indexable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_cursor_Indexable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="cursor.Indexable")


@register_op("cursor.AccessType", "flax")
def _map_flax_cursor_AccessType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_cursor_AccessType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="cursor.AccessType")


@register_op("cursor.ParentKey", "flax")
def _map_flax_cursor_ParentKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_cursor_ParentKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="cursor.ParentKey")


@register_op("cursor.is_named_tuple", "flax")
def _map_flax_cursor_is_named_tuple(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_cursor_is_named_tuple operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="cursor.is_named_tuple")


@register_op("cursor.Cursor", "flax")
def _map_flax_cursor_Cursor(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_cursor_Cursor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="cursor.Cursor")


@register_op("cursor.cursor", "flax")
def _map_flax_cursor_cursor(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_cursor_cursor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="cursor.cursor")


@register_op("serialization.current_path", "flax")
def _map_flax_serialization_current_path(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_serialization_current_path operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="serialization.current_path"
    )


@register_op("serialization.from_state_dict", "flax")
def _map_flax_serialization_from_state_dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_serialization_from_state_dict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="serialization.from_state_dict"
    )


@register_op("serialization.to_state_dict", "flax")
def _map_flax_serialization_to_state_dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_serialization_to_state_dict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="serialization.to_state_dict"
    )


@register_op("serialization.is_serializable", "flax")
def _map_flax_serialization_is_serializable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_serialization_is_serializable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="serialization.is_serializable"
    )


@register_op("serialization.register_serialization_state", "flax")
def _map_flax_serialization_register_serialization_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_serialization_register_serialization_state operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="serialization.register_serialization_state",
    )


@register_op("serialization.MAX_CHUNK_SIZE", "flax")
def _map_flax_serialization_MAX_CHUNK_SIZE(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_serialization_MAX_CHUNK_SIZE operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="serialization.MAX_CHUNK_SIZE"
    )


@register_op("serialization.msgpack_serialize", "flax")
def _map_flax_serialization_msgpack_serialize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_serialization_msgpack_serialize operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="serialization.msgpack_serialize"
    )


@register_op("serialization.msgpack_restore", "flax")
def _map_flax_serialization_msgpack_restore(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_serialization_msgpack_restore operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="serialization.msgpack_restore"
    )


@register_op("serialization.from_bytes", "flax")
def _map_flax_serialization_from_bytes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_serialization_from_bytes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="serialization.from_bytes")


@register_op("serialization.to_bytes", "flax")
def _map_flax_serialization_to_bytes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_serialization_to_bytes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="serialization.to_bytes")


@register_op("io.errors", "flax")
def _map_flax_io_errors(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_errors operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.errors")


@register_op("io.BackendMode", "flax")
def _map_flax_io_BackendMode(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_BackendMode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.BackendMode")


@register_op("io.io_mode", "flax")
def _map_flax_io_io_mode(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_io_mode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.io_mode")


@register_op("io.NotFoundError", "flax")
def _map_flax_io_NotFoundError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_io_NotFoundError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.NotFoundError")


@register_op("io.override_mode", "flax")
def _map_flax_io_override_mode(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_io_override_mode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.override_mode")


@register_op("io.set_mode", "flax")
def _map_flax_io_set_mode(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_set_mode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.set_mode")


@register_op("io.GFile", "flax")
def _map_flax_io_GFile(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_GFile operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.GFile")


@register_op("io.listdir", "flax")
def _map_flax_io_listdir(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_listdir operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.listdir")


@register_op("io.isdir", "flax")
def _map_flax_io_isdir(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_isdir operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.isdir")


@register_op("io.copy", "flax")
def _map_flax_io_copy(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_copy operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.copy")


@register_op("io.rename", "flax")
def _map_flax_io_rename(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_rename operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.rename")


@register_op("io.exists", "flax")
def _map_flax_io_exists(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_exists operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.exists")


@register_op("io.makedirs", "flax")
def _map_flax_io_makedirs(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_makedirs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.makedirs")


@register_op("io.glob", "flax")
def _map_flax_io_glob(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_glob operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.glob")


@register_op("io.remove", "flax")
def _map_flax_io_remove(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_remove operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.remove")


@register_op("io.rmtree", "flax")
def _map_flax_io_rmtree(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_rmtree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.rmtree")


@register_op("io.getsize", "flax")
def _map_flax_io_getsize(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_io_getsize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="io.getsize")


@register_op("traverse_util.VariableDict", "flax")
def _map_flax_traverse_util_VariableDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_VariableDict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.VariableDict"
    )


@register_op("traverse_util.PathParts", "flax")
def _map_flax_traverse_util_PathParts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_PathParts operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.PathParts")


@register_op("traverse_util.struct", "flax")
def _map_flax_traverse_util_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_struct operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.struct")


@register_op("traverse_util.empty_node", "flax")
def _map_flax_traverse_util_empty_node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_empty_node operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.empty_node")


@register_op("traverse_util.flatten_dict", "flax")
def _map_flax_traverse_util_flatten_dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_flatten_dict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.flatten_dict"
    )


@register_op("traverse_util.unflatten_dict", "flax")
def _map_flax_traverse_util_unflatten_dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_unflatten_dict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.unflatten_dict"
    )


@register_op("traverse_util.path_aware_map", "flax")
def _map_flax_traverse_util_path_aware_map(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_path_aware_map operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.path_aware_map"
    )


@register_op("traverse_util.Traversal", "flax")
def _map_flax_traverse_util_Traversal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_Traversal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.Traversal")


@register_op("traverse_util.TraverseId", "flax")
def _map_flax_traverse_util_TraverseId(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_TraverseId operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.TraverseId")


@register_op("traverse_util.t_identity", "flax")
def _map_flax_traverse_util_t_identity(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_t_identity operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.t_identity")


@register_op("traverse_util.TraverseMerge", "flax")
def _map_flax_traverse_util_TraverseMerge(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_TraverseMerge operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.TraverseMerge"
    )


@register_op("traverse_util.TraverseCompose", "flax")
def _map_flax_traverse_util_TraverseCompose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_TraverseCompose operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.TraverseCompose"
    )


@register_op("traverse_util.TraverseFilter", "flax")
def _map_flax_traverse_util_TraverseFilter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_TraverseFilter operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.TraverseFilter"
    )


@register_op("traverse_util.TraverseAttr", "flax")
def _map_flax_traverse_util_TraverseAttr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_TraverseAttr operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.TraverseAttr"
    )


@register_op("traverse_util.TraverseItem", "flax")
def _map_flax_traverse_util_TraverseItem(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_TraverseItem operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.TraverseItem"
    )


@register_op("traverse_util.TraverseEach", "flax")
def _map_flax_traverse_util_TraverseEach(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_TraverseEach operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.TraverseEach"
    )


@register_op("traverse_util.TraverseTree", "flax")
def _map_flax_traverse_util_TraverseTree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_TraverseTree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.TraverseTree"
    )


@register_op("traverse_util.ModelParamTraversal", "flax")
def _map_flax_traverse_util_ModelParamTraversal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_traverse_util_ModelParamTraversal operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="traverse_util.ModelParamTraversal"
    )


@register_op("errors.FlaxError", "flax")
def _map_flax_errors_FlaxError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_FlaxError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.FlaxError")


@register_op("errors.TraceContextError", "flax")
def _map_flax_errors_TraceContextError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_TraceContextError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.TraceContextError")


@register_op("errors.LazyInitError", "flax")
def _map_flax_errors_LazyInitError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_LazyInitError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.LazyInitError")


@register_op("errors.InvalidRngError", "flax")
def _map_flax_errors_InvalidRngError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_InvalidRngError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.InvalidRngError")


@register_op("errors.ApplyScopeInvalidVariablesTypeError", "flax")
def _map_flax_errors_ApplyScopeInvalidVariablesTypeError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_ApplyScopeInvalidVariablesTypeError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.ApplyScopeInvalidVariablesTypeError",
    )


@register_op("errors.ApplyScopeInvalidVariablesStructureError", "flax")
def _map_flax_errors_ApplyScopeInvalidVariablesStructureError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_ApplyScopeInvalidVariablesStructureError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.ApplyScopeInvalidVariablesStructureError",
    )


@register_op("errors.ScopeParamNotFoundError", "flax")
def _map_flax_errors_ScopeParamNotFoundError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_ScopeParamNotFoundError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.ScopeParamNotFoundError"
    )


@register_op("errors.ScopeCollectionNotFound", "flax")
def _map_flax_errors_ScopeCollectionNotFound(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_ScopeCollectionNotFound operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.ScopeCollectionNotFound"
    )


@register_op("errors.ScopeParamShapeError", "flax")
def _map_flax_errors_ScopeParamShapeError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_ScopeParamShapeError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.ScopeParamShapeError"
    )


@register_op("errors.ScopeVariableNotFoundError", "flax")
def _map_flax_errors_ScopeVariableNotFoundError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_ScopeVariableNotFoundError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.ScopeVariableNotFoundError"
    )


@register_op("errors.InvalidFilterError", "flax")
def _map_flax_errors_InvalidFilterError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_InvalidFilterError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.InvalidFilterError"
    )


@register_op("errors.InvalidScopeError", "flax")
def _map_flax_errors_InvalidScopeError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_InvalidScopeError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.InvalidScopeError")


@register_op("errors.ModifyScopeVariableError", "flax")
def _map_flax_errors_ModifyScopeVariableError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_ModifyScopeVariableError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.ModifyScopeVariableError"
    )


@register_op("errors.ImmutableVariableError", "flax")
def _map_flax_errors_ImmutableVariableError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_ImmutableVariableError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.ImmutableVariableError"
    )


@register_op("errors.JaxTransformError", "flax")
def _map_flax_errors_JaxTransformError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_JaxTransformError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.JaxTransformError")


@register_op("errors.PartitioningUnspecifiedError", "flax")
def _map_flax_errors_PartitioningUnspecifiedError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_PartitioningUnspecifiedError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.PartitioningUnspecifiedError",
    )


@register_op("errors.NameInUseError", "flax")
def _map_flax_errors_NameInUseError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_NameInUseError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.NameInUseError")


@register_op("errors.AssignSubModuleError", "flax")
def _map_flax_errors_AssignSubModuleError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_AssignSubModuleError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.AssignSubModuleError"
    )


@register_op("errors.SetAttributeInModuleSetupError", "flax")
def _map_flax_errors_SetAttributeInModuleSetupError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_SetAttributeInModuleSetupError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.SetAttributeInModuleSetupError",
    )


@register_op("errors.SetAttributeFrozenModuleError", "flax")
def _map_flax_errors_SetAttributeFrozenModuleError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_SetAttributeFrozenModuleError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.SetAttributeFrozenModuleError",
    )


@register_op("errors.MultipleMethodsCompactError", "flax")
def _map_flax_errors_MultipleMethodsCompactError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_MultipleMethodsCompactError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.MultipleMethodsCompactError",
    )


@register_op("errors.ReservedModuleAttributeError", "flax")
def _map_flax_errors_ReservedModuleAttributeError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_ReservedModuleAttributeError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.ReservedModuleAttributeError",
    )


@register_op("errors.ApplyModuleInvalidMethodError", "flax")
def _map_flax_errors_ApplyModuleInvalidMethodError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_ApplyModuleInvalidMethodError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.ApplyModuleInvalidMethodError",
    )


@register_op("errors.CallCompactUnboundModuleError", "flax")
def _map_flax_errors_CallCompactUnboundModuleError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_CallCompactUnboundModuleError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.CallCompactUnboundModuleError",
    )


@register_op("errors.CallSetupUnboundModuleError", "flax")
def _map_flax_errors_CallSetupUnboundModuleError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_CallSetupUnboundModuleError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.CallSetupUnboundModuleError",
    )


@register_op("errors.CallUnbindOnUnboundModuleError", "flax")
def _map_flax_errors_CallUnbindOnUnboundModuleError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_CallUnbindOnUnboundModuleError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.CallUnbindOnUnboundModuleError",
    )


@register_op("errors.CallShareScopeOnUnboundModuleError", "flax")
def _map_flax_errors_CallShareScopeOnUnboundModuleError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_CallShareScopeOnUnboundModuleError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.CallShareScopeOnUnboundModuleError",
    )


@register_op("errors.InvalidInstanceModuleError", "flax")
def _map_flax_errors_InvalidInstanceModuleError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_InvalidInstanceModuleError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.InvalidInstanceModuleError"
    )


@register_op("errors.IncorrectPostInitOverrideError", "flax")
def _map_flax_errors_IncorrectPostInitOverrideError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_IncorrectPostInitOverrideError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.IncorrectPostInitOverrideError",
    )


@register_op("errors.DescriptorAttributeError", "flax")
def _map_flax_errors_DescriptorAttributeError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_DescriptorAttributeError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.DescriptorAttributeError"
    )


@register_op("errors.InvalidCheckpointError", "flax")
def _map_flax_errors_InvalidCheckpointError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_InvalidCheckpointError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.InvalidCheckpointError"
    )


@register_op("errors.MPACheckpointingRequiredError", "flax")
def _map_flax_errors_MPACheckpointingRequiredError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_MPACheckpointingRequiredError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.MPACheckpointingRequiredError",
    )


@register_op("errors.MPARestoreTargetRequiredError", "flax")
def _map_flax_errors_MPARestoreTargetRequiredError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_MPARestoreTargetRequiredError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.MPARestoreTargetRequiredError",
    )


@register_op("errors.MPARestoreDataCorruptedError", "flax")
def _map_flax_errors_MPARestoreDataCorruptedError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_MPARestoreDataCorruptedError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.MPARestoreDataCorruptedError",
    )


@register_op("errors.TransformedMethodReturnValueError", "flax")
def _map_flax_errors_TransformedMethodReturnValueError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_TransformedMethodReturnValueError operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="errors.TransformedMethodReturnValueError",
    )


@register_op("errors.TransformTargetError", "flax")
def _map_flax_errors_TransformTargetError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_TransformTargetError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.TransformTargetError"
    )


@register_op("errors.AlreadyExistsError", "flax")
def _map_flax_errors_AlreadyExistsError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_AlreadyExistsError operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="errors.AlreadyExistsError"
    )


@register_op("errors.CursorFindError", "flax")
def _map_flax_errors_CursorFindError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_CursorFindError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.CursorFindError")


@register_op("errors.TraverseTreeError", "flax")
def _map_flax_errors_TraverseTreeError(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_errors_TraverseTreeError operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="errors.TraverseTreeError")


@register_op("linen.DenyList", "flax")
def _map_flax_linen_DenyList(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_DenyList operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.DenyList")


@register_op("linen.FrozenDict", "flax")
def _map_flax_linen_FrozenDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_FrozenDict operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.FrozenDict")


@register_op("linen.broadcast", "flax")
def _map_flax_linen_broadcast(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_broadcast operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.broadcast")


@register_op("linen.meta", "flax")
def _map_flax_linen_meta(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_meta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.meta")


@register_op("linen.PARTITION_NAME", "flax")
def _map_flax_linen_PARTITION_NAME(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_PARTITION_NAME operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.PARTITION_NAME")


@register_op("linen.Partitioned", "flax")
def _map_flax_linen_Partitioned(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_Partitioned operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Partitioned")


@register_op("linen.get_partition_spec", "flax")
def _map_flax_linen_get_partition_spec(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_get_partition_spec operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.get_partition_spec")


@register_op("linen.get_sharding", "flax")
def _map_flax_linen_get_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_get_sharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.get_sharding")


@register_op("linen.unbox", "flax")
def _map_flax_linen_unbox(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_unbox operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.unbox")


@register_op("linen.with_partitioning", "flax")
def _map_flax_linen_with_partitioning(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_with_partitioning operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.with_partitioning")


@register_op("linen.get_logical_axis_rules", "flax")
def _map_flax_linen_get_logical_axis_rules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_get_logical_axis_rules operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.get_logical_axis_rules"
    )


@register_op("linen.logical_axis_rules", "flax")
def _map_flax_linen_logical_axis_rules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_logical_axis_rules operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.logical_axis_rules")


@register_op("linen.set_logical_axis_rules", "flax")
def _map_flax_linen_set_logical_axis_rules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_set_logical_axis_rules operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.set_logical_axis_rules"
    )


@register_op("linen.PReLU", "flax")
def _map_flax_linen_PReLU(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_PReLU operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.PReLU")


@register_op("linen.celu", "flax")
def _map_flax_linen_celu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_celu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.celu")


@register_op("linen.elu", "flax")
def _map_flax_linen_elu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_elu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.elu")


@register_op("linen.gelu", "flax")
def _map_flax_linen_gelu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_gelu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.gelu")


@register_op("linen.glu", "flax")
def _map_flax_linen_glu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_glu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.glu")


@register_op("linen.hard_sigmoid", "flax")
def _map_flax_linen_hard_sigmoid(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_hard_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.hard_sigmoid")


@register_op("linen.hard_silu", "flax")
def _map_flax_linen_hard_silu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_hard_silu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.hard_silu")


@register_op("linen.hard_swish", "flax")
def _map_flax_linen_hard_swish(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_hard_swish operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.hard_swish")


@register_op("linen.hard_tanh", "flax")
def _map_flax_linen_hard_tanh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_hard_tanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.hard_tanh")


@register_op("linen.leaky_relu", "flax")
def _map_flax_linen_leaky_relu(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_leaky_relu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.leaky_relu")


@register_op("linen.log_sigmoid", "flax")
def _map_flax_linen_log_sigmoid(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_log_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.log_sigmoid")


@register_op("linen.log_softmax", "flax")
def _map_flax_linen_log_softmax(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_log_softmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.log_softmax")


@register_op("linen.logsumexp", "flax")
def _map_flax_linen_logsumexp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_logsumexp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.logsumexp")


@register_op("linen.normalize", "flax")
def _map_flax_linen_normalize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalize")


@register_op("linen.one_hot", "flax")
def _map_flax_linen_one_hot(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_one_hot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.one_hot")


@register_op("linen.relu6", "flax")
def _map_flax_linen_relu6(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_relu6 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.relu6")


@register_op("linen.relu", "flax")
def _map_flax_linen_relu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_relu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.relu")


@register_op("linen.selu", "flax")
def _map_flax_linen_selu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_selu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.selu")


@register_op("linen.sigmoid", "flax")
def _map_flax_linen_sigmoid(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.sigmoid")


@register_op("linen.silu", "flax")
def _map_flax_linen_silu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_silu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.silu")


@register_op("linen.soft_sign", "flax")
def _map_flax_linen_soft_sign(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_soft_sign operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.soft_sign")


@register_op("linen.softmax", "flax")
def _map_flax_linen_softmax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_softmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.softmax")


@register_op("linen.softplus", "flax")
def _map_flax_linen_softplus(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_softplus operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.softplus")


@register_op("linen.standardize", "flax")
def _map_flax_linen_standardize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_standardize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.standardize")


@register_op("linen.swish", "flax")
def _map_flax_linen_swish(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_swish operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.swish")


@register_op("linen.tanh", "flax")
def _map_flax_linen_tanh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_tanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.tanh")


@register_op("linen.MultiHeadAttention", "flax")
def _map_flax_linen_MultiHeadAttention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_MultiHeadAttention operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.MultiHeadAttention")


@register_op("linen.MultiHeadDotProductAttention", "flax")
def _map_flax_linen_MultiHeadDotProductAttention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_MultiHeadDotProductAttention operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.MultiHeadDotProductAttention",
    )


@register_op("linen.SelfAttention", "flax")
def _map_flax_linen_SelfAttention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_SelfAttention operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.SelfAttention")


@register_op("linen.combine_masks", "flax")
def _map_flax_linen_combine_masks(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_combine_masks operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.combine_masks")


@register_op("linen.dot_product_attention_weights", "flax")
def _map_flax_linen_dot_product_attention_weights(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_dot_product_attention_weights operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.dot_product_attention_weights",
    )


@register_op("linen.dot_product_attention", "flax")
def _map_flax_linen_dot_product_attention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_dot_product_attention operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.dot_product_attention"
    )


@register_op("linen.make_attention_mask", "flax")
def _map_flax_linen_make_attention_mask(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_make_attention_mask operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.make_attention_mask"
    )


@register_op("linen.make_causal_mask", "flax")
def _map_flax_linen_make_causal_mask(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_make_causal_mask operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.make_causal_mask")


@register_op("linen.BatchApply", "flax")
def _map_flax_linen_BatchApply(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_BatchApply operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.BatchApply")


@register_op("linen.Sequential", "flax")
def _map_flax_linen_Sequential(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_Sequential operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Sequential")


@register_op("linen.Fp8DirectDotGeneralOp", "flax")
def _map_flax_linen_Fp8DirectDotGeneralOp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_Fp8DirectDotGeneralOp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Fp8DirectDotGeneralOp"
    )


@register_op("linen.Fp8DotGeneral", "flax")
def _map_flax_linen_Fp8DotGeneral(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_Fp8DotGeneral operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Fp8DotGeneral")


@register_op("linen.Fp8DotGeneralOp", "flax")
def _map_flax_linen_Fp8DotGeneralOp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_Fp8DotGeneralOp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Fp8DotGeneralOp")


@register_op("linen.Fp8Einsum", "flax")
def _map_flax_linen_Fp8Einsum(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_Fp8Einsum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Fp8Einsum")


@register_op("linen.NANOOFp8DotGeneralOp", "flax")
def _map_flax_linen_NANOOFp8DotGeneralOp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_NANOOFp8DotGeneralOp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.NANOOFp8DotGeneralOp"
    )


@register_op("linen.ones_init", "flax")
def _map_flax_linen_ones_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_ones_init operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.ones_init")


@register_op("linen.ones", "flax")
def _map_flax_linen_ones(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_ones operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.ones")


@register_op("linen.zeros_init", "flax")
def _map_flax_linen_zeros_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_zeros_init operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.zeros_init")


@register_op("linen.zeros", "flax")
def _map_flax_linen_zeros(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_zeros operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.zeros")


@register_op("linen.ConvLocal", "flax")
def _map_flax_linen_ConvLocal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_ConvLocal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.ConvLocal")


@register_op("linen.ConvTranspose", "flax")
def _map_flax_linen_ConvTranspose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_ConvTranspose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.ConvTranspose")


@register_op("linen.Conv", "flax")
def _map_flax_linen_Conv(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_Conv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Conv")


@register_op("linen.DenseGeneral", "flax")
def _map_flax_linen_DenseGeneral(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_DenseGeneral operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.DenseGeneral")


@register_op("linen.Dense", "flax")
def _map_flax_linen_Dense(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_Dense operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Dense")


@register_op("linen.Einsum", "flax")
def _map_flax_linen_Einsum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_Einsum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Einsum")


@register_op("linen.Embed", "flax")
def _map_flax_linen_Embed(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_Embed operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Embed")


@register_op("linen.Module", "flax")
def _map_flax_linen_Module(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Module")


@register_op("linen.Variable", "flax")
def _map_flax_linen_Variable(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_Variable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Variable")


@register_op("linen.apply", "flax")
def _map_flax_linen_apply(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_apply operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.apply")


@register_op("linen.compact_name_scope", "flax")
def _map_flax_linen_compact_name_scope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_compact_name_scope operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.compact_name_scope")


@register_op("linen.compact", "flax")
def _map_flax_linen_compact(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_compact operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.compact")


@register_op("linen.disable_named_call", "flax")
def _map_flax_linen_disable_named_call(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_disable_named_call operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.disable_named_call")


@register_op("linen.enable_named_call", "flax")
def _map_flax_linen_enable_named_call(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_enable_named_call operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.enable_named_call")


@register_op("linen.init_with_output", "flax")
def _map_flax_linen_init_with_output(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_init_with_output operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.init_with_output")


@register_op("linen.init", "flax")
def _map_flax_linen_init(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_init operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.init")


@register_op("linen.intercept_methods", "flax")
def _map_flax_linen_intercept_methods(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_intercept_methods operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.intercept_methods")


@register_op("linen.merge_param", "flax")
def _map_flax_linen_merge_param(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_merge_param operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.merge_param")


@register_op("linen.nowrap", "flax")
def _map_flax_linen_nowrap(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_nowrap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.nowrap")


@register_op("linen.override_named_call", "flax")
def _map_flax_linen_override_named_call(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_override_named_call operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.override_named_call"
    )


@register_op("linen.share_scope", "flax")
def _map_flax_linen_share_scope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_share_scope operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.share_scope")


@register_op("linen.BatchNorm", "flax")
def _map_flax_linen_BatchNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_BatchNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.BatchNorm")


@register_op("linen.GroupNorm", "flax")
def _map_flax_linen_GroupNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_GroupNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.GroupNorm")


@register_op("linen.InstanceNorm", "flax")
def _map_flax_linen_InstanceNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_InstanceNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.InstanceNorm")


@register_op("linen.LayerNorm", "flax")
def _map_flax_linen_LayerNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_LayerNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.LayerNorm")


@register_op("linen.RMSNorm", "flax")
def _map_flax_linen_RMSNorm(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_RMSNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.RMSNorm")


@register_op("linen.SpectralNorm", "flax")
def _map_flax_linen_SpectralNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_SpectralNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.SpectralNorm")


@register_op("linen.WeightNorm", "flax")
def _map_flax_linen_WeightNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_WeightNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.WeightNorm")


@register_op("linen.avg_pool", "flax")
def _map_flax_linen_avg_pool(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_avg_pool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.avg_pool")


@register_op("linen.max_pool", "flax")
def _map_flax_linen_max_pool(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_max_pool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.max_pool")


@register_op("linen.pool", "flax")
def _map_flax_linen_pool(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_pool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.pool")


@register_op("linen.Bidirectional", "flax")
def _map_flax_linen_Bidirectional(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_Bidirectional operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Bidirectional")


@register_op("linen.ConvLSTMCell", "flax")
def _map_flax_linen_ConvLSTMCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_ConvLSTMCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.ConvLSTMCell")


@register_op("linen.GRUCell", "flax")
def _map_flax_linen_GRUCell(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_GRUCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.GRUCell")


@register_op("linen.LSTMCell", "flax")
def _map_flax_linen_LSTMCell(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_LSTMCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.LSTMCell")


@register_op("linen.MGUCell", "flax")
def _map_flax_linen_MGUCell(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_MGUCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.MGUCell")


@register_op("linen.OptimizedLSTMCell", "flax")
def _map_flax_linen_OptimizedLSTMCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_OptimizedLSTMCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.OptimizedLSTMCell")


@register_op("linen.RNNCellBase", "flax")
def _map_flax_linen_RNNCellBase(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_RNNCellBase operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.RNNCellBase")


@register_op("linen.RNN", "flax")
def _map_flax_linen_RNN(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_RNN operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.RNN")


@register_op("linen.SimpleCell", "flax")
def _map_flax_linen_SimpleCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_SimpleCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.SimpleCell")


@register_op("linen.LogicallyPartitioned", "flax")
def _map_flax_linen_LogicallyPartitioned(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_LogicallyPartitioned operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.LogicallyPartitioned"
    )


@register_op("linen.logical_to_mesh", "flax")
def _map_flax_linen_logical_to_mesh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_logical_to_mesh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.logical_to_mesh")


@register_op("linen.logical_to_mesh_axes", "flax")
def _map_flax_linen_logical_to_mesh_axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_logical_to_mesh_axes operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.logical_to_mesh_axes"
    )


@register_op("linen.logical_to_mesh_sharding", "flax")
def _map_flax_linen_logical_to_mesh_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_logical_to_mesh_sharding operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.logical_to_mesh_sharding"
    )


@register_op("linen.with_logical_constraint", "flax")
def _map_flax_linen_with_logical_constraint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_with_logical_constraint operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.with_logical_constraint"
    )


@register_op("linen.with_logical_partitioning", "flax")
def _map_flax_linen_with_logical_partitioning(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_with_logical_partitioning operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.with_logical_partitioning"
    )


@register_op("linen.Dropout", "flax")
def _map_flax_linen_Dropout(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_Dropout operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.Dropout")


@register_op("linen.tabulate", "flax")
def _map_flax_linen_tabulate(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_tabulate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.tabulate")


@register_op("linen.add_metadata_axis", "flax")
def _map_flax_linen_add_metadata_axis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_add_metadata_axis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.add_metadata_axis")


@register_op("linen.checkpoint", "flax")
def _map_flax_linen_checkpoint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_checkpoint operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.checkpoint")


@register_op("linen.cond", "flax")
def _map_flax_linen_cond(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_cond operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.cond")


@register_op("linen.custom_vjp", "flax")
def _map_flax_linen_custom_vjp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_custom_vjp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.custom_vjp")


@register_op("linen.fold_rngs", "flax")
def _map_flax_linen_fold_rngs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fold_rngs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fold_rngs")


@register_op("linen.grad", "flax")
def _map_flax_linen_grad(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_grad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.grad")


@register_op("linen.jit", "flax")
def _map_flax_linen_jit(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_jit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.jit")


@register_op("linen.jvp", "flax")
def _map_flax_linen_jvp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_jvp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.jvp")


@register_op("linen.map_variables", "flax")
def _map_flax_linen_map_variables(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_map_variables operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.map_variables")


@register_op("linen.named_call", "flax")
def _map_flax_linen_named_call(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_named_call operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.named_call")


@register_op("linen.remat_scan", "flax")
def _map_flax_linen_remat_scan(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_remat_scan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.remat_scan")


@register_op("linen.remat", "flax")
def _map_flax_linen_remat(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_remat operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.remat")


@register_op("linen.scan", "flax")
def _map_flax_linen_scan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_scan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.scan")


@register_op("linen.switch", "flax")
def _map_flax_linen_switch(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_switch operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.switch")


@register_op("linen.value_and_grad", "flax")
def _map_flax_linen_value_and_grad(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_value_and_grad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.value_and_grad")


@register_op("linen.vjp", "flax")
def _map_flax_linen_vjp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_vjp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.vjp")


@register_op("linen.vmap", "flax")
def _map_flax_linen_vmap(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_vmap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.vmap")


@register_op("linen.while_loop", "flax")
def _map_flax_linen_while_loop(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_while_loop operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.while_loop")


@register_op("linen.dtypes.Dtype", "flax")
def _map_flax_linen_dtypes_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_dtypes_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.dtypes.Dtype")


@register_op("linen.dtypes.T", "flax")
def _map_flax_linen_dtypes_T(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_dtypes_T operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.dtypes.T")


@register_op("linen.dtypes.canonicalize_dtype", "flax")
def _map_flax_linen_dtypes_canonicalize_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_dtypes_canonicalize_dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.dtypes.canonicalize_dtype"
    )


@register_op("linen.dtypes.promote_dtype", "flax")
def _map_flax_linen_dtypes_promote_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_dtypes_promote_dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.dtypes.promote_dtype"
    )


@register_op("linen.recurrent.FrozenDict", "flax")
def _map_flax_linen_recurrent_FrozenDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_FrozenDict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.FrozenDict"
    )


@register_op("linen.recurrent.CollectionFilter", "flax")
def _map_flax_linen_recurrent_CollectionFilter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_CollectionFilter operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.CollectionFilter"
    )


@register_op("linen.recurrent.PRNGSequenceFilter", "flax")
def _map_flax_linen_recurrent_PRNGSequenceFilter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_PRNGSequenceFilter operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.recurrent.PRNGSequenceFilter",
    )


@register_op("linen.recurrent.initializers", "flax")
def _map_flax_linen_recurrent_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_initializers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.initializers"
    )


@register_op("linen.recurrent.transforms", "flax")
def _map_flax_linen_recurrent_transforms(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_transforms operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.transforms"
    )


@register_op("linen.recurrent.sigmoid", "flax")
def _map_flax_linen_recurrent_sigmoid(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.sigmoid")


@register_op("linen.recurrent.tanh", "flax")
def _map_flax_linen_recurrent_tanh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_tanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.tanh")


@register_op("linen.recurrent.promote_dtype", "flax")
def _map_flax_linen_recurrent_promote_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_promote_dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.promote_dtype"
    )


@register_op("linen.recurrent.Conv", "flax")
def _map_flax_linen_recurrent_Conv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_Conv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.Conv")


@register_op("linen.recurrent.Dense", "flax")
def _map_flax_linen_recurrent_Dense(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_Dense operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.Dense")


@register_op("linen.recurrent.default_kernel_init", "flax")
def _map_flax_linen_recurrent_default_kernel_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_default_kernel_init operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.recurrent.default_kernel_init",
    )


@register_op("linen.recurrent.Module", "flax")
def _map_flax_linen_recurrent_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.Module")


@register_op("linen.recurrent.compact", "flax")
def _map_flax_linen_recurrent_compact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_compact operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.compact")


@register_op("linen.recurrent.nowrap", "flax")
def _map_flax_linen_recurrent_nowrap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_nowrap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.nowrap")


@register_op("linen.recurrent.Array", "flax")
def _map_flax_linen_recurrent_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.Array")


@register_op("linen.recurrent.PRNGKey", "flax")
def _map_flax_linen_recurrent_PRNGKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_PRNGKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.PRNGKey")


@register_op("linen.recurrent.Dtype", "flax")
def _map_flax_linen_recurrent_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.Dtype")


@register_op("linen.recurrent.InOutScanAxis", "flax")
def _map_flax_linen_recurrent_InOutScanAxis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_InOutScanAxis operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.InOutScanAxis"
    )


@register_op("linen.recurrent.Initializer", "flax")
def _map_flax_linen_recurrent_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_Initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.Initializer"
    )


@register_op("linen.recurrent.PrecisionLike", "flax")
def _map_flax_linen_recurrent_PrecisionLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_PrecisionLike operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.PrecisionLike"
    )


@register_op("linen.recurrent.A", "flax")
def _map_flax_linen_recurrent_A(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.A")


@register_op("linen.recurrent.Carry", "flax")
def _map_flax_linen_recurrent_Carry(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_Carry operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.Carry")


@register_op("linen.recurrent.CarryHistory", "flax")
def _map_flax_linen_recurrent_CarryHistory(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_CarryHistory operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.CarryHistory"
    )


@register_op("linen.recurrent.Output", "flax")
def _map_flax_linen_recurrent_Output(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_Output operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.Output")


@register_op("linen.recurrent.RNNCellBase", "flax")
def _map_flax_linen_recurrent_RNNCellBase(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_RNNCellBase operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.RNNCellBase"
    )


@register_op("linen.recurrent.LSTMCell", "flax")
def _map_flax_linen_recurrent_LSTMCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_LSTMCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.LSTMCell")


@register_op("linen.recurrent.DenseParams", "flax")
def _map_flax_linen_recurrent_DenseParams(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_DenseParams operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.DenseParams"
    )


@register_op("linen.recurrent.OptimizedLSTMCell", "flax")
def _map_flax_linen_recurrent_OptimizedLSTMCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_OptimizedLSTMCell operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.OptimizedLSTMCell"
    )


@register_op("linen.recurrent.SimpleCell", "flax")
def _map_flax_linen_recurrent_SimpleCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_SimpleCell operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.SimpleCell"
    )


@register_op("linen.recurrent.GRUCell", "flax")
def _map_flax_linen_recurrent_GRUCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_GRUCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.GRUCell")


@register_op("linen.recurrent.MGUCell", "flax")
def _map_flax_linen_recurrent_MGUCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_MGUCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.MGUCell")


@register_op("linen.recurrent.ConvLSTMCell", "flax")
def _map_flax_linen_recurrent_ConvLSTMCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_ConvLSTMCell operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.ConvLSTMCell"
    )


@register_op("linen.recurrent.RNN", "flax")
def _map_flax_linen_recurrent_RNN(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_RNN operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.RNN")


@register_op("linen.recurrent.flip_sequences", "flax")
def _map_flax_linen_recurrent_flip_sequences(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_flip_sequences operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.flip_sequences"
    )


@register_op("linen.recurrent.RNNBase", "flax")
def _map_flax_linen_recurrent_RNNBase(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_RNNBase operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.RNNBase")


@register_op("linen.recurrent.Bidirectional", "flax")
def _map_flax_linen_recurrent_Bidirectional(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_recurrent_Bidirectional operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.recurrent.Bidirectional"
    )


@register_op("linen.transforms.errors", "flax")
def _map_flax_linen_transforms_errors(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_errors operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.errors")


@register_op("linen.transforms.struct", "flax")
def _map_flax_linen_transforms_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_struct operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.struct")


@register_op("linen.transforms.traceback_util", "flax")
def _map_flax_linen_transforms_traceback_util(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_traceback_util operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.traceback_util"
    )


@register_op("linen.transforms.serialization", "flax")
def _map_flax_linen_transforms_serialization(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_serialization operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.serialization"
    )


@register_op("linen.transforms.Scope", "flax")
def _map_flax_linen_transforms_Scope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_Scope operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.Scope")


@register_op("linen.transforms.lift", "flax")
def _map_flax_linen_transforms_lift(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_lift operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.lift")


@register_op("linen.transforms.meta", "flax")
def _map_flax_linen_transforms_meta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_meta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.meta")


@register_op("linen.transforms.FrozenDict", "flax")
def _map_flax_linen_transforms_FrozenDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_FrozenDict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.FrozenDict"
    )


@register_op("linen.transforms.CollectionFilter", "flax")
def _map_flax_linen_transforms_CollectionFilter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_CollectionFilter operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.CollectionFilter"
    )


@register_op("linen.transforms.LazyRng", "flax")
def _map_flax_linen_transforms_LazyRng(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_LazyRng operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.LazyRng")


@register_op("linen.transforms.PRNGSequenceFilter", "flax")
def _map_flax_linen_transforms_PRNGSequenceFilter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_PRNGSequenceFilter operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.PRNGSequenceFilter",
    )


@register_op("linen.transforms.FlaxId", "flax")
def _map_flax_linen_transforms_FlaxId(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_FlaxId operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.FlaxId")


@register_op("linen.transforms.linen_module", "flax")
def _map_flax_linen_transforms_linen_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_linen_module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.linen_module"
    )


@register_op("linen.transforms.Module", "flax")
def _map_flax_linen_transforms_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.Module")


@register_op("linen.transforms.Variable", "flax")
def _map_flax_linen_transforms_Variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_Variable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.Variable"
    )


@register_op("linen.transforms.wrap_method_once", "flax")
def _map_flax_linen_transforms_wrap_method_once(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_wrap_method_once operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.wrap_method_once"
    )


@register_op("linen.transforms.InOutAxis", "flax")
def _map_flax_linen_transforms_InOutAxis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_InOutAxis operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.InOutAxis"
    )


@register_op("linen.transforms.InOutScanAxis", "flax")
def _map_flax_linen_transforms_InOutScanAxis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_InOutScanAxis operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.InOutScanAxis"
    )


@register_op("linen.transforms.clean_clone", "flax")
def _map_flax_linen_transforms_clean_clone(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_clean_clone operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.clean_clone"
    )


@register_op("linen.transforms.VariablePlaceholder", "flax")
def _map_flax_linen_transforms_VariablePlaceholder(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_VariablePlaceholder operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.VariablePlaceholder",
    )


@register_op("linen.transforms.InstancePlaceholder", "flax")
def _map_flax_linen_transforms_InstancePlaceholder(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_InstancePlaceholder operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.InstancePlaceholder",
    )


@register_op("linen.transforms.get_module_scopes", "flax")
def _map_flax_linen_transforms_get_module_scopes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_get_module_scopes operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.get_module_scopes",
    )


@register_op("linen.transforms.set_module_scopes", "flax")
def _map_flax_linen_transforms_set_module_scopes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_set_module_scopes operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.set_module_scopes",
    )


@register_op("linen.transforms.module_class_lift_transform", "flax")
def _map_flax_linen_transforms_module_class_lift_transform(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_module_class_lift_transform operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.module_class_lift_transform",
    )


@register_op("linen.transforms.decorator_lift_transform", "flax")
def _map_flax_linen_transforms_decorator_lift_transform(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_decorator_lift_transform operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.decorator_lift_transform",
    )


@register_op("linen.transforms.decorator_lift_transform_cached", "flax")
def _map_flax_linen_transforms_decorator_lift_transform_cached(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_decorator_lift_transform_cached operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.decorator_lift_transform_cached",
    )


@register_op("linen.transforms.fork_rngs", "flax")
def _map_flax_linen_transforms_fork_rngs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_fork_rngs operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.fork_rngs"
    )


@register_op("linen.transforms.module_class_lift_transform_cached", "flax")
def _map_flax_linen_transforms_module_class_lift_transform_cached(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_module_class_lift_transform_cached operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.module_class_lift_transform_cached",
    )


@register_op("linen.transforms.TransformTarget", "flax")
def _map_flax_linen_transforms_TransformTarget(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_TransformTarget operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.TransformTarget"
    )


@register_op("linen.transforms.Target", "flax")
def _map_flax_linen_transforms_Target(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_Target operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.Target")


@register_op("linen.transforms.lift_transform", "flax")
def _map_flax_linen_transforms_lift_transform(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_lift_transform operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.lift_transform"
    )


@register_op("linen.transforms.lift_transform_cached", "flax")
def _map_flax_linen_transforms_lift_transform_cached(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_lift_transform_cached operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.lift_transform_cached",
    )


@register_op("linen.transforms.lift_direct_transform", "flax")
def _map_flax_linen_transforms_lift_direct_transform(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_lift_direct_transform operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.lift_direct_transform",
    )


@register_op("linen.transforms.vmap", "flax")
def _map_flax_linen_transforms_vmap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_vmap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.vmap")


@register_op("linen.transforms.jit", "flax")
def _map_flax_linen_transforms_jit(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_jit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.jit")


@register_op("linen.transforms.checkpoint", "flax")
def _map_flax_linen_transforms_checkpoint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_checkpoint operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.checkpoint"
    )


@register_op("linen.transforms.remat", "flax")
def _map_flax_linen_transforms_remat(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_remat operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.remat")


@register_op("linen.transforms.remat_scan", "flax")
def _map_flax_linen_transforms_remat_scan(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_remat_scan operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.remat_scan"
    )


@register_op("linen.transforms.scan", "flax")
def _map_flax_linen_transforms_scan(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_scan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.scan")


@register_op("linen.transforms.map_variables", "flax")
def _map_flax_linen_transforms_map_variables(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_map_variables operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.map_variables"
    )


@register_op("linen.transforms.vjp", "flax")
def _map_flax_linen_transforms_vjp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_vjp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.vjp")


@register_op("linen.transforms.value_and_grad", "flax")
def _map_flax_linen_transforms_value_and_grad(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_value_and_grad operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.value_and_grad"
    )


@register_op("linen.transforms.grad", "flax")
def _map_flax_linen_transforms_grad(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_grad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.grad")


@register_op("linen.transforms.jvp", "flax")
def _map_flax_linen_transforms_jvp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_jvp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.jvp")


@register_op("linen.transforms.ModuleT", "flax")
def _map_flax_linen_transforms_ModuleT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_ModuleT operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.ModuleT")


@register_op("linen.transforms.C", "flax")
def _map_flax_linen_transforms_C(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_C operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.C")


@register_op("linen.transforms.while_loop", "flax")
def _map_flax_linen_transforms_while_loop(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_while_loop operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.while_loop"
    )


@register_op("linen.transforms.cond", "flax")
def _map_flax_linen_transforms_cond(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_cond operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.cond")


@register_op("linen.transforms.switch", "flax")
def _map_flax_linen_transforms_switch(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_switch operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.switch")


@register_op("linen.transforms.custom_vjp", "flax")
def _map_flax_linen_transforms_custom_vjp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_custom_vjp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.custom_vjp"
    )


@register_op("linen.transforms.named_call", "flax")
def _map_flax_linen_transforms_named_call(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_named_call operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.named_call"
    )


@register_op("linen.transforms.add_metadata_axis", "flax")
def _map_flax_linen_transforms_add_metadata_axis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_add_metadata_axis operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.transforms.add_metadata_axis",
    )


@register_op("linen.transforms.fold_rngs", "flax")
def _map_flax_linen_transforms_fold_rngs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_transforms_fold_rngs operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.transforms.fold_rngs"
    )


@register_op("linen.kw_only_dataclasses.M", "flax")
def _map_flax_linen_kw_only_dataclasses_M(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_kw_only_dataclasses_M operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.kw_only_dataclasses.M"
    )


@register_op("linen.kw_only_dataclasses.FieldName", "flax")
def _map_flax_linen_kw_only_dataclasses_FieldName(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_kw_only_dataclasses_FieldName operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.kw_only_dataclasses.FieldName",
    )


@register_op("linen.kw_only_dataclasses.Annotation", "flax")
def _map_flax_linen_kw_only_dataclasses_Annotation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_kw_only_dataclasses_Annotation operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.kw_only_dataclasses.Annotation",
    )


@register_op("linen.kw_only_dataclasses.Default", "flax")
def _map_flax_linen_kw_only_dataclasses_Default(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_kw_only_dataclasses_Default operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.kw_only_dataclasses.Default"
    )


@register_op("linen.kw_only_dataclasses.KW_ONLY", "flax")
def _map_flax_linen_kw_only_dataclasses_KW_ONLY(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_kw_only_dataclasses_KW_ONLY operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.kw_only_dataclasses.KW_ONLY"
    )


@register_op("linen.kw_only_dataclasses.field", "flax")
def _map_flax_linen_kw_only_dataclasses_field(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_kw_only_dataclasses_field operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.kw_only_dataclasses.field"
    )


@register_op("linen.kw_only_dataclasses.dataclass", "flax")
def _map_flax_linen_kw_only_dataclasses_dataclass(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_kw_only_dataclasses_dataclass operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.kw_only_dataclasses.dataclass",
    )


@register_op("linen.attention.initializers", "flax")
def _map_flax_linen_attention_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_initializers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.initializers"
    )


@register_op("linen.attention.promote_dtype", "flax")
def _map_flax_linen_attention_promote_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_promote_dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.promote_dtype"
    )


@register_op("linen.attention.DenseGeneral", "flax")
def _map_flax_linen_attention_DenseGeneral(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_DenseGeneral operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.DenseGeneral"
    )


@register_op("linen.attention.default_kernel_init", "flax")
def _map_flax_linen_attention_default_kernel_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_default_kernel_init operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.attention.default_kernel_init",
    )


@register_op("linen.attention.Module", "flax")
def _map_flax_linen_attention_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.Module")


@register_op("linen.attention.compact", "flax")
def _map_flax_linen_attention_compact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_compact operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.compact")


@register_op("linen.attention.merge_param", "flax")
def _map_flax_linen_attention_merge_param(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_merge_param operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.merge_param"
    )


@register_op("linen.attention.LayerNorm", "flax")
def _map_flax_linen_attention_LayerNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_LayerNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.LayerNorm"
    )


@register_op("linen.attention.Array", "flax")
def _map_flax_linen_attention_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.Array")


@register_op("linen.attention.PRNGKey", "flax")
def _map_flax_linen_attention_PRNGKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_PRNGKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.PRNGKey")


@register_op("linen.attention.Dtype", "flax")
def _map_flax_linen_attention_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.Dtype")


@register_op("linen.attention.Shape", "flax")
def _map_flax_linen_attention_Shape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_Shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.Shape")


@register_op("linen.attention.Initializer", "flax")
def _map_flax_linen_attention_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_Initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.Initializer"
    )


@register_op("linen.attention.PrecisionLike", "flax")
def _map_flax_linen_attention_PrecisionLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_PrecisionLike operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.PrecisionLike"
    )


@register_op("linen.attention.DotGeneralT", "flax")
def _map_flax_linen_attention_DotGeneralT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_DotGeneralT operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.DotGeneralT"
    )


@register_op("linen.attention.dot_product_attention_weights", "flax")
def _map_flax_linen_attention_dot_product_attention_weights(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_dot_product_attention_weights operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.attention.dot_product_attention_weights",
    )


@register_op("linen.attention.dot_product_attention", "flax")
def _map_flax_linen_attention_dot_product_attention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_dot_product_attention operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.attention.dot_product_attention",
    )


@register_op("linen.attention.MultiHeadDotProductAttention", "flax")
def _map_flax_linen_attention_MultiHeadDotProductAttention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_MultiHeadDotProductAttention operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.attention.MultiHeadDotProductAttention",
    )


@register_op("linen.attention.MultiHeadAttention", "flax")
def _map_flax_linen_attention_MultiHeadAttention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_MultiHeadAttention operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.attention.MultiHeadAttention",
    )


@register_op("linen.attention.SelfAttention", "flax")
def _map_flax_linen_attention_SelfAttention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_SelfAttention operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.SelfAttention"
    )


@register_op("linen.attention.make_attention_mask", "flax")
def _map_flax_linen_attention_make_attention_mask(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_make_attention_mask operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.attention.make_attention_mask",
    )


@register_op("linen.attention.make_causal_mask", "flax")
def _map_flax_linen_attention_make_causal_mask(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_make_causal_mask operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.make_causal_mask"
    )


@register_op("linen.attention.combine_masks", "flax")
def _map_flax_linen_attention_combine_masks(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_attention_combine_masks operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.attention.combine_masks"
    )


@register_op("linen.batch_apply.ndim_at_least", "flax")
def _map_flax_linen_batch_apply_ndim_at_least(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_batch_apply_ndim_at_least operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.batch_apply.ndim_at_least"
    )


@register_op("linen.batch_apply.arbitrary_mergeable_leaf", "flax")
def _map_flax_linen_batch_apply_arbitrary_mergeable_leaf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_batch_apply_arbitrary_mergeable_leaf operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.batch_apply.arbitrary_mergeable_leaf",
    )


@register_op("linen.batch_apply.merge_leading_dims", "flax")
def _map_flax_linen_batch_apply_merge_leading_dims(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_batch_apply_merge_leading_dims operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.batch_apply.merge_leading_dims",
    )


@register_op("linen.batch_apply.split_leading_dim", "flax")
def _map_flax_linen_batch_apply_split_leading_dim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_batch_apply_split_leading_dim operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.batch_apply.split_leading_dim",
    )


@register_op("linen.batch_apply.BatchApply", "flax")
def _map_flax_linen_batch_apply_BatchApply(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_batch_apply_BatchApply operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.batch_apply.BatchApply"
    )


@register_op("linen.spmd.struct", "flax")
def _map_flax_linen_spmd_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_struct operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.struct")


@register_op("linen.spmd.meta", "flax")
def _map_flax_linen_spmd_meta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_meta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.meta")


@register_op("linen.spmd.get_logical_axis_rules", "flax")
def _map_flax_linen_spmd_get_logical_axis_rules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_get_logical_axis_rules operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.get_logical_axis_rules"
    )


@register_op("linen.spmd.Array", "flax")
def _map_flax_linen_spmd_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.Array")


@register_op("linen.spmd.LogicalNames", "flax")
def _map_flax_linen_spmd_LogicalNames(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_LogicalNames operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.LogicalNames")


@register_op("linen.spmd.LogicalRules", "flax")
def _map_flax_linen_spmd_LogicalRules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_LogicalRules operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.LogicalRules")


@register_op("linen.spmd.ArrayPytree", "flax")
def _map_flax_linen_spmd_ArrayPytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_ArrayPytree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.ArrayPytree")


@register_op("linen.spmd.LogicalPartitionSpec", "flax")
def _map_flax_linen_spmd_LogicalPartitionSpec(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_LogicalPartitionSpec operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.LogicalPartitionSpec"
    )


@register_op("linen.spmd.LogicalPartitionSpecPytree", "flax")
def _map_flax_linen_spmd_LogicalPartitionSpecPytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_LogicalPartitionSpecPytree operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.spmd.LogicalPartitionSpecPytree",
    )


@register_op("linen.spmd.logical_to_mesh_axes", "flax")
def _map_flax_linen_spmd_logical_to_mesh_axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_logical_to_mesh_axes operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.logical_to_mesh_axes"
    )


@register_op("linen.spmd.logical_to_mesh", "flax")
def _map_flax_linen_spmd_logical_to_mesh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_logical_to_mesh operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.logical_to_mesh"
    )


@register_op("linen.spmd.logical_to_mesh_sharding", "flax")
def _map_flax_linen_spmd_logical_to_mesh_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_logical_to_mesh_sharding operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.spmd.logical_to_mesh_sharding",
    )


@register_op("linen.spmd.RulesFallback", "flax")
def _map_flax_linen_spmd_RulesFallback(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_RulesFallback operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.RulesFallback")


@register_op("linen.spmd.with_logical_constraint", "flax")
def _map_flax_linen_spmd_with_logical_constraint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_with_logical_constraint operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.spmd.with_logical_constraint",
    )


@register_op("linen.spmd.LogicallyPartitioned", "flax")
def _map_flax_linen_spmd_LogicallyPartitioned(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_LogicallyPartitioned operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.spmd.LogicallyPartitioned"
    )


@register_op("linen.spmd.with_logical_partitioning", "flax")
def _map_flax_linen_spmd_with_logical_partitioning(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_spmd_with_logical_partitioning operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.spmd.with_logical_partitioning",
    )


@register_op("linen.stochastic.Module", "flax")
def _map_flax_linen_stochastic_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_stochastic_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.stochastic.Module")


@register_op("linen.stochastic.compact", "flax")
def _map_flax_linen_stochastic_compact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_stochastic_compact operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.stochastic.compact")


@register_op("linen.stochastic.merge_param", "flax")
def _map_flax_linen_stochastic_merge_param(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_stochastic_merge_param operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.stochastic.merge_param"
    )


@register_op("linen.stochastic.PRNGKey", "flax")
def _map_flax_linen_stochastic_PRNGKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_stochastic_PRNGKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.stochastic.PRNGKey")


@register_op("linen.stochastic.Dropout", "flax")
def _map_flax_linen_stochastic_Dropout(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_stochastic_Dropout operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.stochastic.Dropout")


@register_op("linen.pooling.pool", "flax")
def _map_flax_linen_pooling_pool(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_pooling_pool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.pooling.pool")


@register_op("linen.pooling.avg_pool", "flax")
def _map_flax_linen_pooling_avg_pool(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_pooling_avg_pool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.pooling.avg_pool")


@register_op("linen.pooling.max_pool", "flax")
def _map_flax_linen_pooling_max_pool(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_pooling_max_pool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.pooling.max_pool")


@register_op("linen.pooling.min_pool", "flax")
def _map_flax_linen_pooling_min_pool(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_pooling_min_pool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.pooling.min_pool")


@register_op("linen.activation.compact", "flax")
def _map_flax_linen_activation_compact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_activation_compact operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.activation.compact")


@register_op("linen.activation.Module", "flax")
def _map_flax_linen_activation_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_activation_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.activation.Module")


@register_op("linen.activation.Dense", "flax")
def _map_flax_linen_activation_Dense(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_activation_Dense operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.activation.Dense")


@register_op("linen.activation.Array", "flax")
def _map_flax_linen_activation_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_activation_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.activation.Array")


@register_op("linen.activation.Dtype", "flax")
def _map_flax_linen_activation_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_activation_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.activation.Dtype")


@register_op("linen.activation.normalize", "flax")
def _map_flax_linen_activation_normalize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_activation_normalize operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.activation.normalize"
    )


@register_op("linen.activation.PReLU", "flax")
def _map_flax_linen_activation_PReLU(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_activation_PReLU operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.activation.PReLU")


@register_op("linen.combinators.Module", "flax")
def _map_flax_linen_combinators_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_combinators_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.combinators.Module")


@register_op("linen.combinators.compact", "flax")
def _map_flax_linen_combinators_compact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_combinators_compact operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.combinators.compact"
    )


@register_op("linen.combinators.Sequential", "flax")
def _map_flax_linen_combinators_Sequential(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_combinators_Sequential operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.combinators.Sequential"
    )


@register_op("linen.linear.meta", "flax")
def _map_flax_linen_linear_meta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_meta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.meta")


@register_op("linen.linear.initializers", "flax")
def _map_flax_linen_linear_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_initializers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.initializers"
    )


@register_op("linen.linear.module", "flax")
def _map_flax_linen_linear_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.module")


@register_op("linen.linear.promote_dtype", "flax")
def _map_flax_linen_linear_promote_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_promote_dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.promote_dtype"
    )


@register_op("linen.linear.Module", "flax")
def _map_flax_linen_linear_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.Module")


@register_op("linen.linear.compact", "flax")
def _map_flax_linen_linear_compact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_compact operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.compact")


@register_op("linen.linear.Array", "flax")
def _map_flax_linen_linear_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.Array")


@register_op("linen.linear.ConvGeneralDilatedT", "flax")
def _map_flax_linen_linear_ConvGeneralDilatedT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_ConvGeneralDilatedT operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.ConvGeneralDilatedT"
    )


@register_op("linen.linear.DotGeneralT", "flax")
def _map_flax_linen_linear_DotGeneralT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_DotGeneralT operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.DotGeneralT")


@register_op("linen.linear.Dtype", "flax")
def _map_flax_linen_linear_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.Dtype")


@register_op("linen.linear.Initializer", "flax")
def _map_flax_linen_linear_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_Initializer operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.Initializer")


@register_op("linen.linear.LaxPadding", "flax")
def _map_flax_linen_linear_LaxPadding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_LaxPadding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.LaxPadding")


@register_op("linen.linear.PRNGKey", "flax")
def _map_flax_linen_linear_PRNGKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_PRNGKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.PRNGKey")


@register_op("linen.linear.PaddingLike", "flax")
def _map_flax_linen_linear_PaddingLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_PaddingLike operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.PaddingLike")


@register_op("linen.linear.PrecisionLike", "flax")
def _map_flax_linen_linear_PrecisionLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_PrecisionLike operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.PrecisionLike"
    )


@register_op("linen.linear.Shape", "flax")
def _map_flax_linen_linear_Shape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_Shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.Shape")


@register_op("linen.linear.PromoteDtypeFn", "flax")
def _map_flax_linen_linear_PromoteDtypeFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_PromoteDtypeFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.PromoteDtypeFn"
    )


@register_op("linen.linear.default_kernel_init", "flax")
def _map_flax_linen_linear_default_kernel_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_default_kernel_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.default_kernel_init"
    )


@register_op("linen.linear.DenseGeneral", "flax")
def _map_flax_linen_linear_DenseGeneral(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_DenseGeneral operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.DenseGeneral"
    )


@register_op("linen.linear.Dense", "flax")
def _map_flax_linen_linear_Dense(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_Dense operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.Dense")


@register_op("linen.linear.Einsum", "flax")
def _map_flax_linen_linear_Einsum(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_Einsum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.Einsum")


@register_op("linen.linear.canonicalize_padding", "flax")
def _map_flax_linen_linear_canonicalize_padding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_canonicalize_padding operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.canonicalize_padding"
    )


@register_op("linen.linear.Conv", "flax")
def _map_flax_linen_linear_Conv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_Conv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.Conv")


@register_op("linen.linear.ConvLocal", "flax")
def _map_flax_linen_linear_ConvLocal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_ConvLocal operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.ConvLocal")


@register_op("linen.linear.ConvTranspose", "flax")
def _map_flax_linen_linear_ConvTranspose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_ConvTranspose operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.ConvTranspose"
    )


@register_op("linen.linear.default_embed_init", "flax")
def _map_flax_linen_linear_default_embed_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_default_embed_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.default_embed_init"
    )


@register_op("linen.linear.Embed", "flax")
def _map_flax_linen_linear_Embed(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_linear_Embed operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.linear.Embed")


@register_op("linen.initializers.Initializer", "flax")
def _map_flax_linen_initializers_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_initializers_Initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.initializers.Initializer"
    )


@register_op("linen.initializers.zeros_init", "flax")
def _map_flax_linen_initializers_zeros_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_initializers_zeros_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.initializers.zeros_init"
    )


@register_op("linen.initializers.ones_init", "flax")
def _map_flax_linen_initializers_ones_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_initializers_ones_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.initializers.ones_init"
    )


@register_op("linen.partitioning.nn", "flax")
def _map_flax_linen_partitioning_nn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_nn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.nn")


@register_op("linen.partitioning.struct", "flax")
def _map_flax_linen_partitioning_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_struct operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.struct"
    )


@register_op("linen.partitioning.freeze", "flax")
def _map_flax_linen_partitioning_freeze(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_freeze operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.freeze"
    )


@register_op("linen.partitioning.unfreeze", "flax")
def _map_flax_linen_partitioning_unfreeze(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_unfreeze operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.unfreeze"
    )


@register_op("linen.partitioning.CollectionFilter", "flax")
def _map_flax_linen_partitioning_CollectionFilter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_CollectionFilter operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.CollectionFilter",
    )


@register_op("linen.partitioning.PRNGSequenceFilter", "flax")
def _map_flax_linen_partitioning_PRNGSequenceFilter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_PRNGSequenceFilter operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.PRNGSequenceFilter",
    )


@register_op("linen.partitioning.axis_rules", "flax")
def _map_flax_linen_partitioning_axis_rules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_axis_rules operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.axis_rules"
    )


@register_op("linen.partitioning.set_axis_rules", "flax")
def _map_flax_linen_partitioning_set_axis_rules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_set_axis_rules operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.set_axis_rules"
    )


@register_op("linen.partitioning.get_axis_rules", "flax")
def _map_flax_linen_partitioning_get_axis_rules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_get_axis_rules operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.get_axis_rules"
    )


@register_op("linen.partitioning.logical_to_mesh", "flax")
def _map_flax_linen_partitioning_logical_to_mesh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_logical_to_mesh operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.logical_to_mesh",
    )


@register_op("linen.partitioning.logical_to_mesh_axes", "flax")
def _map_flax_linen_partitioning_logical_to_mesh_axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_logical_to_mesh_axes operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.logical_to_mesh_axes",
    )


@register_op("linen.partitioning.RulesFallback", "flax")
def _map_flax_linen_partitioning_RulesFallback(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_RulesFallback operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.RulesFallback"
    )


@register_op("linen.partitioning.with_sharding_constraint", "flax")
def _map_flax_linen_partitioning_with_sharding_constraint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_with_sharding_constraint operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.with_sharding_constraint",
    )


@register_op("linen.partitioning.flatten_dict", "flax")
def _map_flax_linen_partitioning_flatten_dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_flatten_dict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.flatten_dict"
    )


@register_op("linen.partitioning.unflatten_dict", "flax")
def _map_flax_linen_partitioning_unflatten_dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_unflatten_dict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.unflatten_dict"
    )


@register_op("linen.partitioning.Array", "flax")
def _map_flax_linen_partitioning_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.Array")


@register_op("linen.partitioning.ScanIn", "flax")
def _map_flax_linen_partitioning_ScanIn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_ScanIn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.ScanIn"
    )


@register_op("linen.partitioning.ScanOut", "flax")
def _map_flax_linen_partitioning_ScanOut(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_ScanOut operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.ScanOut"
    )


@register_op("linen.partitioning.InOutAxis", "flax")
def _map_flax_linen_partitioning_InOutAxis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_InOutAxis operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.InOutAxis"
    )


@register_op("linen.partitioning.InOutScanAxis", "flax")
def _map_flax_linen_partitioning_InOutScanAxis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_InOutScanAxis operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.InOutScanAxis"
    )


@register_op("linen.partitioning.LogicalRules", "flax")
def _map_flax_linen_partitioning_LogicalRules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_LogicalRules operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.LogicalRules"
    )


@register_op("linen.partitioning.ArrayPytree", "flax")
def _map_flax_linen_partitioning_ArrayPytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_ArrayPytree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.ArrayPytree"
    )


@register_op("linen.partitioning.LogicalPartitionSpec", "flax")
def _map_flax_linen_partitioning_LogicalPartitionSpec(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_LogicalPartitionSpec operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.LogicalPartitionSpec",
    )


@register_op("linen.partitioning.LogicalPartitionSpecPytree", "flax")
def _map_flax_linen_partitioning_LogicalPartitionSpecPytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_LogicalPartitionSpecPytree operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.LogicalPartitionSpecPytree",
    )


@register_op("linen.partitioning.PartitionSpecPytree", "flax")
def _map_flax_linen_partitioning_PartitionSpecPytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_PartitionSpecPytree operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.PartitionSpecPytree",
    )


@register_op("linen.partitioning.AxisMetadata", "flax")
def _map_flax_linen_partitioning_AxisMetadata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_AxisMetadata operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.AxisMetadata"
    )


@register_op("linen.partitioning.param_with_axes", "flax")
def _map_flax_linen_partitioning_param_with_axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_param_with_axes operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.param_with_axes",
    )


@register_op("linen.partitioning.PartitionedVariable", "flax")
def _map_flax_linen_partitioning_PartitionedVariable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_PartitionedVariable operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.PartitionedVariable",
    )


@register_op("linen.partitioning.variable_with_axes", "flax")
def _map_flax_linen_partitioning_variable_with_axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_variable_with_axes operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.variable_with_axes",
    )


@register_op("linen.partitioning.get_axis_names", "flax")
def _map_flax_linen_partitioning_get_axis_names(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_get_axis_names operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.get_axis_names"
    )


@register_op("linen.partitioning.scan_with_axes", "flax")
def _map_flax_linen_partitioning_scan_with_axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_scan_with_axes operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.scan_with_axes"
    )


@register_op("linen.partitioning.vmap_with_axes", "flax")
def _map_flax_linen_partitioning_vmap_with_axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_vmap_with_axes operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.vmap_with_axes"
    )


@register_op("linen.partitioning.core_remat_static", "flax")
def _map_flax_linen_partitioning_core_remat_static(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_core_remat_static operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.partitioning.core_remat_static",
    )


@register_op("linen.partitioning.remat", "flax")
def _map_flax_linen_partitioning_remat(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_partitioning_remat operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.partitioning.remat")


@register_op("linen.normalization.dtypes", "flax")
def _map_flax_linen_normalization_dtypes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_dtypes operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.dtypes"
    )


@register_op("linen.normalization.module", "flax")
def _map_flax_linen_normalization_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.module"
    )


@register_op("linen.normalization.transforms", "flax")
def _map_flax_linen_normalization_transforms(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_transforms operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.transforms"
    )


@register_op("linen.normalization.Array", "flax")
def _map_flax_linen_normalization_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_Array operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.Array"
    )


@register_op("linen.normalization.PRNGKey", "flax")
def _map_flax_linen_normalization_PRNGKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_PRNGKey operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.PRNGKey"
    )


@register_op("linen.normalization.Dtype", "flax")
def _map_flax_linen_normalization_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_Dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.Dtype"
    )


@register_op("linen.normalization.Shape", "flax")
def _map_flax_linen_normalization_Shape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_Shape operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.Shape"
    )


@register_op("linen.normalization.Initializer", "flax")
def _map_flax_linen_normalization_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_Initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.Initializer"
    )


@register_op("linen.normalization.Axes", "flax")
def _map_flax_linen_normalization_Axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_Axes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.Axes")


@register_op("linen.normalization.field", "flax")
def _map_flax_linen_normalization_field(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_field operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.field"
    )


@register_op("linen.normalization.canonicalize_dtype", "flax")
def _map_flax_linen_normalization_canonicalize_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_canonicalize_dtype operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.normalization.canonicalize_dtype",
    )


@register_op("linen.normalization.compact", "flax")
def _map_flax_linen_normalization_compact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_compact operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.compact"
    )


@register_op("linen.normalization.Module", "flax")
def _map_flax_linen_normalization_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_Module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.Module"
    )


@register_op("linen.normalization.merge_param", "flax")
def _map_flax_linen_normalization_merge_param(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_merge_param operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.merge_param"
    )


@register_op("linen.normalization.map_variables", "flax")
def _map_flax_linen_normalization_map_variables(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_map_variables operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.map_variables"
    )


@register_op("linen.normalization.BatchNorm", "flax")
def _map_flax_linen_normalization_BatchNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_BatchNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.BatchNorm"
    )


@register_op("linen.normalization.LayerNorm", "flax")
def _map_flax_linen_normalization_LayerNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_LayerNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.LayerNorm"
    )


@register_op("linen.normalization.RMSNorm", "flax")
def _map_flax_linen_normalization_RMSNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_RMSNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.RMSNorm"
    )


@register_op("linen.normalization.GroupNorm", "flax")
def _map_flax_linen_normalization_GroupNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_GroupNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.GroupNorm"
    )


@register_op("linen.normalization.InstanceNorm", "flax")
def _map_flax_linen_normalization_InstanceNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_InstanceNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.InstanceNorm"
    )


@register_op("linen.normalization.SpectralNorm", "flax")
def _map_flax_linen_normalization_SpectralNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_SpectralNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.SpectralNorm"
    )


@register_op("linen.normalization.WeightNorm", "flax")
def _map_flax_linen_normalization_WeightNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_normalization_WeightNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.normalization.WeightNorm"
    )


@register_op("linen.fp8_ops.DType", "flax")
def _map_flax_linen_fp8_ops_DType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_DType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.DType")


@register_op("linen.fp8_ops.CAN_USE_EARRAY", "flax")
def _map_flax_linen_fp8_ops_CAN_USE_EARRAY(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_CAN_USE_EARRAY operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.CAN_USE_EARRAY"
    )


@register_op("linen.fp8_ops.initializers", "flax")
def _map_flax_linen_fp8_ops_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_initializers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.initializers"
    )


@register_op("linen.fp8_ops.module", "flax")
def _map_flax_linen_fp8_ops_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.module")


@register_op("linen.fp8_ops.OVERWRITE_WITH_GRADIENT", "flax")
def _map_flax_linen_fp8_ops_OVERWRITE_WITH_GRADIENT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_OVERWRITE_WITH_GRADIENT operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.fp8_ops.OVERWRITE_WITH_GRADIENT",
    )


@register_op("linen.fp8_ops.Fp8MetaTyRules", "flax")
def _map_flax_linen_fp8_ops_Fp8MetaTyRules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_Fp8MetaTyRules operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.Fp8MetaTyRules"
    )


@register_op("linen.fp8_ops.fp8_meta_dtype", "flax")
def _map_flax_linen_fp8_ops_fp8_meta_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_fp8_meta_dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.fp8_meta_dtype"
    )


@register_op("linen.fp8_ops.fp8_meta_dtype_wrapper", "flax")
def _map_flax_linen_fp8_ops_fp8_meta_dtype_wrapper(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_fp8_meta_dtype_wrapper operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.fp8_ops.fp8_meta_dtype_wrapper",
    )


@register_op("linen.fp8_ops.fm32", "flax")
def _map_flax_linen_fp8_ops_fm32(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_fm32 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.fm32")


@register_op("linen.fp8_ops.fp32_max_grad", "flax")
def _map_flax_linen_fp8_ops_fp32_max_grad(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_fp32_max_grad operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.fp32_max_grad"
    )


@register_op("linen.fp8_ops.get_fp8_max", "flax")
def _map_flax_linen_fp8_ops_get_fp8_max(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_get_fp8_max operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.get_fp8_max"
    )


@register_op("linen.fp8_ops.quantize", "flax")
def _map_flax_linen_fp8_ops_quantize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_quantize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.quantize")


@register_op("linen.fp8_ops.dequantize", "flax")
def _map_flax_linen_fp8_ops_dequantize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_dequantize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.dequantize")


@register_op("linen.fp8_ops.qdq", "flax")
def _map_flax_linen_fp8_ops_qdq(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_qdq operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.qdq")


@register_op("linen.fp8_ops.compute_scale", "flax")
def _map_flax_linen_fp8_ops_compute_scale(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_compute_scale operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.compute_scale"
    )


@register_op("linen.fp8_ops.compute_amax_history", "flax")
def _map_flax_linen_fp8_ops_compute_amax_history(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_compute_amax_history operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.fp8_ops.compute_amax_history",
    )


@register_op("linen.fp8_ops.update_fp8_meta", "flax")
def _map_flax_linen_fp8_ops_update_fp8_meta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_update_fp8_meta operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.update_fp8_meta"
    )


@register_op("linen.fp8_ops.quantize_dequantize_update", "flax")
def _map_flax_linen_fp8_ops_quantize_dequantize_update(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_quantize_dequantize_update operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.fp8_ops.quantize_dequantize_update",
    )


@register_op("linen.fp8_ops.dot_general_transpose_lhs", "flax")
def _map_flax_linen_fp8_ops_dot_general_transpose_lhs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_dot_general_transpose_lhs operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.fp8_ops.dot_general_transpose_lhs",
    )


@register_op("linen.fp8_ops.dot_general_transpose_rhs", "flax")
def _map_flax_linen_fp8_ops_dot_general_transpose_rhs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_dot_general_transpose_rhs operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.fp8_ops.dot_general_transpose_rhs",
    )


@register_op("linen.fp8_ops.in_qdq", "flax")
def _map_flax_linen_fp8_ops_in_qdq(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_in_qdq operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.in_qdq")


@register_op("linen.fp8_ops.in_qdq_fwd", "flax")
def _map_flax_linen_fp8_ops_in_qdq_fwd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_in_qdq_fwd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.in_qdq_fwd")


@register_op("linen.fp8_ops.in_qdq_bwd", "flax")
def _map_flax_linen_fp8_ops_in_qdq_bwd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_in_qdq_bwd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.in_qdq_bwd")


@register_op("linen.fp8_ops.out_qdq", "flax")
def _map_flax_linen_fp8_ops_out_qdq(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_out_qdq operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.out_qdq")


@register_op("linen.fp8_ops.out_qdq_fwd", "flax")
def _map_flax_linen_fp8_ops_out_qdq_fwd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_out_qdq_fwd operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.out_qdq_fwd"
    )


@register_op("linen.fp8_ops.out_qdq_bwd", "flax")
def _map_flax_linen_fp8_ops_out_qdq_bwd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_out_qdq_bwd operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.out_qdq_bwd"
    )


@register_op("linen.fp8_ops.in_q", "flax")
def _map_flax_linen_fp8_ops_in_q(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_in_q operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.in_q")


@register_op("linen.fp8_ops.in_q_fwd", "flax")
def _map_flax_linen_fp8_ops_in_q_fwd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_in_q_fwd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.in_q_fwd")


@register_op("linen.fp8_ops.in_q_bwd", "flax")
def _map_flax_linen_fp8_ops_in_q_bwd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_in_q_bwd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.in_q_bwd")


@register_op("linen.fp8_ops.out_dq", "flax")
def _map_flax_linen_fp8_ops_out_dq(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_out_dq operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.out_dq")


@register_op("linen.fp8_ops.out_dq_fwd", "flax")
def _map_flax_linen_fp8_ops_out_dq_fwd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_out_dq_fwd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.out_dq_fwd")


@register_op("linen.fp8_ops.out_dq_bwd", "flax")
def _map_flax_linen_fp8_ops_out_dq_bwd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_out_dq_bwd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.out_dq_bwd")


@register_op("linen.fp8_ops.quantized_dot", "flax")
def _map_flax_linen_fp8_ops_quantized_dot(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_quantized_dot operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.quantized_dot"
    )


@register_op("linen.fp8_ops.quantized_dot_fwd", "flax")
def _map_flax_linen_fp8_ops_quantized_dot_fwd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_quantized_dot_fwd operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.quantized_dot_fwd"
    )


@register_op("linen.fp8_ops.quantized_dot_bwd", "flax")
def _map_flax_linen_fp8_ops_quantized_dot_bwd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_quantized_dot_bwd operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.quantized_dot_bwd"
    )


@register_op("linen.fp8_ops.fp8_scaled_dot_general", "flax")
def _map_flax_linen_fp8_ops_fp8_scaled_dot_general(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_fp8_scaled_dot_general operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.fp8_ops.fp8_scaled_dot_general",
    )


@register_op("linen.fp8_ops.dot_general_with_precision", "flax")
def _map_flax_linen_fp8_ops_dot_general_with_precision(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_dot_general_with_precision operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.fp8_ops.dot_general_with_precision",
    )


@register_op("linen.fp8_ops.dot_general_with_precision_jvp", "flax")
def _map_flax_linen_fp8_ops_dot_general_with_precision_jvp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_dot_general_with_precision_jvp operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.fp8_ops.dot_general_with_precision_jvp",
    )


@register_op("linen.fp8_ops.Fp8DotGeneralBase", "flax")
def _map_flax_linen_fp8_ops_Fp8DotGeneralBase(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_Fp8DotGeneralBase operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.Fp8DotGeneralBase"
    )


@register_op("linen.fp8_ops.Fp8DotGeneralOp", "flax")
def _map_flax_linen_fp8_ops_Fp8DotGeneralOp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_Fp8DotGeneralOp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.Fp8DotGeneralOp"
    )


@register_op("linen.fp8_ops.Fp8DirectDotGeneralOp", "flax")
def _map_flax_linen_fp8_ops_Fp8DirectDotGeneralOp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_Fp8DirectDotGeneralOp operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.fp8_ops.Fp8DirectDotGeneralOp",
    )


@register_op("linen.fp8_ops.NANOOFp8DotGeneralOp", "flax")
def _map_flax_linen_fp8_ops_NANOOFp8DotGeneralOp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_NANOOFp8DotGeneralOp operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.fp8_ops.NANOOFp8DotGeneralOp",
    )


@register_op("linen.fp8_ops.Fp8Einsum", "flax")
def _map_flax_linen_fp8_ops_Fp8Einsum(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_Fp8Einsum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.Fp8Einsum")


@register_op("linen.fp8_ops.Fp8DotGeneral", "flax")
def _map_flax_linen_fp8_ops_Fp8DotGeneral(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_fp8_ops_Fp8DotGeneral operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.fp8_ops.Fp8DotGeneral"
    )


@register_op("linen.summary.module_lib", "flax")
def _map_flax_linen_summary_module_lib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_module_lib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.module_lib")


@register_op("linen.summary.meta", "flax")
def _map_flax_linen_summary_meta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_meta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.meta")


@register_op("linen.summary.unfreeze", "flax")
def _map_flax_linen_summary_unfreeze(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_unfreeze operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.unfreeze")


@register_op("linen.summary.CollectionFilter", "flax")
def _map_flax_linen_summary_CollectionFilter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_CollectionFilter operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.CollectionFilter"
    )


@register_op("linen.summary.DenyList", "flax")
def _map_flax_linen_summary_DenyList(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_DenyList operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.DenyList")


@register_op("linen.summary.LazyRng", "flax")
def _map_flax_linen_summary_LazyRng(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_LazyRng operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.LazyRng")


@register_op("linen.summary.Array", "flax")
def _map_flax_linen_summary_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.Array")


@register_op("linen.summary.PRNGKey", "flax")
def _map_flax_linen_summary_PRNGKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_PRNGKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.PRNGKey")


@register_op("linen.summary.RNGSequences", "flax")
def _map_flax_linen_summary_RNGSequences(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_RNGSequences operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.RNGSequences"
    )


@register_op("linen.summary.FrozenVariableDict", "flax")
def _map_flax_linen_summary_FrozenVariableDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_FrozenVariableDict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.FrozenVariableDict"
    )


@register_op("linen.summary.MutableVariableDict", "flax")
def _map_flax_linen_summary_MutableVariableDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_MutableVariableDict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.MutableVariableDict"
    )


@register_op("linen.summary.LogicalNames", "flax")
def _map_flax_linen_summary_LogicalNames(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_LogicalNames operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.LogicalNames"
    )


@register_op("linen.summary.Row", "flax")
def _map_flax_linen_summary_Row(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_Row operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.Row")


@register_op("linen.summary.Table", "flax")
def _map_flax_linen_summary_Table(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_Table operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.Table")


@register_op("linen.summary.tabulate", "flax")
def _map_flax_linen_summary_tabulate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_summary_tabulate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.summary.tabulate")


@register_op("linen.module.nn", "flax")
def _map_flax_linen_module_nn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_nn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.nn")


@register_op("linen.module.config", "flax")
def _map_flax_linen_module_config(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_config operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.config")


@register_op("linen.module.errors", "flax")
def _map_flax_linen_module_errors(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_errors operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.errors")


@register_op("linen.module.serialization", "flax")
def _map_flax_linen_module_serialization(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_serialization operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.serialization"
    )


@register_op("linen.module.traceback_util", "flax")
def _map_flax_linen_module_traceback_util(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_traceback_util operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.traceback_util"
    )


@register_op("linen.module.traverse_util", "flax")
def _map_flax_linen_module_traverse_util(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_traverse_util operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.traverse_util"
    )


@register_op("linen.module.Scope", "flax")
def _map_flax_linen_module_Scope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_Scope operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.Scope")


@register_op("linen.module.meta", "flax")
def _map_flax_linen_module_meta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_meta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.meta")


@register_op("linen.module.partial_eval", "flax")
def _map_flax_linen_module_partial_eval(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_partial_eval operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.partial_eval"
    )


@register_op("linen.module.FrozenDict", "flax")
def _map_flax_linen_module_FrozenDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_FrozenDict operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.FrozenDict")


@register_op("linen.module.CollectionFilter", "flax")
def _map_flax_linen_module_CollectionFilter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_CollectionFilter operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.CollectionFilter"
    )


@register_op("linen.module.DenyList", "flax")
def _map_flax_linen_module_DenyList(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_DenyList operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.DenyList")


@register_op("linen.module.Variable", "flax")
def _map_flax_linen_module_Variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_Variable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.Variable")


@register_op("linen.module.union_filters", "flax")
def _map_flax_linen_module_union_filters(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_union_filters operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.union_filters"
    )


@register_op("linen.module.FlaxId", "flax")
def _map_flax_linen_module_FlaxId(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_FlaxId operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.FlaxId")


@register_op("linen.module.uuid", "flax")
def _map_flax_linen_module_uuid(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_uuid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.uuid")


@register_op("linen.module.kw_only_dataclasses", "flax")
def _map_flax_linen_module_kw_only_dataclasses(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_kw_only_dataclasses operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.kw_only_dataclasses"
    )


@register_op("linen.module.RNGSequences", "flax")
def _map_flax_linen_module_RNGSequences(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_RNGSequences operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.RNGSequences"
    )


@register_op("linen.module.PRNGKey", "flax")
def _map_flax_linen_module_PRNGKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_PRNGKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.PRNGKey")


@register_op("linen.module.FrozenVariableDict", "flax")
def _map_flax_linen_module_FrozenVariableDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_FrozenVariableDict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.FrozenVariableDict"
    )


@register_op("linen.module.VariableDict", "flax")
def _map_flax_linen_module_VariableDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_VariableDict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.VariableDict"
    )


@register_op("linen.module.T", "flax")
def _map_flax_linen_module_T(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_module_T operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.T")


@register_op("linen.module.K", "flax")
def _map_flax_linen_module_K(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_module_K operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.K")


@register_op("linen.module.M", "flax")
def _map_flax_linen_module_M(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_linen_module_M operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.M")


@register_op("linen.module.TestScope", "flax")
def _map_flax_linen_module_TestScope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_TestScope operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.TestScope")


@register_op("linen.module.enable_named_call", "flax")
def _map_flax_linen_module_enable_named_call(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_enable_named_call operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.enable_named_call"
    )


@register_op("linen.module.disable_named_call", "flax")
def _map_flax_linen_module_disable_named_call(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_disable_named_call operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.disable_named_call"
    )


@register_op("linen.module.override_named_call", "flax")
def _map_flax_linen_module_override_named_call(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_override_named_call operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.override_named_call"
    )


@register_op("linen.module.InterceptorContext", "flax")
def _map_flax_linen_module_InterceptorContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_InterceptorContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.InterceptorContext"
    )


@register_op("linen.module.ThreadLocalStack", "flax")
def _map_flax_linen_module_ThreadLocalStack(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_ThreadLocalStack operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.ThreadLocalStack"
    )


@register_op("linen.module.Args", "flax")
def _map_flax_linen_module_Args(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_Args operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.Args")


@register_op("linen.module.Kwargs", "flax")
def _map_flax_linen_module_Kwargs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_Kwargs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.Kwargs")


@register_op("linen.module.NextGetter", "flax")
def _map_flax_linen_module_NextGetter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_NextGetter operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.NextGetter")


@register_op("linen.module.Interceptor", "flax")
def _map_flax_linen_module_Interceptor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_Interceptor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.Interceptor")


@register_op("linen.module.intercept_methods", "flax")
def _map_flax_linen_module_intercept_methods(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_intercept_methods operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.intercept_methods"
    )


@register_op("linen.module.run_interceptors", "flax")
def _map_flax_linen_module_run_interceptors(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_run_interceptors operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.run_interceptors"
    )


@register_op("linen.module.compact", "flax")
def _map_flax_linen_module_compact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_compact operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.compact")


@register_op("linen.module.nowrap", "flax")
def _map_flax_linen_module_nowrap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_nowrap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.nowrap")


@register_op("linen.module.compact_name_scope", "flax")
def _map_flax_linen_module_compact_name_scope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_compact_name_scope operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.compact_name_scope"
    )


@register_op("linen.module.wrap_method_once", "flax")
def _map_flax_linen_module_wrap_method_once(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_wrap_method_once operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.wrap_method_once"
    )


@register_op("linen.module.wrap_descriptor_once", "flax")
def _map_flax_linen_module_wrap_descriptor_once(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_wrap_descriptor_once operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.wrap_descriptor_once"
    )


@register_op("linen.module.SetupState", "flax")
def _map_flax_linen_module_SetupState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_SetupState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.SetupState")


@register_op("linen.module.tuple_reduce", "flax")
def _map_flax_linen_module_tuple_reduce(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_tuple_reduce operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.tuple_reduce"
    )


@register_op("linen.module.tuple_init", "flax")
def _map_flax_linen_module_tuple_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_tuple_init operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.tuple_init")


@register_op("linen.module.capture_call_intermediates", "flax")
def _map_flax_linen_module_capture_call_intermediates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_capture_call_intermediates operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.module.capture_call_intermediates",
    )


@register_op("linen.module.ParentDescriptor", "flax")
def _map_flax_linen_module_ParentDescriptor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_ParentDescriptor operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.ParentDescriptor"
    )


@register_op("linen.module.Descriptor", "flax")
def _map_flax_linen_module_Descriptor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_Descriptor operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.Descriptor")


@register_op("linen.module.DescriptorWrapper", "flax")
def _map_flax_linen_module_DescriptorWrapper(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_DescriptorWrapper operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.DescriptorWrapper"
    )


@register_op("linen.module.create_descriptor_wrapper", "flax")
def _map_flax_linen_module_create_descriptor_wrapper(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_create_descriptor_wrapper operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="linen.module.create_descriptor_wrapper",
    )


@register_op("linen.module.module_field", "flax")
def _map_flax_linen_module_module_field(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_module_field operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.module_field"
    )


@register_op("linen.module.ModuleBase", "flax")
def _map_flax_linen_module_ModuleBase(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_ModuleBase operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.ModuleBase")


@register_op("linen.module.Module", "flax")
def _map_flax_linen_module_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.Module")


@register_op("linen.module.merge_param", "flax")
def _map_flax_linen_module_merge_param(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_merge_param operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.merge_param")


@register_op("linen.module.apply", "flax")
def _map_flax_linen_module_apply(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_apply operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.apply")


@register_op("linen.module.init_with_output", "flax")
def _map_flax_linen_module_init_with_output(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_init_with_output operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.init_with_output"
    )


@register_op("linen.module.init", "flax")
def _map_flax_linen_module_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_init operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.init")


@register_op("linen.module.CompactNameScope", "flax")
def _map_flax_linen_module_CompactNameScope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_CompactNameScope operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.CompactNameScope"
    )


@register_op("linen.module.share_scope", "flax")
def _map_flax_linen_module_share_scope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_linen_module_share_scope operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="linen.module.share_scope")


@register_op("training.lr_schedule.create_constant_learning_rate_schedule", "flax")
def _map_flax_training_lr_schedule_create_constant_learning_rate_schedule(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_lr_schedule_create_constant_learning_rate_schedule operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.lr_schedule.create_constant_learning_rate_schedule",
    )


@register_op("training.lr_schedule.create_stepped_learning_rate_schedule", "flax")
def _map_flax_training_lr_schedule_create_stepped_learning_rate_schedule(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_lr_schedule_create_stepped_learning_rate_schedule operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.lr_schedule.create_stepped_learning_rate_schedule",
    )


@register_op("training.lr_schedule.create_cosine_learning_rate_schedule", "flax")
def _map_flax_training_lr_schedule_create_cosine_learning_rate_schedule(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_lr_schedule_create_cosine_learning_rate_schedule operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.lr_schedule.create_cosine_learning_rate_schedule",
    )


@register_op("training.early_stopping.struct", "flax")
def _map_flax_training_early_stopping_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_early_stopping_struct operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.early_stopping.struct"
    )


@register_op("training.early_stopping.EarlyStopping", "flax")
def _map_flax_training_early_stopping_EarlyStopping(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_early_stopping_EarlyStopping operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.early_stopping.EarlyStopping",
    )


@register_op("training.orbax_utils.PyTree", "flax")
def _map_flax_training_orbax_utils_PyTree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_orbax_utils_PyTree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.orbax_utils.PyTree"
    )


@register_op("training.orbax_utils.is_multi_device_array", "flax")
def _map_flax_training_orbax_utils_is_multi_device_array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_orbax_utils_is_multi_device_array operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.orbax_utils.is_multi_device_array",
    )


@register_op("training.orbax_utils.save_args_from_target", "flax")
def _map_flax_training_orbax_utils_save_args_from_target(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_orbax_utils_save_args_from_target operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.orbax_utils.save_args_from_target",
    )


@register_op("training.orbax_utils.maybe_construct_transformations", "flax")
def _map_flax_training_orbax_utils_maybe_construct_transformations(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_orbax_utils_maybe_construct_transformations operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.orbax_utils.maybe_construct_transformations",
    )


@register_op("training.orbax_utils.restore_args_from_target", "flax")
def _map_flax_training_orbax_utils_restore_args_from_target(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_orbax_utils_restore_args_from_target operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.orbax_utils.restore_args_from_target",
    )


@register_op("training.train_state.struct", "flax")
def _map_flax_training_train_state_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_train_state_struct operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.train_state.struct"
    )


@register_op("training.train_state.OVERWRITE_WITH_GRADIENT", "flax")
def _map_flax_training_train_state_OVERWRITE_WITH_GRADIENT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_train_state_OVERWRITE_WITH_GRADIENT operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.train_state.OVERWRITE_WITH_GRADIENT",
    )


@register_op("training.train_state.TrainState", "flax")
def _map_flax_training_train_state_TrainState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_train_state_TrainState operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.train_state.TrainState"
    )


@register_op("training.checkpoints.config", "flax")
def _map_flax_training_checkpoints_config(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_config operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.checkpoints.config"
    )


@register_op("training.checkpoints.errors", "flax")
def _map_flax_training_checkpoints_errors(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_errors operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.checkpoints.errors"
    )


@register_op("training.checkpoints.io", "flax")
def _map_flax_training_checkpoints_io(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_io operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="training.checkpoints.io")


@register_op("training.checkpoints.serialization", "flax")
def _map_flax_training_checkpoints_serialization(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_serialization operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.serialization",
    )


@register_op("training.checkpoints.traverse_util", "flax")
def _map_flax_training_checkpoints_traverse_util(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_traverse_util operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.traverse_util",
    )


@register_op("training.checkpoints.orbax_utils", "flax")
def _map_flax_training_checkpoints_orbax_utils(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_orbax_utils operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.checkpoints.orbax_utils"
    )


@register_op("training.checkpoints.SIGNED_FLOAT_RE", "flax")
def _map_flax_training_checkpoints_SIGNED_FLOAT_RE(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_SIGNED_FLOAT_RE operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.SIGNED_FLOAT_RE",
    )


@register_op("training.checkpoints.UNSIGNED_FLOAT_RE", "flax")
def _map_flax_training_checkpoints_UNSIGNED_FLOAT_RE(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_UNSIGNED_FLOAT_RE operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.UNSIGNED_FLOAT_RE",
    )


@register_op("training.checkpoints.MODULE_NUM_RE", "flax")
def _map_flax_training_checkpoints_MODULE_NUM_RE(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_MODULE_NUM_RE operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.MODULE_NUM_RE",
    )


@register_op("training.checkpoints.SCHEME_RE", "flax")
def _map_flax_training_checkpoints_SCHEME_RE(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_SCHEME_RE operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.checkpoints.SCHEME_RE"
    )


@register_op("training.checkpoints.MP_ARRAY_POSTFIX", "flax")
def _map_flax_training_checkpoints_MP_ARRAY_POSTFIX(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_MP_ARRAY_POSTFIX operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.MP_ARRAY_POSTFIX",
    )


@register_op("training.checkpoints.MP_ARRAY_PH", "flax")
def _map_flax_training_checkpoints_MP_ARRAY_PH(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_MP_ARRAY_PH operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.checkpoints.MP_ARRAY_PH"
    )


@register_op("training.checkpoints.COMMIT_SUCCESS_FILE", "flax")
def _map_flax_training_checkpoints_COMMIT_SUCCESS_FILE(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_COMMIT_SUCCESS_FILE operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.COMMIT_SUCCESS_FILE",
    )


@register_op("training.checkpoints.ORBAX_CKPT_FILENAME", "flax")
def _map_flax_training_checkpoints_ORBAX_CKPT_FILENAME(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_ORBAX_CKPT_FILENAME operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.ORBAX_CKPT_FILENAME",
    )


@register_op("training.checkpoints.ORBAX_MANIFEST_OCDBT", "flax")
def _map_flax_training_checkpoints_ORBAX_MANIFEST_OCDBT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_ORBAX_MANIFEST_OCDBT operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.ORBAX_MANIFEST_OCDBT",
    )


@register_op("training.checkpoints.ORBAX_METADATA_FILENAME", "flax")
def _map_flax_training_checkpoints_ORBAX_METADATA_FILENAME(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_ORBAX_METADATA_FILENAME operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.ORBAX_METADATA_FILENAME",
    )


@register_op("training.checkpoints.PyTree", "flax")
def _map_flax_training_checkpoints_PyTree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_PyTree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.checkpoints.PyTree"
    )


@register_op("training.checkpoints.MultiprocessArrayType", "flax")
def _map_flax_training_checkpoints_MultiprocessArrayType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_MultiprocessArrayType operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.MultiprocessArrayType",
    )


@register_op("training.checkpoints.AsyncManager", "flax")
def _map_flax_training_checkpoints_AsyncManager(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_AsyncManager operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.checkpoints.AsyncManager"
    )


@register_op("training.checkpoints.natural_sort", "flax")
def _map_flax_training_checkpoints_natural_sort(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_natural_sort operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.checkpoints.natural_sort"
    )


@register_op("training.checkpoints.safe_normpath", "flax")
def _map_flax_training_checkpoints_safe_normpath(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_safe_normpath operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.safe_normpath",
    )


@register_op("training.checkpoints.save_checkpoint", "flax")
def _map_flax_training_checkpoints_save_checkpoint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_save_checkpoint operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.save_checkpoint",
    )


@register_op("training.checkpoints.save_checkpoint_multiprocess", "flax")
def _map_flax_training_checkpoints_save_checkpoint_multiprocess(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_save_checkpoint_multiprocess operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.save_checkpoint_multiprocess",
    )


@register_op("training.checkpoints.latest_checkpoint", "flax")
def _map_flax_training_checkpoints_latest_checkpoint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_latest_checkpoint operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.latest_checkpoint",
    )


@register_op("training.checkpoints.available_steps", "flax")
def _map_flax_training_checkpoints_available_steps(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_available_steps operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.available_steps",
    )


@register_op("training.checkpoints.restore_checkpoint", "flax")
def _map_flax_training_checkpoints_restore_checkpoint(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_restore_checkpoint operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.restore_checkpoint",
    )


@register_op("training.checkpoints.convert_pre_linen", "flax")
def _map_flax_training_checkpoints_convert_pre_linen(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_checkpoints_convert_pre_linen operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.checkpoints.convert_pre_linen",
    )


@register_op("training.dynamic_scale.struct", "flax")
def _map_flax_training_dynamic_scale_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_dynamic_scale_struct operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.dynamic_scale.struct"
    )


@register_op("training.dynamic_scale.Array", "flax")
def _map_flax_training_dynamic_scale_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_dynamic_scale_Array operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.dynamic_scale.Array"
    )


@register_op("training.dynamic_scale.DynamicScaleResult", "flax")
def _map_flax_training_dynamic_scale_DynamicScaleResult(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_dynamic_scale_DynamicScaleResult operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.dynamic_scale.DynamicScaleResult",
    )


@register_op("training.dynamic_scale.DynamicScale", "flax")
def _map_flax_training_dynamic_scale_DynamicScale(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_dynamic_scale_DynamicScale operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.dynamic_scale.DynamicScale",
    )


@register_op("training.common_utils.shard", "flax")
def _map_flax_training_common_utils_shard(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_common_utils_shard operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.common_utils.shard"
    )


@register_op("training.common_utils.shard_prng_key", "flax")
def _map_flax_training_common_utils_shard_prng_key(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_common_utils_shard_prng_key operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.common_utils.shard_prng_key",
    )


@register_op("training.common_utils.stack_forest", "flax")
def _map_flax_training_common_utils_stack_forest(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_common_utils_stack_forest operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.common_utils.stack_forest",
    )


@register_op("training.common_utils.get_metrics", "flax")
def _map_flax_training_common_utils_get_metrics(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_common_utils_get_metrics operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.common_utils.get_metrics"
    )


@register_op("training.common_utils.onehot", "flax")
def _map_flax_training_common_utils_onehot(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_common_utils_onehot operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="training.common_utils.onehot"
    )


@register_op("training.prefetch_iterator.PrefetchIterator", "flax")
def _map_flax_training_prefetch_iterator_PrefetchIterator(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_training_prefetch_iterator_PrefetchIterator operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="training.prefetch_iterator.PrefetchIterator",
    )


@register_op("metrics.tensorboard.io", "flax")
def _map_flax_metrics_tensorboard_io(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_metrics_tensorboard_io operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="metrics.tensorboard.io")


@register_op("metrics.tensorboard.SummaryWriter", "flax")
def _map_flax_metrics_tensorboard_SummaryWriter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_metrics_tensorboard_SummaryWriter operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="metrics.tensorboard.SummaryWriter"
    )


@register_op("nnx.logical_axis_rules", "flax")
def _map_flax_nnx_logical_axis_rules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_logical_axis_rules operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.logical_axis_rules")


@register_op("nnx.avg_pool", "flax")
def _map_flax_nnx_avg_pool(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_avg_pool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.avg_pool")


@register_op("nnx.max_pool", "flax")
def _map_flax_nnx_max_pool(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_max_pool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.max_pool")


@register_op("nnx.min_pool", "flax")
def _map_flax_nnx_min_pool(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_min_pool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.min_pool")


@register_op("nnx.pool", "flax")
def _map_flax_nnx_pool(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_pool operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pool")


@register_op("nnx.Initializer", "flax")
def _map_flax_nnx_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_Initializer operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Initializer")


@register_op("nnx.wrappers", "flax")
def _map_flax_nnx_wrappers(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_wrappers operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.wrappers")


@register_op("nnx.WithTag", "flax")
def _map_flax_nnx_WithTag(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_WithTag operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.WithTag")


@register_op("nnx.PathContains", "flax")
def _map_flax_nnx_PathContains(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_PathContains operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.PathContains")


@register_op("nnx.OfType", "flax")
def _map_flax_nnx_OfType(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_OfType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.OfType")


@register_op("nnx.Any", "flax")
def _map_flax_nnx_Any(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Any operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Any")


@register_op("nnx.All", "flax")
def _map_flax_nnx_All(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_All operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.All")


@register_op("nnx.Not", "flax")
def _map_flax_nnx_Not(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Not operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Not")


@register_op("nnx.Everything", "flax")
def _map_flax_nnx_Everything(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Everything operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Everything")


@register_op("nnx.Nothing", "flax")
def _map_flax_nnx_Nothing(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Nothing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Nothing")


@register_op("nnx.GraphDef", "flax")
def _map_flax_nnx_GraphDef(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_GraphDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.GraphDef")


@register_op("nnx.GraphState", "flax")
def _map_flax_nnx_GraphState(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_GraphState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.GraphState")


@register_op("nnx.PureState", "flax")
def _map_flax_nnx_PureState(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_PureState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.PureState")


@register_op("nnx.object", "flax")
def _map_flax_nnx_object(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_object operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.object")


@register_op("nnx.Pytree", "flax")
def _map_flax_nnx_Pytree(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Pytree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Pytree")


@register_op("nnx.Object", "flax")
def _map_flax_nnx_Object(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Object operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Object")


@register_op("nnx.Data", "flax")
def _map_flax_nnx_Data(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Data operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Data")


@register_op("nnx.Static", "flax")
def _map_flax_nnx_Static(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Static operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Static")


@register_op("nnx.dataclass", "flax")
def _map_flax_nnx_dataclass(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_dataclass operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.dataclass")


@register_op("nnx.data", "flax")
def _map_flax_nnx_data(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_data operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.data")


@register_op("nnx.static", "flax")
def _map_flax_nnx_static(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_static operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.static")


@register_op("nnx.register_data_type", "flax")
def _map_flax_nnx_register_data_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_register_data_type operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.register_data_type")


@register_op("nnx.is_data", "flax")
def _map_flax_nnx_is_data(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_is_data operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.is_data")


@register_op("nnx.has_data", "flax")
def _map_flax_nnx_has_data(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_has_data operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.has_data")


@register_op("nnx.check_pytree", "flax")
def _map_flax_nnx_check_pytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_check_pytree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.check_pytree")


@register_op("nnx.Dict", "flax")
def _map_flax_nnx_Dict(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Dict operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Dict")


@register_op("nnx.List", "flax")
def _map_flax_nnx_List(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_List operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.List")


@register_op("nnx.Sequential", "flax")
def _map_flax_nnx_Sequential(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Sequential operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Sequential")


@register_op("nnx.TrainState", "flax")
def _map_flax_nnx_TrainState(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_TrainState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.TrainState")


@register_op("nnx.M", "flax")
def _map_flax_nnx_M(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_M operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.M")


@register_op("nnx.Module", "flax")
def _map_flax_nnx_Module(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Module")


@register_op("nnx.capture", "flax")
def _map_flax_nnx_capture(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_capture operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.capture")


@register_op("nnx.view", "flax")
def _map_flax_nnx_view(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_view operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.view")


@register_op("nnx.view_info", "flax")
def _map_flax_nnx_view_info(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_view_info operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.view_info")


@register_op("nnx.with_attributes", "flax")
def _map_flax_nnx_with_attributes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_with_attributes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.with_attributes")


@register_op("nnx.iter_children", "flax")
def _map_flax_nnx_iter_children(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_iter_children operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.iter_children")


@register_op("nnx.iter_modules", "flax")
def _map_flax_nnx_iter_modules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_iter_modules operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.iter_modules")


@register_op("nnx.merge", "flax")
def _map_flax_nnx_merge(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_merge operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.merge")


@register_op("nnx.UpdateContext", "flax")
def _map_flax_nnx_UpdateContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_UpdateContext operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.UpdateContext")


@register_op("nnx.update_context", "flax")
def _map_flax_nnx_update_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_update_context operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.update_context")


@register_op("nnx.current_update_context", "flax")
def _map_flax_nnx_current_update_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_current_update_context operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.current_update_context"
    )


@register_op("nnx.split", "flax")
def _map_flax_nnx_split(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_split operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.split")


@register_op("nnx.update", "flax")
def _map_flax_nnx_update(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_update operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.update")


@register_op("nnx.clone", "flax")
def _map_flax_nnx_clone(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_clone operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.clone")


@register_op("nnx.pop", "flax")
def _map_flax_nnx_pop(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_pop operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pop")


@register_op("nnx.state", "flax")
def _map_flax_nnx_state(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.state")


@register_op("nnx.graphdef", "flax")
def _map_flax_nnx_graphdef(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graphdef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphdef")


@register_op("nnx.iter_graph", "flax")
def _map_flax_nnx_iter_graph(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_iter_graph operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.iter_graph")


@register_op("nnx.recursive_map", "flax")
def _map_flax_nnx_recursive_map(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_recursive_map operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.recursive_map")


@register_op("nnx.find_duplicates", "flax")
def _map_flax_nnx_find_duplicates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_find_duplicates operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.find_duplicates")


@register_op("nnx.map", "flax")
def _map_flax_nnx_map(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_map operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.map")


@register_op("nnx.call", "flax")
def _map_flax_nnx_call(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_call operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.call")


@register_op("nnx.set_metadata", "flax")
def _map_flax_nnx_set_metadata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_set_metadata operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.set_metadata")


@register_op("nnx.SplitContext", "flax")
def _map_flax_nnx_SplitContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_SplitContext operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.SplitContext")


@register_op("nnx.split_context", "flax")
def _map_flax_nnx_split_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_split_context operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.split_context")


@register_op("nnx.MergeContext", "flax")
def _map_flax_nnx_MergeContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_MergeContext operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.MergeContext")


@register_op("nnx.merge_context", "flax")
def _map_flax_nnx_merge_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_merge_context operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.merge_context")


@register_op("nnx.variables", "flax")
def _map_flax_nnx_variables(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_variables operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variables")


@register_op("nnx.vars_as", "flax")
def _map_flax_nnx_vars_as(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_vars_as operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.vars_as")


@register_op("nnx.pure", "flax")
def _map_flax_nnx_pure(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_pure operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pure")


@register_op("nnx.cached_partial", "flax")
def _map_flax_nnx_cached_partial(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_cached_partial operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.cached_partial")


@register_op("nnx.flatten", "flax")
def _map_flax_nnx_flatten(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_flatten operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.flatten")


@register_op("nnx.unflatten", "flax")
def _map_flax_nnx_unflatten(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_unflatten operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.unflatten")


@register_op("nnx.set_graph_mode", "flax")
def _map_flax_nnx_set_graph_mode(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_set_graph_mode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.set_graph_mode")


@register_op("nnx.set_graph_updates", "flax")
def _map_flax_nnx_set_graph_updates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_set_graph_updates operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.set_graph_updates")


@register_op("nnx.initializers", "flax")
def _map_flax_nnx_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_initializers operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.initializers")


@register_op("nnx.celu", "flax")
def _map_flax_nnx_celu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_celu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.celu")


@register_op("nnx.elu", "flax")
def _map_flax_nnx_elu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_elu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.elu")


@register_op("nnx.gelu", "flax")
def _map_flax_nnx_gelu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_gelu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.gelu")


@register_op("nnx.glu", "flax")
def _map_flax_nnx_glu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_glu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.glu")


@register_op("nnx.hard_sigmoid", "flax")
def _map_flax_nnx_hard_sigmoid(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_hard_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.hard_sigmoid")


@register_op("nnx.hard_silu", "flax")
def _map_flax_nnx_hard_silu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_hard_silu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.hard_silu")


@register_op("nnx.hard_swish", "flax")
def _map_flax_nnx_hard_swish(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_hard_swish operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.hard_swish")


@register_op("nnx.hard_tanh", "flax")
def _map_flax_nnx_hard_tanh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_hard_tanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.hard_tanh")


@register_op("nnx.leaky_relu", "flax")
def _map_flax_nnx_leaky_relu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_leaky_relu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.leaky_relu")


@register_op("nnx.log_sigmoid", "flax")
def _map_flax_nnx_log_sigmoid(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_log_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.log_sigmoid")


@register_op("nnx.log_softmax", "flax")
def _map_flax_nnx_log_softmax(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_log_softmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.log_softmax")


@register_op("nnx.logsumexp", "flax")
def _map_flax_nnx_logsumexp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_logsumexp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.logsumexp")


@register_op("nnx.one_hot", "flax")
def _map_flax_nnx_one_hot(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_one_hot operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.one_hot")


@register_op("nnx.relu", "flax")
def _map_flax_nnx_relu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_relu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.relu")


@register_op("nnx.relu6", "flax")
def _map_flax_nnx_relu6(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_relu6 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.relu6")


@register_op("nnx.selu", "flax")
def _map_flax_nnx_selu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_selu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.selu")


@register_op("nnx.sigmoid", "flax")
def _map_flax_nnx_sigmoid(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.sigmoid")


@register_op("nnx.identity", "flax")
def _map_flax_nnx_identity(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_identity operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.identity")


@register_op("nnx.silu", "flax")
def _map_flax_nnx_silu(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_silu operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.silu")


@register_op("nnx.soft_sign", "flax")
def _map_flax_nnx_soft_sign(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_soft_sign operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.soft_sign")


@register_op("nnx.softmax", "flax")
def _map_flax_nnx_softmax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_softmax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.softmax")


@register_op("nnx.softplus", "flax")
def _map_flax_nnx_softplus(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_softplus operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.softplus")


@register_op("nnx.standardize", "flax")
def _map_flax_nnx_standardize(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_standardize operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.standardize")


@register_op("nnx.swish", "flax")
def _map_flax_nnx_swish(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_swish operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.swish")


@register_op("nnx.tanh", "flax")
def _map_flax_nnx_tanh(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_tanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.tanh")


@register_op("nnx.PReLU", "flax")
def _map_flax_nnx_PReLU(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_PReLU operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.PReLU")


@register_op("nnx.MultiHeadAttention", "flax")
def _map_flax_nnx_MultiHeadAttention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_MultiHeadAttention operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.MultiHeadAttention")


@register_op("nnx.combine_masks", "flax")
def _map_flax_nnx_combine_masks(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_combine_masks operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.combine_masks")


@register_op("nnx.dot_product_attention", "flax")
def _map_flax_nnx_dot_product_attention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_dot_product_attention operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.dot_product_attention"
    )


@register_op("nnx.make_attention_mask", "flax")
def _map_flax_nnx_make_attention_mask(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_make_attention_mask operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.make_attention_mask")


@register_op("nnx.make_causal_mask", "flax")
def _map_flax_nnx_make_causal_mask(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_make_causal_mask operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.make_causal_mask")


@register_op("nnx.RNNCellBase", "flax")
def _map_flax_nnx_RNNCellBase(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_RNNCellBase operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.RNNCellBase")


@register_op("nnx.LSTMCell", "flax")
def _map_flax_nnx_LSTMCell(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_LSTMCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.LSTMCell")


@register_op("nnx.GRUCell", "flax")
def _map_flax_nnx_GRUCell(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_GRUCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.GRUCell")


@register_op("nnx.OptimizedLSTMCell", "flax")
def _map_flax_nnx_OptimizedLSTMCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_OptimizedLSTMCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.OptimizedLSTMCell")


@register_op("nnx.SimpleCell", "flax")
def _map_flax_nnx_SimpleCell(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_SimpleCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.SimpleCell")


@register_op("nnx.RNN", "flax")
def _map_flax_nnx_RNN(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_RNN operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.RNN")


@register_op("nnx.Bidirectional", "flax")
def _map_flax_nnx_Bidirectional(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_Bidirectional operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Bidirectional")


@register_op("nnx.Conv", "flax")
def _map_flax_nnx_Conv(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Conv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Conv")


@register_op("nnx.ConvTranspose", "flax")
def _map_flax_nnx_ConvTranspose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_ConvTranspose operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.ConvTranspose")


@register_op("nnx.Embed", "flax")
def _map_flax_nnx_Embed(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Embed operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Embed")


@register_op("nnx.Linear", "flax")
def _map_flax_nnx_Linear(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Linear operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Linear")


@register_op("nnx.LinearGeneral", "flax")
def _map_flax_nnx_LinearGeneral(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_LinearGeneral operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.LinearGeneral")


@register_op("nnx.Einsum", "flax")
def _map_flax_nnx_Einsum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Einsum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Einsum")


@register_op("nnx.LoRA", "flax")
def _map_flax_nnx_LoRA(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_LoRA operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.LoRA")


@register_op("nnx.LoRALinear", "flax")
def _map_flax_nnx_LoRALinear(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_LoRALinear operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.LoRALinear")


@register_op("nnx.LoRAParam", "flax")
def _map_flax_nnx_LoRAParam(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_LoRAParam operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.LoRAParam")


@register_op("nnx.BatchNorm", "flax")
def _map_flax_nnx_BatchNorm(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_BatchNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.BatchNorm")


@register_op("nnx.LayerNorm", "flax")
def _map_flax_nnx_LayerNorm(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_LayerNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.LayerNorm")


@register_op("nnx.RMSNorm", "flax")
def _map_flax_nnx_RMSNorm(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_RMSNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.RMSNorm")


@register_op("nnx.GroupNorm", "flax")
def _map_flax_nnx_GroupNorm(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_GroupNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.GroupNorm")


@register_op("nnx.InstanceNorm", "flax")
def _map_flax_nnx_InstanceNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_InstanceNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.InstanceNorm")


@register_op("nnx.SpectralNorm", "flax")
def _map_flax_nnx_SpectralNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_SpectralNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.SpectralNorm")


@register_op("nnx.WeightNorm", "flax")
def _map_flax_nnx_WeightNorm(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_WeightNorm operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.WeightNorm")


@register_op("nnx.Dropout", "flax")
def _map_flax_nnx_Dropout(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Dropout operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Dropout")


@register_op("nnx.Rngs", "flax")
def _map_flax_nnx_Rngs(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Rngs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Rngs")


@register_op("nnx.RngStream", "flax")
def _map_flax_nnx_RngStream(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_RngStream operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.RngStream")


@register_op("nnx.RngState", "flax")
def _map_flax_nnx_RngState(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_RngState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.RngState")


@register_op("nnx.RngKey", "flax")
def _map_flax_nnx_RngKey(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_RngKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.RngKey")


@register_op("nnx.RngCount", "flax")
def _map_flax_nnx_RngCount(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_RngCount operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.RngCount")


@register_op("nnx.fork_rngs", "flax")
def _map_flax_nnx_fork_rngs(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_fork_rngs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.fork_rngs")


@register_op("nnx.reseed", "flax")
def _map_flax_nnx_reseed(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_reseed operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reseed")


@register_op("nnx.split_rngs", "flax")
def _map_flax_nnx_split_rngs(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_split_rngs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.split_rngs")


@register_op("nnx.restore_rngs", "flax")
def _map_flax_nnx_restore_rngs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_restore_rngs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.restore_rngs")


@register_op("nnx.PARTITION_NAME", "flax")
def _map_flax_nnx_PARTITION_NAME(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_PARTITION_NAME operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.PARTITION_NAME")


@register_op("nnx.get_partition_spec", "flax")
def _map_flax_nnx_get_partition_spec(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_get_partition_spec operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.get_partition_spec")


@register_op("nnx.get_named_sharding", "flax")
def _map_flax_nnx_get_named_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_get_named_sharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.get_named_sharding")


@register_op("nnx.with_partitioning", "flax")
def _map_flax_nnx_with_partitioning(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_with_partitioning operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.with_partitioning")


@register_op("nnx.get_abstract_model", "flax")
def _map_flax_nnx_get_abstract_model(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_get_abstract_model operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.get_abstract_model")


@register_op("nnx.abstract_with_sharding", "flax")
def _map_flax_nnx_abstract_with_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_abstract_with_sharding operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.abstract_with_sharding"
    )


@register_op("nnx.FlatState", "flax")
def _map_flax_nnx_FlatState(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_FlatState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.FlatState")


@register_op("nnx.State", "flax")
def _map_flax_nnx_State(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_State operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.State")


@register_op("nnx.to_flat_state", "flax")
def _map_flax_nnx_to_flat_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_to_flat_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.to_flat_state")


@register_op("nnx.from_flat_state", "flax")
def _map_flax_nnx_from_flat_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_from_flat_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.from_flat_state")


@register_op("nnx.to_pure_dict", "flax")
def _map_flax_nnx_to_pure_dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_to_pure_dict operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.to_pure_dict")


@register_op("nnx.replace_by_pure_dict", "flax")
def _map_flax_nnx_replace_by_pure_dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_replace_by_pure_dict operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.replace_by_pure_dict")


@register_op("nnx.restore_int_paths", "flax")
def _map_flax_nnx_restore_int_paths(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_restore_int_paths operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.restore_int_paths")


@register_op("nnx.filter_state", "flax")
def _map_flax_nnx_filter_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filter_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filter_state")


@register_op("nnx.merge_state", "flax")
def _map_flax_nnx_merge_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_merge_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.merge_state")


@register_op("nnx.split_state", "flax")
def _map_flax_nnx_split_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_split_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.split_state")


@register_op("nnx.map_state", "flax")
def _map_flax_nnx_map_state(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_map_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.map_state")


@register_op("nnx.metrics", "flax")
def _map_flax_nnx_metrics(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_metrics operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.metrics")


@register_op("nnx.Param", "flax")
def _map_flax_nnx_Param(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Param operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Param")


@register_op("nnx.optimizer", "flax")
def _map_flax_nnx_optimizer(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_optimizer operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.optimizer")


@register_op("nnx.Metric", "flax")
def _map_flax_nnx_Metric(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Metric operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Metric")


@register_op("nnx.MultiMetric", "flax")
def _map_flax_nnx_MultiMetric(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_MultiMetric operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.MultiMetric")


@register_op("nnx.OptState", "flax")
def _map_flax_nnx_OptState(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_OptState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.OptState")


@register_op("nnx.OptArray", "flax")
def _map_flax_nnx_OptArray(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_OptArray operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.OptArray")


@register_op("nnx.OptVariable", "flax")
def _map_flax_nnx_OptVariable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_OptVariable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.OptVariable")


@register_op("nnx.Optimizer", "flax")
def _map_flax_nnx_Optimizer(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Optimizer operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Optimizer")


@register_op("nnx.ModelAndOptimizer", "flax")
def _map_flax_nnx_ModelAndOptimizer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_ModelAndOptimizer operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.ModelAndOptimizer")


@register_op("nnx.DiffState", "flax")
def _map_flax_nnx_DiffState(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_DiffState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.DiffState")


@register_op("nnx.grad", "flax")
def _map_flax_nnx_grad(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_grad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.grad")


@register_op("nnx.value_and_grad", "flax")
def _map_flax_nnx_value_and_grad(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_value_and_grad operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.value_and_grad")


@register_op("nnx.custom_vjp", "flax")
def _map_flax_nnx_custom_vjp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_custom_vjp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.custom_vjp")


@register_op("nnx.vjp", "flax")
def _map_flax_nnx_vjp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_vjp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.vjp")


@register_op("nnx.jvp", "flax")
def _map_flax_nnx_jvp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_jvp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.jvp")


@register_op("nnx.remat", "flax")
def _map_flax_nnx_remat(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_remat operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.remat")


@register_op("nnx.jit", "flax")
def _map_flax_nnx_jit(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_jit operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.jit")


@register_op("nnx.jit_partial", "flax")
def _map_flax_nnx_jit_partial(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_jit_partial operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.jit_partial")


@register_op("nnx.shard_map", "flax")
def _map_flax_nnx_shard_map(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_shard_map operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.shard_map")


@register_op("nnx.StateSharding", "flax")
def _map_flax_nnx_StateSharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_StateSharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.StateSharding")


@register_op("nnx.Carry", "flax")
def _map_flax_nnx_Carry(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Carry operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Carry")


@register_op("nnx.scan", "flax")
def _map_flax_nnx_scan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_scan operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.scan")


@register_op("nnx.vmap", "flax")
def _map_flax_nnx_vmap(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_vmap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.vmap")


@register_op("nnx.pmap", "flax")
def _map_flax_nnx_pmap(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_pmap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pmap")


@register_op("nnx.eval_shape", "flax")
def _map_flax_nnx_eval_shape(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_eval_shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.eval_shape")


@register_op("nnx.cond", "flax")
def _map_flax_nnx_cond(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_cond operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.cond")


@register_op("nnx.switch", "flax")
def _map_flax_nnx_switch(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_switch operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.switch")


@register_op("nnx.checkify", "flax")
def _map_flax_nnx_checkify(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_checkify operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.checkify")


@register_op("nnx.while_loop", "flax")
def _map_flax_nnx_while_loop(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_while_loop operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.while_loop")


@register_op("nnx.fori_loop", "flax")
def _map_flax_nnx_fori_loop(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_fori_loop operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.fori_loop")


@register_op("nnx.StateAxes", "flax")
def _map_flax_nnx_StateAxes(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_StateAxes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.StateAxes")


@register_op("nnx.transform_metadata", "flax")
def _map_flax_nnx_transform_metadata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transform_metadata operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transform_metadata")


@register_op("nnx.A", "flax")
def _map_flax_nnx_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.A")


@register_op("nnx.BatchStat", "flax")
def _map_flax_nnx_BatchStat(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_BatchStat operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.BatchStat")


@register_op("nnx.Cache", "flax")
def _map_flax_nnx_Cache(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Cache operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Cache")


@register_op("nnx.Intermediate", "flax")
def _map_flax_nnx_Intermediate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_Intermediate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Intermediate")


@register_op("nnx.Perturbation", "flax")
def _map_flax_nnx_Perturbation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_Perturbation operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Perturbation")


@register_op("nnx.Variable", "flax")
def _map_flax_nnx_Variable(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_Variable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.Variable")


@register_op("nnx.VariableMetadata", "flax")
def _map_flax_nnx_VariableMetadata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_VariableMetadata operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.VariableMetadata")


@register_op("nnx.with_metadata", "flax")
def _map_flax_nnx_with_metadata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_with_metadata operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.with_metadata")


@register_op("nnx.variable_type_from_name", "flax")
def _map_flax_nnx_variable_type_from_name(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variable_type_from_name operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variable_type_from_name"
    )


@register_op("nnx.variable_name_from_type", "flax")
def _map_flax_nnx_variable_name_from_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variable_name_from_type operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variable_name_from_type"
    )


@register_op("nnx.register_variable_name", "flax")
def _map_flax_nnx_register_variable_name(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_register_variable_name operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.register_variable_name"
    )


@register_op("nnx.use_eager_sharding", "flax")
def _map_flax_nnx_use_eager_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_use_eager_sharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.use_eager_sharding")


@register_op("nnx.using_eager_sharding", "flax")
def _map_flax_nnx_using_eager_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_using_eager_sharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.using_eager_sharding")


@register_op("nnx.var_defaults", "flax")
def _map_flax_nnx_var_defaults(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_var_defaults operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.var_defaults")


@register_op("nnx.display", "flax")
def _map_flax_nnx_display(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_display operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.display")


@register_op("nnx.to_tree", "flax")
def _map_flax_nnx_to_tree(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_to_tree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.to_tree")


@register_op("nnx.from_tree", "flax")
def _map_flax_nnx_from_tree(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_from_tree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.from_tree")


@register_op("nnx.NodeStates", "flax")
def _map_flax_nnx_NodeStates(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_NodeStates operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.NodeStates")


@register_op("nnx.tabulate", "flax")
def _map_flax_nnx_tabulate(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_tabulate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.tabulate")


@register_op("nnx.VariableState", "flax")
def _map_flax_nnx_VariableState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_VariableState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.VariableState")


@register_op("nnx.traversals.struct", "flax")
def _map_flax_nnx_traversals_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_traversals_struct operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.traversals.struct")


@register_op("nnx.traversals.empty_node", "flax")
def _map_flax_nnx_traversals_empty_node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_traversals_empty_node operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.traversals.empty_node"
    )


@register_op("nnx.traversals.IsLeafCallable", "flax")
def _map_flax_nnx_traversals_IsLeafCallable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_traversals_IsLeafCallable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.traversals.IsLeafCallable"
    )


@register_op("nnx.traversals.flatten_mapping", "flax")
def _map_flax_nnx_traversals_flatten_mapping(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_traversals_flatten_mapping operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.traversals.flatten_mapping"
    )


@register_op("nnx.traversals.flatten_to_sequence", "flax")
def _map_flax_nnx_traversals_flatten_to_sequence(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_traversals_flatten_to_sequence operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.traversals.flatten_to_sequence",
    )


@register_op("nnx.traversals.unflatten_mapping", "flax")
def _map_flax_nnx_traversals_unflatten_mapping(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_traversals_unflatten_mapping operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.traversals.unflatten_mapping"
    )


@register_op("nnx.proxy_caller.A", "flax")
def _map_flax_nnx_proxy_caller_A(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_proxy_caller_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.proxy_caller.A")


@register_op("nnx.proxy_caller.GetItem", "flax")
def _map_flax_nnx_proxy_caller_GetItem(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_proxy_caller_GetItem operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.proxy_caller.GetItem")


@register_op("nnx.proxy_caller.GetAttr", "flax")
def _map_flax_nnx_proxy_caller_GetAttr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_proxy_caller_GetAttr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.proxy_caller.GetAttr")


@register_op("nnx.proxy_caller.DelayedAccessor", "flax")
def _map_flax_nnx_proxy_caller_DelayedAccessor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_proxy_caller_DelayedAccessor operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.proxy_caller.DelayedAccessor"
    )


@register_op("nnx.proxy_caller.CallableProxy", "flax")
def _map_flax_nnx_proxy_caller_CallableProxy(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_proxy_caller_CallableProxy operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.proxy_caller.CallableProxy"
    )


@register_op("nnx.proxy_caller.ApplyCaller", "flax")
def _map_flax_nnx_proxy_caller_ApplyCaller(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_proxy_caller_ApplyCaller operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.proxy_caller.ApplyCaller"
    )


@register_op("nnx.ids.UUIDManager", "flax")
def _map_flax_nnx_ids_UUIDManager(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_ids_UUIDManager operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.ids.UUIDManager")


@register_op("nnx.ids.uuid", "flax")
def _map_flax_nnx_ids_uuid(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_ids_uuid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.ids.uuid")


@register_op("nnx.ids.UUID", "flax")
def _map_flax_nnx_ids_UUID(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_ids_UUID operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.ids.UUID")


@register_op("nnx.rnglib.struct", "flax")
def _map_flax_nnx_rnglib_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_struct operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.struct")


@register_op("nnx.rnglib.typing", "flax")
def _map_flax_nnx_rnglib_typing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_typing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.typing")


@register_op("nnx.rnglib.graphlib", "flax")
def _map_flax_nnx_rnglib_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_graphlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.graphlib")


@register_op("nnx.rnglib.initializers", "flax")
def _map_flax_nnx_rnglib_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_initializers operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.initializers")


@register_op("nnx.rnglib.Variable", "flax")
def _map_flax_nnx_rnglib_Variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_Variable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.Variable")


@register_op("nnx.rnglib.filterlib", "flax")
def _map_flax_nnx_rnglib_filterlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_filterlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.filterlib")


@register_op("nnx.rnglib.Pytree", "flax")
def _map_flax_nnx_rnglib_Pytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_Pytree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.Pytree")


@register_op("nnx.rnglib.MISSING", "flax")
def _map_flax_nnx_rnglib_MISSING(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_MISSING operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.MISSING")


@register_op("nnx.rnglib.Key", "flax")
def _map_flax_nnx_rnglib_Key(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_rnglib_Key operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.Key")


@register_op("nnx.rnglib.Missing", "flax")
def _map_flax_nnx_rnglib_Missing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_Missing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.Missing")


@register_op("nnx.rnglib.F", "flax")
def _map_flax_nnx_rnglib_F(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_rnglib_F operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.F")


@register_op("nnx.rnglib.A", "flax")
def _map_flax_nnx_rnglib_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_rnglib_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.A")


@register_op("nnx.rnglib.Counts", "flax")
def _map_flax_nnx_rnglib_Counts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_Counts operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.Counts")


@register_op("nnx.rnglib.AxesValue", "flax")
def _map_flax_nnx_rnglib_AxesValue(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_AxesValue operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.AxesValue")


@register_op("nnx.rnglib.SplitPattern", "flax")
def _map_flax_nnx_rnglib_SplitPattern(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_SplitPattern operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.SplitPattern")


@register_op("nnx.rnglib.OutShardingType", "flax")
def _map_flax_nnx_rnglib_OutShardingType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_OutShardingType operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.OutShardingType"
    )


@register_op("nnx.rnglib.Fargs", "flax")
def _map_flax_nnx_rnglib_Fargs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_Fargs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.Fargs")


@register_op("nnx.rnglib.KeylessInitializer", "flax")
def _map_flax_nnx_rnglib_KeylessInitializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_KeylessInitializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.KeylessInitializer"
    )


@register_op("nnx.rnglib.RngState", "flax")
def _map_flax_nnx_rnglib_RngState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_RngState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.RngState")


@register_op("nnx.rnglib.RngCount", "flax")
def _map_flax_nnx_rnglib_RngCount(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_RngCount operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.RngCount")


@register_op("nnx.rnglib.RngKey", "flax")
def _map_flax_nnx_rnglib_RngKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_RngKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.RngKey")


@register_op("nnx.rnglib.NotKey", "flax")
def _map_flax_nnx_rnglib_NotKey(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_NotKey operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.NotKey")


@register_op("nnx.rnglib.RngStream", "flax")
def _map_flax_nnx_rnglib_RngStream(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_RngStream operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.RngStream")


@register_op("nnx.rnglib.RngValue", "flax")
def _map_flax_nnx_rnglib_RngValue(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_RngValue operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.RngValue")


@register_op("nnx.rnglib.Rngs", "flax")
def _map_flax_nnx_rnglib_Rngs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_Rngs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.Rngs")


@register_op("nnx.rnglib.StreamBackup", "flax")
def _map_flax_nnx_rnglib_StreamBackup(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_StreamBackup operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.StreamBackup")


@register_op("nnx.rnglib.SplitBackups", "flax")
def _map_flax_nnx_rnglib_SplitBackups(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_SplitBackups operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.SplitBackups")


@register_op("nnx.rnglib.split_rngs", "flax")
def _map_flax_nnx_rnglib_split_rngs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_split_rngs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.split_rngs")


@register_op("nnx.rnglib.fork_rngs", "flax")
def _map_flax_nnx_rnglib_fork_rngs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_fork_rngs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.fork_rngs")


@register_op("nnx.rnglib.backup_keys", "flax")
def _map_flax_nnx_rnglib_backup_keys(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_backup_keys operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.backup_keys")


@register_op("nnx.rnglib.reseed", "flax")
def _map_flax_nnx_rnglib_reseed(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_reseed operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.reseed")


@register_op("nnx.rnglib.restore_rngs", "flax")
def _map_flax_nnx_rnglib_restore_rngs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_rnglib_restore_rngs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.rnglib.restore_rngs")


@register_op("nnx.statelib.filterlib", "flax")
def _map_flax_nnx_statelib_filterlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_filterlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.filterlib")


@register_op("nnx.statelib.reprlib", "flax")
def _map_flax_nnx_statelib_reprlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_reprlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.reprlib")


@register_op("nnx.statelib.traversals", "flax")
def _map_flax_nnx_statelib_traversals(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_traversals operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.traversals")


@register_op("nnx.statelib.variablelib", "flax")
def _map_flax_nnx_statelib_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_variablelib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.variablelib")


@register_op("nnx.statelib.Key", "flax")
def _map_flax_nnx_statelib_Key(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_Key operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.Key")


@register_op("nnx.statelib.PathParts", "flax")
def _map_flax_nnx_statelib_PathParts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_PathParts operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.PathParts")


@register_op("nnx.statelib.A", "flax")
def _map_flax_nnx_statelib_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_statelib_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.A")


@register_op("nnx.statelib.K", "flax")
def _map_flax_nnx_statelib_K(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_statelib_K operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.K")


@register_op("nnx.statelib.S", "flax")
def _map_flax_nnx_statelib_S(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_statelib_S operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.S")


@register_op("nnx.statelib.V", "flax")
def _map_flax_nnx_statelib_V(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_statelib_V operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.V")


@register_op("nnx.statelib.ExtractValueFn", "flax")
def _map_flax_nnx_statelib_ExtractValueFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_ExtractValueFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.ExtractValueFn"
    )


@register_op("nnx.statelib.SetValueFn", "flax")
def _map_flax_nnx_statelib_SetValueFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_SetValueFn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.SetValueFn")


@register_op("nnx.statelib.NestedStateRepr", "flax")
def _map_flax_nnx_statelib_NestedStateRepr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_NestedStateRepr operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.NestedStateRepr"
    )


@register_op("nnx.statelib.FlatState", "flax")
def _map_flax_nnx_statelib_FlatState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_FlatState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.FlatState")


@register_op("nnx.statelib.State", "flax")
def _map_flax_nnx_statelib_State(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_State operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.State")


@register_op("nnx.statelib.map_state", "flax")
def _map_flax_nnx_statelib_map_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_map_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.map_state")


@register_op("nnx.statelib.to_flat_state", "flax")
def _map_flax_nnx_statelib_to_flat_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_to_flat_state operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.to_flat_state"
    )


@register_op("nnx.statelib.from_flat_state", "flax")
def _map_flax_nnx_statelib_from_flat_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_from_flat_state operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.from_flat_state"
    )


@register_op("nnx.statelib.to_pure_dict", "flax")
def _map_flax_nnx_statelib_to_pure_dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_to_pure_dict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.to_pure_dict"
    )


@register_op("nnx.statelib.restore_int_paths", "flax")
def _map_flax_nnx_statelib_restore_int_paths(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_restore_int_paths operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.restore_int_paths"
    )


@register_op("nnx.statelib.replace_by_pure_dict", "flax")
def _map_flax_nnx_statelib_replace_by_pure_dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_replace_by_pure_dict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.replace_by_pure_dict"
    )


@register_op("nnx.statelib.split_state", "flax")
def _map_flax_nnx_statelib_split_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_split_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.split_state")


@register_op("nnx.statelib.filter_state", "flax")
def _map_flax_nnx_statelib_filter_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_filter_state operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.filter_state"
    )


@register_op("nnx.statelib.merge_state", "flax")
def _map_flax_nnx_statelib_merge_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_merge_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.merge_state")


@register_op("nnx.statelib.diff", "flax")
def _map_flax_nnx_statelib_diff(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_diff operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.diff")


@register_op("nnx.statelib.create_path_filters", "flax")
def _map_flax_nnx_statelib_create_path_filters(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_statelib_create_path_filters operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.statelib.create_path_filters"
    )


@register_op("nnx.graphlib.config", "flax")
def _map_flax_nnx_graphlib_config(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_config operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.config")


@register_op("nnx.graphlib.filterlib", "flax")
def _map_flax_nnx_graphlib_filterlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_filterlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.filterlib")


@register_op("nnx.graphlib.reprlib", "flax")
def _map_flax_nnx_graphlib_reprlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_reprlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.reprlib")


@register_op("nnx.graphlib.traversals", "flax")
def _map_flax_nnx_graphlib_traversals(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_traversals operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.traversals")


@register_op("nnx.graphlib.variablelib", "flax")
def _map_flax_nnx_graphlib_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_variablelib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.variablelib")


@register_op("nnx.graphlib.statelib", "flax")
def _map_flax_nnx_graphlib_statelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_statelib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.statelib")


@register_op("nnx.graphlib.ApplyCaller", "flax")
def _map_flax_nnx_graphlib_ApplyCaller(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_ApplyCaller operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.ApplyCaller")


@register_op("nnx.graphlib.CallableProxy", "flax")
def _map_flax_nnx_graphlib_CallableProxy(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_CallableProxy operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.CallableProxy"
    )


@register_op("nnx.graphlib.DelayedAccessor", "flax")
def _map_flax_nnx_graphlib_DelayedAccessor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_DelayedAccessor operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.DelayedAccessor"
    )


@register_op("nnx.graphlib.FlatState", "flax")
def _map_flax_nnx_graphlib_FlatState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_FlatState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.FlatState")


@register_op("nnx.graphlib.State", "flax")
def _map_flax_nnx_graphlib_State(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_State operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.State")


@register_op("nnx.graphlib.map_state", "flax")
def _map_flax_nnx_graphlib_map_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_map_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.map_state")


@register_op("nnx.graphlib.Variable", "flax")
def _map_flax_nnx_graphlib_Variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_Variable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.Variable")


@register_op("nnx.graphlib.is_array_ref", "flax")
def _map_flax_nnx_graphlib_is_array_ref(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_is_array_ref operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.is_array_ref"
    )


@register_op("nnx.graphlib.V", "flax")
def _map_flax_nnx_graphlib_V(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graphlib_V operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.V")


@register_op("nnx.graphlib.BaseConfigContext", "flax")
def _map_flax_nnx_graphlib_BaseConfigContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_BaseConfigContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.BaseConfigContext"
    )


@register_op("nnx.graphlib.HashableMapping", "flax")
def _map_flax_nnx_graphlib_HashableMapping(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_HashableMapping operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.HashableMapping"
    )


@register_op("nnx.graphlib.Key", "flax")
def _map_flax_nnx_graphlib_Key(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_Key operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.Key")


@register_op("nnx.graphlib.PathParts", "flax")
def _map_flax_nnx_graphlib_PathParts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_PathParts operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.PathParts")


@register_op("nnx.graphlib.is_key_like", "flax")
def _map_flax_nnx_graphlib_is_key_like(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_is_key_like operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.is_key_like")


@register_op("nnx.graphlib.A", "flax")
def _map_flax_nnx_graphlib_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graphlib_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.A")


@register_op("nnx.graphlib.B", "flax")
def _map_flax_nnx_graphlib_B(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graphlib_B operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.B")


@register_op("nnx.graphlib.C", "flax")
def _map_flax_nnx_graphlib_C(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graphlib_C operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.C")


@register_op("nnx.graphlib.F", "flax")
def _map_flax_nnx_graphlib_F(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graphlib_F operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.F")


@register_op("nnx.graphlib.HA", "flax")
def _map_flax_nnx_graphlib_HA(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_HA operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.HA")


@register_op("nnx.graphlib.HB", "flax")
def _map_flax_nnx_graphlib_HB(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_HB operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.HB")


@register_op("nnx.graphlib.KeyT", "flax")
def _map_flax_nnx_graphlib_KeyT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_KeyT operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.KeyT")


@register_op("nnx.graphlib.Index", "flax")
def _map_flax_nnx_graphlib_Index(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_Index operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.Index")


@register_op("nnx.graphlib.Names", "flax")
def _map_flax_nnx_graphlib_Names(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_Names operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.Names")


@register_op("nnx.graphlib.Node", "flax")
def _map_flax_nnx_graphlib_Node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_Node operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.Node")


@register_op("nnx.graphlib.Leaf", "flax")
def _map_flax_nnx_graphlib_Leaf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_Leaf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.Leaf")


@register_op("nnx.graphlib.AuxData", "flax")
def _map_flax_nnx_graphlib_AuxData(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_AuxData operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.AuxData")


@register_op("nnx.graphlib.NoUpdate", "flax")
def _map_flax_nnx_graphlib_NoUpdate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_NoUpdate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.NoUpdate")


@register_op("nnx.graphlib.NO_UPDATE", "flax")
def _map_flax_nnx_graphlib_NO_UPDATE(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_NO_UPDATE operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.NO_UPDATE")


@register_op("nnx.graphlib.Repeated", "flax")
def _map_flax_nnx_graphlib_Repeated(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_Repeated operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.Repeated")


@register_op("nnx.graphlib.REPEATED", "flax")
def _map_flax_nnx_graphlib_REPEATED(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_REPEATED operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.REPEATED")


@register_op("nnx.graphlib.ArrayRefOutput", "flax")
def _map_flax_nnx_graphlib_ArrayRefOutput(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_ArrayRefOutput operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.ArrayRefOutput"
    )


@register_op("nnx.graphlib.LeafType", "flax")
def _map_flax_nnx_graphlib_LeafType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_LeafType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.LeafType")


@register_op("nnx.graphlib.GraphState", "flax")
def _map_flax_nnx_graphlib_GraphState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_GraphState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.GraphState")


@register_op("nnx.graphlib.GraphFlatState", "flax")
def _map_flax_nnx_graphlib_GraphFlatState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_GraphFlatState operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.GraphFlatState"
    )


@register_op("nnx.graphlib.is_node_leaf", "flax")
def _map_flax_nnx_graphlib_is_node_leaf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_is_node_leaf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.is_node_leaf"
    )


@register_op("nnx.graphlib.IndexMap", "flax")
def _map_flax_nnx_graphlib_IndexMap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_IndexMap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.IndexMap")


@register_op("nnx.graphlib.RefMap", "flax")
def _map_flax_nnx_graphlib_RefMap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_RefMap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.RefMap")


@register_op("nnx.graphlib.PythonRefMap", "flax")
def _map_flax_nnx_graphlib_PythonRefMap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_PythonRefMap operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.PythonRefMap"
    )


@register_op("nnx.graphlib.NodeImplBase", "flax")
def _map_flax_nnx_graphlib_NodeImplBase(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_NodeImplBase operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.NodeImplBase"
    )


@register_op("nnx.graphlib.GraphNodeImpl", "flax")
def _map_flax_nnx_graphlib_GraphNodeImpl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_GraphNodeImpl operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.GraphNodeImpl"
    )


@register_op("nnx.graphlib.PytreeNodeImpl", "flax")
def _map_flax_nnx_graphlib_PytreeNodeImpl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_PytreeNodeImpl operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.PytreeNodeImpl"
    )


@register_op("nnx.graphlib.NodeImpl", "flax")
def _map_flax_nnx_graphlib_NodeImpl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_NodeImpl operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.NodeImpl")


@register_op("nnx.graphlib.GRAPH_REGISTRY", "flax")
def _map_flax_nnx_graphlib_GRAPH_REGISTRY(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_GRAPH_REGISTRY operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.GRAPH_REGISTRY"
    )


@register_op("nnx.graphlib.PYTREE_REGISTRY", "flax")
def _map_flax_nnx_graphlib_PYTREE_REGISTRY(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_PYTREE_REGISTRY operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.PYTREE_REGISTRY"
    )


@register_op("nnx.graphlib.register_graph_node_type", "flax")
def _map_flax_nnx_graphlib_register_graph_node_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_register_graph_node_type operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.graphlib.register_graph_node_type",
    )


@register_op("nnx.graphlib.register_pytree_node_type", "flax")
def _map_flax_nnx_graphlib_register_pytree_node_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_register_pytree_node_type operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.graphlib.register_pytree_node_type",
    )


@register_op("nnx.graphlib.is_node", "flax")
def _map_flax_nnx_graphlib_is_node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_is_node operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.is_node")


@register_op("nnx.graphlib.is_graph_node", "flax")
def _map_flax_nnx_graphlib_is_graph_node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_is_graph_node operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.is_graph_node"
    )


@register_op("nnx.graphlib.is_node_type", "flax")
def _map_flax_nnx_graphlib_is_node_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_is_node_type operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.is_node_type"
    )


@register_op("nnx.graphlib.get_node_impl", "flax")
def _map_flax_nnx_graphlib_get_node_impl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_get_node_impl operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.get_node_impl"
    )


@register_op("nnx.graphlib.get_node_impl_for_type", "flax")
def _map_flax_nnx_graphlib_get_node_impl_for_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_get_node_impl_for_type operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.graphlib.get_node_impl_for_type",
    )


@register_op("nnx.graphlib.NodeRef", "flax")
def _map_flax_nnx_graphlib_NodeRef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_NodeRef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.NodeRef")


@register_op("nnx.graphlib.VariableDef", "flax")
def _map_flax_nnx_graphlib_VariableDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_VariableDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.VariableDef")


@register_op("nnx.graphlib.ArrayRefDef", "flax")
def _map_flax_nnx_graphlib_ArrayRefDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_ArrayRefDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.ArrayRefDef")


@register_op("nnx.graphlib.NodeDef", "flax")
def _map_flax_nnx_graphlib_NodeDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_NodeDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.NodeDef")


@register_op("nnx.graphlib.TreeNodeDef", "flax")
def _map_flax_nnx_graphlib_TreeNodeDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_TreeNodeDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.TreeNodeDef")


@register_op("nnx.graphlib.NodeDefType", "flax")
def _map_flax_nnx_graphlib_NodeDefType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_NodeDefType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.NodeDefType")


@register_op("nnx.graphlib.NodeAttr", "flax")
def _map_flax_nnx_graphlib_NodeAttr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_NodeAttr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.NodeAttr")


@register_op("nnx.graphlib.NODE_ATTR", "flax")
def _map_flax_nnx_graphlib_NODE_ATTR(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_NODE_ATTR operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.NODE_ATTR")


@register_op("nnx.graphlib.LeafAttr", "flax")
def _map_flax_nnx_graphlib_LeafAttr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_LeafAttr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.LeafAttr")


@register_op("nnx.graphlib.LEAF_ATTR", "flax")
def _map_flax_nnx_graphlib_LEAF_ATTR(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_LEAF_ATTR operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.LEAF_ATTR")


@register_op("nnx.graphlib.AttrType", "flax")
def _map_flax_nnx_graphlib_AttrType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_AttrType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.AttrType")


@register_op("nnx.graphlib.GraphDef", "flax")
def _map_flax_nnx_graphlib_GraphDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_GraphDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.GraphDef")


@register_op("nnx.graphlib.PureState", "flax")
def _map_flax_nnx_graphlib_PureState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_PureState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.PureState")


@register_op("nnx.graphlib.flatten", "flax")
def _map_flax_nnx_graphlib_flatten(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_flatten operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.flatten")


@register_op("nnx.graphlib.DataElem", "flax")
def _map_flax_nnx_graphlib_DataElem(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_DataElem operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.DataElem")


@register_op("nnx.graphlib.StaticElem", "flax")
def _map_flax_nnx_graphlib_StaticElem(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_StaticElem operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.StaticElem")


@register_op("nnx.graphlib.unflatten", "flax")
def _map_flax_nnx_graphlib_unflatten(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_unflatten operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.unflatten")


@register_op("nnx.graphlib.graph_pop", "flax")
def _map_flax_nnx_graphlib_graph_pop(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_graph_pop operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.graph_pop")


@register_op("nnx.graphlib.StaticCache", "flax")
def _map_flax_nnx_graphlib_StaticCache(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_StaticCache operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.StaticCache")


@register_op("nnx.graphlib.GraphContext", "flax")
def _map_flax_nnx_graphlib_GraphContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_GraphContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.GraphContext"
    )


@register_op("nnx.graphlib.GRAPH_CONTEXT", "flax")
def _map_flax_nnx_graphlib_GRAPH_CONTEXT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_GRAPH_CONTEXT operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.GRAPH_CONTEXT"
    )


@register_op("nnx.graphlib.set_graph_mode", "flax")
def _map_flax_nnx_graphlib_set_graph_mode(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_set_graph_mode operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.set_graph_mode"
    )


@register_op("nnx.graphlib.set_graph_updates", "flax")
def _map_flax_nnx_graphlib_set_graph_updates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_set_graph_updates operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.set_graph_updates"
    )


@register_op("nnx.graphlib.static_cache", "flax")
def _map_flax_nnx_graphlib_static_cache(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_static_cache operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.static_cache"
    )


@register_op("nnx.graphlib.cached_partial", "flax")
def _map_flax_nnx_graphlib_cached_partial(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_cached_partial operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.cached_partial"
    )


@register_op("nnx.graphlib.SplitContext", "flax")
def _map_flax_nnx_graphlib_SplitContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_SplitContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.SplitContext"
    )


@register_op("nnx.graphlib.split_context", "flax")
def _map_flax_nnx_graphlib_split_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_split_context operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.split_context"
    )


@register_op("nnx.graphlib.MergeContext", "flax")
def _map_flax_nnx_graphlib_MergeContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_MergeContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.MergeContext"
    )


@register_op("nnx.graphlib.merge_context", "flax")
def _map_flax_nnx_graphlib_merge_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_merge_context operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.merge_context"
    )


@register_op("nnx.graphlib.UpdateContext", "flax")
def _map_flax_nnx_graphlib_UpdateContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_UpdateContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.UpdateContext"
    )


@register_op("nnx.graphlib.UpdateContextManager", "flax")
def _map_flax_nnx_graphlib_UpdateContextManager(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_UpdateContextManager operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.UpdateContextManager"
    )


@register_op("nnx.graphlib.update_context", "flax")
def _map_flax_nnx_graphlib_update_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_update_context operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.update_context"
    )


@register_op("nnx.graphlib.current_update_context", "flax")
def _map_flax_nnx_graphlib_current_update_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_current_update_context operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.graphlib.current_update_context",
    )


@register_op("nnx.graphlib.split", "flax")
def _map_flax_nnx_graphlib_split(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_split operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.split")


@register_op("nnx.graphlib.merge", "flax")
def _map_flax_nnx_graphlib_merge(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_merge operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.merge")


@register_op("nnx.graphlib.update", "flax")
def _map_flax_nnx_graphlib_update(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_update operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.update")


@register_op("nnx.graphlib.state", "flax")
def _map_flax_nnx_graphlib_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.state")


@register_op("nnx.graphlib.variables", "flax")
def _map_flax_nnx_graphlib_variables(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_variables operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.variables")


@register_op("nnx.graphlib.map", "flax")
def _map_flax_nnx_graphlib_map(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_map operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.map")


@register_op("nnx.graphlib.graphdef", "flax")
def _map_flax_nnx_graphlib_graphdef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_graphdef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.graphdef")


@register_op("nnx.graphlib.pop", "flax")
def _map_flax_nnx_graphlib_pop(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_pop operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.pop")


@register_op("nnx.graphlib.clone", "flax")
def _map_flax_nnx_graphlib_clone(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_clone operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.clone")


@register_op("nnx.graphlib.vars_as", "flax")
def _map_flax_nnx_graphlib_vars_as(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_vars_as operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.vars_as")


@register_op("nnx.graphlib.pure", "flax")
def _map_flax_nnx_graphlib_pure(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_pure operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.pure")


@register_op("nnx.graphlib.call", "flax")
def _map_flax_nnx_graphlib_call(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_call operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.call")


@register_op("nnx.graphlib.set_metadata", "flax")
def _map_flax_nnx_graphlib_set_metadata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_set_metadata operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.set_metadata"
    )


@register_op("nnx.graphlib.iter_graph", "flax")
def _map_flax_nnx_graphlib_iter_graph(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_iter_graph operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.iter_graph")


@register_op("nnx.graphlib.iter_children", "flax")
def _map_flax_nnx_graphlib_iter_children(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_iter_children operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.iter_children"
    )


@register_op("nnx.graphlib.recursive_map", "flax")
def _map_flax_nnx_graphlib_recursive_map(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_recursive_map operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.recursive_map"
    )


@register_op("nnx.graphlib.find_duplicates", "flax")
def _map_flax_nnx_graphlib_find_duplicates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_find_duplicates operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.find_duplicates"
    )


@register_op("nnx.graphlib.Static", "flax")
def _map_flax_nnx_graphlib_Static(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_Static operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.Static")


@register_op("nnx.graphlib.GenericPytree", "flax")
def _map_flax_nnx_graphlib_GenericPytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_GenericPytree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.GenericPytree"
    )


@register_op("nnx.graphlib.is_pytree_node", "flax")
def _map_flax_nnx_graphlib_is_pytree_node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_is_pytree_node operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.is_pytree_node"
    )


@register_op("nnx.graphlib.jax_to_nnx_path", "flax")
def _map_flax_nnx_graphlib_jax_to_nnx_path(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_jax_to_nnx_path operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.jax_to_nnx_path"
    )


@register_op("nnx.graphlib.IndexesPytreeDef", "flax")
def _map_flax_nnx_graphlib_IndexesPytreeDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_IndexesPytreeDef operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.IndexesPytreeDef"
    )


@register_op("nnx.graphlib.PYTREE_NODE_IMPL", "flax")
def _map_flax_nnx_graphlib_PYTREE_NODE_IMPL(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graphlib_PYTREE_NODE_IMPL operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graphlib.PYTREE_NODE_IMPL"
    )


@register_op("nnx.spmd.core_spmd", "flax")
def _map_flax_nnx_spmd_core_spmd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_core_spmd operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.core_spmd")


@register_op("nnx.spmd.variablelib", "flax")
def _map_flax_nnx_spmd_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_variablelib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.variablelib")


@register_op("nnx.spmd.graphlib", "flax")
def _map_flax_nnx_spmd_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_graphlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.graphlib")


@register_op("nnx.spmd.eval_shape", "flax")
def _map_flax_nnx_spmd_eval_shape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_eval_shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.eval_shape")


@register_op("nnx.spmd.Sharding", "flax")
def _map_flax_nnx_spmd_Sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_Sharding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.Sharding")


@register_op("nnx.spmd.A", "flax")
def _map_flax_nnx_spmd_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_spmd_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.A")


@register_op("nnx.spmd.F", "flax")
def _map_flax_nnx_spmd_F(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_spmd_F operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.F")


@register_op("nnx.spmd.PARTITION_NAME", "flax")
def _map_flax_nnx_spmd_PARTITION_NAME(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_PARTITION_NAME operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.PARTITION_NAME")


@register_op("nnx.spmd.add_axis", "flax")
def _map_flax_nnx_spmd_add_axis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_add_axis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.add_axis")


@register_op("nnx.spmd.remove_axis", "flax")
def _map_flax_nnx_spmd_remove_axis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_remove_axis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.remove_axis")


@register_op("nnx.spmd.with_partitioning", "flax")
def _map_flax_nnx_spmd_with_partitioning(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_with_partitioning operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.with_partitioning"
    )


@register_op("nnx.spmd.get_var_pspec", "flax")
def _map_flax_nnx_spmd_get_var_pspec(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_get_var_pspec operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.get_var_pspec")


@register_op("nnx.spmd.get_partition_spec", "flax")
def _map_flax_nnx_spmd_get_partition_spec(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_get_partition_spec operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.get_partition_spec"
    )


@register_op("nnx.spmd.get_named_sharding", "flax")
def _map_flax_nnx_spmd_get_named_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_get_named_sharding operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.get_named_sharding"
    )


@register_op("nnx.spmd.get_abstract_model", "flax")
def _map_flax_nnx_spmd_get_abstract_model(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_get_abstract_model operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.get_abstract_model"
    )


@register_op("nnx.spmd.abstract_with_sharding", "flax")
def _map_flax_nnx_spmd_abstract_with_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_spmd_abstract_with_sharding operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.spmd.abstract_with_sharding"
    )


@register_op("nnx.reprlib.flax_config", "flax")
def _map_flax_nnx_reprlib_flax_config(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_flax_config operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.flax_config")


@register_op("nnx.reprlib.A", "flax")
def _map_flax_nnx_reprlib_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_reprlib_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.A")


@register_op("nnx.reprlib.B", "flax")
def _map_flax_nnx_reprlib_B(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_reprlib_B operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.B")


@register_op("nnx.reprlib.supports_color", "flax")
def _map_flax_nnx_reprlib_supports_color(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_supports_color operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.supports_color"
    )


@register_op("nnx.reprlib.Color", "flax")
def _map_flax_nnx_reprlib_Color(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_Color operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.Color")


@register_op("nnx.reprlib.NO_COLOR", "flax")
def _map_flax_nnx_reprlib_NO_COLOR(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_NO_COLOR operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.NO_COLOR")


@register_op("nnx.reprlib.COLOR", "flax")
def _map_flax_nnx_reprlib_COLOR(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_COLOR operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.COLOR")


@register_op("nnx.reprlib.ReprContext", "flax")
def _map_flax_nnx_reprlib_ReprContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_ReprContext operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.ReprContext")


@register_op("nnx.reprlib.REPR_CONTEXT", "flax")
def _map_flax_nnx_reprlib_REPR_CONTEXT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_REPR_CONTEXT operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.REPR_CONTEXT")


@register_op("nnx.reprlib.colorized", "flax")
def _map_flax_nnx_reprlib_colorized(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_colorized operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.colorized")


@register_op("nnx.reprlib.Object", "flax")
def _map_flax_nnx_reprlib_Object(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_Object operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.Object")


@register_op("nnx.reprlib.Attr", "flax")
def _map_flax_nnx_reprlib_Attr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_Attr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.Attr")


@register_op("nnx.reprlib.Representable", "flax")
def _map_flax_nnx_reprlib_Representable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_Representable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.Representable"
    )


@register_op("nnx.reprlib.get_repr", "flax")
def _map_flax_nnx_reprlib_get_repr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_get_repr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.get_repr")


@register_op("nnx.reprlib.MappingReprMixin", "flax")
def _map_flax_nnx_reprlib_MappingReprMixin(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_MappingReprMixin operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.MappingReprMixin"
    )


@register_op("nnx.reprlib.PrettyMapping", "flax")
def _map_flax_nnx_reprlib_PrettyMapping(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_PrettyMapping operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.PrettyMapping"
    )


@register_op("nnx.reprlib.SequenceReprMixin", "flax")
def _map_flax_nnx_reprlib_SequenceReprMixin(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_SequenceReprMixin operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.SequenceReprMixin"
    )


@register_op("nnx.reprlib.PrettySequence", "flax")
def _map_flax_nnx_reprlib_PrettySequence(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_reprlib_PrettySequence operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.reprlib.PrettySequence"
    )


@register_op("nnx.graph.annotations", "flax")
def _map_flax_nnx_graph_annotations(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_annotations operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.annotations")


@register_op("nnx.graph.contextlib", "flax")
def _map_flax_nnx_graph_contextlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_contextlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.contextlib")


@register_op("nnx.graph.dataclasses", "flax")
def _map_flax_nnx_graph_dataclasses(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_dataclasses operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.dataclasses")


@register_op("nnx.graph.functools", "flax")
def _map_flax_nnx_graph_functools(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_functools operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.functools")


@register_op("nnx.graph.threading", "flax")
def _map_flax_nnx_graph_threading(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_threading operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.threading")


@register_op("nnx.graph.tp", "flax")
def _map_flax_nnx_graph_tp(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_tp operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.tp")


@register_op("nnx.graph.builtins", "flax")
def _map_flax_nnx_graph_builtins(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_builtins operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.builtins")


@register_op("nnx.graph.jax", "flax")
def _map_flax_nnx_graph_jax(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_jax operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.jax")


@register_op("nnx.graph.config", "flax")
def _map_flax_nnx_graph_config(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_config operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.config")


@register_op("nnx.graph.filterlib", "flax")
def _map_flax_nnx_graph_filterlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_filterlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.filterlib")


@register_op("nnx.graph.reprlib", "flax")
def _map_flax_nnx_graph_reprlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_reprlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.reprlib")


@register_op("nnx.graph.traversals", "flax")
def _map_flax_nnx_graph_traversals(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_traversals operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.traversals")


@register_op("nnx.graph.variablelib", "flax")
def _map_flax_nnx_graph_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_variablelib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.variablelib")


@register_op("nnx.graph.statelib", "flax")
def _map_flax_nnx_graph_statelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_statelib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.statelib")


@register_op("nnx.graph.ApplyCaller", "flax")
def _map_flax_nnx_graph_ApplyCaller(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_ApplyCaller operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.ApplyCaller")


@register_op("nnx.graph.CallableProxy", "flax")
def _map_flax_nnx_graph_CallableProxy(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_CallableProxy operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.CallableProxy")


@register_op("nnx.graph.DelayedAccessor", "flax")
def _map_flax_nnx_graph_DelayedAccessor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_DelayedAccessor operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.DelayedAccessor"
    )


@register_op("nnx.graph.FlatState", "flax")
def _map_flax_nnx_graph_FlatState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_FlatState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.FlatState")


@register_op("nnx.graph.State", "flax")
def _map_flax_nnx_graph_State(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_State operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.State")


@register_op("nnx.graph.map_state", "flax")
def _map_flax_nnx_graph_map_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_map_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.map_state")


@register_op("nnx.graph.Variable", "flax")
def _map_flax_nnx_graph_Variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_Variable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.Variable")


@register_op("nnx.graph.is_array_ref", "flax")
def _map_flax_nnx_graph_is_array_ref(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_is_array_ref operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.is_array_ref")


@register_op("nnx.graph.V", "flax")
def _map_flax_nnx_graph_V(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_V operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.V")


@register_op("nnx.graph.BaseConfigContext", "flax")
def _map_flax_nnx_graph_BaseConfigContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_BaseConfigContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.BaseConfigContext"
    )


@register_op("nnx.graph.HashableMapping", "flax")
def _map_flax_nnx_graph_HashableMapping(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_HashableMapping operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.HashableMapping"
    )


@register_op("nnx.graph.Key", "flax")
def _map_flax_nnx_graph_Key(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_Key operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.Key")


@register_op("nnx.graph.PathParts", "flax")
def _map_flax_nnx_graph_PathParts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_PathParts operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.PathParts")


@register_op("nnx.graph.is_key_like", "flax")
def _map_flax_nnx_graph_is_key_like(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_is_key_like operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.is_key_like")


@register_op("nnx.graph.np", "flax")
def _map_flax_nnx_graph_np(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_np operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.np")


@register_op("nnx.graph.treescope", "flax")
def _map_flax_nnx_graph_treescope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_treescope operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.treescope")


@register_op("nnx.graph.tpe", "flax")
def _map_flax_nnx_graph_tpe(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_tpe operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.tpe")


@register_op("nnx.graph.A", "flax")
def _map_flax_nnx_graph_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.A")


@register_op("nnx.graph.B", "flax")
def _map_flax_nnx_graph_B(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_B operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.B")


@register_op("nnx.graph.C", "flax")
def _map_flax_nnx_graph_C(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_C operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.C")


@register_op("nnx.graph.F", "flax")
def _map_flax_nnx_graph_F(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_F operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.F")


@register_op("nnx.graph.HA", "flax")
def _map_flax_nnx_graph_HA(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_HA operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.HA")


@register_op("nnx.graph.HB", "flax")
def _map_flax_nnx_graph_HB(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_HB operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.HB")


@register_op("nnx.graph.KeyT", "flax")
def _map_flax_nnx_graph_KeyT(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_KeyT operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.KeyT")


@register_op("nnx.graph.Index", "flax")
def _map_flax_nnx_graph_Index(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_Index operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.Index")


@register_op("nnx.graph.Names", "flax")
def _map_flax_nnx_graph_Names(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_Names operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.Names")


@register_op("nnx.graph.Node", "flax")
def _map_flax_nnx_graph_Node(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_Node operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.Node")


@register_op("nnx.graph.Leaf", "flax")
def _map_flax_nnx_graph_Leaf(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_Leaf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.Leaf")


@register_op("nnx.graph.AuxData", "flax")
def _map_flax_nnx_graph_AuxData(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_AuxData operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.AuxData")


@register_op("nnx.graph.NoUpdate", "flax")
def _map_flax_nnx_graph_NoUpdate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_NoUpdate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.NoUpdate")


@register_op("nnx.graph.NO_UPDATE", "flax")
def _map_flax_nnx_graph_NO_UPDATE(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_NO_UPDATE operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.NO_UPDATE")


@register_op("nnx.graph.Repeated", "flax")
def _map_flax_nnx_graph_Repeated(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_Repeated operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.Repeated")


@register_op("nnx.graph.REPEATED", "flax")
def _map_flax_nnx_graph_REPEATED(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_REPEATED operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.REPEATED")


@register_op("nnx.graph.ArrayRefOutput", "flax")
def _map_flax_nnx_graph_ArrayRefOutput(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_ArrayRefOutput operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.ArrayRefOutput")


@register_op("nnx.graph.LeafType", "flax")
def _map_flax_nnx_graph_LeafType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_LeafType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.LeafType")


@register_op("nnx.graph.GraphState", "flax")
def _map_flax_nnx_graph_GraphState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_GraphState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.GraphState")


@register_op("nnx.graph.GraphFlatState", "flax")
def _map_flax_nnx_graph_GraphFlatState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_GraphFlatState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.GraphFlatState")


@register_op("nnx.graph.is_node_leaf", "flax")
def _map_flax_nnx_graph_is_node_leaf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_is_node_leaf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.is_node_leaf")


@register_op("nnx.graph.IndexMap", "flax")
def _map_flax_nnx_graph_IndexMap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_IndexMap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.IndexMap")


@register_op("nnx.graph.flaxlib", "flax")
def _map_flax_nnx_graph_flaxlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_flaxlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.flaxlib")


@register_op("nnx.graph.RefMap", "flax")
def _map_flax_nnx_graph_RefMap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_RefMap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.RefMap")


@register_op("nnx.graph.PythonRefMap", "flax")
def _map_flax_nnx_graph_PythonRefMap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_PythonRefMap operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.PythonRefMap")


@register_op("nnx.graph.NodeImplBase", "flax")
def _map_flax_nnx_graph_NodeImplBase(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_NodeImplBase operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.NodeImplBase")


@register_op("nnx.graph.GraphNodeImpl", "flax")
def _map_flax_nnx_graph_GraphNodeImpl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_GraphNodeImpl operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.GraphNodeImpl")


@register_op("nnx.graph.PytreeNodeImpl", "flax")
def _map_flax_nnx_graph_PytreeNodeImpl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_PytreeNodeImpl operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.PytreeNodeImpl")


@register_op("nnx.graph.NodeImpl", "flax")
def _map_flax_nnx_graph_NodeImpl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_NodeImpl operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.NodeImpl")


@register_op("nnx.graph.GRAPH_REGISTRY", "flax")
def _map_flax_nnx_graph_GRAPH_REGISTRY(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_GRAPH_REGISTRY operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.GRAPH_REGISTRY")


@register_op("nnx.graph.PYTREE_REGISTRY", "flax")
def _map_flax_nnx_graph_PYTREE_REGISTRY(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_PYTREE_REGISTRY operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.PYTREE_REGISTRY"
    )


@register_op("nnx.graph.register_graph_node_type", "flax")
def _map_flax_nnx_graph_register_graph_node_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_register_graph_node_type operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.graph.register_graph_node_type",
    )


@register_op("nnx.graph.register_pytree_node_type", "flax")
def _map_flax_nnx_graph_register_pytree_node_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_register_pytree_node_type operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.graph.register_pytree_node_type",
    )


@register_op("nnx.graph.is_node", "flax")
def _map_flax_nnx_graph_is_node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_is_node operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.is_node")


@register_op("nnx.graph.is_graph_node", "flax")
def _map_flax_nnx_graph_is_graph_node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_is_graph_node operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.is_graph_node")


@register_op("nnx.graph.is_node_type", "flax")
def _map_flax_nnx_graph_is_node_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_is_node_type operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.is_node_type")


@register_op("nnx.graph.get_node_impl", "flax")
def _map_flax_nnx_graph_get_node_impl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_get_node_impl operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.get_node_impl")


@register_op("nnx.graph.get_node_impl_for_type", "flax")
def _map_flax_nnx_graph_get_node_impl_for_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_get_node_impl_for_type operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.get_node_impl_for_type"
    )


@register_op("nnx.graph.NodeRef", "flax")
def _map_flax_nnx_graph_NodeRef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_NodeRef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.NodeRef")


@register_op("nnx.graph.VariableDef", "flax")
def _map_flax_nnx_graph_VariableDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_VariableDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.VariableDef")


@register_op("nnx.graph.ArrayRefDef", "flax")
def _map_flax_nnx_graph_ArrayRefDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_ArrayRefDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.ArrayRefDef")


@register_op("nnx.graph.NodeDef", "flax")
def _map_flax_nnx_graph_NodeDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_NodeDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.NodeDef")


@register_op("nnx.graph.TreeNodeDef", "flax")
def _map_flax_nnx_graph_TreeNodeDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_TreeNodeDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.TreeNodeDef")


@register_op("nnx.graph.NodeDefType", "flax")
def _map_flax_nnx_graph_NodeDefType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_NodeDefType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.NodeDefType")


@register_op("nnx.graph.NodeAttr", "flax")
def _map_flax_nnx_graph_NodeAttr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_NodeAttr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.NodeAttr")


@register_op("nnx.graph.NODE_ATTR", "flax")
def _map_flax_nnx_graph_NODE_ATTR(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_NODE_ATTR operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.NODE_ATTR")


@register_op("nnx.graph.LeafAttr", "flax")
def _map_flax_nnx_graph_LeafAttr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_LeafAttr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.LeafAttr")


@register_op("nnx.graph.LEAF_ATTR", "flax")
def _map_flax_nnx_graph_LEAF_ATTR(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_LEAF_ATTR operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.LEAF_ATTR")


@register_op("nnx.graph.AttrType", "flax")
def _map_flax_nnx_graph_AttrType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_AttrType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.AttrType")


@register_op("nnx.graph.GraphDef", "flax")
def _map_flax_nnx_graph_GraphDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_GraphDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.GraphDef")


@register_op("nnx.graph.PureState", "flax")
def _map_flax_nnx_graph_PureState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_PureState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.PureState")


@register_op("nnx.graph.flatten", "flax")
def _map_flax_nnx_graph_flatten(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_flatten operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.flatten")


@register_op("nnx.graph.DataElem", "flax")
def _map_flax_nnx_graph_DataElem(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_DataElem operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.DataElem")


@register_op("nnx.graph.StaticElem", "flax")
def _map_flax_nnx_graph_StaticElem(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_StaticElem operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.StaticElem")


@register_op("nnx.graph.unflatten", "flax")
def _map_flax_nnx_graph_unflatten(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_unflatten operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.unflatten")


@register_op("nnx.graph.graph_pop", "flax")
def _map_flax_nnx_graph_graph_pop(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_graph_pop operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.graph_pop")


@register_op("nnx.graph.StaticCache", "flax")
def _map_flax_nnx_graph_StaticCache(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_StaticCache operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.StaticCache")


@register_op("nnx.graph.GraphContext", "flax")
def _map_flax_nnx_graph_GraphContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_GraphContext operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.GraphContext")


@register_op("nnx.graph.GRAPH_CONTEXT", "flax")
def _map_flax_nnx_graph_GRAPH_CONTEXT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_GRAPH_CONTEXT operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.GRAPH_CONTEXT")


@register_op("nnx.graph.set_graph_mode", "flax")
def _map_flax_nnx_graph_set_graph_mode(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_set_graph_mode operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.set_graph_mode")


@register_op("nnx.graph.set_graph_updates", "flax")
def _map_flax_nnx_graph_set_graph_updates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_set_graph_updates operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.set_graph_updates"
    )


@register_op("nnx.graph.static_cache", "flax")
def _map_flax_nnx_graph_static_cache(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_static_cache operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.static_cache")


@register_op("nnx.graph.cached_partial", "flax")
def _map_flax_nnx_graph_cached_partial(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_cached_partial operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.cached_partial")


@register_op("nnx.graph.SplitContext", "flax")
def _map_flax_nnx_graph_SplitContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_SplitContext operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.SplitContext")


@register_op("nnx.graph.split_context", "flax")
def _map_flax_nnx_graph_split_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_split_context operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.split_context")


@register_op("nnx.graph.MergeContext", "flax")
def _map_flax_nnx_graph_MergeContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_MergeContext operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.MergeContext")


@register_op("nnx.graph.merge_context", "flax")
def _map_flax_nnx_graph_merge_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_merge_context operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.merge_context")


@register_op("nnx.graph.UpdateContext", "flax")
def _map_flax_nnx_graph_UpdateContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_UpdateContext operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.UpdateContext")


@register_op("nnx.graph.UpdateContextManager", "flax")
def _map_flax_nnx_graph_UpdateContextManager(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_UpdateContextManager operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.UpdateContextManager"
    )


@register_op("nnx.graph.update_context", "flax")
def _map_flax_nnx_graph_update_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_update_context operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.update_context")


@register_op("nnx.graph.current_update_context", "flax")
def _map_flax_nnx_graph_current_update_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_current_update_context operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.current_update_context"
    )


@register_op("nnx.graph.split", "flax")
def _map_flax_nnx_graph_split(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_split operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.split")


@register_op("nnx.graph.merge", "flax")
def _map_flax_nnx_graph_merge(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_merge operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.merge")


@register_op("nnx.graph.update", "flax")
def _map_flax_nnx_graph_update(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_update operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.update")


@register_op("nnx.graph.state", "flax")
def _map_flax_nnx_graph_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.state")


@register_op("nnx.graph.variables", "flax")
def _map_flax_nnx_graph_variables(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_variables operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.variables")


@register_op("nnx.graph.map", "flax")
def _map_flax_nnx_graph_map(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_map operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.map")


@register_op("nnx.graph.graphdef", "flax")
def _map_flax_nnx_graph_graphdef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_graphdef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.graphdef")


@register_op("nnx.graph.pop", "flax")
def _map_flax_nnx_graph_pop(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_pop operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.pop")


@register_op("nnx.graph.clone", "flax")
def _map_flax_nnx_graph_clone(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_clone operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.clone")


@register_op("nnx.graph.vars_as", "flax")
def _map_flax_nnx_graph_vars_as(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_vars_as operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.vars_as")


@register_op("nnx.graph.pure", "flax")
def _map_flax_nnx_graph_pure(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_pure operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.pure")


@register_op("nnx.graph.call", "flax")
def _map_flax_nnx_graph_call(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_graph_call operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.call")


@register_op("nnx.graph.set_metadata", "flax")
def _map_flax_nnx_graph_set_metadata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_set_metadata operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.set_metadata")


@register_op("nnx.graph.iter_graph", "flax")
def _map_flax_nnx_graph_iter_graph(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_iter_graph operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.iter_graph")


@register_op("nnx.graph.iter_children", "flax")
def _map_flax_nnx_graph_iter_children(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_iter_children operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.iter_children")


@register_op("nnx.graph.recursive_map", "flax")
def _map_flax_nnx_graph_recursive_map(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_recursive_map operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.recursive_map")


@register_op("nnx.graph.find_duplicates", "flax")
def _map_flax_nnx_graph_find_duplicates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_find_duplicates operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.find_duplicates"
    )


@register_op("nnx.graph.Static", "flax")
def _map_flax_nnx_graph_Static(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_Static operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.Static")


@register_op("nnx.graph.GenericPytree", "flax")
def _map_flax_nnx_graph_GenericPytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_GenericPytree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.GenericPytree")


@register_op("nnx.graph.JAX_PYTREE_REGISTRY", "flax")
def _map_flax_nnx_graph_JAX_PYTREE_REGISTRY(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_JAX_PYTREE_REGISTRY operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.JAX_PYTREE_REGISTRY"
    )


@register_op("nnx.graph.is_pytree_node", "flax")
def _map_flax_nnx_graph_is_pytree_node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_is_pytree_node operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.is_pytree_node")


@register_op("nnx.graph.jax_to_nnx_path", "flax")
def _map_flax_nnx_graph_jax_to_nnx_path(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_jax_to_nnx_path operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.jax_to_nnx_path"
    )


@register_op("nnx.graph.IndexesPytreeDef", "flax")
def _map_flax_nnx_graph_IndexesPytreeDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_IndexesPytreeDef operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.IndexesPytreeDef"
    )


@register_op("nnx.graph.PYTREE_NODE_IMPL", "flax")
def _map_flax_nnx_graph_PYTREE_NODE_IMPL(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_graph_PYTREE_NODE_IMPL operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.graph.PYTREE_NODE_IMPL"
    )


@register_op("nnx.visualization.in_ipython", "flax")
def _map_flax_nnx_visualization_in_ipython(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_visualization_in_ipython operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.visualization.in_ipython"
    )


@register_op("nnx.visualization.display", "flax")
def _map_flax_nnx_visualization_display(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_visualization_display operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.visualization.display"
    )


@register_op("nnx.visualization.render_object_constructor", "flax")
def _map_flax_nnx_visualization_render_object_constructor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_visualization_render_object_constructor operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.visualization.render_object_constructor",
    )


@register_op("nnx.variablelib.config", "flax")
def _map_flax_nnx_variablelib_config(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_config operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.config")


@register_op("nnx.variablelib.errors", "flax")
def _map_flax_nnx_variablelib_errors(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_errors operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.errors")


@register_op("nnx.variablelib.core_spmd", "flax")
def _map_flax_nnx_variablelib_core_spmd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_core_spmd operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.core_spmd"
    )


@register_op("nnx.variablelib.reprlib", "flax")
def _map_flax_nnx_variablelib_reprlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_reprlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.reprlib")


@register_op("nnx.variablelib.tracers", "flax")
def _map_flax_nnx_variablelib_tracers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_tracers operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.tracers")


@register_op("nnx.variablelib.visualization", "flax")
def _map_flax_nnx_variablelib_visualization(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_visualization operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.visualization"
    )


@register_op("nnx.variablelib.BaseConfigContext", "flax")
def _map_flax_nnx_variablelib_BaseConfigContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_BaseConfigContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.BaseConfigContext"
    )


@register_op("nnx.variablelib.MISSING", "flax")
def _map_flax_nnx_variablelib_MISSING(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_MISSING operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.MISSING")


@register_op("nnx.variablelib.Missing", "flax")
def _map_flax_nnx_variablelib_Missing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_Missing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.Missing")


@register_op("nnx.variablelib.SizeBytes", "flax")
def _map_flax_nnx_variablelib_SizeBytes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_SizeBytes operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.SizeBytes"
    )


@register_op("nnx.variablelib.A", "flax")
def _map_flax_nnx_variablelib_A(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.A")


@register_op("nnx.variablelib.B", "flax")
def _map_flax_nnx_variablelib_B(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_B operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.B")


@register_op("nnx.variablelib.C", "flax")
def _map_flax_nnx_variablelib_C(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_C operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.C")


@register_op("nnx.variablelib.F", "flax")
def _map_flax_nnx_variablelib_F(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_F operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.F")


@register_op("nnx.variablelib.P", "flax")
def _map_flax_nnx_variablelib_P(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_P operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.P")


@register_op("nnx.variablelib.V", "flax")
def _map_flax_nnx_variablelib_V(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_V operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.V")


@register_op("nnx.variablelib.GetValueHook", "flax")
def _map_flax_nnx_variablelib_GetValueHook(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_GetValueHook operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.GetValueHook"
    )


@register_op("nnx.variablelib.SetValueHook", "flax")
def _map_flax_nnx_variablelib_SetValueHook(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_SetValueHook operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.SetValueHook"
    )


@register_op("nnx.variablelib.CreateValueHook", "flax")
def _map_flax_nnx_variablelib_CreateValueHook(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_CreateValueHook operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.CreateValueHook"
    )


@register_op("nnx.variablelib.AxisName", "flax")
def _map_flax_nnx_variablelib_AxisName(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_AxisName operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.AxisName")


@register_op("nnx.variablelib.AxisIndex", "flax")
def _map_flax_nnx_variablelib_AxisIndex(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_AxisIndex operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.AxisIndex"
    )


@register_op("nnx.variablelib.AddAxisHook", "flax")
def _map_flax_nnx_variablelib_AddAxisHook(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_AddAxisHook operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.AddAxisHook"
    )


@register_op("nnx.variablelib.RemoveAxisHook", "flax")
def _map_flax_nnx_variablelib_RemoveAxisHook(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_RemoveAxisHook operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.RemoveAxisHook"
    )


@register_op("nnx.variablelib.VariableContext", "flax")
def _map_flax_nnx_variablelib_VariableContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_VariableContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.VariableContext"
    )


@register_op("nnx.variablelib.VARIABLE_CONTEXT", "flax")
def _map_flax_nnx_variablelib_VARIABLE_CONTEXT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_VARIABLE_CONTEXT operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.VARIABLE_CONTEXT"
    )


@register_op("nnx.variablelib.use_eager_sharding", "flax")
def _map_flax_nnx_variablelib_use_eager_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_use_eager_sharding operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.variablelib.use_eager_sharding",
    )


@register_op("nnx.variablelib.using_eager_sharding", "flax")
def _map_flax_nnx_variablelib_using_eager_sharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_using_eager_sharding operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.variablelib.using_eager_sharding",
    )


@register_op("nnx.variablelib.VarDefaults", "flax")
def _map_flax_nnx_variablelib_VarDefaults(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_VarDefaults operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.VarDefaults"
    )


@register_op("nnx.variablelib.var_defaults", "flax")
def _map_flax_nnx_variablelib_var_defaults(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_var_defaults operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.var_defaults"
    )


@register_op("nnx.variablelib.VarDefaultsContext", "flax")
def _map_flax_nnx_variablelib_VarDefaultsContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_VarDefaultsContext operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.variablelib.VarDefaultsContext",
    )


@register_op("nnx.variablelib.is_array_ref", "flax")
def _map_flax_nnx_variablelib_is_array_ref(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_is_array_ref operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.is_array_ref"
    )


@register_op("nnx.variablelib.VariableMetadata", "flax")
def _map_flax_nnx_variablelib_VariableMetadata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_VariableMetadata operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.VariableMetadata"
    )


@register_op("nnx.variablelib.PyTreeDef", "flax")
def _map_flax_nnx_variablelib_PyTreeDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_PyTreeDef operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.PyTreeDef"
    )


@register_op("nnx.variablelib.Leaf", "flax")
def _map_flax_nnx_variablelib_Leaf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_Leaf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.Leaf")


@register_op("nnx.variablelib.VariableQDD", "flax")
def _map_flax_nnx_variablelib_VariableQDD(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_VariableQDD operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.VariableQDD"
    )


@register_op("nnx.variablelib.VariableEffect", "flax")
def _map_flax_nnx_variablelib_VariableEffect(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_VariableEffect operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.VariableEffect"
    )


@register_op("nnx.variablelib.variable_effect", "flax")
def _map_flax_nnx_variablelib_variable_effect(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_variable_effect operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.variable_effect"
    )


@register_op("nnx.variablelib.NewVariable", "flax")
def _map_flax_nnx_variablelib_NewVariable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_NewVariable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.NewVariable"
    )


@register_op("nnx.variablelib.new_variable_p", "flax")
def _map_flax_nnx_variablelib_new_variable_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_new_variable_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.new_variable_p"
    )


@register_op("nnx.variablelib.SetVariable", "flax")
def _map_flax_nnx_variablelib_SetVariable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_SetVariable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.SetVariable"
    )


@register_op("nnx.variablelib.set_variable_p", "flax")
def _map_flax_nnx_variablelib_set_variable_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_set_variable_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.set_variable_p"
    )


@register_op("nnx.variablelib.GetVariable", "flax")
def _map_flax_nnx_variablelib_GetVariable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_GetVariable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.GetVariable"
    )


@register_op("nnx.variablelib.get_variable_p", "flax")
def _map_flax_nnx_variablelib_get_variable_p(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_get_variable_p operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.get_variable_p"
    )


@register_op("nnx.variablelib.HijaxVariableMeta", "flax")
def _map_flax_nnx_variablelib_HijaxVariableMeta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_HijaxVariableMeta operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.HijaxVariableMeta"
    )


@register_op("nnx.variablelib.HijaxVariable", "flax")
def _map_flax_nnx_variablelib_HijaxVariable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_HijaxVariable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.HijaxVariable"
    )


@register_op("nnx.variablelib.AbstractVariable", "flax")
def _map_flax_nnx_variablelib_AbstractVariable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_AbstractVariable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.AbstractVariable"
    )


@register_op("nnx.variablelib.VariableMeta", "flax")
def _map_flax_nnx_variablelib_VariableMeta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_VariableMeta operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.VariableMeta"
    )


@register_op("nnx.variablelib.Variable", "flax")
def _map_flax_nnx_variablelib_Variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_Variable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.Variable")


@register_op("nnx.variablelib.VariableState", "flax")
def _map_flax_nnx_variablelib_VariableState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_VariableState operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.VariableState"
    )


@register_op("nnx.variablelib.Param", "flax")
def _map_flax_nnx_variablelib_Param(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_Param operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.Param")


@register_op("nnx.variablelib.BatchStat", "flax")
def _map_flax_nnx_variablelib_BatchStat(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_BatchStat operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.BatchStat"
    )


@register_op("nnx.variablelib.Cache", "flax")
def _map_flax_nnx_variablelib_Cache(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_Cache operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.Cache")


@register_op("nnx.variablelib.Intermediate", "flax")
def _map_flax_nnx_variablelib_Intermediate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_Intermediate operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.Intermediate"
    )


@register_op("nnx.variablelib.Perturbation", "flax")
def _map_flax_nnx_variablelib_Perturbation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_Perturbation operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.Perturbation"
    )


@register_op("nnx.variablelib.with_metadata", "flax")
def _map_flax_nnx_variablelib_with_metadata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_with_metadata operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.with_metadata"
    )


@register_op("nnx.variablelib.VariableTypeCache", "flax")
def _map_flax_nnx_variablelib_VariableTypeCache(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_VariableTypeCache operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.variablelib.VariableTypeCache"
    )


@register_op("nnx.variablelib.variable_type_from_name", "flax")
def _map_flax_nnx_variablelib_variable_type_from_name(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_variable_type_from_name operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.variablelib.variable_type_from_name",
    )


@register_op("nnx.variablelib.variable_name_from_type", "flax")
def _map_flax_nnx_variablelib_variable_name_from_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_variable_name_from_type operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.variablelib.variable_name_from_type",
    )


@register_op("nnx.variablelib.register_variable_name", "flax")
def _map_flax_nnx_variablelib_register_variable_name(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_variablelib_register_variable_name operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.variablelib.register_variable_name",
    )


@register_op("nnx.pytreelib.variablelib", "flax")
def _map_flax_nnx_pytreelib_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_variablelib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.variablelib"
    )


@register_op("nnx.pytreelib.errors", "flax")
def _map_flax_nnx_pytreelib_errors(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_errors operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.errors")


@register_op("nnx.pytreelib.nnx", "flax")
def _map_flax_nnx_pytreelib_nnx(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_nnx operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.nnx")


@register_op("nnx.pytreelib.graphlib", "flax")
def _map_flax_nnx_pytreelib_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_graphlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.graphlib")


@register_op("nnx.pytreelib.reprlib", "flax")
def _map_flax_nnx_pytreelib_reprlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_reprlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.reprlib")


@register_op("nnx.pytreelib.tracers", "flax")
def _map_flax_nnx_pytreelib_tracers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_tracers operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.tracers")


@register_op("nnx.pytreelib.visualization", "flax")
def _map_flax_nnx_pytreelib_visualization(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_visualization operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.visualization"
    )


@register_op("nnx.pytreelib.config", "flax")
def _map_flax_nnx_pytreelib_config(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_config operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.config")


@register_op("nnx.pytreelib.Variable", "flax")
def _map_flax_nnx_pytreelib_Variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_Variable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.Variable")


@register_op("nnx.pytreelib.MISSING", "flax")
def _map_flax_nnx_pytreelib_MISSING(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_MISSING operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.MISSING")


@register_op("nnx.pytreelib.Missing", "flax")
def _map_flax_nnx_pytreelib_Missing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_Missing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.Missing")


@register_op("nnx.pytreelib.SizeBytes", "flax")
def _map_flax_nnx_pytreelib_SizeBytes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_SizeBytes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.SizeBytes")


@register_op("nnx.pytreelib.BUILDING_DOCS", "flax")
def _map_flax_nnx_pytreelib_BUILDING_DOCS(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_BUILDING_DOCS operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.BUILDING_DOCS"
    )


@register_op("nnx.pytreelib.A", "flax")
def _map_flax_nnx_pytreelib_A(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.A")


@register_op("nnx.pytreelib.P", "flax")
def _map_flax_nnx_pytreelib_P(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_P operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.P")


@register_op("nnx.pytreelib.T", "flax")
def _map_flax_nnx_pytreelib_T(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_T operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.T")


@register_op("nnx.pytreelib.DataAnnotation", "flax")
def _map_flax_nnx_pytreelib_DataAnnotation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_DataAnnotation operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.DataAnnotation"
    )


@register_op("nnx.pytreelib.Data", "flax")
def _map_flax_nnx_pytreelib_Data(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_Data operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.Data")


@register_op("nnx.pytreelib.DATA_REGISTRY", "flax")
def _map_flax_nnx_pytreelib_DATA_REGISTRY(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_DATA_REGISTRY operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.DATA_REGISTRY"
    )


@register_op("nnx.pytreelib.data", "flax")
def _map_flax_nnx_pytreelib_data(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_data operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.data")


@register_op("nnx.pytreelib.register_data_type", "flax")
def _map_flax_nnx_pytreelib_register_data_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_register_data_type operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.register_data_type"
    )


@register_op("nnx.pytreelib.is_data", "flax")
def _map_flax_nnx_pytreelib_is_data(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_is_data operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.is_data")


@register_op("nnx.pytreelib.has_data", "flax")
def _map_flax_nnx_pytreelib_has_data(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_has_data operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.has_data")


@register_op("nnx.pytreelib.StaticAnnotation", "flax")
def _map_flax_nnx_pytreelib_StaticAnnotation(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_StaticAnnotation operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.StaticAnnotation"
    )


@register_op("nnx.pytreelib.Static", "flax")
def _map_flax_nnx_pytreelib_Static(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_Static operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.Static")


@register_op("nnx.pytreelib.static", "flax")
def _map_flax_nnx_pytreelib_static(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_static operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.static")


@register_op("nnx.pytreelib.dataclass", "flax")
def _map_flax_nnx_pytreelib_dataclass(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_dataclass operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.dataclass")


@register_op("nnx.pytreelib.ObjectContext", "flax")
def _map_flax_nnx_pytreelib_ObjectContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_ObjectContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.ObjectContext"
    )


@register_op("nnx.pytreelib.OBJECT_CONTEXT", "flax")
def _map_flax_nnx_pytreelib_OBJECT_CONTEXT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_OBJECT_CONTEXT operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.OBJECT_CONTEXT"
    )


@register_op("nnx.pytreelib.PytreeState", "flax")
def _map_flax_nnx_pytreelib_PytreeState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_PytreeState operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.PytreeState"
    )


@register_op("nnx.pytreelib.check_pytree", "flax")
def _map_flax_nnx_pytreelib_check_pytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_check_pytree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.check_pytree"
    )


@register_op("nnx.pytreelib.PytreeMeta", "flax")
def _map_flax_nnx_pytreelib_PytreeMeta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_PytreeMeta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.PytreeMeta")


@register_op("nnx.pytreelib.ObjectMeta", "flax")
def _map_flax_nnx_pytreelib_ObjectMeta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_ObjectMeta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.ObjectMeta")


@register_op("nnx.pytreelib.ArrayRepr", "flax")
def _map_flax_nnx_pytreelib_ArrayRepr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_ArrayRepr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.ArrayRepr")


@register_op("nnx.pytreelib.VariableRepr", "flax")
def _map_flax_nnx_pytreelib_VariableRepr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_VariableRepr operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.VariableRepr"
    )


@register_op("nnx.pytreelib.MutableArrayRepr", "flax")
def _map_flax_nnx_pytreelib_MutableArrayRepr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_MutableArrayRepr operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.MutableArrayRepr"
    )


@register_op("nnx.pytreelib.AttributeStatus", "flax")
def _map_flax_nnx_pytreelib_AttributeStatus(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_AttributeStatus operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.AttributeStatus"
    )


@register_op("nnx.pytreelib.Pytree", "flax")
def _map_flax_nnx_pytreelib_Pytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_Pytree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.Pytree")


@register_op("nnx.pytreelib.Object", "flax")
def _map_flax_nnx_pytreelib_Object(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_pytreelib_Object operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.pytreelib.Object")


@register_op("nnx.helpers.graphlib", "flax")
def _map_flax_nnx_helpers_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_graphlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.graphlib")


@register_op("nnx.helpers.reprlib", "flax")
def _map_flax_nnx_helpers_reprlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_reprlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.reprlib")


@register_op("nnx.helpers.GraphDef", "flax")
def _map_flax_nnx_helpers_GraphDef(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_GraphDef operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.GraphDef")


@register_op("nnx.helpers.Module", "flax")
def _map_flax_nnx_helpers_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.Module")


@register_op("nnx.helpers.ApplyCaller", "flax")
def _map_flax_nnx_helpers_ApplyCaller(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_ApplyCaller operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.ApplyCaller")


@register_op("nnx.helpers.Rngs", "flax")
def _map_flax_nnx_helpers_Rngs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_Rngs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.Rngs")


@register_op("nnx.helpers.State", "flax")
def _map_flax_nnx_helpers_State(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_State operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.State")


@register_op("nnx.helpers.struct", "flax")
def _map_flax_nnx_helpers_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_struct operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.struct")


@register_op("nnx.helpers.Variable", "flax")
def _map_flax_nnx_helpers_Variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_Variable operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.Variable")


@register_op("nnx.helpers.A", "flax")
def _map_flax_nnx_helpers_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_helpers_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.A")


@register_op("nnx.helpers.M", "flax")
def _map_flax_nnx_helpers_M(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_helpers_M operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.M")


@register_op("nnx.helpers.TS", "flax")
def _map_flax_nnx_helpers_TS(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_helpers_TS operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.TS")


@register_op("nnx.helpers.Dict", "flax")
def _map_flax_nnx_helpers_Dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_Dict operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.Dict")


@register_op("nnx.helpers.List", "flax")
def _map_flax_nnx_helpers_List(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_List operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.List")


@register_op("nnx.helpers.Sequential", "flax")
def _map_flax_nnx_helpers_Sequential(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_Sequential operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.Sequential")


@register_op("nnx.helpers.ModuleDefApply", "flax")
def _map_flax_nnx_helpers_ModuleDefApply(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_ModuleDefApply operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.ModuleDefApply"
    )


@register_op("nnx.helpers.TrainState", "flax")
def _map_flax_nnx_helpers_TrainState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_TrainState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.TrainState")


@register_op("nnx.helpers.has_keyword_arg", "flax")
def _map_flax_nnx_helpers_has_keyword_arg(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_helpers_has_keyword_arg operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.helpers.has_keyword_arg"
    )


@register_op("nnx.extract.struct", "flax")
def _map_flax_nnx_extract_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_struct operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.struct")


@register_op("nnx.extract.typing", "flax")
def _map_flax_nnx_extract_typing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_typing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.typing")


@register_op("nnx.extract.Pytree", "flax")
def _map_flax_nnx_extract_Pytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_Pytree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.Pytree")


@register_op("nnx.extract.Missing", "flax")
def _map_flax_nnx_extract_Missing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_Missing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.Missing")


@register_op("nnx.extract.PathParts", "flax")
def _map_flax_nnx_extract_PathParts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_PathParts operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.PathParts")


@register_op("nnx.extract.graphlib", "flax")
def _map_flax_nnx_extract_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_graphlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.graphlib")


@register_op("nnx.extract.variablelib", "flax")
def _map_flax_nnx_extract_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_variablelib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.variablelib")


@register_op("nnx.extract.A", "flax")
def _map_flax_nnx_extract_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_extract_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.A")


@register_op("nnx.extract.Index", "flax")
def _map_flax_nnx_extract_Index(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_Index operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.Index")


@register_op("nnx.extract.KeyPath", "flax")
def _map_flax_nnx_extract_KeyPath(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_KeyPath operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.KeyPath")


@register_op("nnx.extract.Prefix", "flax")
def _map_flax_nnx_extract_Prefix(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_Prefix operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.Prefix")


@register_op("nnx.extract.Leaf", "flax")
def _map_flax_nnx_extract_Leaf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_Leaf operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.Leaf")


@register_op("nnx.extract.PrefixMapping", "flax")
def _map_flax_nnx_extract_PrefixMapping(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_PrefixMapping operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.PrefixMapping"
    )


@register_op("nnx.extract.check_consistent_aliasing", "flax")
def _map_flax_nnx_extract_check_consistent_aliasing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_check_consistent_aliasing operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.extract.check_consistent_aliasing",
    )


@register_op("nnx.extract.check_consistent_aliasing2", "flax")
def _map_flax_nnx_extract_check_consistent_aliasing2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_check_consistent_aliasing2 operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.extract.check_consistent_aliasing2",
    )


@register_op("nnx.extract.broadcast_prefix", "flax")
def _map_flax_nnx_extract_broadcast_prefix(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_broadcast_prefix operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.broadcast_prefix"
    )


@register_op("nnx.extract.broadcast_prefix2", "flax")
def _map_flax_nnx_extract_broadcast_prefix2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_broadcast_prefix2 operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.broadcast_prefix2"
    )


@register_op("nnx.extract.broadcast_prefix_map", "flax")
def _map_flax_nnx_extract_broadcast_prefix_map(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_broadcast_prefix_map operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.broadcast_prefix_map"
    )


@register_op("nnx.extract.GraphDefState", "flax")
def _map_flax_nnx_extract_GraphDefState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_GraphDefState operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.GraphDefState"
    )


@register_op("nnx.extract.S", "flax")
def _map_flax_nnx_extract_S(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_extract_S operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.S")


@register_op("nnx.extract.NodeStates", "flax")
def _map_flax_nnx_extract_NodeStates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_NodeStates operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.NodeStates")


@register_op("nnx.extract.default_split_fn", "flax")
def _map_flax_nnx_extract_default_split_fn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_default_split_fn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.default_split_fn"
    )


@register_op("nnx.extract.to_tree", "flax")
def _map_flax_nnx_extract_to_tree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_to_tree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.to_tree")


@register_op("nnx.extract.to_tree2", "flax")
def _map_flax_nnx_extract_to_tree2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_to_tree2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.to_tree2")


@register_op("nnx.extract.from_tree2", "flax")
def _map_flax_nnx_extract_from_tree2(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_from_tree2 operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.from_tree2")


@register_op("nnx.extract.merge_tree_node", "flax")
def _map_flax_nnx_extract_merge_tree_node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_merge_tree_node operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.merge_tree_node"
    )


@register_op("nnx.extract.is_tree_node", "flax")
def _map_flax_nnx_extract_is_tree_node(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_is_tree_node operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.is_tree_node")


@register_op("nnx.extract.from_tree", "flax")
def _map_flax_nnx_extract_from_tree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_from_tree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.from_tree")


@register_op("nnx.extract.clear_non_graph_nodes", "flax")
def _map_flax_nnx_extract_clear_non_graph_nodes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_clear_non_graph_nodes operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.clear_non_graph_nodes"
    )


@register_op("nnx.extract.Mask", "flax")
def _map_flax_nnx_extract_Mask(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_Mask operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.Mask")


@register_op("nnx.extract.mask_at", "flax")
def _map_flax_nnx_extract_mask_at(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_mask_at operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.mask_at")


@register_op("nnx.extract.replace_at", "flax")
def _map_flax_nnx_extract_replace_at(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_replace_at operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.replace_at")


@register_op("nnx.extract.updates_and_snapshot", "flax")
def _map_flax_nnx_extract_updates_and_snapshot(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_updates_and_snapshot operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.updates_and_snapshot"
    )


@register_op("nnx.extract.check_no_aliases", "flax")
def _map_flax_nnx_extract_check_no_aliases(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_check_no_aliases operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.check_no_aliases"
    )


@register_op("nnx.extract.check_prefix", "flax")
def _map_flax_nnx_extract_check_prefix(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_check_prefix operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.check_prefix")


@register_op("nnx.extract.variable_changed", "flax")
def _map_flax_nnx_extract_variable_changed(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_variable_changed operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.variable_changed"
    )


@register_op("nnx.extract.KeepFn", "flax")
def _map_flax_nnx_extract_KeepFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_KeepFn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.KeepFn")


@register_op("nnx.extract.mask_variable_updates", "flax")
def _map_flax_nnx_extract_mask_variable_updates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_mask_variable_updates operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.mask_variable_updates"
    )


@register_op("nnx.extract.apply_variable_updates", "flax")
def _map_flax_nnx_extract_apply_variable_updates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_apply_variable_updates operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.extract.apply_variable_updates",
    )


@register_op("nnx.extract.treemap_copy_args", "flax")
def _map_flax_nnx_extract_treemap_copy_args(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_treemap_copy_args operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.treemap_copy_args"
    )


@register_op("nnx.extract.check_same_variables", "flax")
def _map_flax_nnx_extract_check_same_variables(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_check_same_variables operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.extract.check_same_variables"
    )


@register_op("nnx.extract.update_carry_variables", "flax")
def _map_flax_nnx_extract_update_carry_variables(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_extract_update_carry_variables operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.extract.update_carry_variables",
    )


@register_op("nnx.filterlib.Key", "flax")
def _map_flax_nnx_filterlib_Key(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_Key operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.Key")


@register_op("nnx.filterlib.PathParts", "flax")
def _map_flax_nnx_filterlib_PathParts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_PathParts operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.PathParts")


@register_op("nnx.filterlib.ellipsis", "flax")
def _map_flax_nnx_filterlib_ellipsis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_ellipsis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.ellipsis")


@register_op("nnx.filterlib.Predicate", "flax")
def _map_flax_nnx_filterlib_Predicate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_Predicate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.Predicate")


@register_op("nnx.filterlib.FilterLiteral", "flax")
def _map_flax_nnx_filterlib_FilterLiteral(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_FilterLiteral operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.FilterLiteral"
    )


@register_op("nnx.filterlib.Filter", "flax")
def _map_flax_nnx_filterlib_Filter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_Filter operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.Filter")


@register_op("nnx.filterlib.to_predicate", "flax")
def _map_flax_nnx_filterlib_to_predicate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_to_predicate operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.to_predicate"
    )


@register_op("nnx.filterlib.filters_to_predicates", "flax")
def _map_flax_nnx_filterlib_filters_to_predicates(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_filters_to_predicates operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.filterlib.filters_to_predicates",
    )


@register_op("nnx.filterlib.HasTag", "flax")
def _map_flax_nnx_filterlib_HasTag(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_HasTag operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.HasTag")


@register_op("nnx.filterlib.WithTag", "flax")
def _map_flax_nnx_filterlib_WithTag(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_WithTag operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.WithTag")


@register_op("nnx.filterlib.PathContains", "flax")
def _map_flax_nnx_filterlib_PathContains(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_PathContains operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.PathContains"
    )


@register_op("nnx.filterlib.PathIn", "flax")
def _map_flax_nnx_filterlib_PathIn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_PathIn operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.PathIn")


@register_op("nnx.filterlib.OfType", "flax")
def _map_flax_nnx_filterlib_OfType(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_OfType operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.OfType")


@register_op("nnx.filterlib.Any", "flax")
def _map_flax_nnx_filterlib_Any(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_Any operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.Any")


@register_op("nnx.filterlib.All", "flax")
def _map_flax_nnx_filterlib_All(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_All operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.All")


@register_op("nnx.filterlib.Not", "flax")
def _map_flax_nnx_filterlib_Not(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_Not operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.Not")


@register_op("nnx.filterlib.Everything", "flax")
def _map_flax_nnx_filterlib_Everything(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_Everything operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.Everything")


@register_op("nnx.filterlib.Nothing", "flax")
def _map_flax_nnx_filterlib_Nothing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_filterlib_Nothing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.filterlib.Nothing")


@register_op("nnx.tracers.reprlib", "flax")
def _map_flax_nnx_tracers_reprlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_tracers_reprlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.tracers.reprlib")


@register_op("nnx.tracers.current_jax_trace", "flax")
def _map_flax_nnx_tracers_current_jax_trace(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_tracers_current_jax_trace operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.tracers.current_jax_trace"
    )


@register_op("nnx.tracers.TraceState", "flax")
def _map_flax_nnx_tracers_TraceState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_tracers_TraceState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.tracers.TraceState")


@register_op("nnx.summary.nnx", "flax")
def _map_flax_nnx_summary_nnx(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_nnx operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.nnx")


@register_op("nnx.summary.typing", "flax")
def _map_flax_nnx_summary_typing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_typing operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.typing")


@register_op("nnx.summary.graphlib", "flax")
def _map_flax_nnx_summary_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_graphlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.graphlib")


@register_op("nnx.summary.statelib", "flax")
def _map_flax_nnx_summary_statelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_statelib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.statelib")


@register_op("nnx.summary.variablelib", "flax")
def _map_flax_nnx_summary_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_variablelib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.variablelib")


@register_op("nnx.summary.in_ipython", "flax")
def _map_flax_nnx_summary_in_ipython(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_in_ipython operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.in_ipython")


@register_op("nnx.summary.NoneDumper", "flax")
def _map_flax_nnx_summary_NoneDumper(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_NoneDumper operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.NoneDumper")


@register_op("nnx.summary.SizeBytes", "flax")
def _map_flax_nnx_summary_SizeBytes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_SizeBytes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.SizeBytes")


@register_op("nnx.summary.ObjectInfo", "flax")
def _map_flax_nnx_summary_ObjectInfo(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_ObjectInfo operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.ObjectInfo")


@register_op("nnx.summary.NodeStats", "flax")
def _map_flax_nnx_summary_NodeStats(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_NodeStats operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.NodeStats")


@register_op("nnx.summary.ArrayRepr", "flax")
def _map_flax_nnx_summary_ArrayRepr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_ArrayRepr operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.ArrayRepr")


@register_op("nnx.summary.CallInfo", "flax")
def _map_flax_nnx_summary_CallInfo(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_CallInfo operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.CallInfo")


@register_op("nnx.summary.SimpleObjectRepr", "flax")
def _map_flax_nnx_summary_SimpleObjectRepr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_SimpleObjectRepr operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.SimpleObjectRepr"
    )


@register_op("nnx.summary.filter_rng_streams", "flax")
def _map_flax_nnx_summary_filter_rng_streams(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_filter_rng_streams operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.filter_rng_streams"
    )


@register_op("nnx.summary.tabulate", "flax")
def _map_flax_nnx_summary_tabulate(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_summary_tabulate operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.summary.tabulate")


@register_op("nnx.module.filterlib", "flax")
def _map_flax_nnx_module_filterlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_filterlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.filterlib")


@register_op("nnx.module.graphlib", "flax")
def _map_flax_nnx_module_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_graphlib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.graphlib")


@register_op("nnx.module.pytreelib", "flax")
def _map_flax_nnx_module_pytreelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_pytreelib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.pytreelib")


@register_op("nnx.module.variableslib", "flax")
def _map_flax_nnx_module_variableslib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_variableslib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.variableslib")


@register_op("nnx.module.Pytree", "flax")
def _map_flax_nnx_module_Pytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_Pytree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.Pytree")


@register_op("nnx.module.PytreeMeta", "flax")
def _map_flax_nnx_module_PytreeMeta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_PytreeMeta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.PytreeMeta")


@register_op("nnx.module.GraphState", "flax")
def _map_flax_nnx_module_GraphState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_GraphState operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.GraphState")


@register_op("nnx.module.split_state", "flax")
def _map_flax_nnx_module_split_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_split_state operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.split_state")


@register_op("nnx.module.State", "flax")
def _map_flax_nnx_module_State(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_State operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.State")


@register_op("nnx.module.Key", "flax")
def _map_flax_nnx_module_Key(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_module_Key operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.Key")


@register_op("nnx.module.Path", "flax")
def _map_flax_nnx_module_Path(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_Path operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.Path")


@register_op("nnx.module.PathParts", "flax")
def _map_flax_nnx_module_PathParts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_PathParts operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.PathParts")


@register_op("nnx.module.A", "flax")
def _map_flax_nnx_module_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_module_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.A")


@register_op("nnx.module.B", "flax")
def _map_flax_nnx_module_B(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_module_B operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.B")


@register_op("nnx.module.M", "flax")
def _map_flax_nnx_module_M(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_module_M operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.M")


@register_op("nnx.module.S", "flax")
def _map_flax_nnx_module_S(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_module_S operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.S")


@register_op("nnx.module.V", "flax")
def _map_flax_nnx_module_V(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_module_V operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.V")


@register_op("nnx.module.F", "flax")
def _map_flax_nnx_module_F(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_module_F operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.F")


@register_op("nnx.module.StateMapping", "flax")
def _map_flax_nnx_module_StateMapping(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_StateMapping operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.StateMapping")


@register_op("nnx.module.tuple_reduce", "flax")
def _map_flax_nnx_module_tuple_reduce(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_tuple_reduce operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.tuple_reduce")


@register_op("nnx.module.tuple_init", "flax")
def _map_flax_nnx_module_tuple_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_tuple_init operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.tuple_init")


@register_op("nnx.module.ModuleMeta", "flax")
def _map_flax_nnx_module_ModuleMeta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_ModuleMeta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.ModuleMeta")


@register_op("nnx.module.Module", "flax")
def _map_flax_nnx_module_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.Module")


@register_op("nnx.module.view", "flax")
def _map_flax_nnx_module_view(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_view operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.view")


@register_op("nnx.module.with_attributes", "flax")
def _map_flax_nnx_module_with_attributes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_with_attributes operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.with_attributes"
    )


@register_op("nnx.module.view_info", "flax")
def _map_flax_nnx_module_view_info(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_view_info operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.view_info")


@register_op("nnx.module.first_from", "flax")
def _map_flax_nnx_module_first_from(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_first_from operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.first_from")


@register_op("nnx.module.iter_modules", "flax")
def _map_flax_nnx_module_iter_modules(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_iter_modules operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.iter_modules")


@register_op("nnx.module.iter_children", "flax")
def _map_flax_nnx_module_iter_children(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_iter_children operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.iter_children")


@register_op("nnx.module.P", "flax")
def _map_flax_nnx_module_P(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_module_P operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.P")


@register_op("nnx.module.R", "flax")
def _map_flax_nnx_module_R(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_module_R operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.R")


@register_op("nnx.module.capture", "flax")
def _map_flax_nnx_module_capture(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_module_capture operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.module.capture")


@register_op("nnx.nn.activations.nnx", "flax")
def _map_flax_nnx_nn_activations_nnx(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_activations_nnx operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.activations.nnx")


@register_op("nnx.nn.activations.dtypes", "flax")
def _map_flax_nnx_nn_activations_dtypes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_activations_dtypes operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.activations.dtypes"
    )


@register_op("nnx.nn.activations.Array", "flax")
def _map_flax_nnx_nn_activations_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_activations_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.activations.Array")


@register_op("nnx.nn.activations.Dtype", "flax")
def _map_flax_nnx_nn_activations_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_activations_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.activations.Dtype")


@register_op("nnx.nn.activations.PromoteDtypeFn", "flax")
def _map_flax_nnx_nn_activations_PromoteDtypeFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_activations_PromoteDtypeFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.activations.PromoteDtypeFn"
    )


@register_op("nnx.nn.activations.PReLU", "flax")
def _map_flax_nnx_nn_activations_PReLU(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_activations_PReLU operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.activations.PReLU")


@register_op("nnx.nn.dtypes.Dtype", "flax")
def _map_flax_nnx_nn_dtypes_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_dtypes_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.dtypes.Dtype")


@register_op("nnx.nn.dtypes.T", "flax")
def _map_flax_nnx_nn_dtypes_T(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_dtypes_T operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.dtypes.T")


@register_op("nnx.nn.dtypes.canonicalize_dtype", "flax")
def _map_flax_nnx_nn_dtypes_canonicalize_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_dtypes_canonicalize_dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.dtypes.canonicalize_dtype"
    )


@register_op("nnx.nn.dtypes.promote_dtype", "flax")
def _map_flax_nnx_nn_dtypes_promote_dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_dtypes_promote_dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.dtypes.promote_dtype"
    )


@register_op("nnx.nn.recurrent.nnx", "flax")
def _map_flax_nnx_nn_recurrent_nnx(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_nnx operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.nnx")


@register_op("nnx.nn.recurrent.filterlib", "flax")
def _map_flax_nnx_nn_recurrent_filterlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_filterlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.filterlib"
    )


@register_op("nnx.nn.recurrent.rnglib", "flax")
def _map_flax_nnx_nn_recurrent_rnglib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_rnglib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.rnglib")


@register_op("nnx.nn.recurrent.Module", "flax")
def _map_flax_nnx_nn_recurrent_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.Module")


@register_op("nnx.nn.recurrent.initializers", "flax")
def _map_flax_nnx_nn_recurrent_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_initializers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.initializers"
    )


@register_op("nnx.nn.recurrent.dtypes", "flax")
def _map_flax_nnx_nn_recurrent_dtypes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_dtypes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.dtypes")


@register_op("nnx.nn.recurrent.Linear", "flax")
def _map_flax_nnx_nn_recurrent_Linear(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_Linear operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.Linear")


@register_op("nnx.nn.recurrent.sigmoid", "flax")
def _map_flax_nnx_nn_recurrent_sigmoid(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_sigmoid operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.sigmoid")


@register_op("nnx.nn.recurrent.tanh", "flax")
def _map_flax_nnx_nn_recurrent_tanh(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_tanh operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.tanh")


@register_op("nnx.nn.recurrent.iteration", "flax")
def _map_flax_nnx_nn_recurrent_iteration(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_iteration operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.iteration"
    )


@register_op("nnx.nn.recurrent.Dtype", "flax")
def _map_flax_nnx_nn_recurrent_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.Dtype")


@register_op("nnx.nn.recurrent.Initializer", "flax")
def _map_flax_nnx_nn_recurrent_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_Initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.Initializer"
    )


@register_op("nnx.nn.recurrent.PromoteDtypeFn", "flax")
def _map_flax_nnx_nn_recurrent_PromoteDtypeFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_PromoteDtypeFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.PromoteDtypeFn"
    )


@register_op("nnx.nn.recurrent.Shape", "flax")
def _map_flax_nnx_nn_recurrent_Shape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_Shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.Shape")


@register_op("nnx.nn.recurrent.default_kernel_init", "flax")
def _map_flax_nnx_nn_recurrent_default_kernel_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_default_kernel_init operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.recurrent.default_kernel_init",
    )


@register_op("nnx.nn.recurrent.default_bias_init", "flax")
def _map_flax_nnx_nn_recurrent_default_bias_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_default_bias_init operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.recurrent.default_bias_init",
    )


@register_op("nnx.nn.recurrent.A", "flax")
def _map_flax_nnx_nn_recurrent_A(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.A")


@register_op("nnx.nn.recurrent.Array", "flax")
def _map_flax_nnx_nn_recurrent_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.Array")


@register_op("nnx.nn.recurrent.Output", "flax")
def _map_flax_nnx_nn_recurrent_Output(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_Output operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.Output")


@register_op("nnx.nn.recurrent.Carry", "flax")
def _map_flax_nnx_nn_recurrent_Carry(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_Carry operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.Carry")


@register_op("nnx.nn.recurrent.RNNCellBase", "flax")
def _map_flax_nnx_nn_recurrent_RNNCellBase(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_RNNCellBase operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.RNNCellBase"
    )


@register_op("nnx.nn.recurrent.modified_orthogonal", "flax")
def _map_flax_nnx_nn_recurrent_modified_orthogonal(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_modified_orthogonal operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.recurrent.modified_orthogonal",
    )


@register_op("nnx.nn.recurrent.LSTMCell", "flax")
def _map_flax_nnx_nn_recurrent_LSTMCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_LSTMCell operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.LSTMCell"
    )


@register_op("nnx.nn.recurrent.OptimizedLSTMCell", "flax")
def _map_flax_nnx_nn_recurrent_OptimizedLSTMCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_OptimizedLSTMCell operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.recurrent.OptimizedLSTMCell",
    )


@register_op("nnx.nn.recurrent.SimpleCell", "flax")
def _map_flax_nnx_nn_recurrent_SimpleCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_SimpleCell operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.SimpleCell"
    )


@register_op("nnx.nn.recurrent.GRUCell", "flax")
def _map_flax_nnx_nn_recurrent_GRUCell(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_GRUCell operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.GRUCell")


@register_op("nnx.nn.recurrent.RNN", "flax")
def _map_flax_nnx_nn_recurrent_RNN(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_RNN operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.RNN")


@register_op("nnx.nn.recurrent.flip_sequences", "flax")
def _map_flax_nnx_nn_recurrent_flip_sequences(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_flip_sequences operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.flip_sequences"
    )


@register_op("nnx.nn.recurrent.RNNBase", "flax")
def _map_flax_nnx_nn_recurrent_RNNBase(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_RNNBase operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.RNNBase")


@register_op("nnx.nn.recurrent.Bidirectional", "flax")
def _map_flax_nnx_nn_recurrent_Bidirectional(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_recurrent_Bidirectional operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.recurrent.Bidirectional"
    )


@register_op("nnx.nn.attention.nnx", "flax")
def _map_flax_nnx_nn_attention_nnx(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_nnx operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.nnx")


@register_op("nnx.nn.attention.rnglib", "flax")
def _map_flax_nnx_nn_attention_rnglib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_rnglib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.rnglib")


@register_op("nnx.nn.attention.Module", "flax")
def _map_flax_nnx_nn_attention_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.Module")


@register_op("nnx.nn.attention.first_from", "flax")
def _map_flax_nnx_nn_attention_first_from(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_first_from operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.first_from"
    )


@register_op("nnx.nn.attention.initializers", "flax")
def _map_flax_nnx_nn_attention_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_initializers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.initializers"
    )


@register_op("nnx.nn.attention.dtypes", "flax")
def _map_flax_nnx_nn_attention_dtypes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_dtypes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.dtypes")


@register_op("nnx.nn.attention.LinearGeneral", "flax")
def _map_flax_nnx_nn_attention_LinearGeneral(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_LinearGeneral operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.LinearGeneral"
    )


@register_op("nnx.nn.attention.default_kernel_init", "flax")
def _map_flax_nnx_nn_attention_default_kernel_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_default_kernel_init operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.attention.default_kernel_init",
    )


@register_op("nnx.nn.attention.LayerNorm", "flax")
def _map_flax_nnx_nn_attention_LayerNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_LayerNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.LayerNorm"
    )


@register_op("nnx.nn.attention.Dtype", "flax")
def _map_flax_nnx_nn_attention_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.Dtype")


@register_op("nnx.nn.attention.PromoteDtypeFn", "flax")
def _map_flax_nnx_nn_attention_PromoteDtypeFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_PromoteDtypeFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.PromoteDtypeFn"
    )


@register_op("nnx.nn.attention.Shape", "flax")
def _map_flax_nnx_nn_attention_Shape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_Shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.Shape")


@register_op("nnx.nn.attention.Initializer", "flax")
def _map_flax_nnx_nn_attention_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_Initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.Initializer"
    )


@register_op("nnx.nn.attention.PrecisionLike", "flax")
def _map_flax_nnx_nn_attention_PrecisionLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_PrecisionLike operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.PrecisionLike"
    )


@register_op("nnx.nn.attention.DotGeneralT", "flax")
def _map_flax_nnx_nn_attention_DotGeneralT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_DotGeneralT operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.DotGeneralT"
    )


@register_op("nnx.nn.attention.Array", "flax")
def _map_flax_nnx_nn_attention_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.Array")


@register_op("nnx.nn.attention.dot_product_attention_weights", "flax")
def _map_flax_nnx_nn_attention_dot_product_attention_weights(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_dot_product_attention_weights operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.attention.dot_product_attention_weights",
    )


@register_op("nnx.nn.attention.dot_product_attention", "flax")
def _map_flax_nnx_nn_attention_dot_product_attention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_dot_product_attention operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.attention.dot_product_attention",
    )


@register_op("nnx.nn.attention.MultiHeadAttention", "flax")
def _map_flax_nnx_nn_attention_MultiHeadAttention(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_MultiHeadAttention operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.attention.MultiHeadAttention",
    )


@register_op("nnx.nn.attention.make_attention_mask", "flax")
def _map_flax_nnx_nn_attention_make_attention_mask(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_make_attention_mask operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.attention.make_attention_mask",
    )


@register_op("nnx.nn.attention.make_causal_mask", "flax")
def _map_flax_nnx_nn_attention_make_causal_mask(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_make_causal_mask operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.make_causal_mask"
    )


@register_op("nnx.nn.attention.combine_masks", "flax")
def _map_flax_nnx_nn_attention_combine_masks(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_attention_combine_masks operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.attention.combine_masks"
    )


@register_op("nnx.nn.stochastic.rnglib", "flax")
def _map_flax_nnx_nn_stochastic_rnglib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_stochastic_rnglib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.stochastic.rnglib")


@register_op("nnx.nn.stochastic.Module", "flax")
def _map_flax_nnx_nn_stochastic_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_stochastic_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.stochastic.Module")


@register_op("nnx.nn.stochastic.first_from", "flax")
def _map_flax_nnx_nn_stochastic_first_from(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_stochastic_first_from operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.stochastic.first_from"
    )


@register_op("nnx.nn.stochastic.nnx", "flax")
def _map_flax_nnx_nn_stochastic_nnx(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_stochastic_nnx operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.stochastic.nnx")


@register_op("nnx.nn.stochastic.Dropout", "flax")
def _map_flax_nnx_nn_stochastic_Dropout(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_stochastic_Dropout operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.stochastic.Dropout"
    )


@register_op("nnx.nn.linear.FrozenDict", "flax")
def _map_flax_nnx_nn_linear_FrozenDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_FrozenDict operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.FrozenDict")


@register_op("nnx.nn.linear.nnx", "flax")
def _map_flax_nnx_nn_linear_nnx(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_nnx operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.nnx")


@register_op("nnx.nn.linear.rnglib", "flax")
def _map_flax_nnx_nn_linear_rnglib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_rnglib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.rnglib")


@register_op("nnx.nn.linear.variablelib", "flax")
def _map_flax_nnx_nn_linear_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_variablelib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.variablelib"
    )


@register_op("nnx.nn.linear.Module", "flax")
def _map_flax_nnx_nn_linear_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.Module")


@register_op("nnx.nn.linear.first_from", "flax")
def _map_flax_nnx_nn_linear_first_from(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_first_from operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.first_from")


@register_op("nnx.nn.linear.dtypes", "flax")
def _map_flax_nnx_nn_linear_dtypes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_dtypes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.dtypes")


@register_op("nnx.nn.linear.initializers", "flax")
def _map_flax_nnx_nn_linear_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_initializers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.initializers"
    )


@register_op("nnx.nn.linear.Dtype", "flax")
def _map_flax_nnx_nn_linear_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.Dtype")


@register_op("nnx.nn.linear.Shape", "flax")
def _map_flax_nnx_nn_linear_Shape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_Shape operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.Shape")


@register_op("nnx.nn.linear.Initializer", "flax")
def _map_flax_nnx_nn_linear_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_Initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.Initializer"
    )


@register_op("nnx.nn.linear.PrecisionLike", "flax")
def _map_flax_nnx_nn_linear_PrecisionLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_PrecisionLike operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.PrecisionLike"
    )


@register_op("nnx.nn.linear.DotGeneralT", "flax")
def _map_flax_nnx_nn_linear_DotGeneralT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_DotGeneralT operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.DotGeneralT"
    )


@register_op("nnx.nn.linear.ConvGeneralDilatedT", "flax")
def _map_flax_nnx_nn_linear_ConvGeneralDilatedT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_ConvGeneralDilatedT operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.ConvGeneralDilatedT"
    )


@register_op("nnx.nn.linear.PaddingLike", "flax")
def _map_flax_nnx_nn_linear_PaddingLike(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_PaddingLike operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.PaddingLike"
    )


@register_op("nnx.nn.linear.LaxPadding", "flax")
def _map_flax_nnx_nn_linear_LaxPadding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_LaxPadding operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.LaxPadding")


@register_op("nnx.nn.linear.PromoteDtypeFn", "flax")
def _map_flax_nnx_nn_linear_PromoteDtypeFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_PromoteDtypeFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.PromoteDtypeFn"
    )


@register_op("nnx.nn.linear.EinsumT", "flax")
def _map_flax_nnx_nn_linear_EinsumT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_EinsumT operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.EinsumT")


@register_op("nnx.nn.linear.Array", "flax")
def _map_flax_nnx_nn_linear_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.Array")


@register_op("nnx.nn.linear.Axis", "flax")
def _map_flax_nnx_nn_linear_Axis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_Axis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.Axis")


@register_op("nnx.nn.linear.Size", "flax")
def _map_flax_nnx_nn_linear_Size(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_Size operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.Size")


@register_op("nnx.nn.linear.default_kernel_init", "flax")
def _map_flax_nnx_nn_linear_default_kernel_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_default_kernel_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.default_kernel_init"
    )


@register_op("nnx.nn.linear.default_bias_init", "flax")
def _map_flax_nnx_nn_linear_default_bias_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_default_bias_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.default_bias_init"
    )


@register_op("nnx.nn.linear.canonicalize_padding", "flax")
def _map_flax_nnx_nn_linear_canonicalize_padding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_canonicalize_padding operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.linear.canonicalize_padding",
    )


@register_op("nnx.nn.linear.LinearGeneral", "flax")
def _map_flax_nnx_nn_linear_LinearGeneral(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_LinearGeneral operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.LinearGeneral"
    )


@register_op("nnx.nn.linear.Linear", "flax")
def _map_flax_nnx_nn_linear_Linear(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_Linear operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.Linear")


@register_op("nnx.nn.linear.Einsum", "flax")
def _map_flax_nnx_nn_linear_Einsum(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_Einsum operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.Einsum")


@register_op("nnx.nn.linear.Conv", "flax")
def _map_flax_nnx_nn_linear_Conv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_Conv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.Conv")


@register_op("nnx.nn.linear.ConvTranspose", "flax")
def _map_flax_nnx_nn_linear_ConvTranspose(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_ConvTranspose operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.ConvTranspose"
    )


@register_op("nnx.nn.linear.default_embed_init", "flax")
def _map_flax_nnx_nn_linear_default_embed_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_default_embed_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.default_embed_init"
    )


@register_op("nnx.nn.linear.Embed", "flax")
def _map_flax_nnx_nn_linear_Embed(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_linear_Embed operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.linear.Embed")


@register_op("nnx.nn.initializers.Initializer", "flax")
def _map_flax_nnx_nn_initializers_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_initializers_Initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.initializers.Initializer"
    )


@register_op("nnx.nn.initializers.DtypeLikeInexact", "flax")
def _map_flax_nnx_nn_initializers_DtypeLikeInexact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_initializers_DtypeLikeInexact operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.initializers.DtypeLikeInexact",
    )


@register_op("nnx.nn.initializers.zeros_init", "flax")
def _map_flax_nnx_nn_initializers_zeros_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_initializers_zeros_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.initializers.zeros_init"
    )


@register_op("nnx.nn.initializers.ones_init", "flax")
def _map_flax_nnx_nn_initializers_ones_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_initializers_ones_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.initializers.ones_init"
    )


@register_op("nnx.nn.normalization.nnx", "flax")
def _map_flax_nnx_nn_normalization_nnx(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_nnx operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.nnx")


@register_op("nnx.nn.normalization.rnglib", "flax")
def _map_flax_nnx_nn_normalization_rnglib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_rnglib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.rnglib"
    )


@register_op("nnx.nn.normalization.Module", "flax")
def _map_flax_nnx_nn_normalization_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_Module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.Module"
    )


@register_op("nnx.nn.normalization.first_from", "flax")
def _map_flax_nnx_nn_normalization_first_from(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_first_from operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.first_from"
    )


@register_op("nnx.nn.normalization.dtypes", "flax")
def _map_flax_nnx_nn_normalization_dtypes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_dtypes operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.dtypes"
    )


@register_op("nnx.nn.normalization.initializers", "flax")
def _map_flax_nnx_nn_normalization_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_initializers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.initializers"
    )


@register_op("nnx.nn.normalization.Array", "flax")
def _map_flax_nnx_nn_normalization_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_Array operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.Array"
    )


@register_op("nnx.nn.normalization.Dtype", "flax")
def _map_flax_nnx_nn_normalization_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_Dtype operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.Dtype"
    )


@register_op("nnx.nn.normalization.Initializer", "flax")
def _map_flax_nnx_nn_normalization_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_Initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.Initializer"
    )


@register_op("nnx.nn.normalization.Axes", "flax")
def _map_flax_nnx_nn_normalization_Axes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_Axes operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.Axes"
    )


@register_op("nnx.nn.normalization.PromoteDtypeFn", "flax")
def _map_flax_nnx_nn_normalization_PromoteDtypeFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_PromoteDtypeFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.nn.normalization.PromoteDtypeFn",
    )


@register_op("nnx.nn.normalization.BatchNorm", "flax")
def _map_flax_nnx_nn_normalization_BatchNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_BatchNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.BatchNorm"
    )


@register_op("nnx.nn.normalization.LayerNorm", "flax")
def _map_flax_nnx_nn_normalization_LayerNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_LayerNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.LayerNorm"
    )


@register_op("nnx.nn.normalization.RMSNorm", "flax")
def _map_flax_nnx_nn_normalization_RMSNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_RMSNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.RMSNorm"
    )


@register_op("nnx.nn.normalization.GroupNorm", "flax")
def _map_flax_nnx_nn_normalization_GroupNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_GroupNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.GroupNorm"
    )


@register_op("nnx.nn.normalization.WeightNorm", "flax")
def _map_flax_nnx_nn_normalization_WeightNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_WeightNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.WeightNorm"
    )


@register_op("nnx.nn.normalization.InstanceNorm", "flax")
def _map_flax_nnx_nn_normalization_InstanceNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_InstanceNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.InstanceNorm"
    )


@register_op("nnx.nn.normalization.SpectralNorm", "flax")
def _map_flax_nnx_nn_normalization_SpectralNorm(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_normalization_SpectralNorm operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.normalization.SpectralNorm"
    )


@register_op("nnx.nn.lora.rnglib", "flax")
def _map_flax_nnx_nn_lora_rnglib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_rnglib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.rnglib")


@register_op("nnx.nn.lora.variablelib", "flax")
def _map_flax_nnx_nn_lora_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_variablelib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.variablelib")


@register_op("nnx.nn.lora.Module", "flax")
def _map_flax_nnx_nn_lora_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.Module")


@register_op("nnx.nn.lora.initializers", "flax")
def _map_flax_nnx_nn_lora_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_initializers operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.initializers")


@register_op("nnx.nn.lora.dtypes", "flax")
def _map_flax_nnx_nn_lora_dtypes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_dtypes operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.dtypes")


@register_op("nnx.nn.lora.Linear", "flax")
def _map_flax_nnx_nn_lora_Linear(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_Linear operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.Linear")


@register_op("nnx.nn.lora.Dtype", "flax")
def _map_flax_nnx_nn_lora_Dtype(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_Dtype operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.Dtype")


@register_op("nnx.nn.lora.Initializer", "flax")
def _map_flax_nnx_nn_lora_Initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_Initializer operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.Initializer")


@register_op("nnx.nn.lora.PromoteDtypeFn", "flax")
def _map_flax_nnx_nn_lora_PromoteDtypeFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_PromoteDtypeFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.PromoteDtypeFn"
    )


@register_op("nnx.nn.lora.Array", "flax")
def _map_flax_nnx_nn_lora_Array(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_Array operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.Array")


@register_op("nnx.nn.lora.Axis", "flax")
def _map_flax_nnx_nn_lora_Axis(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_Axis operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.Axis")


@register_op("nnx.nn.lora.Size", "flax")
def _map_flax_nnx_nn_lora_Size(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_Size operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.Size")


@register_op("nnx.nn.lora.A", "flax")
def _map_flax_nnx_nn_lora_A(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_flax_nnx_nn_lora_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.A")


@register_op("nnx.nn.lora.default_a_initializer", "flax")
def _map_flax_nnx_nn_lora_default_a_initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_default_a_initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.default_a_initializer"
    )


@register_op("nnx.nn.lora.default_b_initializer", "flax")
def _map_flax_nnx_nn_lora_default_b_initializer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_default_b_initializer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.default_b_initializer"
    )


@register_op("nnx.nn.lora.LoRAParam", "flax")
def _map_flax_nnx_nn_lora_LoRAParam(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_LoRAParam operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.LoRAParam")


@register_op("nnx.nn.lora.LoRA", "flax")
def _map_flax_nnx_nn_lora_LoRA(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_LoRA operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.LoRA")


@register_op("nnx.nn.lora.LoRALinear", "flax")
def _map_flax_nnx_nn_lora_LoRALinear(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_nn_lora_LoRALinear operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.nn.lora.LoRALinear")


@register_op("nnx.bridge.functional", "flax")
def _map_flax_nnx_bridge_functional(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_functional operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.functional")


@register_op("nnx.bridge.Functional", "flax")
def _map_flax_nnx_bridge_Functional(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_Functional operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.Functional")


@register_op("nnx.bridge.ToNNX", "flax")
def _map_flax_nnx_bridge_ToNNX(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_ToNNX operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.ToNNX")


@register_op("nnx.bridge.lazy_init", "flax")
def _map_flax_nnx_bridge_lazy_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_lazy_init operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.lazy_init")


@register_op("nnx.bridge.ToLinen", "flax")
def _map_flax_nnx_bridge_ToLinen(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_ToLinen operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.ToLinen")


@register_op("nnx.bridge.to_linen", "flax")
def _map_flax_nnx_bridge_to_linen(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_to_linen operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.to_linen")


@register_op("nnx.bridge.NNXMeta", "flax")
def _map_flax_nnx_bridge_NNXMeta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_NNXMeta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.NNXMeta")


@register_op("nnx.bridge.with_partitioning", "flax")
def _map_flax_nnx_bridge_with_partitioning(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_with_partitioning operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.with_partitioning"
    )


@register_op("nnx.bridge.Module", "flax")
def _map_flax_nnx_bridge_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.Module")


@register_op("nnx.bridge.Scope", "flax")
def _map_flax_nnx_bridge_Scope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_Scope operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.Scope")


@register_op("nnx.bridge.AttrPriority", "flax")
def _map_flax_nnx_bridge_AttrPriority(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_AttrPriority operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.AttrPriority")


@register_op("nnx.bridge.compact", "flax")
def _map_flax_nnx_bridge_compact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_compact operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.compact")


@register_op("nnx.bridge.current_context", "flax")
def _map_flax_nnx_bridge_current_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_current_context operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.current_context"
    )


@register_op("nnx.bridge.current_module", "flax")
def _map_flax_nnx_bridge_current_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_current_module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.current_module"
    )


@register_op("nnx.bridge.nnx_in_bridge_mdl", "flax")
def _map_flax_nnx_bridge_nnx_in_bridge_mdl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_nnx_in_bridge_mdl operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.nnx_in_bridge_mdl"
    )


@register_op("nnx.bridge.linen_in_bridge_mdl", "flax")
def _map_flax_nnx_bridge_linen_in_bridge_mdl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_linen_in_bridge_mdl operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.linen_in_bridge_mdl"
    )


@register_op("nnx.bridge.initializers", "flax")
def _map_flax_nnx_bridge_initializers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_initializers operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.initializers")


@register_op("nnx.bridge.wrappers.linen", "flax")
def _map_flax_nnx_bridge_wrappers_linen(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_linen operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.linen"
    )


@register_op("nnx.bridge.wrappers.nnx", "flax")
def _map_flax_nnx_bridge_wrappers_nnx(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_nnx operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.nnx")


@register_op("nnx.bridge.wrappers.FrozenDict", "flax")
def _map_flax_nnx_bridge_wrappers_FrozenDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_FrozenDict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.FrozenDict"
    )


@register_op("nnx.bridge.wrappers.meta", "flax")
def _map_flax_nnx_bridge_wrappers_meta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_meta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.meta")


@register_op("nnx.bridge.wrappers.graphlib", "flax")
def _map_flax_nnx_bridge_wrappers_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_graphlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.graphlib"
    )


@register_op("nnx.bridge.wrappers.variablelib", "flax")
def _map_flax_nnx_bridge_wrappers_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_variablelib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.variablelib"
    )


@register_op("nnx.bridge.wrappers.bv", "flax")
def _map_flax_nnx_bridge_wrappers_bv(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_bv operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.bv")


@register_op("nnx.bridge.wrappers.bdg_module", "flax")
def _map_flax_nnx_bridge_wrappers_bdg_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_bdg_module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.bdg_module"
    )


@register_op("nnx.bridge.wrappers.Module", "flax")
def _map_flax_nnx_bridge_wrappers_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_Module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.Module"
    )


@register_op("nnx.bridge.wrappers.State", "flax")
def _map_flax_nnx_bridge_wrappers_State(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_State operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.State"
    )


@register_op("nnx.bridge.wrappers.Pytree", "flax")
def _map_flax_nnx_bridge_wrappers_Pytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_Pytree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.Pytree"
    )


@register_op("nnx.bridge.wrappers.Rngs", "flax")
def _map_flax_nnx_bridge_wrappers_Rngs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_Rngs operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.Rngs")


@register_op("nnx.bridge.wrappers.M", "flax")
def _map_flax_nnx_bridge_wrappers_M(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_M operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.M")


@register_op("nnx.bridge.wrappers.Functional", "flax")
def _map_flax_nnx_bridge_wrappers_Functional(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_Functional operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.Functional"
    )


@register_op("nnx.bridge.wrappers.functional", "flax")
def _map_flax_nnx_bridge_wrappers_functional(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_functional operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.functional"
    )


@register_op("nnx.bridge.wrappers.lazy_init", "flax")
def _map_flax_nnx_bridge_wrappers_lazy_init(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_lazy_init operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.lazy_init"
    )


@register_op("nnx.bridge.wrappers.current_linen_module", "flax")
def _map_flax_nnx_bridge_wrappers_current_linen_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_current_linen_module operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.wrappers.current_linen_module",
    )


@register_op("nnx.bridge.wrappers.ToNNX", "flax")
def _map_flax_nnx_bridge_wrappers_ToNNX(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_ToNNX operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.ToNNX"
    )


@register_op("nnx.bridge.wrappers.linen_rngs_dict", "flax")
def _map_flax_nnx_bridge_wrappers_linen_rngs_dict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_linen_rngs_dict operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.wrappers.linen_rngs_dict",
    )


@register_op("nnx.bridge.wrappers.ToLinen", "flax")
def _map_flax_nnx_bridge_wrappers_ToLinen(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_ToLinen operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.ToLinen"
    )


@register_op("nnx.bridge.wrappers.to_linen", "flax")
def _map_flax_nnx_bridge_wrappers_to_linen(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_to_linen operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.wrappers.to_linen"
    )


@register_op("nnx.bridge.wrappers.to_linen_class", "flax")
def _map_flax_nnx_bridge_wrappers_to_linen_class(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_wrappers_to_linen_class operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.wrappers.to_linen_class",
    )


@register_op("nnx.bridge.interop.nn_module", "flax")
def _map_flax_nnx_bridge_interop_nn_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_interop_nn_module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.interop.nn_module"
    )


@register_op("nnx.bridge.interop.graphlib", "flax")
def _map_flax_nnx_bridge_interop_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_interop_graphlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.interop.graphlib"
    )


@register_op("nnx.bridge.interop.rnglib", "flax")
def _map_flax_nnx_bridge_interop_rnglib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_interop_rnglib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.interop.rnglib"
    )


@register_op("nnx.bridge.interop.wrappers", "flax")
def _map_flax_nnx_bridge_interop_wrappers(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_interop_wrappers operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.interop.wrappers"
    )


@register_op("nnx.bridge.interop.bdg_module", "flax")
def _map_flax_nnx_bridge_interop_bdg_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_interop_bdg_module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.interop.bdg_module"
    )


@register_op("nnx.bridge.interop.nnx_module", "flax")
def _map_flax_nnx_bridge_interop_nnx_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_interop_nnx_module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.interop.nnx_module"
    )


@register_op("nnx.bridge.interop.nnx_eval_shape", "flax")
def _map_flax_nnx_bridge_interop_nnx_eval_shape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_interop_nnx_eval_shape operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.interop.nnx_eval_shape"
    )


@register_op("nnx.bridge.interop.nnx_jit", "flax")
def _map_flax_nnx_bridge_interop_nnx_jit(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_interop_nnx_jit operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.interop.nnx_jit"
    )


@register_op("nnx.bridge.interop.nnx_in_bridge_mdl", "flax")
def _map_flax_nnx_bridge_interop_nnx_in_bridge_mdl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_interop_nnx_in_bridge_mdl operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.interop.nnx_in_bridge_mdl",
    )


@register_op("nnx.bridge.interop.linen_in_bridge_mdl", "flax")
def _map_flax_nnx_bridge_interop_linen_in_bridge_mdl(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_interop_linen_in_bridge_mdl operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.interop.linen_in_bridge_mdl",
    )


@register_op("nnx.bridge.variables.struct", "flax")
def _map_flax_nnx_bridge_variables_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_struct operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.struct"
    )


@register_op("nnx.bridge.variables.meta", "flax")
def _map_flax_nnx_bridge_variables_meta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_meta operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.meta"
    )


@register_op("nnx.bridge.variables.spmd", "flax")
def _map_flax_nnx_bridge_variables_spmd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_spmd operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.spmd"
    )


@register_op("nnx.bridge.variables.traversals", "flax")
def _map_flax_nnx_bridge_variables_traversals(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_traversals operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.traversals"
    )


@register_op("nnx.bridge.variables.variablelib", "flax")
def _map_flax_nnx_bridge_variables_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_variablelib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.variablelib"
    )


@register_op("nnx.bridge.variables.LogicalNames", "flax")
def _map_flax_nnx_bridge_variables_LogicalNames(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_LogicalNames operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.LogicalNames"
    )


@register_op("nnx.bridge.variables.A", "flax")
def _map_flax_nnx_bridge_variables_A(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.A")


@register_op("nnx.bridge.variables.B", "flax")
def _map_flax_nnx_bridge_variables_B(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_B operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.B")


@register_op("nnx.bridge.variables.sort_variable_types", "flax")
def _map_flax_nnx_bridge_variables_sort_variable_types(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_sort_variable_types operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.variables.sort_variable_types",
    )


@register_op("nnx.bridge.variables.NNXMeta", "flax")
def _map_flax_nnx_bridge_variables_NNXMeta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_NNXMeta operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.NNXMeta"
    )


@register_op("nnx.bridge.variables.is_vanilla_variable", "flax")
def _map_flax_nnx_bridge_variables_is_vanilla_variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_is_vanilla_variable operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.variables.is_vanilla_variable",
    )


@register_op("nnx.bridge.variables.to_linen_var", "flax")
def _map_flax_nnx_bridge_variables_to_linen_var(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_to_linen_var operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.to_linen_var"
    )


@register_op("nnx.bridge.variables.get_col_name", "flax")
def _map_flax_nnx_bridge_variables_get_col_name(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_get_col_name operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.get_col_name"
    )


@register_op("nnx.bridge.variables.to_nnx_var", "flax")
def _map_flax_nnx_bridge_variables_to_nnx_var(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_to_nnx_var operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.variables.to_nnx_var"
    )


@register_op("nnx.bridge.variables.linen_vars_to_nnx_attrs", "flax")
def _map_flax_nnx_bridge_variables_linen_vars_to_nnx_attrs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_linen_vars_to_nnx_attrs operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.variables.linen_vars_to_nnx_attrs",
    )


@register_op("nnx.bridge.variables.nnx_attrs_to_linen_vars", "flax")
def _map_flax_nnx_bridge_variables_nnx_attrs_to_linen_vars(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_nnx_attrs_to_linen_vars operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.variables.nnx_attrs_to_linen_vars",
    )


@register_op("nnx.bridge.variables.with_partitioning", "flax")
def _map_flax_nnx_bridge_variables_with_partitioning(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_variables_with_partitioning operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.variables.with_partitioning",
    )


@register_op("nnx.bridge.module.errors", "flax")
def _map_flax_nnx_bridge_module_errors(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_errors operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.errors")


@register_op("nnx.bridge.module.meta", "flax")
def _map_flax_nnx_bridge_module_meta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_meta operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.meta")


@register_op("nnx.bridge.module.CollectionFilter", "flax")
def _map_flax_nnx_bridge_module_CollectionFilter(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_CollectionFilter operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.module.CollectionFilter",
    )


@register_op("nnx.bridge.module.FrozenDict", "flax")
def _map_flax_nnx_bridge_module_FrozenDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_FrozenDict operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.FrozenDict"
    )


@register_op("nnx.bridge.module.graphlib", "flax")
def _map_flax_nnx_bridge_module_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_graphlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.graphlib"
    )


@register_op("nnx.bridge.module.rnglib", "flax")
def _map_flax_nnx_bridge_module_rnglib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_rnglib operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.rnglib")


@register_op("nnx.bridge.module.statelib", "flax")
def _map_flax_nnx_bridge_module_statelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_statelib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.statelib"
    )


@register_op("nnx.bridge.module.traversals", "flax")
def _map_flax_nnx_bridge_module_traversals(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_traversals operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.traversals"
    )


@register_op("nnx.bridge.module.variablelib", "flax")
def _map_flax_nnx_bridge_module_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_variablelib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.variablelib"
    )


@register_op("nnx.bridge.module.nnx_module", "flax")
def _map_flax_nnx_bridge_module_nnx_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_nnx_module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.nnx_module"
    )


@register_op("nnx.bridge.module.Pytree", "flax")
def _map_flax_nnx_bridge_module_Pytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_Pytree operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.Pytree")


@register_op("nnx.bridge.module.bridge_variables", "flax")
def _map_flax_nnx_bridge_module_bridge_variables(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_bridge_variables operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.module.bridge_variables",
    )


@register_op("nnx.bridge.module.A", "flax")
def _map_flax_nnx_bridge_module_A(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.A")


@register_op("nnx.bridge.module.M", "flax")
def _map_flax_nnx_bridge_module_M(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_M operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.M")


@register_op("nnx.bridge.module.F", "flax")
def _map_flax_nnx_bridge_module_F(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_F operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.F")


@register_op("nnx.bridge.module.ModuleStackEntry", "flax")
def _map_flax_nnx_bridge_module_ModuleStackEntry(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_ModuleStackEntry operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.module.ModuleStackEntry",
    )


@register_op("nnx.bridge.module.ModuleContext", "flax")
def _map_flax_nnx_bridge_module_ModuleContext(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_ModuleContext operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.ModuleContext"
    )


@register_op("nnx.bridge.module.MODULE_CONTEXT", "flax")
def _map_flax_nnx_bridge_module_MODULE_CONTEXT(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_MODULE_CONTEXT operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.MODULE_CONTEXT"
    )


@register_op("nnx.bridge.module.ModuleState", "flax")
def _map_flax_nnx_bridge_module_ModuleState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_ModuleState operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.ModuleState"
    )


@register_op("nnx.bridge.module.register_data_type", "flax")
def _map_flax_nnx_bridge_module_register_data_type(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_register_data_type operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.bridge.module.register_data_type",
    )


@register_op("nnx.bridge.module.Scope", "flax")
def _map_flax_nnx_bridge_module_Scope(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_Scope operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.Scope")


@register_op("nnx.bridge.module.has_setup", "flax")
def _map_flax_nnx_bridge_module_has_setup(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_has_setup operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.has_setup"
    )


@register_op("nnx.bridge.module.current_context", "flax")
def _map_flax_nnx_bridge_module_current_context(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_current_context operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.current_context"
    )


@register_op("nnx.bridge.module.current_module", "flax")
def _map_flax_nnx_bridge_module_current_module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_current_module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.current_module"
    )


@register_op("nnx.bridge.module.ModuleMeta", "flax")
def _map_flax_nnx_bridge_module_ModuleMeta(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_ModuleMeta operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.ModuleMeta"
    )


@register_op("nnx.bridge.module.AttrPriority", "flax")
def _map_flax_nnx_bridge_module_AttrPriority(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_AttrPriority operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.AttrPriority"
    )


@register_op("nnx.bridge.module.PriorityStr", "flax")
def _map_flax_nnx_bridge_module_PriorityStr(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_PriorityStr operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.PriorityStr"
    )


@register_op("nnx.bridge.module.ModuleBase", "flax")
def _map_flax_nnx_bridge_module_ModuleBase(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_ModuleBase operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.ModuleBase"
    )


@register_op("nnx.bridge.module.Module", "flax")
def _map_flax_nnx_bridge_module_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_Module operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.Module")


@register_op("nnx.bridge.module.compact", "flax")
def _map_flax_nnx_bridge_module_compact(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_bridge_module_compact operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.bridge.module.compact"
    )


@register_op("nnx.transforms.transforms.extract", "flax")
def _map_flax_nnx_transforms_transforms_extract(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_extract operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.extract"
    )


@register_op("nnx.transforms.transforms.graphlib", "flax")
def _map_flax_nnx_transforms_transforms_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_graphlib operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.graphlib",
    )


@register_op("nnx.transforms.transforms.variablelib", "flax")
def _map_flax_nnx_transforms_transforms_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_variablelib operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.variablelib",
    )


@register_op("nnx.transforms.transforms.Module", "flax")
def _map_flax_nnx_transforms_transforms_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_Module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.Module"
    )


@register_op("nnx.transforms.transforms.CallableProxy", "flax")
def _map_flax_nnx_transforms_transforms_CallableProxy(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_CallableProxy operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.CallableProxy",
    )


@register_op("nnx.transforms.transforms.DelayedAccessor", "flax")
def _map_flax_nnx_transforms_transforms_DelayedAccessor(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_DelayedAccessor operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.DelayedAccessor",
    )


@register_op("nnx.transforms.transforms.general", "flax")
def _map_flax_nnx_transforms_transforms_general(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_general operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.general"
    )


@register_op("nnx.transforms.transforms.MISSING", "flax")
def _map_flax_nnx_transforms_transforms_MISSING(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_MISSING operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.MISSING"
    )


@register_op("nnx.transforms.transforms.Leaf", "flax")
def _map_flax_nnx_transforms_transforms_Leaf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_Leaf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.Leaf"
    )


@register_op("nnx.transforms.transforms.Missing", "flax")
def _map_flax_nnx_transforms_transforms_Missing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_Missing operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.Missing"
    )


@register_op("nnx.transforms.transforms.A", "flax")
def _map_flax_nnx_transforms_transforms_A(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_A operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.A"
    )


@register_op("nnx.transforms.transforms.C", "flax")
def _map_flax_nnx_transforms_transforms_C(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_C operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.C"
    )


@register_op("nnx.transforms.transforms.B", "flax")
def _map_flax_nnx_transforms_transforms_B(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_B operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.B"
    )


@register_op("nnx.transforms.transforms.F", "flax")
def _map_flax_nnx_transforms_transforms_F(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_F operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.F"
    )


@register_op("nnx.transforms.transforms.G", "flax")
def _map_flax_nnx_transforms_transforms_G(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_G operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.G"
    )


@register_op("nnx.transforms.transforms.M", "flax")
def _map_flax_nnx_transforms_transforms_M(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_M operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.M"
    )


@register_op("nnx.transforms.transforms.MA", "flax")
def _map_flax_nnx_transforms_transforms_MA(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_MA operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.MA"
    )


@register_op("nnx.transforms.transforms.N", "flax")
def _map_flax_nnx_transforms_transforms_N(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_N operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.N"
    )


@register_op("nnx.transforms.transforms.StrInt", "flax")
def _map_flax_nnx_transforms_transforms_StrInt(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_StrInt operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.StrInt"
    )


@register_op("nnx.transforms.transforms.AxisName", "flax")
def _map_flax_nnx_transforms_transforms_AxisName(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_AxisName operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.AxisName",
    )


@register_op("nnx.transforms.transforms.Leaves", "flax")
def _map_flax_nnx_transforms_transforms_Leaves(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_Leaves operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.Leaves"
    )


@register_op("nnx.transforms.transforms.Index", "flax")
def _map_flax_nnx_transforms_transforms_Index(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_Index operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.Index"
    )


@register_op("nnx.transforms.transforms.resolve_kwargs", "flax")
def _map_flax_nnx_transforms_transforms_resolve_kwargs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_resolve_kwargs operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.resolve_kwargs",
    )


@register_op("nnx.transforms.transforms.LiftedModule", "flax")
def _map_flax_nnx_transforms_transforms_LiftedModule(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_LiftedModule operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.LiftedModule",
    )


@register_op("nnx.transforms.transforms.ValueMetadata", "flax")
def _map_flax_nnx_transforms_transforms_ValueMetadata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_ValueMetadata operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.ValueMetadata",
    )


@register_op("nnx.transforms.transforms.SimpleEvalShapeFn", "flax")
def _map_flax_nnx_transforms_transforms_SimpleEvalShapeFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_SimpleEvalShapeFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.SimpleEvalShapeFn",
    )


@register_op("nnx.transforms.transforms.eval_shape", "flax")
def _map_flax_nnx_transforms_transforms_eval_shape(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_eval_shape operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.eval_shape",
    )


@register_op("nnx.transforms.transforms.CheckifyFn", "flax")
def _map_flax_nnx_transforms_transforms_CheckifyFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_CheckifyFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.CheckifyFn",
    )


@register_op("nnx.transforms.transforms.SimpleCheckifyFn", "flax")
def _map_flax_nnx_transforms_transforms_SimpleCheckifyFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_SimpleCheckifyFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.SimpleCheckifyFn",
    )


@register_op("nnx.transforms.transforms.checkify", "flax")
def _map_flax_nnx_transforms_transforms_checkify(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_checkify operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.checkify",
    )


@register_op("nnx.transforms.transforms.SimpleCondFn", "flax")
def _map_flax_nnx_transforms_transforms_SimpleCondFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_SimpleCondFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.transforms.SimpleCondFn",
    )


@register_op("nnx.transforms.transforms.cond", "flax")
def _map_flax_nnx_transforms_transforms_cond(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_cond operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.cond"
    )


@register_op("nnx.transforms.transforms.switch", "flax")
def _map_flax_nnx_transforms_transforms_switch(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_transforms_switch operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.transforms.switch"
    )


@register_op("nnx.transforms.iteration.struct", "flax")
def _map_flax_nnx_transforms_iteration_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_struct operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.struct"
    )


@register_op("nnx.transforms.iteration.typing", "flax")
def _map_flax_nnx_transforms_iteration_typing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_typing operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.typing"
    )


@register_op("nnx.transforms.iteration.FrozenDict", "flax")
def _map_flax_nnx_transforms_iteration_FrozenDict(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_FrozenDict operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.FrozenDict",
    )


@register_op("nnx.transforms.iteration.extract", "flax")
def _map_flax_nnx_transforms_iteration_extract(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_extract operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.extract"
    )


@register_op("nnx.transforms.iteration.filterlib", "flax")
def _map_flax_nnx_transforms_iteration_filterlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_filterlib operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.filterlib",
    )


@register_op("nnx.transforms.iteration.graphlib", "flax")
def _map_flax_nnx_transforms_iteration_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_graphlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.graphlib"
    )


@register_op("nnx.transforms.iteration.spmd", "flax")
def _map_flax_nnx_transforms_iteration_spmd(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_spmd operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.spmd"
    )


@register_op("nnx.transforms.iteration.variablelib", "flax")
def _map_flax_nnx_transforms_iteration_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_variablelib operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.variablelib",
    )


@register_op("nnx.transforms.iteration.statelib", "flax")
def _map_flax_nnx_transforms_iteration_statelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_statelib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.statelib"
    )


@register_op("nnx.transforms.iteration.Module", "flax")
def _map_flax_nnx_transforms_iteration_Module(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_Module operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.Module"
    )


@register_op("nnx.transforms.iteration.State", "flax")
def _map_flax_nnx_transforms_iteration_State(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_State operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.State"
    )


@register_op("nnx.transforms.iteration.resolve_kwargs", "flax")
def _map_flax_nnx_transforms_iteration_resolve_kwargs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_resolve_kwargs operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.resolve_kwargs",
    )


@register_op("nnx.transforms.iteration.Leaf", "flax")
def _map_flax_nnx_transforms_iteration_Leaf(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_Leaf operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.Leaf"
    )


@register_op("nnx.transforms.iteration.Missing", "flax")
def _map_flax_nnx_transforms_iteration_Missing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_Missing operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.Missing"
    )


@register_op("nnx.transforms.iteration.PytreeDeque", "flax")
def _map_flax_nnx_transforms_iteration_PytreeDeque(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_PytreeDeque operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.PytreeDeque",
    )


@register_op("nnx.transforms.iteration.A", "flax")
def _map_flax_nnx_transforms_iteration_A(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_A operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.A"
    )


@register_op("nnx.transforms.iteration.C", "flax")
def _map_flax_nnx_transforms_iteration_C(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_C operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.C"
    )


@register_op("nnx.transforms.iteration.B", "flax")
def _map_flax_nnx_transforms_iteration_B(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_B operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.B"
    )


@register_op("nnx.transforms.iteration.F", "flax")
def _map_flax_nnx_transforms_iteration_F(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_F operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.F"
    )


@register_op("nnx.transforms.iteration.G", "flax")
def _map_flax_nnx_transforms_iteration_G(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_G operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.G"
    )


@register_op("nnx.transforms.iteration.M", "flax")
def _map_flax_nnx_transforms_iteration_M(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_M operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.M"
    )


@register_op("nnx.transforms.iteration.MA", "flax")
def _map_flax_nnx_transforms_iteration_MA(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_MA operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.MA"
    )


@register_op("nnx.transforms.iteration.N", "flax")
def _map_flax_nnx_transforms_iteration_N(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_N operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.N"
    )


@register_op("nnx.transforms.iteration.T", "flax")
def _map_flax_nnx_transforms_iteration_T(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_T operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.T"
    )


@register_op("nnx.transforms.iteration.StrInt", "flax")
def _map_flax_nnx_transforms_iteration_StrInt(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_StrInt operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.StrInt"
    )


@register_op("nnx.transforms.iteration.AxisName", "flax")
def _map_flax_nnx_transforms_iteration_AxisName(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_AxisName operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.AxisName"
    )


@register_op("nnx.transforms.iteration.Leaves", "flax")
def _map_flax_nnx_transforms_iteration_Leaves(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_Leaves operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.Leaves"
    )


@register_op("nnx.transforms.iteration.Index", "flax")
def _map_flax_nnx_transforms_iteration_Index(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_Index operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.Index"
    )


@register_op("nnx.transforms.iteration.Carry", "flax")
def _map_flax_nnx_transforms_iteration_Carry(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_Carry operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.Carry"
    )


@register_op("nnx.transforms.iteration.transform_metadata", "flax")
def _map_flax_nnx_transforms_iteration_transform_metadata(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_transform_metadata operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.transform_metadata",
    )


@register_op("nnx.transforms.iteration.StateAxes", "flax")
def _map_flax_nnx_transforms_iteration_StateAxes(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_StateAxes operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.StateAxes",
    )


@register_op("nnx.transforms.iteration.AxisFn", "flax")
def _map_flax_nnx_transforms_iteration_AxisFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_AxisFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.AxisFn"
    )


@register_op("nnx.transforms.iteration.SimpleVmapFn", "flax")
def _map_flax_nnx_transforms_iteration_SimpleVmapFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_SimpleVmapFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.SimpleVmapFn",
    )


@register_op("nnx.transforms.iteration.SimplePmapFn", "flax")
def _map_flax_nnx_transforms_iteration_SimplePmapFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_SimplePmapFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.SimplePmapFn",
    )


@register_op("nnx.transforms.iteration.VmapFn", "flax")
def _map_flax_nnx_transforms_iteration_VmapFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_VmapFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.VmapFn"
    )


@register_op("nnx.transforms.iteration.vmap", "flax")
def _map_flax_nnx_transforms_iteration_vmap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_vmap operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.vmap"
    )


@register_op("nnx.transforms.iteration.PmapFn", "flax")
def _map_flax_nnx_transforms_iteration_PmapFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_PmapFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.PmapFn"
    )


@register_op("nnx.transforms.iteration.pmap", "flax")
def _map_flax_nnx_transforms_iteration_pmap(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_pmap operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.pmap"
    )


@register_op("nnx.transforms.iteration.Broadcasted", "flax")
def _map_flax_nnx_transforms_iteration_Broadcasted(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_Broadcasted operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.Broadcasted",
    )


@register_op("nnx.transforms.iteration.ScanFn", "flax")
def _map_flax_nnx_transforms_iteration_ScanFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_ScanFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.ScanFn"
    )


@register_op("nnx.transforms.iteration.SimpleScanFn", "flax")
def _map_flax_nnx_transforms_iteration_SimpleScanFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_SimpleScanFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.SimpleScanFn",
    )


@register_op("nnx.transforms.iteration.scan", "flax")
def _map_flax_nnx_transforms_iteration_scan(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_scan operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.iteration.scan"
    )


@register_op("nnx.transforms.iteration.pure_jax_fancy_scan", "flax")
def _map_flax_nnx_transforms_iteration_pure_jax_fancy_scan(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_pure_jax_fancy_scan operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.pure_jax_fancy_scan",
    )


@register_op("nnx.transforms.iteration.SimpleWhileLoopBodyFn", "flax")
def _map_flax_nnx_transforms_iteration_SimpleWhileLoopBodyFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_SimpleWhileLoopBodyFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.SimpleWhileLoopBodyFn",
    )


@register_op("nnx.transforms.iteration.SimpleWhileLoopCondFn", "flax")
def _map_flax_nnx_transforms_iteration_SimpleWhileLoopCondFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_SimpleWhileLoopCondFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.SimpleWhileLoopCondFn",
    )


@register_op("nnx.transforms.iteration.WhileLoopCondFn", "flax")
def _map_flax_nnx_transforms_iteration_WhileLoopCondFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_WhileLoopCondFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.WhileLoopCondFn",
    )


@register_op("nnx.transforms.iteration.WhileLoopBodyFn", "flax")
def _map_flax_nnx_transforms_iteration_WhileLoopBodyFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_WhileLoopBodyFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.WhileLoopBodyFn",
    )


@register_op("nnx.transforms.iteration.while_loop", "flax")
def _map_flax_nnx_transforms_iteration_while_loop(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_while_loop operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.while_loop",
    )


@register_op("nnx.transforms.iteration.SimpleForiLoopBodyFn", "flax")
def _map_flax_nnx_transforms_iteration_SimpleForiLoopBodyFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_SimpleForiLoopBodyFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.SimpleForiLoopBodyFn",
    )


@register_op("nnx.transforms.iteration.ForiLoopBodyFn", "flax")
def _map_flax_nnx_transforms_iteration_ForiLoopBodyFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_ForiLoopBodyFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.ForiLoopBodyFn",
    )


@register_op("nnx.transforms.iteration.fori_loop", "flax")
def _map_flax_nnx_transforms_iteration_fori_loop(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_iteration_fori_loop operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.iteration.fori_loop",
    )


@register_op("nnx.transforms.compilation.extract", "flax")
def _map_flax_nnx_transforms_compilation_extract(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_extract operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.extract",
    )


@register_op("nnx.transforms.compilation.filterlib", "flax")
def _map_flax_nnx_transforms_compilation_filterlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_filterlib operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.filterlib",
    )


@register_op("nnx.transforms.compilation.graphlib", "flax")
def _map_flax_nnx_transforms_compilation_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_graphlib operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.graphlib",
    )


@register_op("nnx.transforms.compilation.statelib", "flax")
def _map_flax_nnx_transforms_compilation_statelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_statelib operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.statelib",
    )


@register_op("nnx.transforms.compilation.variablelib", "flax")
def _map_flax_nnx_transforms_compilation_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_variablelib operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.variablelib",
    )


@register_op("nnx.transforms.compilation.MISSING", "flax")
def _map_flax_nnx_transforms_compilation_MISSING(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_MISSING operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.MISSING",
    )


@register_op("nnx.transforms.compilation.Missing", "flax")
def _map_flax_nnx_transforms_compilation_Missing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_Missing operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.Missing",
    )


@register_op("nnx.transforms.compilation.PathParts", "flax")
def _map_flax_nnx_transforms_compilation_PathParts(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_PathParts operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.PathParts",
    )


@register_op("nnx.transforms.compilation.F", "flax")
def _map_flax_nnx_transforms_compilation_F(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_F operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.compilation.F"
    )


@register_op("nnx.transforms.compilation.P", "flax")
def _map_flax_nnx_transforms_compilation_P(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_P operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.compilation.P"
    )


@register_op("nnx.transforms.compilation.R", "flax")
def _map_flax_nnx_transforms_compilation_R(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_R operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.compilation.R"
    )


@register_op("nnx.transforms.compilation.Specs", "flax")
def _map_flax_nnx_transforms_compilation_Specs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_Specs operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.compilation.Specs"
    )


@register_op("nnx.transforms.compilation.AxisName", "flax")
def _map_flax_nnx_transforms_compilation_AxisName(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_AxisName operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.AxisName",
    )


@register_op("nnx.transforms.compilation.StateSharding", "flax")
def _map_flax_nnx_transforms_compilation_StateSharding(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_StateSharding operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.StateSharding",
    )


@register_op("nnx.transforms.compilation.JitFn", "flax")
def _map_flax_nnx_transforms_compilation_JitFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_JitFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.compilation.JitFn"
    )


@register_op("nnx.transforms.compilation.jit", "flax")
def _map_flax_nnx_transforms_compilation_jit(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_jit operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.compilation.jit"
    )


@register_op("nnx.transforms.compilation.PartialState", "flax")
def _map_flax_nnx_transforms_compilation_PartialState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_PartialState operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.PartialState",
    )


@register_op("nnx.transforms.compilation.SimpleJitFn", "flax")
def _map_flax_nnx_transforms_compilation_SimpleJitFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_SimpleJitFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.SimpleJitFn",
    )


@register_op("nnx.transforms.compilation.SimpleJitWrapped", "flax")
def _map_flax_nnx_transforms_compilation_SimpleJitWrapped(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_SimpleJitWrapped operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.SimpleJitWrapped",
    )


@register_op("nnx.transforms.compilation.jit_partial", "flax")
def _map_flax_nnx_transforms_compilation_jit_partial(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_jit_partial operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.jit_partial",
    )


@register_op("nnx.transforms.compilation.JitWrapped", "flax")
def _map_flax_nnx_transforms_compilation_JitWrapped(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_JitWrapped operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.JitWrapped",
    )


@register_op("nnx.transforms.compilation.Stage", "flax")
def _map_flax_nnx_transforms_compilation_Stage(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_Stage operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.compilation.Stage"
    )


@register_op("nnx.transforms.compilation.Compiled", "flax")
def _map_flax_nnx_transforms_compilation_Compiled(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_Compiled operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.Compiled",
    )


@register_op("nnx.transforms.compilation.Lowered", "flax")
def _map_flax_nnx_transforms_compilation_Lowered(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_Lowered operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.Lowered",
    )


@register_op("nnx.transforms.compilation.Traced", "flax")
def _map_flax_nnx_transforms_compilation_Traced(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_Traced operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.compilation.Traced"
    )


@register_op("nnx.transforms.compilation.SimpleCompiled", "flax")
def _map_flax_nnx_transforms_compilation_SimpleCompiled(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_SimpleCompiled operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.SimpleCompiled",
    )


@register_op("nnx.transforms.compilation.SimpleLowered", "flax")
def _map_flax_nnx_transforms_compilation_SimpleLowered(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_SimpleLowered operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.SimpleLowered",
    )


@register_op("nnx.transforms.compilation.SimpleTraced", "flax")
def _map_flax_nnx_transforms_compilation_SimpleTraced(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_SimpleTraced operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.SimpleTraced",
    )


@register_op("nnx.transforms.compilation.SimpleShardMapFn", "flax")
def _map_flax_nnx_transforms_compilation_SimpleShardMapFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_SimpleShardMapFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.SimpleShardMapFn",
    )


@register_op("nnx.transforms.compilation.ShardMapFn", "flax")
def _map_flax_nnx_transforms_compilation_ShardMapFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_ShardMapFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.ShardMapFn",
    )


@register_op("nnx.transforms.compilation.shard_map", "flax")
def _map_flax_nnx_transforms_compilation_shard_map(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_compilation_shard_map operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.compilation.shard_map",
    )


@register_op("nnx.transforms.general.extract", "flax")
def _map_flax_nnx_transforms_general_extract(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_general_extract operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.general.extract"
    )


@register_op("nnx.transforms.general.graphlib", "flax")
def _map_flax_nnx_transforms_general_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_general_graphlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.general.graphlib"
    )


@register_op("nnx.transforms.general.MISSING", "flax")
def _map_flax_nnx_transforms_general_MISSING(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_general_MISSING operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.general.MISSING"
    )


@register_op("nnx.transforms.general.Missing", "flax")
def _map_flax_nnx_transforms_general_Missing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_general_Missing operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.general.Missing"
    )


@register_op("nnx.transforms.general.A", "flax")
def _map_flax_nnx_transforms_general_A(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_general_A operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.general.A")


@register_op("nnx.transforms.general.F", "flax")
def _map_flax_nnx_transforms_general_F(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_general_F operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.general.F")


@register_op("nnx.transforms.general.split_inputs", "flax")
def _map_flax_nnx_transforms_general_split_inputs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_general_split_inputs operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.general.split_inputs",
    )


@register_op("nnx.transforms.general.merge_inputs", "flax")
def _map_flax_nnx_transforms_general_merge_inputs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_general_merge_inputs operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.general.merge_inputs",
    )


@register_op("nnx.transforms.autodiff.struct", "flax")
def _map_flax_nnx_transforms_autodiff_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_struct operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.struct"
    )


@register_op("nnx.transforms.autodiff.extract", "flax")
def _map_flax_nnx_transforms_autodiff_extract(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_extract operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.extract"
    )


@register_op("nnx.transforms.autodiff.filterlib", "flax")
def _map_flax_nnx_transforms_autodiff_filterlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_filterlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.filterlib"
    )


@register_op("nnx.transforms.autodiff.graphlib", "flax")
def _map_flax_nnx_transforms_autodiff_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_graphlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.graphlib"
    )


@register_op("nnx.transforms.autodiff.variablelib", "flax")
def _map_flax_nnx_transforms_autodiff_variablelib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_variablelib operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.variablelib",
    )


@register_op("nnx.transforms.autodiff.State", "flax")
def _map_flax_nnx_transforms_autodiff_State(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_State operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.State"
    )


@register_op("nnx.transforms.autodiff.general", "flax")
def _map_flax_nnx_transforms_autodiff_general(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_general operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.general"
    )


@register_op("nnx.transforms.autodiff.resolve_kwargs", "flax")
def _map_flax_nnx_transforms_autodiff_resolve_kwargs(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_resolve_kwargs operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.resolve_kwargs",
    )


@register_op("nnx.transforms.autodiff.MISSING", "flax")
def _map_flax_nnx_transforms_autodiff_MISSING(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_MISSING operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.MISSING"
    )


@register_op("nnx.transforms.autodiff.Missing", "flax")
def _map_flax_nnx_transforms_autodiff_Missing(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_Missing operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.Missing"
    )


@register_op("nnx.transforms.autodiff.A", "flax")
def _map_flax_nnx_transforms_autodiff_A(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_A operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.A"
    )


@register_op("nnx.transforms.autodiff.F", "flax")
def _map_flax_nnx_transforms_autodiff_F(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_F operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.F"
    )


@register_op("nnx.transforms.autodiff.AxisName", "flax")
def _map_flax_nnx_transforms_autodiff_AxisName(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_AxisName operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.AxisName"
    )


@register_op("nnx.transforms.autodiff.DiffState", "flax")
def _map_flax_nnx_transforms_autodiff_DiffState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_DiffState operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.DiffState"
    )


@register_op("nnx.transforms.autodiff.SimpleGradFn", "flax")
def _map_flax_nnx_transforms_autodiff_SimpleGradFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_SimpleGradFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.SimpleGradFn",
    )


@register_op("nnx.transforms.autodiff.GradFn", "flax")
def _map_flax_nnx_transforms_autodiff_GradFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_GradFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.GradFn"
    )


@register_op("nnx.transforms.autodiff.grad", "flax")
def _map_flax_nnx_transforms_autodiff_grad(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_grad operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.grad"
    )


@register_op("nnx.transforms.autodiff.value_and_grad", "flax")
def _map_flax_nnx_transforms_autodiff_value_and_grad(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_value_and_grad operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.value_and_grad",
    )


@register_op("nnx.transforms.autodiff.SimpleVjpFn", "flax")
def _map_flax_nnx_transforms_autodiff_SimpleVjpFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_SimpleVjpFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.SimpleVjpFn",
    )


@register_op("nnx.transforms.autodiff.vjp", "flax")
def _map_flax_nnx_transforms_autodiff_vjp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_vjp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.vjp"
    )


@register_op("nnx.transforms.autodiff.SimpleJvpFn", "flax")
def _map_flax_nnx_transforms_autodiff_SimpleJvpFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_SimpleJvpFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.SimpleJvpFn",
    )


@register_op("nnx.transforms.autodiff.jvp", "flax")
def _map_flax_nnx_transforms_autodiff_jvp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_jvp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.jvp"
    )


@register_op("nnx.transforms.autodiff.SimpleCustomVjpFn", "flax")
def _map_flax_nnx_transforms_autodiff_SimpleCustomVjpFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_SimpleCustomVjpFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.SimpleCustomVjpFn",
    )


@register_op("nnx.transforms.autodiff.SimpleFwdFn", "flax")
def _map_flax_nnx_transforms_autodiff_SimpleFwdFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_SimpleFwdFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.SimpleFwdFn",
    )


@register_op("nnx.transforms.autodiff.SimpleBwdFn", "flax")
def _map_flax_nnx_transforms_autodiff_SimpleBwdFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_SimpleBwdFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.SimpleBwdFn",
    )


@register_op("nnx.transforms.autodiff.SimpleCustomVjp", "flax")
def _map_flax_nnx_transforms_autodiff_SimpleCustomVjp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_SimpleCustomVjp operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.SimpleCustomVjp",
    )


@register_op("nnx.transforms.autodiff.CustomVjpFnWrapper", "flax")
def _map_flax_nnx_transforms_autodiff_CustomVjpFnWrapper(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_CustomVjpFnWrapper operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.CustomVjpFnWrapper",
    )


@register_op("nnx.transforms.autodiff.FwdFn", "flax")
def _map_flax_nnx_transforms_autodiff_FwdFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_FwdFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.FwdFn"
    )


@register_op("nnx.transforms.autodiff.BwdFn", "flax")
def _map_flax_nnx_transforms_autodiff_BwdFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_BwdFn operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.BwdFn"
    )


@register_op("nnx.transforms.autodiff.CustomVjp", "flax")
def _map_flax_nnx_transforms_autodiff_CustomVjp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_CustomVjp operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.CustomVjp"
    )


@register_op("nnx.transforms.autodiff.custom_vjp", "flax")
def _map_flax_nnx_transforms_autodiff_custom_vjp(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_custom_vjp operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.custom_vjp",
    )


@register_op("nnx.transforms.autodiff.SimpleRematFn", "flax")
def _map_flax_nnx_transforms_autodiff_SimpleRematFn(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_SimpleRematFn operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.transforms.autodiff.SimpleRematFn",
    )


@register_op("nnx.transforms.autodiff.remat", "flax")
def _map_flax_nnx_transforms_autodiff_remat(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_transforms_autodiff_remat operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.transforms.autodiff.remat"
    )


@register_op("nnx.training.optimizer.nnx", "flax")
def _map_flax_nnx_training_optimizer_nnx(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_nnx operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.optimizer.nnx"
    )


@register_op("nnx.training.optimizer.filterlib", "flax")
def _map_flax_nnx_training_optimizer_filterlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_filterlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.optimizer.filterlib"
    )


@register_op("nnx.training.optimizer.Pytree", "flax")
def _map_flax_nnx_training_optimizer_Pytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_Pytree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.optimizer.Pytree"
    )


@register_op("nnx.training.optimizer.Variable", "flax")
def _map_flax_nnx_training_optimizer_Variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_Variable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.optimizer.Variable"
    )


@register_op("nnx.training.optimizer.M", "flax")
def _map_flax_nnx_training_optimizer_M(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_M operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.optimizer.M")


@register_op("nnx.training.optimizer.F", "flax")
def _map_flax_nnx_training_optimizer_F(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_F operation."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.optimizer.F")


@register_op("nnx.training.optimizer.OptState", "flax")
def _map_flax_nnx_training_optimizer_OptState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_OptState operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.optimizer.OptState"
    )


@register_op("nnx.training.optimizer.OptArray", "flax")
def _map_flax_nnx_training_optimizer_OptArray(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_OptArray operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.optimizer.OptArray"
    )


@register_op("nnx.training.optimizer.OptVariable", "flax")
def _map_flax_nnx_training_optimizer_OptVariable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_OptVariable operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.training.optimizer.OptVariable",
    )


@register_op("nnx.training.optimizer.to_opt_state", "flax")
def _map_flax_nnx_training_optimizer_to_opt_state(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_to_opt_state operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.training.optimizer.to_opt_state",
    )


@register_op("nnx.training.optimizer.MISSING", "flax")
def _map_flax_nnx_training_optimizer_MISSING(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_MISSING operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.optimizer.MISSING"
    )


@register_op("nnx.training.optimizer.Optimizer", "flax")
def _map_flax_nnx_training_optimizer_Optimizer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_Optimizer operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.optimizer.Optimizer"
    )


@register_op("nnx.training.optimizer.ModelAndOptimizer", "flax")
def _map_flax_nnx_training_optimizer_ModelAndOptimizer(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_optimizer_ModelAndOptimizer operation."""
    return Node(
        op_type="Identity",
        inputs=inputs,
        outputs=outputs,
        name="nnx.training.optimizer.ModelAndOptimizer",
    )


@register_op("nnx.training.metrics.struct", "flax")
def _map_flax_nnx_training_metrics_struct(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_struct operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.struct"
    )


@register_op("nnx.training.metrics.filterlib", "flax")
def _map_flax_nnx_training_metrics_filterlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_filterlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.filterlib"
    )


@register_op("nnx.training.metrics.graphlib", "flax")
def _map_flax_nnx_training_metrics_graphlib(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_graphlib operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.graphlib"
    )


@register_op("nnx.training.metrics.Pytree", "flax")
def _map_flax_nnx_training_metrics_Pytree(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_Pytree operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.Pytree"
    )


@register_op("nnx.training.metrics.Variable", "flax")
def _map_flax_nnx_training_metrics_Variable(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_Variable operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.Variable"
    )


@register_op("nnx.training.metrics.MetricState", "flax")
def _map_flax_nnx_training_metrics_MetricState(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_MetricState operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.MetricState"
    )


@register_op("nnx.training.metrics.Metric", "flax")
def _map_flax_nnx_training_metrics_Metric(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_Metric operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.Metric"
    )


@register_op("nnx.training.metrics.Average", "flax")
def _map_flax_nnx_training_metrics_Average(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_Average operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.Average"
    )


@register_op("nnx.training.metrics.Statistics", "flax")
def _map_flax_nnx_training_metrics_Statistics(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_Statistics operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.Statistics"
    )


@register_op("nnx.training.metrics.Welford", "flax")
def _map_flax_nnx_training_metrics_Welford(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_Welford operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.Welford"
    )


@register_op("nnx.training.metrics.Accuracy", "flax")
def _map_flax_nnx_training_metrics_Accuracy(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_Accuracy operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.Accuracy"
    )


@register_op("nnx.training.metrics.MultiMetric", "flax")
def _map_flax_nnx_training_metrics_MultiMetric(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_flax_nnx_training_metrics_MultiMetric operation."""
    return Node(
        op_type="Identity", inputs=inputs, outputs=outputs, name="nnx.training.metrics.MultiMetric"
    )
