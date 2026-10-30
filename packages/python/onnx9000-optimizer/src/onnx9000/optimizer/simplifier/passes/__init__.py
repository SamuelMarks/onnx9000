"""Optimization and Analysis Passes.

Contains all graph transformation passes (DCE, Fusion, Layout, Quantization)
and analytical tools (Memory Planning, Validation) for the IR.
"""

from .broadcast import optimize_broadcasting
from .constant_folding import constant_folding
from .dce import dead_code_elimination
from .debug import inject_probes
from .flattening import flatten_subgraphs
from .fusion import fuse_consecutive_transpose, fuse_linear_activation, fuse_matmul_add
from .layout import transform_nchw_to_nhwc, transform_nhwc_to_nchw
from .memory_planning import estimate_memory_consumption, plan_tensor_lifecycles
from .partitioning import partition_for_multi_device
from .quantization import convert_to_int8, insert_qat_nodes
from .shapes import extract_rnn_states, resolve_dynamic_batch, resolve_dynamic_sequence
from .validation import detect_cycles
from .versioning import apply_opset_fallbacks, enforce_opset_18
from .webgpu import optimize_for_webgpu, polyfill_webgpu_unsupported

__all__ = [
    "apply_opset_fallbacks",
    "constant_folding",
    "convert_to_int8",
    "dead_code_elimination",
    "detect_cycles",
    "enforce_opset_18",
    "estimate_memory_consumption",
    "extract_rnn_states",
    "flatten_subgraphs",
    "fuse_consecutive_transpose",
    "fuse_linear_activation",
    "fuse_matmul_add",
    "inject_probes",
    "insert_qat_nodes",
    "optimize_broadcasting",
    "optimize_for_webgpu",
    "partition_for_multi_device",
    "plan_tensor_lifecycles",
    "polyfill_webgpu_unsupported",
    "resolve_dynamic_batch",
    "resolve_dynamic_sequence",
    "transform_nchw_to_nhwc",
    "transform_nhwc_to_nchw",
]
