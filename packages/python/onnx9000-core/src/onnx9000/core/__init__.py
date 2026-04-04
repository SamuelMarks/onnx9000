"""ONNX9000 Core IR Package."""

from onnx9000.core.exceptions import ONNXParseError, ShapeInferenceError
from onnx9000.core.execution import (
    Environment,
    ExecutionProvider,
    RunOptions,
    SessionOptions,
)
from onnx9000.core.ir import DynamicDim, Graph, Node, Tensor
from onnx9000.core.macros import MacroExpander, MacroMatcher, ir_macro
from onnx9000.core.primitives import (
    AlibiBias,
    BaseActivation,
    BaseNorm,
    BatchNormalization,
    ConvFamily,
    ConvND,
    DepthwiseConv,
    FlashAttention,
    Gelu,
    Gemm,
    GroupedQueryAttention,
    GroupNorm,
    InstanceNorm,
    LayerNormalization,
    LeakyRelu,
    MatMul,
    Mish,
    MultiHeadAttention,
    Relu,
    RMSNorm,
    RoPE,
    Sigmoid,
    Silu,
    Swish,
    Tanh,
)
from onnx9000.core.registry import register_op
from onnx9000.core.sharding import (
    AutoShardingPass,
    SPMDLoweringPass,
    all_gather,
    all_reduce,
    all_to_all,
    reduce_scatter,
)

__all__ = [
    "Graph",
    "Node",
    "Tensor",
    "DynamicDim",
    "ONNXParseError",
    "ShapeInferenceError",
    "register_op",
    "SessionOptions",
    "RunOptions",
    "Environment",
    "ExecutionProvider",
    "BaseNorm",
    "BatchNormalization",
    "LayerNormalization",
    "RMSNorm",
    "GroupNorm",
    "InstanceNorm",
    "BaseActivation",
    "Relu",
    "Sigmoid",
    "Tanh",
    "LeakyRelu",
    "Gelu",
    "Silu",
    "Swish",
    "Mish",
    "ConvFamily",
    "ConvND",
    "DepthwiseConv",
    "MatMul",
    "Gemm",
    "MultiHeadAttention",
    "FlashAttention",
    "GroupedQueryAttention",
    "RoPE",
    "AlibiBias",
    "ir_macro",
    "MacroExpander",
    "MacroMatcher",
    "AutoShardingPass",
    "SPMDLoweringPass",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
]
from onnx9000.core.memory_planner import simulate_memory_plan
from onnx9000.core.profiler import ProfilerResult, profile
from onnx9000.core.profiler_checks import OptimizationAnalyzer
from onnx9000.core.profiler_grouping import (
    export_csv,
    export_hierarchical_json,
    group_by_namespace,
    to_pandas_dataframe,
)
