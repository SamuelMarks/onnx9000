"""PyTorch-like drop-in replacement namespace."""

from onnx9000.converters.frontend import nn
from onnx9000.converters.frontend.exporter import export
from onnx9000.converters.frontend.tensor import Parameter, Tensor
from onnx9000.converters.frontend.tracer import script, trace
from onnx9000.core.dtypes import DType


def tensor(data, dtype=None):
    """Implement the tensor method or operation."""
    if dtype is None:
        return Tensor(data=data)
    return Tensor(data=data, dtype=dtype)


def zeros(*shape, dtype=DType.FLOAT32):
    """Implement the zeros method or operation."""
    import numpy as np

    return Tensor(shape=shape, dtype=dtype, data=np.zeros(shape, dtype=np.float32))


def ones(*shape, dtype=DType.FLOAT32):
    """Implement the ones method or operation."""
    import numpy as np

    return Tensor(shape=shape, dtype=dtype, data=np.ones(shape, dtype=np.float32))


def randn(*shape, dtype=DType.FLOAT32):
    """Implement the randn method or operation."""
    import numpy as np

    return Tensor(shape=shape, dtype=dtype, data=np.random.randn(*shape).astype(np.float32))


float32 = DType.FLOAT32
float64 = DType.FLOAT64
int32 = DType.INT32
int64 = DType.INT64
bool = DType.BOOL


class jit:
    """Class jit implementation."""

    @staticmethod
    def trace(*args, **kwargs):
        """Implement the trace method or operation."""
        if len(args) == 2:
            return trace(args[0], args[1])
        return trace(*args, **kwargs)

    @staticmethod
    def script(*args, **kwargs):
        """Implement the script method or operation."""
        return script(*args, **kwargs)


class onnx:
    """Class onnx implementation."""

    @staticmethod
    def export(*args, **kwargs):
        """Implement the export method or operation."""
        return export(*args, **kwargs)


__all__ = [
    "Tensor",
    "Parameter",
    "nn",
    "export",
    "trace",
    "script",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "float32",
    "float64",
    "int32",
    "int64",
    "bool",
    "jit",
    "onnx",
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "CharStorage",
    "CharTensor",
    "DoubleStorage",
    "DoubleTensor",
    "FloatStorage",
    "FloatTensor",
    "GradScaler",
    "IntStorage",
    "IntTensor",
    "LongStorage",
    "LongTensor",
    "ShortStorage",
    "ShortTensor",
    "SymBool",
    "SymFloat",
    "SymInt",
    "TypedStorage",
    "UntypedStorage",
    "are_deterministic_algorithms_enabled",
    "autocast",
    "chunk",
    "compile",
    "cond",
    "enable_grad",
    "export_AdditionalInputs",
    "export_Constraint",
    "export_CustomDecompTable",
    "export_default_decompositions",
    "export_Dim",
    "export_dims",
    "export_draft_export",
    "export_export",
    "export_ExportBackwardSignature",
    "export_ExportedProgram",
    "export_ExportGraphSignature",
    "export_FlatArgsAdapter",
    "export_load",
    "export_ModuleCallEntry",
    "export_ModuleCallSignature",
    "export_register_dataclass",
    "export_save",
    "export_ShapesCollection",
    "export_unflatten",
    "export_UnflattenedModule",
    "get_default_device",
    "get_deterministic_debug_mode",
    "get_device_module",
    "get_float32_matmul_precision",
    "get_rng_state",
    "inference_mode",
    "initial_seed",
    "is_deterministic_algorithms_warn_only_enabled",
    "is_storage",
    "is_tensor",
    "is_warn_always_enabled",
    "load",
    "lobpcg",
    "manual_seed",
    "matmul",
    "no_grad",
    "rand",
    "save",
    "seed",
    "set_default_device",
    "set_default_tensor_type",
    "set_deterministic_debug_mode",
    "set_float32_matmul_precision",
    "set_printoptions",
    "set_rng_state",
    "set_warn_always",
    "split",
    "stack",
    "sym_float",
    "sym_fresh_size",
    "sym_int",
    "sym_ite",
    "sym_max",
    "sym_min",
    "sym_not",
    "sym_sum",
    "typename",
    "unravel_index",
    "use_deterministic_algorithms",
    "vmap",
    "sym_sqrt",
    "AVG",
    "AcceleratorError",
    "AggregationType",
    "AliasDb",
    "AnyType",
    "Argument",
    "ArgumentSpec",
    "AwaitType",
    "BenchmarkConfig",
    "BenchmarkExecutionStats",
    "Block",
    "BoolType",
    "BufferDict",
    "CallStack",
    "Capsule",
    "ClassType",
    "Code",
    "CompilationUnit",
    "CompleteArgumentSpec",
    "ComplexType",
    "ConcreteModuleType",
    "ConcreteModuleTypeBuilder",
    "DeepCopyMemoTable",
    "DeserializationStorageContext",
    "DeviceObjType",
    "DictType",
    "DisableTorchFunction",
    "DisableTorchFunctionSubclass",
    "DispatchKey",
    "DispatchKeySet",
    "EnumType",
    "ErrorReport",
    "Event",
    "ExcludeDispatchKeyGuard",
    "ExecutionPlan",
    "FatalError",
    "FileCheck",
    "FloatType",
    "FunctionSchema",
    "Future",
    "FutureType",
    "Generator",
    "Gradient",
    "Graph",
    "GraphExecutorState",
    "IODescriptor",
    "InferredType",
    "IntType",
    "InterfaceType",
    "JITException",
    "ListType",
    "LiteScriptModule",
    "LockingLogger",
    "ModuleDict",
    "Node",
    "NoneType",
    "NoopLogger",
    "NumberType",
    "OperatorInfo",
    "OptionalType",
    "OutOfMemoryError",
    "ParameterDict",
    "PyObjectType",
    "PyTorchFileReader",
    "PyTorchFileWriter",
    "RRefType",
    "SUM",
    "ScriptClass",
    "ScriptClassFunction",
    "ScriptDict",
    "ScriptDictIterator",
    "ScriptDictKeyIterator",
    "ScriptFunction",
    "ScriptList",
    "ScriptListIterator",
    "ScriptMethod",
    "ScriptModule",
    "ScriptModuleSerializer",
    "ScriptObject",
    "ScriptObjectProperty",
    "SerializationStorageContext",
    "Size",
    "StaticModule",
    "Stream",
    "StreamObjType",
    "StringType",
    "SymBoolType",
    "SymIntType",
    "Tag",
    "TensorType",
    "ThroughputBenchmark",
    "TracingState",
    "TupleType",
    "Type",
    "UnionType",
    "Use",
    "Value",
    "autocast_decrement_nesting",
    "autocast_increment_nesting",
    "clear_autocast_cache",
    "cpp_OrderedModuleDict",
    "cpp_OrderedTensorDict",
    "cpp_nn_Module",
    "default_generator",
    "device",
    "dtype",
    "finfo",
    "fork",
    "get_autocast_cpu_dtype",
    "get_autocast_dtype",
    "get_autocast_gpu_dtype",
    "get_autocast_ipu_dtype",
    "get_autocast_xla_dtype",
    "get_default_dtype",
    "get_num_interop_threads",
    "get_num_threads",
    "has_lapack",
    "has_mkl",
    "has_openmp",
    "has_spectral",
    "iinfo",
    "import_ir_module",
    "import_ir_module_from_buffer",
    "init_num_threads",
    "is_anomaly_check_nan_enabled",
    "is_anomaly_enabled",
    "is_autocast_cache_enabled",
    "is_autocast_cpu_enabled",
    "is_autocast_enabled",
    "is_autocast_ipu_enabled",
    "is_autocast_xla_enabled",
    "is_grad_enabled",
    "is_inference_mode_enabled",
    "layout",
    "memory_format",
    "merge_type_from_type_comment",
    "parse_ir",
    "parse_schema",
    "parse_type_comment",
    "qscheme",
    "read_vitals",
    "set_anomaly_enabled",
    "set_autocast_cache_enabled",
    "set_autocast_cpu_dtype",
    "set_autocast_cpu_enabled",
    "set_autocast_dtype",
    "set_autocast_enabled",
    "set_autocast_gpu_dtype",
    "set_autocast_ipu_dtype",
    "set_autocast_ipu_enabled",
    "set_autocast_xla_dtype",
    "set_autocast_xla_enabled",
    "set_flush_denormal",
    "set_num_interop_threads",
    "set_num_threads",
    "set_vital",
    "unify_type_list",
    "vitals_enabled",
    "wait",
    "e",
    "pi",
    "nan",
    "inf",
    "newaxis",
    "abs",
    "abs_",
    "absolute",
    "acos",
    "acos_",
    "acosh",
    "acosh_",
    "adaptive_avg_pool1d",
    "adaptive_max_pool1d",
    "add",
    "addbmm",
    "addcdiv",
    "addcmul",
    "addmm",
    "addmv",
    "addmv_",
    "addr",
    "adjoint",
    "affine_grid_generator",
    "alias_copy",
    "align_tensors",
    "all",
    "allclose",
    "alpha_dropout",
    "alpha_dropout_",
    "amax",
    "amin",
    "aminmax",
    "angle",
    "any",
    "arange",
    "arccos",
    "arccos_",
    "arccosh",
    "arccosh_",
    "arcsin",
    "arcsin_",
    "arcsinh",
    "arcsinh_",
    "arctan",
    "arctan2",
    "arctan_",
    "arctanh",
    "arctanh_",
    "argmax",
    "argmin",
    "argsort",
    "argwhere",
    "as_strided",
    "as_strided_",
    "as_strided_copy",
    "as_strided_scatter",
    "as_tensor",
    "asarray",
    "asin",
    "asin_",
    "asinh",
    "asinh_",
    "atan",
    "atan2",
    "atan_",
    "atanh",
    "atanh_",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "avg_pool1d",
    "baddbmm",
    "bartlett_window",
    "batch_norm",
    "batch_norm_backward_elemt",
    "batch_norm_backward_reduce",
    "batch_norm_elemt",
    "batch_norm_gather_stats",
    "batch_norm_gather_stats_with_counts",
    "batch_norm_stats",
    "batch_norm_update_stats",
    "bernoulli",
    "bilinear",
    "binary_cross_entropy_with_logits",
    "bincount",
    "binomial",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "blackman_window",
    "block_diag",
    "bmm",
    "broadcast_tensors",
    "broadcast_to",
    "bucketize",
    "can_cast",
    "cartesian_prod",
    "cat",
    "ccol_indices_copy",
    "cdist",
    "ceil",
    "ceil_",
    "celu",
    "celu_",
    "chain_matmul",
    "channel_shuffle",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "choose_qparams_optimized",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "clip",
    "clip_",
    "clone",
    "col_indices_copy",
    "column_stack",
    "combinations",
    "complex",
    "concat",
    "concatenate",
    "conj",
    "conj_physical",
    "conj_physical_",
    "constant_pad_nd",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "convolution",
    "copysign",
    "corrcoef",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "cosine_embedding_loss",
    "cosine_similarity",
    "count_nonzero",
    "cov",
    "cross",
    "crow_indices_copy",
    "ctc_loss",
    "cudnn_affine_grid_generator",
    "cudnn_batch_norm",
    "cudnn_convolution",
    "cudnn_convolution_add_relu",
    "cudnn_convolution_relu",
    "cudnn_convolution_transpose",
    "cudnn_grid_sampler",
    "cudnn_is_acceptable",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "cumulative_trapezoid",
    "deg2rad",
    "deg2rad_",
    "dequantize",
    "det",
    "detach",
    "detach_",
    "detach_copy",
    "diag",
    "diag_embed",
    "diagflat",
    "diagonal",
    "diagonal_copy",
    "diagonal_scatter",
    "diff",
    "digamma",
    "dist",
    "div",
    "divide",
    "dot",
    "dropout",
    "dropout_",
    "dsmm",
    "dsplit",
    "dstack",
    "einsum",
    "embedding",
    "embedding_bag",
    "embedding_renorm_",
    "empty",
    "empty_like",
    "empty_permuted",
    "empty_quantized",
    "empty_strided",
    "eq",
    "equal",
    "erf",
    "erf_",
    "erfc",
    "erfc_",
    "erfinv",
    "exp",
    "exp2",
    "exp2_",
    "exp_",
    "expand_copy",
    "expm1",
    "expm1_",
    "eye",
    "fake_quantize_per_channel_affine",
    "fake_quantize_per_tensor_affine",
    "fbgemm_linear_fp16_weight",
    "fbgemm_linear_fp16_weight_fp32_activation",
    "fbgemm_linear_int8_weight",
    "fbgemm_linear_int8_weight_fp32_activation",
    "fbgemm_linear_quantize_weight",
    "fbgemm_pack_gemm_matrix_fp16",
    "fbgemm_pack_quantized_matrix",
    "feature_alpha_dropout",
    "feature_alpha_dropout_",
    "feature_dropout",
    "feature_dropout_",
    "fill",
    "fill_",
    "fix",
    "fix_",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "float_power",
    "floor",
    "floor_",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frac",
    "frac_",
    "frexp",
    "frobenius_norm",
    "from_file",
    "from_numpy",
    "frombuffer",
    "full",
    "full_like",
    "fused_moving_avg_obs_fake_quant",
    "gather",
    "gcd",
    "gcd_",
    "ge",
    "geqrf",
    "ger",
    "get_device",
    "gradient",
    "greater",
    "greater_equal",
    "grid_sampler",
    "grid_sampler_2d",
    "grid_sampler_3d",
    "group_norm",
    "gru",
    "gru_cell",
    "gt",
    "hamming_window",
    "hann_window",
    "hardshrink",
    "hash_tensor",
    "heaviside",
    "hinge_embedding_loss",
    "histc",
    "histogram",
    "histogramdd",
    "hsmm",
    "hsplit",
    "hspmm",
    "hstack",
    "hypot",
    "i0",
    "i0_",
    "igamma",
    "igammac",
    "imag",
    "index_add",
    "index_copy",
    "index_fill",
    "index_put",
    "index_put_",
    "index_reduce",
    "index_select",
    "indices_copy",
    "inner",
    "instance_norm",
    "int_repr",
    "inverse",
    "is_complex",
    "is_conj",
    "is_distributed",
    "is_floating_point",
    "is_inference",
    "is_neg",
    "is_nonzero",
    "is_same_size",
    "is_signed",
    "is_vulkan_available",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "istft",
    "kaiser_window",
    "kl_div",
    "kron",
    "kthvalue",
    "layer_norm",
    "lcm",
    "lcm_",
    "ldexp",
    "ldexp_",
    "le",
    "lerp",
    "less",
    "less_equal",
    "lgamma",
    "linspace",
    "log",
    "log10",
    "log10_",
    "log1p",
    "log1p_",
    "log2",
    "log2_",
    "log_",
    "log_softmax",
    "logaddexp",
    "logaddexp2",
    "logcumsumexp",
    "logdet",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logit",
    "logit_",
    "logspace",
    "logsumexp",
    "lstm",
    "lstm_cell",
    "lt",
    "lu_solve",
    "lu_unpack",
    "margin_ranking_loss",
    "masked_fill",
    "masked_scatter",
    "masked_select",
    "matrix_exp",
    "matrix_power",
    "max",
    "max_pool1d",
    "max_pool1d_with_indices",
    "max_pool2d",
    "max_pool3d",
    "maximum",
    "mean",
    "median",
    "meshgrid",
    "min",
    "minimum",
    "miopen_batch_norm",
    "miopen_convolution",
    "miopen_convolution_add_relu",
    "miopen_convolution_relu",
    "miopen_convolution_transpose",
    "miopen_ctc_loss",
    "miopen_depthwise_convolution",
    "miopen_rnn",
    "mkldnn_adaptive_avg_pool2d",
    "mkldnn_convolution",
    "mkldnn_linear_backward_weights",
    "mkldnn_max_pool2d",
    "mkldnn_max_pool3d",
    "mkldnn_rnn_layer",
    "mm",
    "mode",
    "moveaxis",
    "movedim",
    "msort",
    "mul",
    "multinomial",
    "multiply",
    "mv",
    "mvlgamma",
    "nan_to_num",
    "nan_to_num_",
    "nanmean",
    "nanmedian",
    "nanquantile",
    "nansum",
    "narrow",
    "narrow_copy",
    "native_batch_norm",
    "native_channel_shuffle",
    "native_dropout",
    "native_group_norm",
    "native_layer_norm",
    "native_norm",
    "ne",
    "neg",
    "neg_",
    "negative",
    "negative_",
    "nextafter",
    "nonzero",
    "nonzero_static",
    "norm",
    "norm_except_dim",
    "normal",
    "not_equal",
    "nuclear_norm",
    "numel",
    "ones_like",
    "orgqr",
    "ormqr",
    "outer",
    "pairwise_distance",
    "pdist",
    "permute",
    "permute_copy",
    "pinverse",
    "pixel_shuffle",
    "pixel_unshuffle",
    "poisson",
    "poisson_nll_loss",
    "polar",
    "polygamma",
    "positive",
    "pow",
    "prelu",
    "prod",
    "promote_types",
    "put",
    "q_per_channel_axis",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
    "q_scale",
    "q_zero_point",
    "qr",
    "quantile",
    "quantize_per_channel",
    "quantize_per_tensor",
    "quantize_per_tensor_dynamic",
    "quantized_batch_norm",
    "quantized_gru_cell",
    "quantized_lstm_cell",
    "quantized_max_pool1d",
    "quantized_max_pool2d",
    "quantized_max_pool3d",
    "quantized_rnn_relu_cell",
    "quantized_rnn_tanh_cell",
    "rad2deg",
    "rad2deg_",
    "rand_like",
    "randint",
    "randint_like",
    "randn_like",
    "randperm",
    "range",
    "ravel",
    "real",
    "reciprocal",
    "reciprocal_",
    "relu",
    "relu_",
    "remainder",
    "renorm",
    "repeat_interleave",
    "reshape",
    "resize_as_",
    "resize_as_sparse_",
    "resolve_conj",
    "resolve_neg",
    "result_type",
    "rms_norm",
    "rnn_relu",
    "rnn_relu_cell",
    "rnn_tanh",
    "rnn_tanh_cell",
    "roll",
    "rot90",
    "round",
    "round_",
    "row_indices_copy",
    "row_stack",
    "rrelu",
    "rrelu_",
    "rsqrt",
    "rsqrt_",
    "rsub",
    "saddmm",
    "scalar_tensor",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "searchsorted",
    "select",
    "select_copy",
    "select_scatter",
    "selu",
    "selu_",
    "sgn",
    "sigmoid",
    "sigmoid_",
    "sign",
    "signbit",
    "sin",
    "sin_",
    "sinc",
    "sinc_",
    "sinh",
    "sinh_",
    "slice_copy",
    "slice_inverse",
    "slice_scatter",
    "slogdet",
    "smm",
    "softmax",
    "sort",
    "sparse_bsc_tensor",
    "sparse_bsr_tensor",
    "sparse_compressed_tensor",
    "sparse_coo_tensor",
    "sparse_csc_tensor",
    "sparse_csr_tensor",
    "split_copy",
    "split_with_sizes",
    "split_with_sizes_copy",
    "spmm",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "squeeze",
    "squeeze_copy",
    "sspaddmm",
    "std",
    "std_mean",
    "stft",
    "sub",
    "subtract",
    "sum",
    "svd",
    "swapaxes",
    "swapdims",
    "sym_constrain_range",
    "sym_constrain_range_for_size",
    "t",
    "t_copy",
    "take",
    "take_along_dim",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tensor_split",
    "tensordot",
    "threshold",
    "threshold_",
    "tile",
    "topk",
    "transpose",
    "transpose_copy",
    "trapezoid",
    "trapz",
    "triangular_solve",
    "tril",
    "tril_indices",
    "triplet_margin_loss",
    "triu",
    "triu_indices",
    "true_divide",
    "trunc",
    "trunc_",
    "unbind",
    "unbind_copy",
    "unflatten",
    "unfold_copy",
    "unique_consecutive",
    "unsafe_chunk",
    "unsafe_split",
    "unsafe_split_with_sizes",
    "unsqueeze",
    "unsqueeze_copy",
    "values_copy",
    "vander",
    "var",
    "var_mean",
    "vdot",
    "view_as_complex",
    "view_as_complex_copy",
    "view_as_real",
    "view_as_real_copy",
    "view_copy",
    "vsplit",
    "vstack",
    "where",
    "xlogy",
    "xlogy_",
    "zero_",
    "zeros_like",
    "bfloat16",
    "bit",
    "bits16",
    "bits1x8",
    "bits2x4",
    "bits4x2",
    "bits8",
    "cdouble",
    "cfloat",
    "chalf",
    "complex128",
    "complex32",
    "complex64",
    "double",
    "float",
    "float16",
    "float4_e2m1fn_x2",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    "half",
    "int",
    "int1",
    "int16",
    "int2",
    "int3",
    "int4",
    "int5",
    "int6",
    "int7",
    "int8",
    "long",
    "qint32",
    "qint8",
    "quint2x4",
    "quint4x2",
    "quint8",
    "short",
    "uint1",
    "uint16",
    "uint2",
    "uint3",
    "uint32",
    "uint4",
    "uint5",
    "uint6",
    "uint64",
    "uint7",
    "uint8",
]


def BoolStorage(input_tensor, *args, **kwargs):
    """
    Apply BoolStorage to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the BoolStorage operation.
    """
    from onnx9000.core.ops.torch_auto import BoolStorage as core_BoolStorage

    return core_BoolStorage(input_tensor, *args, **kwargs)


def BoolTensor(input_tensor, *args, **kwargs):
    """
    Apply BoolTensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the BoolTensor operation.
    """
    from onnx9000.core.ops.torch_auto import BoolTensor as core_BoolTensor

    return core_BoolTensor(input_tensor, *args, **kwargs)


def ByteStorage(input_tensor, *args, **kwargs):
    """
    Apply ByteStorage to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ByteStorage operation.
    """
    from onnx9000.core.ops.torch_auto import ByteStorage as core_ByteStorage

    return core_ByteStorage(input_tensor, *args, **kwargs)


def ByteTensor(input_tensor, *args, **kwargs):
    """
    Apply ByteTensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ByteTensor operation.
    """
    from onnx9000.core.ops.torch_auto import ByteTensor as core_ByteTensor

    return core_ByteTensor(input_tensor, *args, **kwargs)


def CharStorage(input_tensor, *args, **kwargs):
    """
    Apply CharStorage to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the CharStorage operation.
    """
    from onnx9000.core.ops.torch_auto import CharStorage as core_CharStorage

    return core_CharStorage(input_tensor, *args, **kwargs)


def CharTensor(input_tensor, *args, **kwargs):
    """
    Apply CharTensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the CharTensor operation.
    """
    from onnx9000.core.ops.torch_auto import CharTensor as core_CharTensor

    return core_CharTensor(input_tensor, *args, **kwargs)


def DoubleStorage(input_tensor, *args, **kwargs):
    """
    Apply DoubleStorage to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the DoubleStorage operation.
    """
    from onnx9000.core.ops.torch_auto import DoubleStorage as core_DoubleStorage

    return core_DoubleStorage(input_tensor, *args, **kwargs)


def DoubleTensor(input_tensor, *args, **kwargs):
    """
    Apply DoubleTensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the DoubleTensor operation.
    """
    from onnx9000.core.ops.torch_auto import DoubleTensor as core_DoubleTensor

    return core_DoubleTensor(input_tensor, *args, **kwargs)


def FloatStorage(input_tensor, *args, **kwargs):
    """
    Apply FloatStorage to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the FloatStorage operation.
    """
    from onnx9000.core.ops.torch_auto import FloatStorage as core_FloatStorage

    return core_FloatStorage(input_tensor, *args, **kwargs)


def FloatTensor(input_tensor, *args, **kwargs):
    """
    Apply FloatTensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the FloatTensor operation.
    """
    from onnx9000.core.ops.torch_auto import FloatTensor as core_FloatTensor

    return core_FloatTensor(input_tensor, *args, **kwargs)


def GradScaler(input_tensor, *args, **kwargs):
    """
    Apply GradScaler to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the GradScaler operation.
    """
    from onnx9000.core.ops.torch_auto import GradScaler as core_GradScaler

    return core_GradScaler(input_tensor, *args, **kwargs)


def IntStorage(input_tensor, *args, **kwargs):
    """
    Apply IntStorage to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the IntStorage operation.
    """
    from onnx9000.core.ops.torch_auto import IntStorage as core_IntStorage

    return core_IntStorage(input_tensor, *args, **kwargs)


def IntTensor(input_tensor, *args, **kwargs):
    """
    Apply IntTensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the IntTensor operation.
    """
    from onnx9000.core.ops.torch_auto import IntTensor as core_IntTensor

    return core_IntTensor(input_tensor, *args, **kwargs)


def LongStorage(input_tensor, *args, **kwargs):
    """
    Apply LongStorage to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the LongStorage operation.
    """
    from onnx9000.core.ops.torch_auto import LongStorage as core_LongStorage

    return core_LongStorage(input_tensor, *args, **kwargs)


def LongTensor(input_tensor, *args, **kwargs):
    """
    Apply LongTensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the LongTensor operation.
    """
    from onnx9000.core.ops.torch_auto import LongTensor as core_LongTensor

    return core_LongTensor(input_tensor, *args, **kwargs)


def ShortStorage(input_tensor, *args, **kwargs):
    """
    Apply ShortStorage to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ShortStorage operation.
    """
    from onnx9000.core.ops.torch_auto import ShortStorage as core_ShortStorage

    return core_ShortStorage(input_tensor, *args, **kwargs)


def ShortTensor(input_tensor, *args, **kwargs):
    """
    Apply ShortTensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ShortTensor operation.
    """
    from onnx9000.core.ops.torch_auto import ShortTensor as core_ShortTensor

    return core_ShortTensor(input_tensor, *args, **kwargs)


def SymBool(input_tensor, *args, **kwargs):
    """
    Apply SymBool to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the SymBool operation.
    """
    from onnx9000.core.ops.torch_auto import SymBool as core_SymBool

    return core_SymBool(input_tensor, *args, **kwargs)


def SymFloat(input_tensor, *args, **kwargs):
    """
    Apply SymFloat to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the SymFloat operation.
    """
    from onnx9000.core.ops.torch_auto import SymFloat as core_SymFloat

    return core_SymFloat(input_tensor, *args, **kwargs)


def SymInt(input_tensor, *args, **kwargs):
    """
    Apply SymInt to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the SymInt operation.
    """
    from onnx9000.core.ops.torch_auto import SymInt as core_SymInt

    return core_SymInt(input_tensor, *args, **kwargs)


def TypedStorage(input_tensor, *args, **kwargs):
    """
    Apply TypedStorage to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the TypedStorage operation.
    """
    from onnx9000.core.ops.torch_auto import TypedStorage as core_TypedStorage

    return core_TypedStorage(input_tensor, *args, **kwargs)


def UntypedStorage(input_tensor, *args, **kwargs):
    """
    Apply UntypedStorage to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the UntypedStorage operation.
    """
    from onnx9000.core.ops.torch_auto import UntypedStorage as core_UntypedStorage

    return core_UntypedStorage(input_tensor, *args, **kwargs)


def are_deterministic_algorithms_enabled(input_tensor, *args, **kwargs):
    """
    Apply are_deterministic_algorithms_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the are_deterministic_algorithms_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import (
        are_deterministic_algorithms_enabled as core_are_deterministic_algorithms_enabled,
    )

    return core_are_deterministic_algorithms_enabled(input_tensor, *args, **kwargs)


def autocast(input_tensor, *args, **kwargs):
    """
    Apply autocast to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the autocast operation.
    """
    from onnx9000.core.ops.torch_auto import autocast as core_autocast

    return core_autocast(input_tensor, *args, **kwargs)


def chunk(input_tensor, *args, **kwargs):
    """
    Apply chunk to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the chunk operation.
    """
    from onnx9000.core.ops.torch_auto import chunk as core_chunk

    return core_chunk(input_tensor, *args, **kwargs)


def compile(input_tensor, *args, **kwargs):
    """
    Apply compile to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the compile operation.
    """
    from onnx9000.core.ops.torch_auto import compile as core_compile

    return core_compile(input_tensor, *args, **kwargs)


def cond(input_tensor, *args, **kwargs):
    """
    Apply cond to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cond operation.
    """
    from onnx9000.core.ops.torch_auto import cond as core_cond

    return core_cond(input_tensor, *args, **kwargs)


def enable_grad(input_tensor, *args, **kwargs):
    """
    Apply enable_grad to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the enable_grad operation.
    """
    from onnx9000.core.ops.torch_auto import enable_grad as core_enable_grad

    return core_enable_grad(input_tensor, *args, **kwargs)


def export_AdditionalInputs(input_tensor, *args, **kwargs):
    """
    Apply export_AdditionalInputs to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_AdditionalInputs operation.
    """
    from onnx9000.core.ops.torch_auto import export_AdditionalInputs as core_export_AdditionalInputs

    return core_export_AdditionalInputs(input_tensor, *args, **kwargs)


def export_Constraint(input_tensor, *args, **kwargs):
    """
    Apply export_Constraint to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_Constraint operation.
    """
    from onnx9000.core.ops.torch_auto import export_Constraint as core_export_Constraint

    return core_export_Constraint(input_tensor, *args, **kwargs)


def export_CustomDecompTable(input_tensor, *args, **kwargs):
    """
    Apply export_CustomDecompTable to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_CustomDecompTable operation.
    """
    from onnx9000.core.ops.torch_auto import (
        export_CustomDecompTable as core_export_CustomDecompTable,
    )

    return core_export_CustomDecompTable(input_tensor, *args, **kwargs)


def export_default_decompositions(input_tensor, *args, **kwargs):
    """
    Apply export_default_decompositions to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_default_decompositions operation.
    """
    from onnx9000.core.ops.torch_auto import (
        export_default_decompositions as core_export_default_decompositions,
    )

    return core_export_default_decompositions(input_tensor, *args, **kwargs)


def export_Dim(input_tensor, *args, **kwargs):
    """
    Apply export_Dim to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_Dim operation.
    """
    from onnx9000.core.ops.torch_auto import export_Dim as core_export_Dim

    return core_export_Dim(input_tensor, *args, **kwargs)


def export_dims(input_tensor, *args, **kwargs):
    """
    Apply export_dims to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_dims operation.
    """
    from onnx9000.core.ops.torch_auto import export_dims as core_export_dims

    return core_export_dims(input_tensor, *args, **kwargs)


def export_draft_export(input_tensor, *args, **kwargs):
    """
    Apply export_draft_export to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_draft_export operation.
    """
    from onnx9000.core.ops.torch_auto import export_draft_export as core_export_draft_export

    return core_export_draft_export(input_tensor, *args, **kwargs)


def export_export(input_tensor, *args, **kwargs):
    """
    Apply export_export to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_export operation.
    """
    from onnx9000.core.ops.torch_auto import export_export as core_export_export

    return core_export_export(input_tensor, *args, **kwargs)


def export_ExportBackwardSignature(input_tensor, *args, **kwargs):
    """
    Apply export_ExportBackwardSignature to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_ExportBackwardSignature operation.
    """
    from onnx9000.core.ops.torch_auto import (
        export_ExportBackwardSignature as core_export_ExportBackwardSignature,
    )

    return core_export_ExportBackwardSignature(input_tensor, *args, **kwargs)


def export_ExportedProgram(input_tensor, *args, **kwargs):
    """
    Apply export_ExportedProgram to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_ExportedProgram operation.
    """
    from onnx9000.core.ops.torch_auto import export_ExportedProgram as core_export_ExportedProgram

    return core_export_ExportedProgram(input_tensor, *args, **kwargs)


def export_ExportGraphSignature(input_tensor, *args, **kwargs):
    """
    Apply export_ExportGraphSignature to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_ExportGraphSignature operation.
    """
    from onnx9000.core.ops.torch_auto import (
        export_ExportGraphSignature as core_export_ExportGraphSignature,
    )

    return core_export_ExportGraphSignature(input_tensor, *args, **kwargs)


def export_FlatArgsAdapter(input_tensor, *args, **kwargs):
    """
    Apply export_FlatArgsAdapter to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_FlatArgsAdapter operation.
    """
    from onnx9000.core.ops.torch_auto import export_FlatArgsAdapter as core_export_FlatArgsAdapter

    return core_export_FlatArgsAdapter(input_tensor, *args, **kwargs)


def export_load(input_tensor, *args, **kwargs):
    """
    Apply export_load to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_load operation.
    """
    from onnx9000.core.ops.torch_auto import export_load as core_export_load

    return core_export_load(input_tensor, *args, **kwargs)


def export_ModuleCallEntry(input_tensor, *args, **kwargs):
    """
    Apply export_ModuleCallEntry to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_ModuleCallEntry operation.
    """
    from onnx9000.core.ops.torch_auto import export_ModuleCallEntry as core_export_ModuleCallEntry

    return core_export_ModuleCallEntry(input_tensor, *args, **kwargs)


def export_ModuleCallSignature(input_tensor, *args, **kwargs):
    """
    Apply export_ModuleCallSignature to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_ModuleCallSignature operation.
    """
    from onnx9000.core.ops.torch_auto import (
        export_ModuleCallSignature as core_export_ModuleCallSignature,
    )

    return core_export_ModuleCallSignature(input_tensor, *args, **kwargs)


def export_register_dataclass(input_tensor, *args, **kwargs):
    """
    Apply export_register_dataclass to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_register_dataclass operation.
    """
    from onnx9000.core.ops.torch_auto import (
        export_register_dataclass as core_export_register_dataclass,
    )

    return core_export_register_dataclass(input_tensor, *args, **kwargs)


def export_save(input_tensor, *args, **kwargs):
    """
    Apply export_save to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_save operation.
    """
    from onnx9000.core.ops.torch_auto import export_save as core_export_save

    return core_export_save(input_tensor, *args, **kwargs)


def export_ShapesCollection(input_tensor, *args, **kwargs):
    """
    Apply export_ShapesCollection to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_ShapesCollection operation.
    """
    from onnx9000.core.ops.torch_auto import export_ShapesCollection as core_export_ShapesCollection

    return core_export_ShapesCollection(input_tensor, *args, **kwargs)


def export_unflatten(input_tensor, *args, **kwargs):
    """
    Apply export_unflatten to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_unflatten operation.
    """
    from onnx9000.core.ops.torch_auto import export_unflatten as core_export_unflatten

    return core_export_unflatten(input_tensor, *args, **kwargs)


def export_UnflattenedModule(input_tensor, *args, **kwargs):
    """
    Apply export_UnflattenedModule to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the export_UnflattenedModule operation.
    """
    from onnx9000.core.ops.torch_auto import (
        export_UnflattenedModule as core_export_UnflattenedModule,
    )

    return core_export_UnflattenedModule(input_tensor, *args, **kwargs)


def get_default_device(input_tensor, *args, **kwargs):
    """
    Apply get_default_device to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_default_device operation.
    """
    from onnx9000.core.ops.torch_auto import get_default_device as core_get_default_device

    return core_get_default_device(input_tensor, *args, **kwargs)


def get_deterministic_debug_mode(input_tensor, *args, **kwargs):
    """
    Apply get_deterministic_debug_mode to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_deterministic_debug_mode operation.
    """
    from onnx9000.core.ops.torch_auto import (
        get_deterministic_debug_mode as core_get_deterministic_debug_mode,
    )

    return core_get_deterministic_debug_mode(input_tensor, *args, **kwargs)


def get_device_module(input_tensor, *args, **kwargs):
    """
    Apply get_device_module to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_device_module operation.
    """
    from onnx9000.core.ops.torch_auto import get_device_module as core_get_device_module

    return core_get_device_module(input_tensor, *args, **kwargs)


def get_float32_matmul_precision(input_tensor, *args, **kwargs):
    """
    Apply get_float32_matmul_precision to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_float32_matmul_precision operation.
    """
    from onnx9000.core.ops.torch_auto import (
        get_float32_matmul_precision as core_get_float32_matmul_precision,
    )

    return core_get_float32_matmul_precision(input_tensor, *args, **kwargs)


def get_rng_state(input_tensor, *args, **kwargs):
    """
    Apply get_rng_state to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_rng_state operation.
    """
    from onnx9000.core.ops.torch_auto import get_rng_state as core_get_rng_state

    return core_get_rng_state(input_tensor, *args, **kwargs)


def inference_mode(input_tensor, *args, **kwargs):
    """
    Apply inference_mode to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the inference_mode operation.
    """
    from onnx9000.core.ops.torch_auto import inference_mode as core_inference_mode

    return core_inference_mode(input_tensor, *args, **kwargs)


def initial_seed(input_tensor, *args, **kwargs):
    """
    Apply initial_seed to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the initial_seed operation.
    """
    from onnx9000.core.ops.torch_auto import initial_seed as core_initial_seed

    return core_initial_seed(input_tensor, *args, **kwargs)


def is_deterministic_algorithms_warn_only_enabled(input_tensor, *args, **kwargs):
    """
    Apply is_deterministic_algorithms_warn_only_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_deterministic_algorithms_warn_only_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import (
        is_deterministic_algorithms_warn_only_enabled as core_is_deterministic_algorithms_warn_only_enabled,
    )

    return core_is_deterministic_algorithms_warn_only_enabled(input_tensor, *args, **kwargs)


def is_storage(input_tensor, *args, **kwargs):
    """
    Apply is_storage to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_storage operation.
    """
    from onnx9000.core.ops.torch_auto import is_storage as core_is_storage

    return core_is_storage(input_tensor, *args, **kwargs)


def is_tensor(input_tensor, *args, **kwargs):
    """
    Apply is_tensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_tensor operation.
    """
    from onnx9000.core.ops.torch_auto import is_tensor as core_is_tensor

    return core_is_tensor(input_tensor, *args, **kwargs)


def is_warn_always_enabled(input_tensor, *args, **kwargs):
    """
    Apply is_warn_always_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_warn_always_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import is_warn_always_enabled as core_is_warn_always_enabled

    return core_is_warn_always_enabled(input_tensor, *args, **kwargs)


def load(input_tensor, *args, **kwargs):
    """
    Apply load to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the load operation.
    """
    from onnx9000.core.ops.torch_auto import load as core_load

    return core_load(input_tensor, *args, **kwargs)


def lobpcg(input_tensor, *args, **kwargs):
    """
    Apply lobpcg to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the lobpcg operation.
    """
    from onnx9000.core.ops.torch_auto import lobpcg as core_lobpcg

    return core_lobpcg(input_tensor, *args, **kwargs)


def manual_seed(input_tensor, *args, **kwargs):
    """
    Apply manual_seed to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the manual_seed operation.
    """
    from onnx9000.core.ops.torch_auto import manual_seed as core_manual_seed

    return core_manual_seed(input_tensor, *args, **kwargs)


def matmul(input_tensor, *args, **kwargs):
    """
    Apply matmul to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the matmul operation.
    """
    from onnx9000.core.ops import matmul as core_matmul

    return core_matmul(input_tensor, *args, **kwargs)


def no_grad(input_tensor, *args, **kwargs):
    """
    Apply no_grad to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the no_grad operation.
    """
    from onnx9000.core.ops.torch_auto import no_grad as core_no_grad

    return core_no_grad(input_tensor, *args, **kwargs)


def rand(input_tensor, *args, **kwargs):
    """
    Apply rand to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rand operation.
    """
    from onnx9000.core.ops.torch_auto import rand as core_rand

    return core_rand(input_tensor, *args, **kwargs)


def save(input_tensor, *args, **kwargs):
    """
    Apply save to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the save operation.
    """
    from onnx9000.core.ops.torch_auto import save as core_save

    return core_save(input_tensor, *args, **kwargs)


def seed(input_tensor, *args, **kwargs):
    """
    Apply seed to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the seed operation.
    """
    from onnx9000.core.ops.torch_auto import seed as core_seed

    return core_seed(input_tensor, *args, **kwargs)


def set_default_device(input_tensor, *args, **kwargs):
    """
    Apply set_default_device to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_default_device operation.
    """
    from onnx9000.core.ops.torch_auto import set_default_device as core_set_default_device

    return core_set_default_device(input_tensor, *args, **kwargs)


def set_default_tensor_type(input_tensor, *args, **kwargs):
    """
    Apply set_default_tensor_type to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_default_tensor_type operation.
    """
    from onnx9000.core.ops.torch_auto import set_default_tensor_type as core_set_default_tensor_type

    return core_set_default_tensor_type(input_tensor, *args, **kwargs)


def set_deterministic_debug_mode(input_tensor, *args, **kwargs):
    """
    Apply set_deterministic_debug_mode to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_deterministic_debug_mode operation.
    """
    from onnx9000.core.ops.torch_auto import (
        set_deterministic_debug_mode as core_set_deterministic_debug_mode,
    )

    return core_set_deterministic_debug_mode(input_tensor, *args, **kwargs)


def set_float32_matmul_precision(input_tensor, *args, **kwargs):
    """
    Apply set_float32_matmul_precision to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_float32_matmul_precision operation.
    """
    from onnx9000.core.ops.torch_auto import (
        set_float32_matmul_precision as core_set_float32_matmul_precision,
    )

    return core_set_float32_matmul_precision(input_tensor, *args, **kwargs)


def set_printoptions(input_tensor, *args, **kwargs):
    """
    Apply set_printoptions to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_printoptions operation.
    """
    from onnx9000.core.ops.torch_auto import set_printoptions as core_set_printoptions

    return core_set_printoptions(input_tensor, *args, **kwargs)


def set_rng_state(input_tensor, *args, **kwargs):
    """
    Apply set_rng_state to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_rng_state operation.
    """
    from onnx9000.core.ops.torch_auto import set_rng_state as core_set_rng_state

    return core_set_rng_state(input_tensor, *args, **kwargs)


def set_warn_always(input_tensor, *args, **kwargs):
    """
    Apply set_warn_always to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_warn_always operation.
    """
    from onnx9000.core.ops.torch_auto import set_warn_always as core_set_warn_always

    return core_set_warn_always(input_tensor, *args, **kwargs)


def split(input_tensor, *args, **kwargs):
    """
    Apply split to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the split operation.
    """
    from onnx9000.core.ops.torch_auto import split as core_split

    return core_split(input_tensor, *args, **kwargs)


def stack(input_tensor, *args, **kwargs):
    """
    Apply stack to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the stack operation.
    """
    from onnx9000.core.ops.torch_auto import stack as core_stack

    return core_stack(input_tensor, *args, **kwargs)


def sym_float(input_tensor, *args, **kwargs):
    """
    Apply sym_float to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sym_float operation.
    """
    from onnx9000.core.ops.torch_auto import sym_float as core_sym_float

    return core_sym_float(input_tensor, *args, **kwargs)


def sym_fresh_size(input_tensor, *args, **kwargs):
    """
    Apply sym_fresh_size to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sym_fresh_size operation.
    """
    from onnx9000.core.ops.torch_auto import sym_fresh_size as core_sym_fresh_size

    return core_sym_fresh_size(input_tensor, *args, **kwargs)


def sym_int(input_tensor, *args, **kwargs):
    """
    Apply sym_int to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sym_int operation.
    """
    from onnx9000.core.ops.torch_auto import sym_int as core_sym_int

    return core_sym_int(input_tensor, *args, **kwargs)


def sym_ite(input_tensor, *args, **kwargs):
    """
    Apply sym_ite to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sym_ite operation.
    """
    from onnx9000.core.ops.torch_auto import sym_ite as core_sym_ite

    return core_sym_ite(input_tensor, *args, **kwargs)


def sym_max(input_tensor, *args, **kwargs):
    """
    Apply sym_max to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sym_max operation.
    """
    from onnx9000.core.ops.torch_auto import sym_max as core_sym_max

    return core_sym_max(input_tensor, *args, **kwargs)


def sym_min(input_tensor, *args, **kwargs):
    """
    Apply sym_min to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sym_min operation.
    """
    from onnx9000.core.ops.torch_auto import sym_min as core_sym_min

    return core_sym_min(input_tensor, *args, **kwargs)


def sym_not(input_tensor, *args, **kwargs):
    """
    Apply sym_not to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sym_not operation.
    """
    from onnx9000.core.ops.torch_auto import sym_not as core_sym_not

    return core_sym_not(input_tensor, *args, **kwargs)


def sym_sum(input_tensor, *args, **kwargs):
    """
    Apply sym_sum to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sym_sum operation.
    """
    from onnx9000.core.ops.torch_auto import sym_sum as core_sym_sum

    return core_sym_sum(input_tensor, *args, **kwargs)


def typename(input_tensor, *args, **kwargs):
    """
    Apply typename to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the typename operation.
    """
    from onnx9000.core.ops.torch_auto import typename as core_typename

    return core_typename(input_tensor, *args, **kwargs)


def unravel_index(input_tensor, *args, **kwargs):
    """
    Apply unravel_index to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unravel_index operation.
    """
    from onnx9000.core.ops.torch_auto import unravel_index as core_unravel_index

    return core_unravel_index(input_tensor, *args, **kwargs)


def use_deterministic_algorithms(input_tensor, *args, **kwargs):
    """
    Apply use_deterministic_algorithms to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the use_deterministic_algorithms operation.
    """
    from onnx9000.core.ops.torch_auto import (
        use_deterministic_algorithms as core_use_deterministic_algorithms,
    )

    return core_use_deterministic_algorithms(input_tensor, *args, **kwargs)


def vmap(input_tensor, *args, **kwargs):
    """
    Apply vmap to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the vmap operation.
    """
    from onnx9000.core.ops.torch_auto import vmap as core_vmap

    return core_vmap(input_tensor, *args, **kwargs)


def sym_sqrt(input_tensor, *args, **kwargs):
    """
    Apply sym_sqrt to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sym_sqrt operation.
    """
    from onnx9000.core.ops.torch_auto import sym_sqrt as core_sym_sqrt

    return core_sym_sqrt(input_tensor, *args, **kwargs)


def AVG(input_tensor, *args, **kwargs):
    """
    Apply AVG to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the AVG operation.
    """
    from onnx9000.core.ops.torch_auto import AVG as core_AVG

    return core_AVG(input_tensor, *args, **kwargs)


def AcceleratorError(input_tensor, *args, **kwargs):
    """
    Apply AcceleratorError to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the AcceleratorError operation.
    """
    from onnx9000.core.ops.torch_auto import AcceleratorError as core_AcceleratorError

    return core_AcceleratorError(input_tensor, *args, **kwargs)


def AggregationType(input_tensor, *args, **kwargs):
    """
    Apply AggregationType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the AggregationType operation.
    """
    from onnx9000.core.ops.torch_auto import AggregationType as core_AggregationType

    return core_AggregationType(input_tensor, *args, **kwargs)


def AliasDb(input_tensor, *args, **kwargs):
    """
    Apply AliasDb to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the AliasDb operation.
    """
    from onnx9000.core.ops.torch_auto import AliasDb as core_AliasDb

    return core_AliasDb(input_tensor, *args, **kwargs)


def AnyType(input_tensor, *args, **kwargs):
    """
    Apply AnyType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the AnyType operation.
    """
    from onnx9000.core.ops.torch_auto import AnyType as core_AnyType

    return core_AnyType(input_tensor, *args, **kwargs)


def Argument(input_tensor, *args, **kwargs):
    """
    Apply Argument to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Argument operation.
    """
    from onnx9000.core.ops.torch_auto import Argument as core_Argument

    return core_Argument(input_tensor, *args, **kwargs)


def ArgumentSpec(input_tensor, *args, **kwargs):
    """
    Apply ArgumentSpec to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ArgumentSpec operation.
    """
    from onnx9000.core.ops.torch_auto import ArgumentSpec as core_ArgumentSpec

    return core_ArgumentSpec(input_tensor, *args, **kwargs)


def AwaitType(input_tensor, *args, **kwargs):
    """
    Apply AwaitType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the AwaitType operation.
    """
    from onnx9000.core.ops.torch_auto import AwaitType as core_AwaitType

    return core_AwaitType(input_tensor, *args, **kwargs)


def BenchmarkConfig(input_tensor, *args, **kwargs):
    """
    Apply BenchmarkConfig to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the BenchmarkConfig operation.
    """
    from onnx9000.core.ops.torch_auto import BenchmarkConfig as core_BenchmarkConfig

    return core_BenchmarkConfig(input_tensor, *args, **kwargs)


def BenchmarkExecutionStats(input_tensor, *args, **kwargs):
    """
    Apply BenchmarkExecutionStats to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the BenchmarkExecutionStats operation.
    """
    from onnx9000.core.ops.torch_auto import BenchmarkExecutionStats as core_BenchmarkExecutionStats

    return core_BenchmarkExecutionStats(input_tensor, *args, **kwargs)


def Block(input_tensor, *args, **kwargs):
    """
    Apply Block to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Block operation.
    """
    from onnx9000.core.ops.torch_auto import Block as core_Block

    return core_Block(input_tensor, *args, **kwargs)


def BoolType(input_tensor, *args, **kwargs):
    """
    Apply BoolType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the BoolType operation.
    """
    from onnx9000.core.ops.torch_auto import BoolType as core_BoolType

    return core_BoolType(input_tensor, *args, **kwargs)


def BufferDict(input_tensor, *args, **kwargs):
    """
    Apply BufferDict to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the BufferDict operation.
    """
    from onnx9000.core.ops.torch_auto import BufferDict as core_BufferDict

    return core_BufferDict(input_tensor, *args, **kwargs)


def CallStack(input_tensor, *args, **kwargs):
    """
    Apply CallStack to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the CallStack operation.
    """
    from onnx9000.core.ops.torch_auto import CallStack as core_CallStack

    return core_CallStack(input_tensor, *args, **kwargs)


def Capsule(input_tensor, *args, **kwargs):
    """
    Apply Capsule to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Capsule operation.
    """
    from onnx9000.core.ops.torch_auto import Capsule as core_Capsule

    return core_Capsule(input_tensor, *args, **kwargs)


def ClassType(input_tensor, *args, **kwargs):
    """
    Apply ClassType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ClassType operation.
    """
    from onnx9000.core.ops.torch_auto import ClassType as core_ClassType

    return core_ClassType(input_tensor, *args, **kwargs)


def Code(input_tensor, *args, **kwargs):
    """
    Apply Code to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Code operation.
    """
    from onnx9000.core.ops.torch_auto import Code as core_Code

    return core_Code(input_tensor, *args, **kwargs)


def CompilationUnit(input_tensor, *args, **kwargs):
    """
    Apply CompilationUnit to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the CompilationUnit operation.
    """
    from onnx9000.core.ops.torch_auto import CompilationUnit as core_CompilationUnit

    return core_CompilationUnit(input_tensor, *args, **kwargs)


def CompleteArgumentSpec(input_tensor, *args, **kwargs):
    """
    Apply CompleteArgumentSpec to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the CompleteArgumentSpec operation.
    """
    from onnx9000.core.ops.torch_auto import CompleteArgumentSpec as core_CompleteArgumentSpec

    return core_CompleteArgumentSpec(input_tensor, *args, **kwargs)


def ComplexType(input_tensor, *args, **kwargs):
    """
    Apply ComplexType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ComplexType operation.
    """
    from onnx9000.core.ops.torch_auto import ComplexType as core_ComplexType

    return core_ComplexType(input_tensor, *args, **kwargs)


def ConcreteModuleType(input_tensor, *args, **kwargs):
    """
    Apply ConcreteModuleType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ConcreteModuleType operation.
    """
    from onnx9000.core.ops.torch_auto import ConcreteModuleType as core_ConcreteModuleType

    return core_ConcreteModuleType(input_tensor, *args, **kwargs)


def ConcreteModuleTypeBuilder(input_tensor, *args, **kwargs):
    """
    Apply ConcreteModuleTypeBuilder to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ConcreteModuleTypeBuilder operation.
    """
    from onnx9000.core.ops.torch_auto import (
        ConcreteModuleTypeBuilder as core_ConcreteModuleTypeBuilder,
    )

    return core_ConcreteModuleTypeBuilder(input_tensor, *args, **kwargs)


def DeepCopyMemoTable(input_tensor, *args, **kwargs):
    """
    Apply DeepCopyMemoTable to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the DeepCopyMemoTable operation.
    """
    from onnx9000.core.ops.torch_auto import DeepCopyMemoTable as core_DeepCopyMemoTable

    return core_DeepCopyMemoTable(input_tensor, *args, **kwargs)


def DeserializationStorageContext(input_tensor, *args, **kwargs):
    """
    Apply DeserializationStorageContext to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the DeserializationStorageContext operation.
    """
    from onnx9000.core.ops.torch_auto import (
        DeserializationStorageContext as core_DeserializationStorageContext,
    )

    return core_DeserializationStorageContext(input_tensor, *args, **kwargs)


def DeviceObjType(input_tensor, *args, **kwargs):
    """
    Apply DeviceObjType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the DeviceObjType operation.
    """
    from onnx9000.core.ops.torch_auto import DeviceObjType as core_DeviceObjType

    return core_DeviceObjType(input_tensor, *args, **kwargs)


def DictType(input_tensor, *args, **kwargs):
    """
    Apply DictType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the DictType operation.
    """
    from onnx9000.core.ops.torch_auto import DictType as core_DictType

    return core_DictType(input_tensor, *args, **kwargs)


def DisableTorchFunction(input_tensor, *args, **kwargs):
    """
    Apply DisableTorchFunction to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the DisableTorchFunction operation.
    """
    from onnx9000.core.ops.torch_auto import DisableTorchFunction as core_DisableTorchFunction

    return core_DisableTorchFunction(input_tensor, *args, **kwargs)


def DisableTorchFunctionSubclass(input_tensor, *args, **kwargs):
    """
    Apply DisableTorchFunctionSubclass to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the DisableTorchFunctionSubclass operation.
    """
    from onnx9000.core.ops.torch_auto import (
        DisableTorchFunctionSubclass as core_DisableTorchFunctionSubclass,
    )

    return core_DisableTorchFunctionSubclass(input_tensor, *args, **kwargs)


def DispatchKey(input_tensor, *args, **kwargs):
    """
    Apply DispatchKey to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the DispatchKey operation.
    """
    from onnx9000.core.ops.torch_auto import DispatchKey as core_DispatchKey

    return core_DispatchKey(input_tensor, *args, **kwargs)


def DispatchKeySet(input_tensor, *args, **kwargs):
    """
    Apply DispatchKeySet to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the DispatchKeySet operation.
    """
    from onnx9000.core.ops.torch_auto import DispatchKeySet as core_DispatchKeySet

    return core_DispatchKeySet(input_tensor, *args, **kwargs)


def EnumType(input_tensor, *args, **kwargs):
    """
    Apply EnumType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the EnumType operation.
    """
    from onnx9000.core.ops.torch_auto import EnumType as core_EnumType

    return core_EnumType(input_tensor, *args, **kwargs)


def ErrorReport(input_tensor, *args, **kwargs):
    """
    Apply ErrorReport to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ErrorReport operation.
    """
    from onnx9000.core.ops.torch_auto import ErrorReport as core_ErrorReport

    return core_ErrorReport(input_tensor, *args, **kwargs)


def Event(input_tensor, *args, **kwargs):
    """
    Apply Event to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Event operation.
    """
    from onnx9000.core.ops.torch_auto import Event as core_Event

    return core_Event(input_tensor, *args, **kwargs)


def ExcludeDispatchKeyGuard(input_tensor, *args, **kwargs):
    """
    Apply ExcludeDispatchKeyGuard to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ExcludeDispatchKeyGuard operation.
    """
    from onnx9000.core.ops.torch_auto import ExcludeDispatchKeyGuard as core_ExcludeDispatchKeyGuard

    return core_ExcludeDispatchKeyGuard(input_tensor, *args, **kwargs)


def ExecutionPlan(input_tensor, *args, **kwargs):
    """
    Apply ExecutionPlan to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ExecutionPlan operation.
    """
    from onnx9000.core.ops.torch_auto import ExecutionPlan as core_ExecutionPlan

    return core_ExecutionPlan(input_tensor, *args, **kwargs)


def FatalError(input_tensor, *args, **kwargs):
    """
    Apply FatalError to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the FatalError operation.
    """
    from onnx9000.core.ops.torch_auto import FatalError as core_FatalError

    return core_FatalError(input_tensor, *args, **kwargs)


def FileCheck(input_tensor, *args, **kwargs):
    """
    Apply FileCheck to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the FileCheck operation.
    """
    from onnx9000.core.ops.torch_auto import FileCheck as core_FileCheck

    return core_FileCheck(input_tensor, *args, **kwargs)


def FloatType(input_tensor, *args, **kwargs):
    """
    Apply FloatType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the FloatType operation.
    """
    from onnx9000.core.ops.torch_auto import FloatType as core_FloatType

    return core_FloatType(input_tensor, *args, **kwargs)


def FunctionSchema(input_tensor, *args, **kwargs):
    """
    Apply FunctionSchema to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the FunctionSchema operation.
    """
    from onnx9000.core.ops.torch_auto import FunctionSchema as core_FunctionSchema

    return core_FunctionSchema(input_tensor, *args, **kwargs)


def Future(input_tensor, *args, **kwargs):
    """
    Apply Future to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Future operation.
    """
    from onnx9000.core.ops.torch_auto import Future as core_Future

    return core_Future(input_tensor, *args, **kwargs)


def FutureType(input_tensor, *args, **kwargs):
    """
    Apply FutureType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the FutureType operation.
    """
    from onnx9000.core.ops.torch_auto import FutureType as core_FutureType

    return core_FutureType(input_tensor, *args, **kwargs)


def Generator(input_tensor, *args, **kwargs):
    """
    Apply Generator to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Generator operation.
    """
    from onnx9000.core.ops.torch_auto import Generator as core_Generator

    return core_Generator(input_tensor, *args, **kwargs)


def Gradient(input_tensor, *args, **kwargs):
    """
    Apply Gradient to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Gradient operation.
    """
    from onnx9000.core.ops.torch_auto import Gradient as core_Gradient

    return core_Gradient(input_tensor, *args, **kwargs)


def Graph(input_tensor, *args, **kwargs):
    """
    Apply Graph to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Graph operation.
    """
    from onnx9000.core.ops.torch_auto import Graph as core_Graph

    return core_Graph(input_tensor, *args, **kwargs)


def GraphExecutorState(input_tensor, *args, **kwargs):
    """
    Apply GraphExecutorState to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the GraphExecutorState operation.
    """
    from onnx9000.core.ops.torch_auto import GraphExecutorState as core_GraphExecutorState

    return core_GraphExecutorState(input_tensor, *args, **kwargs)


def IODescriptor(input_tensor, *args, **kwargs):
    """
    Apply IODescriptor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the IODescriptor operation.
    """
    from onnx9000.core.ops.torch_auto import IODescriptor as core_IODescriptor

    return core_IODescriptor(input_tensor, *args, **kwargs)


def InferredType(input_tensor, *args, **kwargs):
    """
    Apply InferredType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the InferredType operation.
    """
    from onnx9000.core.ops.torch_auto import InferredType as core_InferredType

    return core_InferredType(input_tensor, *args, **kwargs)


def IntType(input_tensor, *args, **kwargs):
    """
    Apply IntType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the IntType operation.
    """
    from onnx9000.core.ops.torch_auto import IntType as core_IntType

    return core_IntType(input_tensor, *args, **kwargs)


def InterfaceType(input_tensor, *args, **kwargs):
    """
    Apply InterfaceType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the InterfaceType operation.
    """
    from onnx9000.core.ops.torch_auto import InterfaceType as core_InterfaceType

    return core_InterfaceType(input_tensor, *args, **kwargs)


def JITException(input_tensor, *args, **kwargs):
    """
    Apply JITException to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the JITException operation.
    """
    from onnx9000.core.ops.torch_auto import JITException as core_JITException

    return core_JITException(input_tensor, *args, **kwargs)


def ListType(input_tensor, *args, **kwargs):
    """
    Apply ListType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ListType operation.
    """
    from onnx9000.core.ops.torch_auto import ListType as core_ListType

    return core_ListType(input_tensor, *args, **kwargs)


def LiteScriptModule(input_tensor, *args, **kwargs):
    """
    Apply LiteScriptModule to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the LiteScriptModule operation.
    """
    from onnx9000.core.ops.torch_auto import LiteScriptModule as core_LiteScriptModule

    return core_LiteScriptModule(input_tensor, *args, **kwargs)


def LockingLogger(input_tensor, *args, **kwargs):
    """
    Apply LockingLogger to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the LockingLogger operation.
    """
    from onnx9000.core.ops.torch_auto import LockingLogger as core_LockingLogger

    return core_LockingLogger(input_tensor, *args, **kwargs)


def ModuleDict(input_tensor, *args, **kwargs):
    """
    Apply ModuleDict to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ModuleDict operation.
    """
    from onnx9000.core.ops.torch_auto import ModuleDict as core_ModuleDict

    return core_ModuleDict(input_tensor, *args, **kwargs)


def Node(input_tensor, *args, **kwargs):
    """
    Apply Node to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Node operation.
    """
    from onnx9000.core.ops.torch_auto import Node as core_Node

    return core_Node(input_tensor, *args, **kwargs)


def NoneType(input_tensor, *args, **kwargs):
    """
    Apply NoneType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the NoneType operation.
    """
    from onnx9000.core.ops.torch_auto import NoneType as core_NoneType

    return core_NoneType(input_tensor, *args, **kwargs)


def NoopLogger(input_tensor, *args, **kwargs):
    """
    Apply NoopLogger to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the NoopLogger operation.
    """
    from onnx9000.core.ops.torch_auto import NoopLogger as core_NoopLogger

    return core_NoopLogger(input_tensor, *args, **kwargs)


def NumberType(input_tensor, *args, **kwargs):
    """
    Apply NumberType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the NumberType operation.
    """
    from onnx9000.core.ops.torch_auto import NumberType as core_NumberType

    return core_NumberType(input_tensor, *args, **kwargs)


def OperatorInfo(input_tensor, *args, **kwargs):
    """
    Apply OperatorInfo to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the OperatorInfo operation.
    """
    from onnx9000.core.ops.torch_auto import OperatorInfo as core_OperatorInfo

    return core_OperatorInfo(input_tensor, *args, **kwargs)


def OptionalType(input_tensor, *args, **kwargs):
    """
    Apply OptionalType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the OptionalType operation.
    """
    from onnx9000.core.ops.torch_auto import OptionalType as core_OptionalType

    return core_OptionalType(input_tensor, *args, **kwargs)


def OutOfMemoryError(input_tensor, *args, **kwargs):
    """
    Apply OutOfMemoryError to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the OutOfMemoryError operation.
    """
    from onnx9000.core.ops.torch_auto import OutOfMemoryError as core_OutOfMemoryError

    return core_OutOfMemoryError(input_tensor, *args, **kwargs)


def ParameterDict(input_tensor, *args, **kwargs):
    """
    Apply ParameterDict to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ParameterDict operation.
    """
    from onnx9000.core.ops.torch_auto import ParameterDict as core_ParameterDict

    return core_ParameterDict(input_tensor, *args, **kwargs)


def PyObjectType(input_tensor, *args, **kwargs):
    """
    Apply PyObjectType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the PyObjectType operation.
    """
    from onnx9000.core.ops.torch_auto import PyObjectType as core_PyObjectType

    return core_PyObjectType(input_tensor, *args, **kwargs)


def PyTorchFileReader(input_tensor, *args, **kwargs):
    """
    Apply PyTorchFileReader to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the PyTorchFileReader operation.
    """
    from onnx9000.core.ops.torch_auto import PyTorchFileReader as core_PyTorchFileReader

    return core_PyTorchFileReader(input_tensor, *args, **kwargs)


def PyTorchFileWriter(input_tensor, *args, **kwargs):
    """
    Apply PyTorchFileWriter to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the PyTorchFileWriter operation.
    """
    from onnx9000.core.ops.torch_auto import PyTorchFileWriter as core_PyTorchFileWriter

    return core_PyTorchFileWriter(input_tensor, *args, **kwargs)


def RRefType(input_tensor, *args, **kwargs):
    """
    Apply RRefType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the RRefType operation.
    """
    from onnx9000.core.ops.torch_auto import RRefType as core_RRefType

    return core_RRefType(input_tensor, *args, **kwargs)


def SUM(input_tensor, *args, **kwargs):
    """
    Apply SUM to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the SUM operation.
    """
    from onnx9000.core.ops.torch_auto import SUM as core_SUM

    return core_SUM(input_tensor, *args, **kwargs)


def ScriptClass(input_tensor, *args, **kwargs):
    """
    Apply ScriptClass to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptClass operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptClass as core_ScriptClass

    return core_ScriptClass(input_tensor, *args, **kwargs)


def ScriptClassFunction(input_tensor, *args, **kwargs):
    """
    Apply ScriptClassFunction to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptClassFunction operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptClassFunction as core_ScriptClassFunction

    return core_ScriptClassFunction(input_tensor, *args, **kwargs)


def ScriptDict(input_tensor, *args, **kwargs):
    """
    Apply ScriptDict to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptDict operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptDict as core_ScriptDict

    return core_ScriptDict(input_tensor, *args, **kwargs)


def ScriptDictIterator(input_tensor, *args, **kwargs):
    """
    Apply ScriptDictIterator to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptDictIterator operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptDictIterator as core_ScriptDictIterator

    return core_ScriptDictIterator(input_tensor, *args, **kwargs)


def ScriptDictKeyIterator(input_tensor, *args, **kwargs):
    """
    Apply ScriptDictKeyIterator to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptDictKeyIterator operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptDictKeyIterator as core_ScriptDictKeyIterator

    return core_ScriptDictKeyIterator(input_tensor, *args, **kwargs)


def ScriptFunction(input_tensor, *args, **kwargs):
    """
    Apply ScriptFunction to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptFunction operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptFunction as core_ScriptFunction

    return core_ScriptFunction(input_tensor, *args, **kwargs)


def ScriptList(input_tensor, *args, **kwargs):
    """
    Apply ScriptList to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptList operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptList as core_ScriptList

    return core_ScriptList(input_tensor, *args, **kwargs)


def ScriptListIterator(input_tensor, *args, **kwargs):
    """
    Apply ScriptListIterator to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptListIterator operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptListIterator as core_ScriptListIterator

    return core_ScriptListIterator(input_tensor, *args, **kwargs)


def ScriptMethod(input_tensor, *args, **kwargs):
    """
    Apply ScriptMethod to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptMethod operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptMethod as core_ScriptMethod

    return core_ScriptMethod(input_tensor, *args, **kwargs)


def ScriptModule(input_tensor, *args, **kwargs):
    """
    Apply ScriptModule to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptModule operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptModule as core_ScriptModule

    return core_ScriptModule(input_tensor, *args, **kwargs)


def ScriptModuleSerializer(input_tensor, *args, **kwargs):
    """
    Apply ScriptModuleSerializer to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptModuleSerializer operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptModuleSerializer as core_ScriptModuleSerializer

    return core_ScriptModuleSerializer(input_tensor, *args, **kwargs)


def ScriptObject(input_tensor, *args, **kwargs):
    """
    Apply ScriptObject to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptObject operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptObject as core_ScriptObject

    return core_ScriptObject(input_tensor, *args, **kwargs)


def ScriptObjectProperty(input_tensor, *args, **kwargs):
    """
    Apply ScriptObjectProperty to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ScriptObjectProperty operation.
    """
    from onnx9000.core.ops.torch_auto import ScriptObjectProperty as core_ScriptObjectProperty

    return core_ScriptObjectProperty(input_tensor, *args, **kwargs)


def SerializationStorageContext(input_tensor, *args, **kwargs):
    """
    Apply SerializationStorageContext to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the SerializationStorageContext operation.
    """
    from onnx9000.core.ops.torch_auto import (
        SerializationStorageContext as core_SerializationStorageContext,
    )

    return core_SerializationStorageContext(input_tensor, *args, **kwargs)


def Size(input_tensor, *args, **kwargs):
    """
    Apply Size to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Size operation.
    """
    from onnx9000.core.ops.torch_auto import Size as core_Size

    return core_Size(input_tensor, *args, **kwargs)


def StaticModule(input_tensor, *args, **kwargs):
    """
    Apply StaticModule to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the StaticModule operation.
    """
    from onnx9000.core.ops.torch_auto import StaticModule as core_StaticModule

    return core_StaticModule(input_tensor, *args, **kwargs)


def Stream(input_tensor, *args, **kwargs):
    """
    Apply Stream to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Stream operation.
    """
    from onnx9000.core.ops.torch_auto import Stream as core_Stream

    return core_Stream(input_tensor, *args, **kwargs)


def StreamObjType(input_tensor, *args, **kwargs):
    """
    Apply StreamObjType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the StreamObjType operation.
    """
    from onnx9000.core.ops.torch_auto import StreamObjType as core_StreamObjType

    return core_StreamObjType(input_tensor, *args, **kwargs)


def StringType(input_tensor, *args, **kwargs):
    """
    Apply StringType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the StringType operation.
    """
    from onnx9000.core.ops.torch_auto import StringType as core_StringType

    return core_StringType(input_tensor, *args, **kwargs)


def SymBoolType(input_tensor, *args, **kwargs):
    """
    Apply SymBoolType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the SymBoolType operation.
    """
    from onnx9000.core.ops.torch_auto import SymBoolType as core_SymBoolType

    return core_SymBoolType(input_tensor, *args, **kwargs)


def SymIntType(input_tensor, *args, **kwargs):
    """
    Apply SymIntType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the SymIntType operation.
    """
    from onnx9000.core.ops.torch_auto import SymIntType as core_SymIntType

    return core_SymIntType(input_tensor, *args, **kwargs)


def Tag(input_tensor, *args, **kwargs):
    """
    Apply Tag to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Tag operation.
    """
    from onnx9000.core.ops.torch_auto import Tag as core_Tag

    return core_Tag(input_tensor, *args, **kwargs)


def TensorType(input_tensor, *args, **kwargs):
    """
    Apply TensorType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the TensorType operation.
    """
    from onnx9000.core.ops.torch_auto import TensorType as core_TensorType

    return core_TensorType(input_tensor, *args, **kwargs)


def ThroughputBenchmark(input_tensor, *args, **kwargs):
    """
    Apply ThroughputBenchmark to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ThroughputBenchmark operation.
    """
    from onnx9000.core.ops.torch_auto import ThroughputBenchmark as core_ThroughputBenchmark

    return core_ThroughputBenchmark(input_tensor, *args, **kwargs)


def TracingState(input_tensor, *args, **kwargs):
    """
    Apply TracingState to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the TracingState operation.
    """
    from onnx9000.core.ops.torch_auto import TracingState as core_TracingState

    return core_TracingState(input_tensor, *args, **kwargs)


def TupleType(input_tensor, *args, **kwargs):
    """
    Apply TupleType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the TupleType operation.
    """
    from onnx9000.core.ops.torch_auto import TupleType as core_TupleType

    return core_TupleType(input_tensor, *args, **kwargs)


def Type(input_tensor, *args, **kwargs):
    """
    Apply Type to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Type operation.
    """
    from onnx9000.core.ops.torch_auto import Type as core_Type

    return core_Type(input_tensor, *args, **kwargs)


def UnionType(input_tensor, *args, **kwargs):
    """
    Apply UnionType to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the UnionType operation.
    """
    from onnx9000.core.ops.torch_auto import UnionType as core_UnionType

    return core_UnionType(input_tensor, *args, **kwargs)


def Use(input_tensor, *args, **kwargs):
    """
    Apply Use to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Use operation.
    """
    from onnx9000.core.ops.torch_auto import Use as core_Use

    return core_Use(input_tensor, *args, **kwargs)


def Value(input_tensor, *args, **kwargs):
    """
    Apply Value to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the Value operation.
    """
    from onnx9000.core.ops.torch_auto import Value as core_Value

    return core_Value(input_tensor, *args, **kwargs)


def autocast_decrement_nesting(input_tensor, *args, **kwargs):
    """
    Apply autocast_decrement_nesting to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the autocast_decrement_nesting operation.
    """
    from onnx9000.core.ops.torch_auto import (
        autocast_decrement_nesting as core_autocast_decrement_nesting,
    )

    return core_autocast_decrement_nesting(input_tensor, *args, **kwargs)


def autocast_increment_nesting(input_tensor, *args, **kwargs):
    """
    Apply autocast_increment_nesting to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the autocast_increment_nesting operation.
    """
    from onnx9000.core.ops.torch_auto import (
        autocast_increment_nesting as core_autocast_increment_nesting,
    )

    return core_autocast_increment_nesting(input_tensor, *args, **kwargs)


def clear_autocast_cache(input_tensor, *args, **kwargs):
    """
    Apply clear_autocast_cache to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the clear_autocast_cache operation.
    """
    from onnx9000.core.ops.torch_auto import clear_autocast_cache as core_clear_autocast_cache

    return core_clear_autocast_cache(input_tensor, *args, **kwargs)


def cpp_OrderedModuleDict(input_tensor, *args, **kwargs):
    """
    Apply cpp_OrderedModuleDict to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cpp_OrderedModuleDict operation.
    """
    from onnx9000.core.ops.torch_auto import cpp_OrderedModuleDict as core_cpp_OrderedModuleDict

    return core_cpp_OrderedModuleDict(input_tensor, *args, **kwargs)


def cpp_OrderedTensorDict(input_tensor, *args, **kwargs):
    """
    Apply cpp_OrderedTensorDict to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cpp_OrderedTensorDict operation.
    """
    from onnx9000.core.ops.torch_auto import cpp_OrderedTensorDict as core_cpp_OrderedTensorDict

    return core_cpp_OrderedTensorDict(input_tensor, *args, **kwargs)


def cpp_nn_Module(input_tensor, *args, **kwargs):
    """
    Apply cpp_nn_Module to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cpp_nn_Module operation.
    """
    from onnx9000.core.ops.torch_auto import cpp_nn_Module as core_cpp_nn_Module

    return core_cpp_nn_Module(input_tensor, *args, **kwargs)


def default_generator(input_tensor, *args, **kwargs):
    """
    Apply default_generator to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the default_generator operation.
    """
    from onnx9000.core.ops.torch_auto import default_generator as core_default_generator

    return core_default_generator(input_tensor, *args, **kwargs)


def device(input_tensor, *args, **kwargs):
    """
    Apply device to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the device operation.
    """
    from onnx9000.core.ops.torch_auto import device as core_device

    return core_device(input_tensor, *args, **kwargs)


def dtype(input_tensor, *args, **kwargs):
    """
    Apply dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the dtype operation.
    """
    from onnx9000.core.ops.torch_auto import dtype as core_dtype

    return core_dtype(input_tensor, *args, **kwargs)


def finfo(input_tensor, *args, **kwargs):
    """
    Apply finfo to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the finfo operation.
    """
    from onnx9000.core.ops.torch_auto import finfo as core_finfo

    return core_finfo(input_tensor, *args, **kwargs)


def fork(input_tensor, *args, **kwargs):
    """
    Apply fork to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fork operation.
    """
    from onnx9000.core.ops.torch_auto import fork as core_fork

    return core_fork(input_tensor, *args, **kwargs)


def get_autocast_cpu_dtype(input_tensor, *args, **kwargs):
    """
    Apply get_autocast_cpu_dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_autocast_cpu_dtype operation.
    """
    from onnx9000.core.ops.torch_auto import get_autocast_cpu_dtype as core_get_autocast_cpu_dtype

    return core_get_autocast_cpu_dtype(input_tensor, *args, **kwargs)


def get_autocast_dtype(input_tensor, *args, **kwargs):
    """
    Apply get_autocast_dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_autocast_dtype operation.
    """
    from onnx9000.core.ops.torch_auto import get_autocast_dtype as core_get_autocast_dtype

    return core_get_autocast_dtype(input_tensor, *args, **kwargs)


def get_autocast_gpu_dtype(input_tensor, *args, **kwargs):
    """
    Apply get_autocast_gpu_dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_autocast_gpu_dtype operation.
    """
    from onnx9000.core.ops.torch_auto import get_autocast_gpu_dtype as core_get_autocast_gpu_dtype

    return core_get_autocast_gpu_dtype(input_tensor, *args, **kwargs)


def get_autocast_ipu_dtype(input_tensor, *args, **kwargs):
    """
    Apply get_autocast_ipu_dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_autocast_ipu_dtype operation.
    """
    from onnx9000.core.ops.torch_auto import get_autocast_ipu_dtype as core_get_autocast_ipu_dtype

    return core_get_autocast_ipu_dtype(input_tensor, *args, **kwargs)


def get_autocast_xla_dtype(input_tensor, *args, **kwargs):
    """
    Apply get_autocast_xla_dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_autocast_xla_dtype operation.
    """
    from onnx9000.core.ops.torch_auto import get_autocast_xla_dtype as core_get_autocast_xla_dtype

    return core_get_autocast_xla_dtype(input_tensor, *args, **kwargs)


def get_default_dtype(input_tensor, *args, **kwargs):
    """
    Apply get_default_dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_default_dtype operation.
    """
    from onnx9000.core.ops.torch_auto import get_default_dtype as core_get_default_dtype

    return core_get_default_dtype(input_tensor, *args, **kwargs)


def get_num_interop_threads(input_tensor, *args, **kwargs):
    """
    Apply get_num_interop_threads to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_num_interop_threads operation.
    """
    from onnx9000.core.ops.torch_auto import get_num_interop_threads as core_get_num_interop_threads

    return core_get_num_interop_threads(input_tensor, *args, **kwargs)


def get_num_threads(input_tensor, *args, **kwargs):
    """
    Apply get_num_threads to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_num_threads operation.
    """
    from onnx9000.core.ops.torch_auto import get_num_threads as core_get_num_threads

    return core_get_num_threads(input_tensor, *args, **kwargs)


def has_lapack(input_tensor, *args, **kwargs):
    """
    Apply has_lapack to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the has_lapack operation.
    """
    from onnx9000.core.ops.torch_auto import has_lapack as core_has_lapack

    return core_has_lapack(input_tensor, *args, **kwargs)


def has_mkl(input_tensor, *args, **kwargs):
    """
    Apply has_mkl to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the has_mkl operation.
    """
    from onnx9000.core.ops.torch_auto import has_mkl as core_has_mkl

    return core_has_mkl(input_tensor, *args, **kwargs)


def has_openmp(input_tensor, *args, **kwargs):
    """
    Apply has_openmp to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the has_openmp operation.
    """
    from onnx9000.core.ops.torch_auto import has_openmp as core_has_openmp

    return core_has_openmp(input_tensor, *args, **kwargs)


def has_spectral(input_tensor, *args, **kwargs):
    """
    Apply has_spectral to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the has_spectral operation.
    """
    from onnx9000.core.ops.torch_auto import has_spectral as core_has_spectral

    return core_has_spectral(input_tensor, *args, **kwargs)


def iinfo(input_tensor, *args, **kwargs):
    """
    Apply iinfo to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the iinfo operation.
    """
    from onnx9000.core.ops.torch_auto import iinfo as core_iinfo

    return core_iinfo(input_tensor, *args, **kwargs)


def import_ir_module(input_tensor, *args, **kwargs):
    """
    Apply import_ir_module to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the import_ir_module operation.
    """
    from onnx9000.core.ops.torch_auto import import_ir_module as core_import_ir_module

    return core_import_ir_module(input_tensor, *args, **kwargs)


def import_ir_module_from_buffer(input_tensor, *args, **kwargs):
    """
    Apply import_ir_module_from_buffer to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the import_ir_module_from_buffer operation.
    """
    from onnx9000.core.ops.torch_auto import (
        import_ir_module_from_buffer as core_import_ir_module_from_buffer,
    )

    return core_import_ir_module_from_buffer(input_tensor, *args, **kwargs)


def init_num_threads(input_tensor, *args, **kwargs):
    """
    Apply init_num_threads to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the init_num_threads operation.
    """
    from onnx9000.core.ops.torch_auto import init_num_threads as core_init_num_threads

    return core_init_num_threads(input_tensor, *args, **kwargs)


def is_anomaly_check_nan_enabled(input_tensor, *args, **kwargs):
    """
    Apply is_anomaly_check_nan_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_anomaly_check_nan_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import (
        is_anomaly_check_nan_enabled as core_is_anomaly_check_nan_enabled,
    )

    return core_is_anomaly_check_nan_enabled(input_tensor, *args, **kwargs)


def is_anomaly_enabled(input_tensor, *args, **kwargs):
    """
    Apply is_anomaly_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_anomaly_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import is_anomaly_enabled as core_is_anomaly_enabled

    return core_is_anomaly_enabled(input_tensor, *args, **kwargs)


def is_autocast_cache_enabled(input_tensor, *args, **kwargs):
    """
    Apply is_autocast_cache_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_autocast_cache_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import (
        is_autocast_cache_enabled as core_is_autocast_cache_enabled,
    )

    return core_is_autocast_cache_enabled(input_tensor, *args, **kwargs)


def is_autocast_cpu_enabled(input_tensor, *args, **kwargs):
    """
    Apply is_autocast_cpu_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_autocast_cpu_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import is_autocast_cpu_enabled as core_is_autocast_cpu_enabled

    return core_is_autocast_cpu_enabled(input_tensor, *args, **kwargs)


def is_autocast_enabled(input_tensor, *args, **kwargs):
    """
    Apply is_autocast_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_autocast_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import is_autocast_enabled as core_is_autocast_enabled

    return core_is_autocast_enabled(input_tensor, *args, **kwargs)


def is_autocast_ipu_enabled(input_tensor, *args, **kwargs):
    """
    Apply is_autocast_ipu_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_autocast_ipu_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import is_autocast_ipu_enabled as core_is_autocast_ipu_enabled

    return core_is_autocast_ipu_enabled(input_tensor, *args, **kwargs)


def is_autocast_xla_enabled(input_tensor, *args, **kwargs):
    """
    Apply is_autocast_xla_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_autocast_xla_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import is_autocast_xla_enabled as core_is_autocast_xla_enabled

    return core_is_autocast_xla_enabled(input_tensor, *args, **kwargs)


def is_grad_enabled(input_tensor, *args, **kwargs):
    """
    Apply is_grad_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_grad_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import is_grad_enabled as core_is_grad_enabled

    return core_is_grad_enabled(input_tensor, *args, **kwargs)


def is_inference_mode_enabled(input_tensor, *args, **kwargs):
    """
    Apply is_inference_mode_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_inference_mode_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import (
        is_inference_mode_enabled as core_is_inference_mode_enabled,
    )

    return core_is_inference_mode_enabled(input_tensor, *args, **kwargs)


def layout(input_tensor, *args, **kwargs):
    """
    Apply layout to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the layout operation.
    """
    from onnx9000.core.ops.torch_auto import layout as core_layout

    return core_layout(input_tensor, *args, **kwargs)


def memory_format(input_tensor, *args, **kwargs):
    """
    Apply memory_format to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the memory_format operation.
    """
    from onnx9000.core.ops.torch_auto import memory_format as core_memory_format

    return core_memory_format(input_tensor, *args, **kwargs)


def merge_type_from_type_comment(input_tensor, *args, **kwargs):
    """
    Apply merge_type_from_type_comment to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the merge_type_from_type_comment operation.
    """
    from onnx9000.core.ops.torch_auto import (
        merge_type_from_type_comment as core_merge_type_from_type_comment,
    )

    return core_merge_type_from_type_comment(input_tensor, *args, **kwargs)


def parse_ir(input_tensor, *args, **kwargs):
    """
    Apply parse_ir to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the parse_ir operation.
    """
    from onnx9000.core.ops.torch_auto import parse_ir as core_parse_ir

    return core_parse_ir(input_tensor, *args, **kwargs)


def parse_schema(input_tensor, *args, **kwargs):
    """
    Apply parse_schema to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the parse_schema operation.
    """
    from onnx9000.core.ops.torch_auto import parse_schema as core_parse_schema

    return core_parse_schema(input_tensor, *args, **kwargs)


def parse_type_comment(input_tensor, *args, **kwargs):
    """
    Apply parse_type_comment to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the parse_type_comment operation.
    """
    from onnx9000.core.ops.torch_auto import parse_type_comment as core_parse_type_comment

    return core_parse_type_comment(input_tensor, *args, **kwargs)


def qscheme(input_tensor, *args, **kwargs):
    """
    Apply qscheme to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the qscheme operation.
    """
    from onnx9000.core.ops.torch_auto import qscheme as core_qscheme

    return core_qscheme(input_tensor, *args, **kwargs)


def read_vitals(input_tensor, *args, **kwargs):
    """
    Apply read_vitals to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the read_vitals operation.
    """
    from onnx9000.core.ops.torch_auto import read_vitals as core_read_vitals

    return core_read_vitals(input_tensor, *args, **kwargs)


def set_anomaly_enabled(input_tensor, *args, **kwargs):
    """
    Apply set_anomaly_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_anomaly_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import set_anomaly_enabled as core_set_anomaly_enabled

    return core_set_anomaly_enabled(input_tensor, *args, **kwargs)


def set_autocast_cache_enabled(input_tensor, *args, **kwargs):
    """
    Apply set_autocast_cache_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_autocast_cache_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import (
        set_autocast_cache_enabled as core_set_autocast_cache_enabled,
    )

    return core_set_autocast_cache_enabled(input_tensor, *args, **kwargs)


def set_autocast_cpu_dtype(input_tensor, *args, **kwargs):
    """
    Apply set_autocast_cpu_dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_autocast_cpu_dtype operation.
    """
    from onnx9000.core.ops.torch_auto import set_autocast_cpu_dtype as core_set_autocast_cpu_dtype

    return core_set_autocast_cpu_dtype(input_tensor, *args, **kwargs)


def set_autocast_cpu_enabled(input_tensor, *args, **kwargs):
    """
    Apply set_autocast_cpu_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_autocast_cpu_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import (
        set_autocast_cpu_enabled as core_set_autocast_cpu_enabled,
    )

    return core_set_autocast_cpu_enabled(input_tensor, *args, **kwargs)


def set_autocast_dtype(input_tensor, *args, **kwargs):
    """
    Apply set_autocast_dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_autocast_dtype operation.
    """
    from onnx9000.core.ops.torch_auto import set_autocast_dtype as core_set_autocast_dtype

    return core_set_autocast_dtype(input_tensor, *args, **kwargs)


def set_autocast_enabled(input_tensor, *args, **kwargs):
    """
    Apply set_autocast_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_autocast_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import set_autocast_enabled as core_set_autocast_enabled

    return core_set_autocast_enabled(input_tensor, *args, **kwargs)


def set_autocast_gpu_dtype(input_tensor, *args, **kwargs):
    """
    Apply set_autocast_gpu_dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_autocast_gpu_dtype operation.
    """
    from onnx9000.core.ops.torch_auto import set_autocast_gpu_dtype as core_set_autocast_gpu_dtype

    return core_set_autocast_gpu_dtype(input_tensor, *args, **kwargs)


def set_autocast_ipu_dtype(input_tensor, *args, **kwargs):
    """
    Apply set_autocast_ipu_dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_autocast_ipu_dtype operation.
    """
    from onnx9000.core.ops.torch_auto import set_autocast_ipu_dtype as core_set_autocast_ipu_dtype

    return core_set_autocast_ipu_dtype(input_tensor, *args, **kwargs)


def set_autocast_ipu_enabled(input_tensor, *args, **kwargs):
    """
    Apply set_autocast_ipu_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_autocast_ipu_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import (
        set_autocast_ipu_enabled as core_set_autocast_ipu_enabled,
    )

    return core_set_autocast_ipu_enabled(input_tensor, *args, **kwargs)


def set_autocast_xla_dtype(input_tensor, *args, **kwargs):
    """
    Apply set_autocast_xla_dtype to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_autocast_xla_dtype operation.
    """
    from onnx9000.core.ops.torch_auto import set_autocast_xla_dtype as core_set_autocast_xla_dtype

    return core_set_autocast_xla_dtype(input_tensor, *args, **kwargs)


def set_autocast_xla_enabled(input_tensor, *args, **kwargs):
    """
    Apply set_autocast_xla_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_autocast_xla_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import (
        set_autocast_xla_enabled as core_set_autocast_xla_enabled,
    )

    return core_set_autocast_xla_enabled(input_tensor, *args, **kwargs)


def set_flush_denormal(input_tensor, *args, **kwargs):
    """
    Apply set_flush_denormal to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_flush_denormal operation.
    """
    from onnx9000.core.ops.torch_auto import set_flush_denormal as core_set_flush_denormal

    return core_set_flush_denormal(input_tensor, *args, **kwargs)


def set_num_interop_threads(input_tensor, *args, **kwargs):
    """
    Apply set_num_interop_threads to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_num_interop_threads operation.
    """
    from onnx9000.core.ops.torch_auto import set_num_interop_threads as core_set_num_interop_threads

    return core_set_num_interop_threads(input_tensor, *args, **kwargs)


def set_num_threads(input_tensor, *args, **kwargs):
    """
    Apply set_num_threads to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_num_threads operation.
    """
    from onnx9000.core.ops.torch_auto import set_num_threads as core_set_num_threads

    return core_set_num_threads(input_tensor, *args, **kwargs)


def set_vital(input_tensor, *args, **kwargs):
    """
    Apply set_vital to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the set_vital operation.
    """
    from onnx9000.core.ops.torch_auto import set_vital as core_set_vital

    return core_set_vital(input_tensor, *args, **kwargs)


def unify_type_list(input_tensor, *args, **kwargs):
    """
    Apply unify_type_list to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unify_type_list operation.
    """
    from onnx9000.core.ops.torch_auto import unify_type_list as core_unify_type_list

    return core_unify_type_list(input_tensor, *args, **kwargs)


def vitals_enabled(input_tensor, *args, **kwargs):
    """
    Apply vitals_enabled to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the vitals_enabled operation.
    """
    from onnx9000.core.ops.torch_auto import vitals_enabled as core_vitals_enabled

    return core_vitals_enabled(input_tensor, *args, **kwargs)


def wait(input_tensor, *args, **kwargs):
    """
    Apply wait to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the wait operation.
    """
    from onnx9000.core.ops.torch_auto import wait as core_wait

    return core_wait(input_tensor, *args, **kwargs)


def e(input_tensor, *args, **kwargs):
    """
    Apply e to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the e operation.
    """
    from onnx9000.core.ops.torch_auto import e as core_e

    return core_e(input_tensor, *args, **kwargs)


def pi(input_tensor, *args, **kwargs):
    """
    Apply pi to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the pi operation.
    """
    from onnx9000.core.ops.torch_auto import pi as core_pi

    return core_pi(input_tensor, *args, **kwargs)


def nan(input_tensor, *args, **kwargs):
    """
    Apply nan to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the nan operation.
    """
    from onnx9000.core.ops.torch_auto import nan as core_nan

    return core_nan(input_tensor, *args, **kwargs)


def inf(input_tensor, *args, **kwargs):
    """
    Apply inf to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the inf operation.
    """
    from onnx9000.core.ops.torch_auto import inf as core_inf

    return core_inf(input_tensor, *args, **kwargs)


def newaxis(input_tensor, *args, **kwargs):
    """
    Apply newaxis to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the newaxis operation.
    """
    from onnx9000.core.ops.torch_auto import newaxis as core_newaxis

    return core_newaxis(input_tensor, *args, **kwargs)


def abs(input_tensor, *args, **kwargs):
    """
    Apply abs to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the abs operation.
    """
    from onnx9000.core.ops import abs as core_abs

    return core_abs(input_tensor, *args, **kwargs)


def abs_(input_tensor, *args, **kwargs):
    """
    Apply abs_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the abs_ operation.
    """
    from onnx9000.core.ops.torch_auto import abs_ as core_abs_

    return core_abs_(input_tensor, *args, **kwargs)


def absolute(input_tensor, *args, **kwargs):
    """
    Apply absolute to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the absolute operation.
    """
    from onnx9000.core.ops.torch_auto import absolute as core_absolute

    return core_absolute(input_tensor, *args, **kwargs)


def acos(input_tensor, *args, **kwargs):
    """
    Apply acos to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the acos operation.
    """
    from onnx9000.core.ops import acos as core_acos

    return core_acos(input_tensor, *args, **kwargs)


def acos_(input_tensor, *args, **kwargs):
    """
    Apply acos_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the acos_ operation.
    """
    from onnx9000.core.ops.torch_auto import acos_ as core_acos_

    return core_acos_(input_tensor, *args, **kwargs)


def acosh(input_tensor, *args, **kwargs):
    """
    Apply acosh to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the acosh operation.
    """
    from onnx9000.core.ops import acosh as core_acosh

    return core_acosh(input_tensor, *args, **kwargs)


def acosh_(input_tensor, *args, **kwargs):
    """
    Apply acosh_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the acosh_ operation.
    """
    from onnx9000.core.ops.torch_auto import acosh_ as core_acosh_

    return core_acosh_(input_tensor, *args, **kwargs)


def adaptive_avg_pool1d(input_tensor, *args, **kwargs):
    """
    Apply adaptive_avg_pool1d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the adaptive_avg_pool1d operation.
    """
    from onnx9000.core.ops.torch_auto import adaptive_avg_pool1d as core_adaptive_avg_pool1d

    return core_adaptive_avg_pool1d(input_tensor, *args, **kwargs)


def adaptive_max_pool1d(input_tensor, *args, **kwargs):
    """
    Apply adaptive_max_pool1d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the adaptive_max_pool1d operation.
    """
    from onnx9000.core.ops.torch_auto import adaptive_max_pool1d as core_adaptive_max_pool1d

    return core_adaptive_max_pool1d(input_tensor, *args, **kwargs)


def add(input_tensor, *args, **kwargs):
    """
    Apply add to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the add operation.
    """
    from onnx9000.core.ops import add as core_add

    return core_add(input_tensor, *args, **kwargs)


def addbmm(input_tensor, *args, **kwargs):
    """
    Apply addbmm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the addbmm operation.
    """
    from onnx9000.core.ops.torch_auto import addbmm as core_addbmm

    return core_addbmm(input_tensor, *args, **kwargs)


def addcdiv(input_tensor, *args, **kwargs):
    """
    Apply addcdiv to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the addcdiv operation.
    """
    from onnx9000.core.ops.torch_auto import addcdiv as core_addcdiv

    return core_addcdiv(input_tensor, *args, **kwargs)


def addcmul(input_tensor, *args, **kwargs):
    """
    Apply addcmul to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the addcmul operation.
    """
    from onnx9000.core.ops.torch_auto import addcmul as core_addcmul

    return core_addcmul(input_tensor, *args, **kwargs)


def addmm(input_tensor, *args, **kwargs):
    """
    Apply addmm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the addmm operation.
    """
    from onnx9000.core.ops.torch_auto import addmm as core_addmm

    return core_addmm(input_tensor, *args, **kwargs)


def addmv(input_tensor, *args, **kwargs):
    """
    Apply addmv to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the addmv operation.
    """
    from onnx9000.core.ops.torch_auto import addmv as core_addmv

    return core_addmv(input_tensor, *args, **kwargs)


def addmv_(input_tensor, *args, **kwargs):
    """
    Apply addmv_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the addmv_ operation.
    """
    from onnx9000.core.ops.torch_auto import addmv_ as core_addmv_

    return core_addmv_(input_tensor, *args, **kwargs)


def addr(input_tensor, *args, **kwargs):
    """
    Apply addr to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the addr operation.
    """
    from onnx9000.core.ops.torch_auto import addr as core_addr

    return core_addr(input_tensor, *args, **kwargs)


def adjoint(input_tensor, *args, **kwargs):
    """
    Apply adjoint to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the adjoint operation.
    """
    from onnx9000.core.ops.torch_auto import adjoint as core_adjoint

    return core_adjoint(input_tensor, *args, **kwargs)


def affine_grid_generator(input_tensor, *args, **kwargs):
    """
    Apply affine_grid_generator to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the affine_grid_generator operation.
    """
    from onnx9000.core.ops.torch_auto import affine_grid_generator as core_affine_grid_generator

    return core_affine_grid_generator(input_tensor, *args, **kwargs)


def alias_copy(input_tensor, *args, **kwargs):
    """
    Apply alias_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the alias_copy operation.
    """
    from onnx9000.core.ops.torch_auto import alias_copy as core_alias_copy

    return core_alias_copy(input_tensor, *args, **kwargs)


def align_tensors(input_tensor, *args, **kwargs):
    """
    Apply align_tensors to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the align_tensors operation.
    """
    from onnx9000.core.ops.torch_auto import align_tensors as core_align_tensors

    return core_align_tensors(input_tensor, *args, **kwargs)


def all(input_tensor, *args, **kwargs):
    """
    Apply all to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the all operation.
    """
    from onnx9000.core.ops.torch_auto import all as core_all

    return core_all(input_tensor, *args, **kwargs)


def allclose(input_tensor, *args, **kwargs):
    """
    Apply allclose to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the allclose operation.
    """
    from onnx9000.core.ops.torch_auto import allclose as core_allclose

    return core_allclose(input_tensor, *args, **kwargs)


def alpha_dropout(input_tensor, *args, **kwargs):
    """
    Apply alpha_dropout to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the alpha_dropout operation.
    """
    from onnx9000.core.ops.torch_auto import alpha_dropout as core_alpha_dropout

    return core_alpha_dropout(input_tensor, *args, **kwargs)


def alpha_dropout_(input_tensor, *args, **kwargs):
    """
    Apply alpha_dropout_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the alpha_dropout_ operation.
    """
    from onnx9000.core.ops.torch_auto import alpha_dropout_ as core_alpha_dropout_

    return core_alpha_dropout_(input_tensor, *args, **kwargs)


def amax(input_tensor, *args, **kwargs):
    """
    Apply amax to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the amax operation.
    """
    from onnx9000.core.ops.torch_auto import amax as core_amax

    return core_amax(input_tensor, *args, **kwargs)


def amin(input_tensor, *args, **kwargs):
    """
    Apply amin to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the amin operation.
    """
    from onnx9000.core.ops.torch_auto import amin as core_amin

    return core_amin(input_tensor, *args, **kwargs)


def aminmax(input_tensor, *args, **kwargs):
    """
    Apply aminmax to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the aminmax operation.
    """
    from onnx9000.core.ops.torch_auto import aminmax as core_aminmax

    return core_aminmax(input_tensor, *args, **kwargs)


def angle(input_tensor, *args, **kwargs):
    """
    Apply angle to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the angle operation.
    """
    from onnx9000.core.ops.torch_auto import angle as core_angle

    return core_angle(input_tensor, *args, **kwargs)


def any(input_tensor, *args, **kwargs):
    """
    Apply any to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the any operation.
    """
    from onnx9000.core.ops.torch_auto import any as core_any

    return core_any(input_tensor, *args, **kwargs)


def arange(input_tensor, *args, **kwargs):
    """
    Apply arange to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arange operation.
    """
    from onnx9000.core.ops.torch_auto import arange as core_arange

    return core_arange(input_tensor, *args, **kwargs)


def arccos(input_tensor, *args, **kwargs):
    """
    Apply arccos to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arccos operation.
    """
    from onnx9000.core.ops.torch_auto import arccos as core_arccos

    return core_arccos(input_tensor, *args, **kwargs)


def arccos_(input_tensor, *args, **kwargs):
    """
    Apply arccos_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arccos_ operation.
    """
    from onnx9000.core.ops.torch_auto import arccos_ as core_arccos_

    return core_arccos_(input_tensor, *args, **kwargs)


def arccosh(input_tensor, *args, **kwargs):
    """
    Apply arccosh to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arccosh operation.
    """
    from onnx9000.core.ops.torch_auto import arccosh as core_arccosh

    return core_arccosh(input_tensor, *args, **kwargs)


def arccosh_(input_tensor, *args, **kwargs):
    """
    Apply arccosh_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arccosh_ operation.
    """
    from onnx9000.core.ops.torch_auto import arccosh_ as core_arccosh_

    return core_arccosh_(input_tensor, *args, **kwargs)


def arcsin(input_tensor, *args, **kwargs):
    """
    Apply arcsin to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arcsin operation.
    """
    from onnx9000.core.ops.torch_auto import arcsin as core_arcsin

    return core_arcsin(input_tensor, *args, **kwargs)


def arcsin_(input_tensor, *args, **kwargs):
    """
    Apply arcsin_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arcsin_ operation.
    """
    from onnx9000.core.ops.torch_auto import arcsin_ as core_arcsin_

    return core_arcsin_(input_tensor, *args, **kwargs)


def arcsinh(input_tensor, *args, **kwargs):
    """
    Apply arcsinh to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arcsinh operation.
    """
    from onnx9000.core.ops.torch_auto import arcsinh as core_arcsinh

    return core_arcsinh(input_tensor, *args, **kwargs)


def arcsinh_(input_tensor, *args, **kwargs):
    """
    Apply arcsinh_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arcsinh_ operation.
    """
    from onnx9000.core.ops.torch_auto import arcsinh_ as core_arcsinh_

    return core_arcsinh_(input_tensor, *args, **kwargs)


def arctan(input_tensor, *args, **kwargs):
    """
    Apply arctan to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arctan operation.
    """
    from onnx9000.core.ops.torch_auto import arctan as core_arctan

    return core_arctan(input_tensor, *args, **kwargs)


def arctan2(input_tensor, *args, **kwargs):
    """
    Apply arctan2 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arctan2 operation.
    """
    from onnx9000.core.ops.torch_auto import arctan2 as core_arctan2

    return core_arctan2(input_tensor, *args, **kwargs)


def arctan_(input_tensor, *args, **kwargs):
    """
    Apply arctan_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arctan_ operation.
    """
    from onnx9000.core.ops.torch_auto import arctan_ as core_arctan_

    return core_arctan_(input_tensor, *args, **kwargs)


def arctanh(input_tensor, *args, **kwargs):
    """
    Apply arctanh to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arctanh operation.
    """
    from onnx9000.core.ops.torch_auto import arctanh as core_arctanh

    return core_arctanh(input_tensor, *args, **kwargs)


def arctanh_(input_tensor, *args, **kwargs):
    """
    Apply arctanh_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the arctanh_ operation.
    """
    from onnx9000.core.ops.torch_auto import arctanh_ as core_arctanh_

    return core_arctanh_(input_tensor, *args, **kwargs)


def argmax(input_tensor, *args, **kwargs):
    """
    Apply argmax to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the argmax operation.
    """
    from onnx9000.core.ops import argmax as core_argmax

    return core_argmax(input_tensor, *args, **kwargs)


def argmin(input_tensor, *args, **kwargs):
    """
    Apply argmin to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the argmin operation.
    """
    from onnx9000.core.ops import argmin as core_argmin

    return core_argmin(input_tensor, *args, **kwargs)


def argsort(input_tensor, *args, **kwargs):
    """
    Apply argsort to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the argsort operation.
    """
    from onnx9000.core.ops.torch_auto import argsort as core_argsort

    return core_argsort(input_tensor, *args, **kwargs)


def argwhere(input_tensor, *args, **kwargs):
    """
    Apply argwhere to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the argwhere operation.
    """
    from onnx9000.core.ops.torch_auto import argwhere as core_argwhere

    return core_argwhere(input_tensor, *args, **kwargs)


def as_strided(input_tensor, *args, **kwargs):
    """
    Apply as_strided to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the as_strided operation.
    """
    from onnx9000.core.ops.torch_auto import as_strided as core_as_strided

    return core_as_strided(input_tensor, *args, **kwargs)


def as_strided_(input_tensor, *args, **kwargs):
    """
    Apply as_strided_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the as_strided_ operation.
    """
    from onnx9000.core.ops.torch_auto import as_strided_ as core_as_strided_

    return core_as_strided_(input_tensor, *args, **kwargs)


def as_strided_copy(input_tensor, *args, **kwargs):
    """
    Apply as_strided_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the as_strided_copy operation.
    """
    from onnx9000.core.ops.torch_auto import as_strided_copy as core_as_strided_copy

    return core_as_strided_copy(input_tensor, *args, **kwargs)


def as_strided_scatter(input_tensor, *args, **kwargs):
    """
    Apply as_strided_scatter to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the as_strided_scatter operation.
    """
    from onnx9000.core.ops.torch_auto import as_strided_scatter as core_as_strided_scatter

    return core_as_strided_scatter(input_tensor, *args, **kwargs)


def as_tensor(input_tensor, *args, **kwargs):
    """
    Apply as_tensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the as_tensor operation.
    """
    from onnx9000.core.ops.torch_auto import as_tensor as core_as_tensor

    return core_as_tensor(input_tensor, *args, **kwargs)


def asarray(input_tensor, *args, **kwargs):
    """
    Apply asarray to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the asarray operation.
    """
    from onnx9000.core.ops.torch_auto import asarray as core_asarray

    return core_asarray(input_tensor, *args, **kwargs)


def asin(input_tensor, *args, **kwargs):
    """
    Apply asin to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the asin operation.
    """
    from onnx9000.core.ops import asin as core_asin

    return core_asin(input_tensor, *args, **kwargs)


def asin_(input_tensor, *args, **kwargs):
    """
    Apply asin_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the asin_ operation.
    """
    from onnx9000.core.ops.torch_auto import asin_ as core_asin_

    return core_asin_(input_tensor, *args, **kwargs)


def asinh(input_tensor, *args, **kwargs):
    """
    Apply asinh to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the asinh operation.
    """
    from onnx9000.core.ops import asinh as core_asinh

    return core_asinh(input_tensor, *args, **kwargs)


def asinh_(input_tensor, *args, **kwargs):
    """
    Apply asinh_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the asinh_ operation.
    """
    from onnx9000.core.ops.torch_auto import asinh_ as core_asinh_

    return core_asinh_(input_tensor, *args, **kwargs)


def atan(input_tensor, *args, **kwargs):
    """
    Apply atan to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the atan operation.
    """
    from onnx9000.core.ops import atan as core_atan

    return core_atan(input_tensor, *args, **kwargs)


def atan2(input_tensor, *args, **kwargs):
    """
    Apply atan2 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the atan2 operation.
    """
    from onnx9000.core.ops.torch_auto import atan2 as core_atan2

    return core_atan2(input_tensor, *args, **kwargs)


def atan_(input_tensor, *args, **kwargs):
    """
    Apply atan_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the atan_ operation.
    """
    from onnx9000.core.ops.torch_auto import atan_ as core_atan_

    return core_atan_(input_tensor, *args, **kwargs)


def atanh(input_tensor, *args, **kwargs):
    """
    Apply atanh to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the atanh operation.
    """
    from onnx9000.core.ops import atanh as core_atanh

    return core_atanh(input_tensor, *args, **kwargs)


def atanh_(input_tensor, *args, **kwargs):
    """
    Apply atanh_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the atanh_ operation.
    """
    from onnx9000.core.ops.torch_auto import atanh_ as core_atanh_

    return core_atanh_(input_tensor, *args, **kwargs)


def atleast_1d(input_tensor, *args, **kwargs):
    """
    Apply atleast_1d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the atleast_1d operation.
    """
    from onnx9000.core.ops.torch_auto import atleast_1d as core_atleast_1d

    return core_atleast_1d(input_tensor, *args, **kwargs)


def atleast_2d(input_tensor, *args, **kwargs):
    """
    Apply atleast_2d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the atleast_2d operation.
    """
    from onnx9000.core.ops.torch_auto import atleast_2d as core_atleast_2d

    return core_atleast_2d(input_tensor, *args, **kwargs)


def atleast_3d(input_tensor, *args, **kwargs):
    """
    Apply atleast_3d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the atleast_3d operation.
    """
    from onnx9000.core.ops.torch_auto import atleast_3d as core_atleast_3d

    return core_atleast_3d(input_tensor, *args, **kwargs)


def avg_pool1d(input_tensor, *args, **kwargs):
    """
    Apply avg_pool1d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the avg_pool1d operation.
    """
    from onnx9000.core.ops.torch_auto import avg_pool1d as core_avg_pool1d

    return core_avg_pool1d(input_tensor, *args, **kwargs)


def baddbmm(input_tensor, *args, **kwargs):
    """
    Apply baddbmm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the baddbmm operation.
    """
    from onnx9000.core.ops.torch_auto import baddbmm as core_baddbmm

    return core_baddbmm(input_tensor, *args, **kwargs)


def bartlett_window(input_tensor, *args, **kwargs):
    """
    Apply bartlett_window to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bartlett_window operation.
    """
    from onnx9000.core.ops.torch_auto import bartlett_window as core_bartlett_window

    return core_bartlett_window(input_tensor, *args, **kwargs)


def batch_norm(input_tensor, *args, **kwargs):
    """
    Apply batch_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the batch_norm operation.
    """
    from onnx9000.core.ops.torch_auto import batch_norm as core_batch_norm

    return core_batch_norm(input_tensor, *args, **kwargs)


def batch_norm_backward_elemt(input_tensor, *args, **kwargs):
    """
    Apply batch_norm_backward_elemt to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the batch_norm_backward_elemt operation.
    """
    from onnx9000.core.ops.torch_auto import (
        batch_norm_backward_elemt as core_batch_norm_backward_elemt,
    )

    return core_batch_norm_backward_elemt(input_tensor, *args, **kwargs)


def batch_norm_backward_reduce(input_tensor, *args, **kwargs):
    """
    Apply batch_norm_backward_reduce to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the batch_norm_backward_reduce operation.
    """
    from onnx9000.core.ops.torch_auto import (
        batch_norm_backward_reduce as core_batch_norm_backward_reduce,
    )

    return core_batch_norm_backward_reduce(input_tensor, *args, **kwargs)


def batch_norm_elemt(input_tensor, *args, **kwargs):
    """
    Apply batch_norm_elemt to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the batch_norm_elemt operation.
    """
    from onnx9000.core.ops.torch_auto import batch_norm_elemt as core_batch_norm_elemt

    return core_batch_norm_elemt(input_tensor, *args, **kwargs)


def batch_norm_gather_stats(input_tensor, *args, **kwargs):
    """
    Apply batch_norm_gather_stats to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the batch_norm_gather_stats operation.
    """
    from onnx9000.core.ops.torch_auto import batch_norm_gather_stats as core_batch_norm_gather_stats

    return core_batch_norm_gather_stats(input_tensor, *args, **kwargs)


def batch_norm_gather_stats_with_counts(input_tensor, *args, **kwargs):
    """
    Apply batch_norm_gather_stats_with_counts to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the batch_norm_gather_stats_with_counts operation.
    """
    from onnx9000.core.ops.torch_auto import (
        batch_norm_gather_stats_with_counts as core_batch_norm_gather_stats_with_counts,
    )

    return core_batch_norm_gather_stats_with_counts(input_tensor, *args, **kwargs)


def batch_norm_stats(input_tensor, *args, **kwargs):
    """
    Apply batch_norm_stats to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the batch_norm_stats operation.
    """
    from onnx9000.core.ops.torch_auto import batch_norm_stats as core_batch_norm_stats

    return core_batch_norm_stats(input_tensor, *args, **kwargs)


def batch_norm_update_stats(input_tensor, *args, **kwargs):
    """
    Apply batch_norm_update_stats to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the batch_norm_update_stats operation.
    """
    from onnx9000.core.ops.torch_auto import batch_norm_update_stats as core_batch_norm_update_stats

    return core_batch_norm_update_stats(input_tensor, *args, **kwargs)


def bernoulli(input_tensor, *args, **kwargs):
    """
    Apply bernoulli to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bernoulli operation.
    """
    from onnx9000.core.ops import bernoulli as core_bernoulli

    return core_bernoulli(input_tensor, *args, **kwargs)


def bilinear(input_tensor, *args, **kwargs):
    """
    Apply bilinear to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bilinear operation.
    """
    from onnx9000.core.ops.torch_auto import bilinear as core_bilinear

    return core_bilinear(input_tensor, *args, **kwargs)


def binary_cross_entropy_with_logits(input_tensor, *args, **kwargs):
    """
    Apply binary_cross_entropy_with_logits to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the binary_cross_entropy_with_logits operation.
    """
    from onnx9000.core.ops.torch_auto import (
        binary_cross_entropy_with_logits as core_binary_cross_entropy_with_logits,
    )

    return core_binary_cross_entropy_with_logits(input_tensor, *args, **kwargs)


def bincount(input_tensor, *args, **kwargs):
    """
    Apply bincount to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bincount operation.
    """
    from onnx9000.core.ops.torch_auto import bincount as core_bincount

    return core_bincount(input_tensor, *args, **kwargs)


def binomial(input_tensor, *args, **kwargs):
    """
    Apply binomial to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the binomial operation.
    """
    from onnx9000.core.ops.torch_auto import binomial as core_binomial

    return core_binomial(input_tensor, *args, **kwargs)


def bitwise_and(input_tensor, *args, **kwargs):
    """
    Apply bitwise_and to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bitwise_and operation.
    """
    from onnx9000.core.ops import bitwise_and as core_bitwise_and

    return core_bitwise_and(input_tensor, *args, **kwargs)


def bitwise_left_shift(input_tensor, *args, **kwargs):
    """
    Apply bitwise_left_shift to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bitwise_left_shift operation.
    """
    from onnx9000.core.ops.torch_auto import bitwise_left_shift as core_bitwise_left_shift

    return core_bitwise_left_shift(input_tensor, *args, **kwargs)


def bitwise_not(input_tensor, *args, **kwargs):
    """
    Apply bitwise_not to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bitwise_not operation.
    """
    from onnx9000.core.ops import bitwise_not as core_bitwise_not

    return core_bitwise_not(input_tensor, *args, **kwargs)


def bitwise_or(input_tensor, *args, **kwargs):
    """
    Apply bitwise_or to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bitwise_or operation.
    """
    from onnx9000.core.ops import bitwise_or as core_bitwise_or

    return core_bitwise_or(input_tensor, *args, **kwargs)


def bitwise_right_shift(input_tensor, *args, **kwargs):
    """
    Apply bitwise_right_shift to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bitwise_right_shift operation.
    """
    from onnx9000.core.ops.torch_auto import bitwise_right_shift as core_bitwise_right_shift

    return core_bitwise_right_shift(input_tensor, *args, **kwargs)


def bitwise_xor(input_tensor, *args, **kwargs):
    """
    Apply bitwise_xor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bitwise_xor operation.
    """
    from onnx9000.core.ops import bitwise_xor as core_bitwise_xor

    return core_bitwise_xor(input_tensor, *args, **kwargs)


def blackman_window(input_tensor, *args, **kwargs):
    """
    Apply blackman_window to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the blackman_window operation.
    """
    from onnx9000.core.ops import blackman_window as core_blackman_window

    return core_blackman_window(input_tensor, *args, **kwargs)


def block_diag(input_tensor, *args, **kwargs):
    """
    Apply block_diag to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the block_diag operation.
    """
    from onnx9000.core.ops.torch_auto import block_diag as core_block_diag

    return core_block_diag(input_tensor, *args, **kwargs)


def bmm(input_tensor, *args, **kwargs):
    """
    Apply bmm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bmm operation.
    """
    from onnx9000.core.ops.torch_auto import bmm as core_bmm

    return core_bmm(input_tensor, *args, **kwargs)


def broadcast_tensors(input_tensor, *args, **kwargs):
    """
    Apply broadcast_tensors to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the broadcast_tensors operation.
    """
    from onnx9000.core.ops.torch_auto import broadcast_tensors as core_broadcast_tensors

    return core_broadcast_tensors(input_tensor, *args, **kwargs)


def broadcast_to(input_tensor, *args, **kwargs):
    """
    Apply broadcast_to to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the broadcast_to operation.
    """
    from onnx9000.core.ops.torch_auto import broadcast_to as core_broadcast_to

    return core_broadcast_to(input_tensor, *args, **kwargs)


def bucketize(input_tensor, *args, **kwargs):
    """
    Apply bucketize to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bucketize operation.
    """
    from onnx9000.core.ops.torch_auto import bucketize as core_bucketize

    return core_bucketize(input_tensor, *args, **kwargs)


def can_cast(input_tensor, *args, **kwargs):
    """
    Apply can_cast to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the can_cast operation.
    """
    from onnx9000.core.ops.torch_auto import can_cast as core_can_cast

    return core_can_cast(input_tensor, *args, **kwargs)


def cartesian_prod(input_tensor, *args, **kwargs):
    """
    Apply cartesian_prod to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cartesian_prod operation.
    """
    from onnx9000.core.ops.torch_auto import cartesian_prod as core_cartesian_prod

    return core_cartesian_prod(input_tensor, *args, **kwargs)


def cat(input_tensor, *args, **kwargs):
    """
    Apply cat to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cat operation.
    """
    from onnx9000.core.ops.torch_auto import cat as core_cat

    return core_cat(input_tensor, *args, **kwargs)


def ccol_indices_copy(input_tensor, *args, **kwargs):
    """
    Apply ccol_indices_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ccol_indices_copy operation.
    """
    from onnx9000.core.ops.torch_auto import ccol_indices_copy as core_ccol_indices_copy

    return core_ccol_indices_copy(input_tensor, *args, **kwargs)


def cdist(input_tensor, *args, **kwargs):
    """
    Apply cdist to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cdist operation.
    """
    from onnx9000.core.ops.torch_auto import cdist as core_cdist

    return core_cdist(input_tensor, *args, **kwargs)


def ceil(input_tensor, *args, **kwargs):
    """
    Apply ceil to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ceil operation.
    """
    from onnx9000.core.ops import ceil as core_ceil

    return core_ceil(input_tensor, *args, **kwargs)


def ceil_(input_tensor, *args, **kwargs):
    """
    Apply ceil_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ceil_ operation.
    """
    from onnx9000.core.ops.torch_auto import ceil_ as core_ceil_

    return core_ceil_(input_tensor, *args, **kwargs)


def celu(input_tensor, *args, **kwargs):
    """
    Apply celu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the celu operation.
    """
    from onnx9000.core.ops import celu as core_celu

    return core_celu(input_tensor, *args, **kwargs)


def celu_(input_tensor, *args, **kwargs):
    """
    Apply celu_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the celu_ operation.
    """
    from onnx9000.core.ops.torch_auto import celu_ as core_celu_

    return core_celu_(input_tensor, *args, **kwargs)


def chain_matmul(input_tensor, *args, **kwargs):
    """
    Apply chain_matmul to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the chain_matmul operation.
    """
    from onnx9000.core.ops.torch_auto import chain_matmul as core_chain_matmul

    return core_chain_matmul(input_tensor, *args, **kwargs)


def channel_shuffle(input_tensor, *args, **kwargs):
    """
    Apply channel_shuffle to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the channel_shuffle operation.
    """
    from onnx9000.core.ops.torch_auto import channel_shuffle as core_channel_shuffle

    return core_channel_shuffle(input_tensor, *args, **kwargs)


def cholesky(input_tensor, *args, **kwargs):
    """
    Apply cholesky to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cholesky operation.
    """
    from onnx9000.core.ops.torch_auto import cholesky as core_cholesky

    return core_cholesky(input_tensor, *args, **kwargs)


def cholesky_inverse(input_tensor, *args, **kwargs):
    """
    Apply cholesky_inverse to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cholesky_inverse operation.
    """
    from onnx9000.core.ops.torch_auto import cholesky_inverse as core_cholesky_inverse

    return core_cholesky_inverse(input_tensor, *args, **kwargs)


def cholesky_solve(input_tensor, *args, **kwargs):
    """
    Apply cholesky_solve to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cholesky_solve operation.
    """
    from onnx9000.core.ops.torch_auto import cholesky_solve as core_cholesky_solve

    return core_cholesky_solve(input_tensor, *args, **kwargs)


def choose_qparams_optimized(input_tensor, *args, **kwargs):
    """
    Apply choose_qparams_optimized to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the choose_qparams_optimized operation.
    """
    from onnx9000.core.ops.torch_auto import (
        choose_qparams_optimized as core_choose_qparams_optimized,
    )

    return core_choose_qparams_optimized(input_tensor, *args, **kwargs)


def clamp(input_tensor, *args, **kwargs):
    """
    Apply clamp to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the clamp operation.
    """
    from onnx9000.core.ops.torch_auto import clamp as core_clamp

    return core_clamp(input_tensor, *args, **kwargs)


def clamp_(input_tensor, *args, **kwargs):
    """
    Apply clamp_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the clamp_ operation.
    """
    from onnx9000.core.ops.torch_auto import clamp_ as core_clamp_

    return core_clamp_(input_tensor, *args, **kwargs)


def clamp_max(input_tensor, *args, **kwargs):
    """
    Apply clamp_max to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the clamp_max operation.
    """
    from onnx9000.core.ops.torch_auto import clamp_max as core_clamp_max

    return core_clamp_max(input_tensor, *args, **kwargs)


def clamp_max_(input_tensor, *args, **kwargs):
    """
    Apply clamp_max_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the clamp_max_ operation.
    """
    from onnx9000.core.ops.torch_auto import clamp_max_ as core_clamp_max_

    return core_clamp_max_(input_tensor, *args, **kwargs)


def clamp_min(input_tensor, *args, **kwargs):
    """
    Apply clamp_min to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the clamp_min operation.
    """
    from onnx9000.core.ops.torch_auto import clamp_min as core_clamp_min

    return core_clamp_min(input_tensor, *args, **kwargs)


def clamp_min_(input_tensor, *args, **kwargs):
    """
    Apply clamp_min_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the clamp_min_ operation.
    """
    from onnx9000.core.ops.torch_auto import clamp_min_ as core_clamp_min_

    return core_clamp_min_(input_tensor, *args, **kwargs)


def clip(input_tensor, *args, **kwargs):
    """
    Apply clip to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the clip operation.
    """
    from onnx9000.core.ops import clip as core_clip

    return core_clip(input_tensor, *args, **kwargs)


def clip_(input_tensor, *args, **kwargs):
    """
    Apply clip_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the clip_ operation.
    """
    from onnx9000.core.ops.torch_auto import clip_ as core_clip_

    return core_clip_(input_tensor, *args, **kwargs)


def clone(input_tensor, *args, **kwargs):
    """
    Apply clone to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the clone operation.
    """
    from onnx9000.core.ops.torch_auto import clone as core_clone

    return core_clone(input_tensor, *args, **kwargs)


def col_indices_copy(input_tensor, *args, **kwargs):
    """
    Apply col_indices_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the col_indices_copy operation.
    """
    from onnx9000.core.ops.torch_auto import col_indices_copy as core_col_indices_copy

    return core_col_indices_copy(input_tensor, *args, **kwargs)


def column_stack(input_tensor, *args, **kwargs):
    """
    Apply column_stack to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the column_stack operation.
    """
    from onnx9000.core.ops.torch_auto import column_stack as core_column_stack

    return core_column_stack(input_tensor, *args, **kwargs)


def combinations(input_tensor, *args, **kwargs):
    """
    Apply combinations to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the combinations operation.
    """
    from onnx9000.core.ops.torch_auto import combinations as core_combinations

    return core_combinations(input_tensor, *args, **kwargs)


def complex(input_tensor, *args, **kwargs):
    """
    Apply complex to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the complex operation.
    """
    from onnx9000.core.ops.torch_auto import complex as core_complex

    return core_complex(input_tensor, *args, **kwargs)


def concat(input_tensor, *args, **kwargs):
    """
    Apply concat to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the concat operation.
    """
    from onnx9000.core.ops import concat as core_concat

    return core_concat(input_tensor, *args, **kwargs)


def concatenate(input_tensor, *args, **kwargs):
    """
    Apply concatenate to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the concatenate operation.
    """
    from onnx9000.core.ops.torch_auto import concatenate as core_concatenate

    return core_concatenate(input_tensor, *args, **kwargs)


def conj(input_tensor, *args, **kwargs):
    """
    Apply conj to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the conj operation.
    """
    from onnx9000.core.ops.torch_auto import conj as core_conj

    return core_conj(input_tensor, *args, **kwargs)


def conj_physical(input_tensor, *args, **kwargs):
    """
    Apply conj_physical to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the conj_physical operation.
    """
    from onnx9000.core.ops.torch_auto import conj_physical as core_conj_physical

    return core_conj_physical(input_tensor, *args, **kwargs)


def conj_physical_(input_tensor, *args, **kwargs):
    """
    Apply conj_physical_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the conj_physical_ operation.
    """
    from onnx9000.core.ops.torch_auto import conj_physical_ as core_conj_physical_

    return core_conj_physical_(input_tensor, *args, **kwargs)


def constant_pad_nd(input_tensor, *args, **kwargs):
    """
    Apply constant_pad_nd to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the constant_pad_nd operation.
    """
    from onnx9000.core.ops.torch_auto import constant_pad_nd as core_constant_pad_nd

    return core_constant_pad_nd(input_tensor, *args, **kwargs)


def conv1d(input_tensor, *args, **kwargs):
    """
    Apply conv1d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the conv1d operation.
    """
    from onnx9000.core.ops.torch_auto import conv1d as core_conv1d

    return core_conv1d(input_tensor, *args, **kwargs)


def conv2d(input_tensor, *args, **kwargs):
    """
    Apply conv2d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the conv2d operation.
    """
    from onnx9000.core.ops.torch_auto import conv2d as core_conv2d

    return core_conv2d(input_tensor, *args, **kwargs)


def conv3d(input_tensor, *args, **kwargs):
    """
    Apply conv3d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the conv3d operation.
    """
    from onnx9000.core.ops.torch_auto import conv3d as core_conv3d

    return core_conv3d(input_tensor, *args, **kwargs)


def conv_tbc(input_tensor, *args, **kwargs):
    """
    Apply conv_tbc to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the conv_tbc operation.
    """
    from onnx9000.core.ops.torch_auto import conv_tbc as core_conv_tbc

    return core_conv_tbc(input_tensor, *args, **kwargs)


def conv_transpose1d(input_tensor, *args, **kwargs):
    """
    Apply conv_transpose1d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the conv_transpose1d operation.
    """
    from onnx9000.core.ops.torch_auto import conv_transpose1d as core_conv_transpose1d

    return core_conv_transpose1d(input_tensor, *args, **kwargs)


def conv_transpose2d(input_tensor, *args, **kwargs):
    """
    Apply conv_transpose2d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the conv_transpose2d operation.
    """
    from onnx9000.core.ops.torch_auto import conv_transpose2d as core_conv_transpose2d

    return core_conv_transpose2d(input_tensor, *args, **kwargs)


def conv_transpose3d(input_tensor, *args, **kwargs):
    """
    Apply conv_transpose3d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the conv_transpose3d operation.
    """
    from onnx9000.core.ops.torch_auto import conv_transpose3d as core_conv_transpose3d

    return core_conv_transpose3d(input_tensor, *args, **kwargs)


def convolution(input_tensor, *args, **kwargs):
    """
    Apply convolution to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the convolution operation.
    """
    from onnx9000.core.ops.torch_auto import convolution as core_convolution

    return core_convolution(input_tensor, *args, **kwargs)


def copysign(input_tensor, *args, **kwargs):
    """
    Apply copysign to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the copysign operation.
    """
    from onnx9000.core.ops.torch_auto import copysign as core_copysign

    return core_copysign(input_tensor, *args, **kwargs)


def corrcoef(input_tensor, *args, **kwargs):
    """
    Apply corrcoef to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the corrcoef operation.
    """
    from onnx9000.core.ops.torch_auto import corrcoef as core_corrcoef

    return core_corrcoef(input_tensor, *args, **kwargs)


def cos(input_tensor, *args, **kwargs):
    """
    Apply cos to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cos operation.
    """
    from onnx9000.core.ops import cos as core_cos

    return core_cos(input_tensor, *args, **kwargs)


def cos_(input_tensor, *args, **kwargs):
    """
    Apply cos_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cos_ operation.
    """
    from onnx9000.core.ops.torch_auto import cos_ as core_cos_

    return core_cos_(input_tensor, *args, **kwargs)


def cosh(input_tensor, *args, **kwargs):
    """
    Apply cosh to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cosh operation.
    """
    from onnx9000.core.ops import cosh as core_cosh

    return core_cosh(input_tensor, *args, **kwargs)


def cosh_(input_tensor, *args, **kwargs):
    """
    Apply cosh_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cosh_ operation.
    """
    from onnx9000.core.ops.torch_auto import cosh_ as core_cosh_

    return core_cosh_(input_tensor, *args, **kwargs)


def cosine_embedding_loss(input_tensor, *args, **kwargs):
    """
    Apply cosine_embedding_loss to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cosine_embedding_loss operation.
    """
    from onnx9000.core.ops.torch_auto import cosine_embedding_loss as core_cosine_embedding_loss

    return core_cosine_embedding_loss(input_tensor, *args, **kwargs)


def cosine_similarity(input_tensor, *args, **kwargs):
    """
    Apply cosine_similarity to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cosine_similarity operation.
    """
    from onnx9000.core.ops.torch_auto import cosine_similarity as core_cosine_similarity

    return core_cosine_similarity(input_tensor, *args, **kwargs)


def count_nonzero(input_tensor, *args, **kwargs):
    """
    Apply count_nonzero to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the count_nonzero operation.
    """
    from onnx9000.core.ops.torch_auto import count_nonzero as core_count_nonzero

    return core_count_nonzero(input_tensor, *args, **kwargs)


def cov(input_tensor, *args, **kwargs):
    """
    Apply cov to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cov operation.
    """
    from onnx9000.core.ops.torch_auto import cov as core_cov

    return core_cov(input_tensor, *args, **kwargs)


def cross(input_tensor, *args, **kwargs):
    """
    Apply cross to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cross operation.
    """
    from onnx9000.core.ops.torch_auto import cross as core_cross

    return core_cross(input_tensor, *args, **kwargs)


def crow_indices_copy(input_tensor, *args, **kwargs):
    """
    Apply crow_indices_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the crow_indices_copy operation.
    """
    from onnx9000.core.ops.torch_auto import crow_indices_copy as core_crow_indices_copy

    return core_crow_indices_copy(input_tensor, *args, **kwargs)


def ctc_loss(input_tensor, *args, **kwargs):
    """
    Apply ctc_loss to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ctc_loss operation.
    """
    from onnx9000.core.ops.torch_auto import ctc_loss as core_ctc_loss

    return core_ctc_loss(input_tensor, *args, **kwargs)


def cudnn_affine_grid_generator(input_tensor, *args, **kwargs):
    """
    Apply cudnn_affine_grid_generator to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cudnn_affine_grid_generator operation.
    """
    from onnx9000.core.ops.torch_auto import (
        cudnn_affine_grid_generator as core_cudnn_affine_grid_generator,
    )

    return core_cudnn_affine_grid_generator(input_tensor, *args, **kwargs)


def cudnn_batch_norm(input_tensor, *args, **kwargs):
    """
    Apply cudnn_batch_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cudnn_batch_norm operation.
    """
    from onnx9000.core.ops.torch_auto import cudnn_batch_norm as core_cudnn_batch_norm

    return core_cudnn_batch_norm(input_tensor, *args, **kwargs)


def cudnn_convolution(input_tensor, *args, **kwargs):
    """
    Apply cudnn_convolution to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cudnn_convolution operation.
    """
    from onnx9000.core.ops.torch_auto import cudnn_convolution as core_cudnn_convolution

    return core_cudnn_convolution(input_tensor, *args, **kwargs)


def cudnn_convolution_add_relu(input_tensor, *args, **kwargs):
    """
    Apply cudnn_convolution_add_relu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cudnn_convolution_add_relu operation.
    """
    from onnx9000.core.ops.torch_auto import (
        cudnn_convolution_add_relu as core_cudnn_convolution_add_relu,
    )

    return core_cudnn_convolution_add_relu(input_tensor, *args, **kwargs)


def cudnn_convolution_relu(input_tensor, *args, **kwargs):
    """
    Apply cudnn_convolution_relu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cudnn_convolution_relu operation.
    """
    from onnx9000.core.ops.torch_auto import cudnn_convolution_relu as core_cudnn_convolution_relu

    return core_cudnn_convolution_relu(input_tensor, *args, **kwargs)


def cudnn_convolution_transpose(input_tensor, *args, **kwargs):
    """
    Apply cudnn_convolution_transpose to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cudnn_convolution_transpose operation.
    """
    from onnx9000.core.ops.torch_auto import (
        cudnn_convolution_transpose as core_cudnn_convolution_transpose,
    )

    return core_cudnn_convolution_transpose(input_tensor, *args, **kwargs)


def cudnn_grid_sampler(input_tensor, *args, **kwargs):
    """
    Apply cudnn_grid_sampler to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cudnn_grid_sampler operation.
    """
    from onnx9000.core.ops.torch_auto import cudnn_grid_sampler as core_cudnn_grid_sampler

    return core_cudnn_grid_sampler(input_tensor, *args, **kwargs)


def cudnn_is_acceptable(input_tensor, *args, **kwargs):
    """
    Apply cudnn_is_acceptable to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cudnn_is_acceptable operation.
    """
    from onnx9000.core.ops.torch_auto import cudnn_is_acceptable as core_cudnn_is_acceptable

    return core_cudnn_is_acceptable(input_tensor, *args, **kwargs)


def cummax(input_tensor, *args, **kwargs):
    """
    Apply cummax to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cummax operation.
    """
    from onnx9000.core.ops.torch_auto import cummax as core_cummax

    return core_cummax(input_tensor, *args, **kwargs)


def cummin(input_tensor, *args, **kwargs):
    """
    Apply cummin to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cummin operation.
    """
    from onnx9000.core.ops.torch_auto import cummin as core_cummin

    return core_cummin(input_tensor, *args, **kwargs)


def cumprod(input_tensor, *args, **kwargs):
    """
    Apply cumprod to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cumprod operation.
    """
    from onnx9000.core.ops.torch_auto import cumprod as core_cumprod

    return core_cumprod(input_tensor, *args, **kwargs)


def cumsum(input_tensor, *args, **kwargs):
    """
    Apply cumsum to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cumsum operation.
    """
    from onnx9000.core.ops import cumsum as core_cumsum

    return core_cumsum(input_tensor, *args, **kwargs)


def cumulative_trapezoid(input_tensor, *args, **kwargs):
    """
    Apply cumulative_trapezoid to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cumulative_trapezoid operation.
    """
    from onnx9000.core.ops.torch_auto import cumulative_trapezoid as core_cumulative_trapezoid

    return core_cumulative_trapezoid(input_tensor, *args, **kwargs)


def deg2rad(input_tensor, *args, **kwargs):
    """
    Apply deg2rad to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the deg2rad operation.
    """
    from onnx9000.core.ops.torch_auto import deg2rad as core_deg2rad

    return core_deg2rad(input_tensor, *args, **kwargs)


def deg2rad_(input_tensor, *args, **kwargs):
    """
    Apply deg2rad_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the deg2rad_ operation.
    """
    from onnx9000.core.ops.torch_auto import deg2rad_ as core_deg2rad_

    return core_deg2rad_(input_tensor, *args, **kwargs)


def dequantize(input_tensor, *args, **kwargs):
    """
    Apply dequantize to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the dequantize operation.
    """
    from onnx9000.core.ops.torch_auto import dequantize as core_dequantize

    return core_dequantize(input_tensor, *args, **kwargs)


def det(input_tensor, *args, **kwargs):
    """
    Apply det to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the det operation.
    """
    from onnx9000.core.ops import det as core_det

    return core_det(input_tensor, *args, **kwargs)


def detach(input_tensor, *args, **kwargs):
    """
    Apply detach to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the detach operation.
    """
    from onnx9000.core.ops.torch_auto import detach as core_detach

    return core_detach(input_tensor, *args, **kwargs)


def detach_(input_tensor, *args, **kwargs):
    """
    Apply detach_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the detach_ operation.
    """
    from onnx9000.core.ops.torch_auto import detach_ as core_detach_

    return core_detach_(input_tensor, *args, **kwargs)


def detach_copy(input_tensor, *args, **kwargs):
    """
    Apply detach_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the detach_copy operation.
    """
    from onnx9000.core.ops.torch_auto import detach_copy as core_detach_copy

    return core_detach_copy(input_tensor, *args, **kwargs)


def diag(input_tensor, *args, **kwargs):
    """
    Apply diag to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the diag operation.
    """
    from onnx9000.core.ops.torch_auto import diag as core_diag

    return core_diag(input_tensor, *args, **kwargs)


def diag_embed(input_tensor, *args, **kwargs):
    """
    Apply diag_embed to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the diag_embed operation.
    """
    from onnx9000.core.ops.torch_auto import diag_embed as core_diag_embed

    return core_diag_embed(input_tensor, *args, **kwargs)


def diagflat(input_tensor, *args, **kwargs):
    """
    Apply diagflat to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the diagflat operation.
    """
    from onnx9000.core.ops.torch_auto import diagflat as core_diagflat

    return core_diagflat(input_tensor, *args, **kwargs)


def diagonal(input_tensor, *args, **kwargs):
    """
    Apply diagonal to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the diagonal operation.
    """
    from onnx9000.core.ops.torch_auto import diagonal as core_diagonal

    return core_diagonal(input_tensor, *args, **kwargs)


def diagonal_copy(input_tensor, *args, **kwargs):
    """
    Apply diagonal_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the diagonal_copy operation.
    """
    from onnx9000.core.ops.torch_auto import diagonal_copy as core_diagonal_copy

    return core_diagonal_copy(input_tensor, *args, **kwargs)


def diagonal_scatter(input_tensor, *args, **kwargs):
    """
    Apply diagonal_scatter to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the diagonal_scatter operation.
    """
    from onnx9000.core.ops.torch_auto import diagonal_scatter as core_diagonal_scatter

    return core_diagonal_scatter(input_tensor, *args, **kwargs)


def diff(input_tensor, *args, **kwargs):
    """
    Apply diff to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the diff operation.
    """
    from onnx9000.core.ops.torch_auto import diff as core_diff

    return core_diff(input_tensor, *args, **kwargs)


def digamma(input_tensor, *args, **kwargs):
    """
    Apply digamma to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the digamma operation.
    """
    from onnx9000.core.ops.torch_auto import digamma as core_digamma

    return core_digamma(input_tensor, *args, **kwargs)


def dist(input_tensor, *args, **kwargs):
    """
    Apply dist to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the dist operation.
    """
    from onnx9000.core.ops.torch_auto import dist as core_dist

    return core_dist(input_tensor, *args, **kwargs)


def div(input_tensor, *args, **kwargs):
    """
    Apply div to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the div operation.
    """
    from onnx9000.core.ops import div as core_div

    return core_div(input_tensor, *args, **kwargs)


def divide(input_tensor, *args, **kwargs):
    """
    Apply divide to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the divide operation.
    """
    from onnx9000.core.ops.torch_auto import divide as core_divide

    return core_divide(input_tensor, *args, **kwargs)


def dot(input_tensor, *args, **kwargs):
    """
    Apply dot to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the dot operation.
    """
    from onnx9000.core.ops.torch_auto import dot as core_dot

    return core_dot(input_tensor, *args, **kwargs)


def dropout(input_tensor, *args, **kwargs):
    """
    Apply dropout to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the dropout operation.
    """
    from onnx9000.core.ops import dropout as core_dropout

    return core_dropout(input_tensor, *args, **kwargs)


def dropout_(input_tensor, *args, **kwargs):
    """
    Apply dropout_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the dropout_ operation.
    """
    from onnx9000.core.ops.torch_auto import dropout_ as core_dropout_

    return core_dropout_(input_tensor, *args, **kwargs)


def dsmm(input_tensor, *args, **kwargs):
    """
    Apply dsmm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the dsmm operation.
    """
    from onnx9000.core.ops.torch_auto import dsmm as core_dsmm

    return core_dsmm(input_tensor, *args, **kwargs)


def dsplit(input_tensor, *args, **kwargs):
    """
    Apply dsplit to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the dsplit operation.
    """
    from onnx9000.core.ops.torch_auto import dsplit as core_dsplit

    return core_dsplit(input_tensor, *args, **kwargs)


def dstack(input_tensor, *args, **kwargs):
    """
    Apply dstack to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the dstack operation.
    """
    from onnx9000.core.ops.torch_auto import dstack as core_dstack

    return core_dstack(input_tensor, *args, **kwargs)


def einsum(input_tensor, *args, **kwargs):
    """
    Apply einsum to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the einsum operation.
    """
    from onnx9000.core.ops import einsum as core_einsum

    return core_einsum(input_tensor, *args, **kwargs)


def embedding(input_tensor, *args, **kwargs):
    """
    Apply embedding to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the embedding operation.
    """
    from onnx9000.core.ops.torch_auto import embedding as core_embedding

    return core_embedding(input_tensor, *args, **kwargs)


def embedding_bag(input_tensor, *args, **kwargs):
    """
    Apply embedding_bag to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the embedding_bag operation.
    """
    from onnx9000.core.ops.torch_auto import embedding_bag as core_embedding_bag

    return core_embedding_bag(input_tensor, *args, **kwargs)


def embedding_renorm_(input_tensor, *args, **kwargs):
    """
    Apply embedding_renorm_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the embedding_renorm_ operation.
    """
    from onnx9000.core.ops.torch_auto import embedding_renorm_ as core_embedding_renorm_

    return core_embedding_renorm_(input_tensor, *args, **kwargs)


def empty(input_tensor, *args, **kwargs):
    """
    Apply empty to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the empty operation.
    """
    from onnx9000.core.ops.torch_auto import empty as core_empty

    return core_empty(input_tensor, *args, **kwargs)


def empty_like(input_tensor, *args, **kwargs):
    """
    Apply empty_like to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the empty_like operation.
    """
    from onnx9000.core.ops.torch_auto import empty_like as core_empty_like

    return core_empty_like(input_tensor, *args, **kwargs)


def empty_permuted(input_tensor, *args, **kwargs):
    """
    Apply empty_permuted to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the empty_permuted operation.
    """
    from onnx9000.core.ops.torch_auto import empty_permuted as core_empty_permuted

    return core_empty_permuted(input_tensor, *args, **kwargs)


def empty_quantized(input_tensor, *args, **kwargs):
    """
    Apply empty_quantized to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the empty_quantized operation.
    """
    from onnx9000.core.ops.torch_auto import empty_quantized as core_empty_quantized

    return core_empty_quantized(input_tensor, *args, **kwargs)


def empty_strided(input_tensor, *args, **kwargs):
    """
    Apply empty_strided to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the empty_strided operation.
    """
    from onnx9000.core.ops.torch_auto import empty_strided as core_empty_strided

    return core_empty_strided(input_tensor, *args, **kwargs)


def eq(input_tensor, *args, **kwargs):
    """
    Apply eq to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the eq operation.
    """
    from onnx9000.core.ops.torch_auto import eq as core_eq

    return core_eq(input_tensor, *args, **kwargs)


def equal(input_tensor, *args, **kwargs):
    """
    Apply equal to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the equal operation.
    """
    from onnx9000.core.ops import equal as core_equal

    return core_equal(input_tensor, *args, **kwargs)


def erf(input_tensor, *args, **kwargs):
    """
    Apply erf to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the erf operation.
    """
    from onnx9000.core.ops import erf as core_erf

    return core_erf(input_tensor, *args, **kwargs)


def erf_(input_tensor, *args, **kwargs):
    """
    Apply erf_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the erf_ operation.
    """
    from onnx9000.core.ops.torch_auto import erf_ as core_erf_

    return core_erf_(input_tensor, *args, **kwargs)


def erfc(input_tensor, *args, **kwargs):
    """
    Apply erfc to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the erfc operation.
    """
    from onnx9000.core.ops.torch_auto import erfc as core_erfc

    return core_erfc(input_tensor, *args, **kwargs)


def erfc_(input_tensor, *args, **kwargs):
    """
    Apply erfc_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the erfc_ operation.
    """
    from onnx9000.core.ops.torch_auto import erfc_ as core_erfc_

    return core_erfc_(input_tensor, *args, **kwargs)


def erfinv(input_tensor, *args, **kwargs):
    """
    Apply erfinv to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the erfinv operation.
    """
    from onnx9000.core.ops.torch_auto import erfinv as core_erfinv

    return core_erfinv(input_tensor, *args, **kwargs)


def exp(input_tensor, *args, **kwargs):
    """
    Apply exp to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the exp operation.
    """
    from onnx9000.core.ops import exp as core_exp

    return core_exp(input_tensor, *args, **kwargs)


def exp2(input_tensor, *args, **kwargs):
    """
    Apply exp2 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the exp2 operation.
    """
    from onnx9000.core.ops.torch_auto import exp2 as core_exp2

    return core_exp2(input_tensor, *args, **kwargs)


def exp2_(input_tensor, *args, **kwargs):
    """
    Apply exp2_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the exp2_ operation.
    """
    from onnx9000.core.ops.torch_auto import exp2_ as core_exp2_

    return core_exp2_(input_tensor, *args, **kwargs)


def exp_(input_tensor, *args, **kwargs):
    """
    Apply exp_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the exp_ operation.
    """
    from onnx9000.core.ops.torch_auto import exp_ as core_exp_

    return core_exp_(input_tensor, *args, **kwargs)


def expand_copy(input_tensor, *args, **kwargs):
    """
    Apply expand_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the expand_copy operation.
    """
    from onnx9000.core.ops.torch_auto import expand_copy as core_expand_copy

    return core_expand_copy(input_tensor, *args, **kwargs)


def expm1(input_tensor, *args, **kwargs):
    """
    Apply expm1 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the expm1 operation.
    """
    from onnx9000.core.ops.torch_auto import expm1 as core_expm1

    return core_expm1(input_tensor, *args, **kwargs)


def expm1_(input_tensor, *args, **kwargs):
    """
    Apply expm1_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the expm1_ operation.
    """
    from onnx9000.core.ops.torch_auto import expm1_ as core_expm1_

    return core_expm1_(input_tensor, *args, **kwargs)


def eye(input_tensor, *args, **kwargs):
    """
    Apply eye to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the eye operation.
    """
    from onnx9000.core.ops.torch_auto import eye as core_eye

    return core_eye(input_tensor, *args, **kwargs)


def fake_quantize_per_channel_affine(input_tensor, *args, **kwargs):
    """
    Apply fake_quantize_per_channel_affine to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fake_quantize_per_channel_affine operation.
    """
    from onnx9000.core.ops.torch_auto import (
        fake_quantize_per_channel_affine as core_fake_quantize_per_channel_affine,
    )

    return core_fake_quantize_per_channel_affine(input_tensor, *args, **kwargs)


def fake_quantize_per_tensor_affine(input_tensor, *args, **kwargs):
    """
    Apply fake_quantize_per_tensor_affine to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fake_quantize_per_tensor_affine operation.
    """
    from onnx9000.core.ops.torch_auto import (
        fake_quantize_per_tensor_affine as core_fake_quantize_per_tensor_affine,
    )

    return core_fake_quantize_per_tensor_affine(input_tensor, *args, **kwargs)


def fbgemm_linear_fp16_weight(input_tensor, *args, **kwargs):
    """
    Apply fbgemm_linear_fp16_weight to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fbgemm_linear_fp16_weight operation.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_linear_fp16_weight as core_fbgemm_linear_fp16_weight,
    )

    return core_fbgemm_linear_fp16_weight(input_tensor, *args, **kwargs)


def fbgemm_linear_fp16_weight_fp32_activation(input_tensor, *args, **kwargs):
    """
    Apply fbgemm_linear_fp16_weight_fp32_activation to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fbgemm_linear_fp16_weight_fp32_activation operation.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_linear_fp16_weight_fp32_activation as core_fbgemm_linear_fp16_weight_fp32_activation,
    )

    return core_fbgemm_linear_fp16_weight_fp32_activation(input_tensor, *args, **kwargs)


def fbgemm_linear_int8_weight(input_tensor, *args, **kwargs):
    """
    Apply fbgemm_linear_int8_weight to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fbgemm_linear_int8_weight operation.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_linear_int8_weight as core_fbgemm_linear_int8_weight,
    )

    return core_fbgemm_linear_int8_weight(input_tensor, *args, **kwargs)


def fbgemm_linear_int8_weight_fp32_activation(input_tensor, *args, **kwargs):
    """
    Apply fbgemm_linear_int8_weight_fp32_activation to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fbgemm_linear_int8_weight_fp32_activation operation.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_linear_int8_weight_fp32_activation as core_fbgemm_linear_int8_weight_fp32_activation,
    )

    return core_fbgemm_linear_int8_weight_fp32_activation(input_tensor, *args, **kwargs)


def fbgemm_linear_quantize_weight(input_tensor, *args, **kwargs):
    """
    Apply fbgemm_linear_quantize_weight to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fbgemm_linear_quantize_weight operation.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_linear_quantize_weight as core_fbgemm_linear_quantize_weight,
    )

    return core_fbgemm_linear_quantize_weight(input_tensor, *args, **kwargs)


def fbgemm_pack_gemm_matrix_fp16(input_tensor, *args, **kwargs):
    """
    Apply fbgemm_pack_gemm_matrix_fp16 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fbgemm_pack_gemm_matrix_fp16 operation.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_pack_gemm_matrix_fp16 as core_fbgemm_pack_gemm_matrix_fp16,
    )

    return core_fbgemm_pack_gemm_matrix_fp16(input_tensor, *args, **kwargs)


def fbgemm_pack_quantized_matrix(input_tensor, *args, **kwargs):
    """
    Apply fbgemm_pack_quantized_matrix to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fbgemm_pack_quantized_matrix operation.
    """
    from onnx9000.core.ops.torch_auto import (
        fbgemm_pack_quantized_matrix as core_fbgemm_pack_quantized_matrix,
    )

    return core_fbgemm_pack_quantized_matrix(input_tensor, *args, **kwargs)


def feature_alpha_dropout(input_tensor, *args, **kwargs):
    """
    Apply feature_alpha_dropout to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the feature_alpha_dropout operation.
    """
    from onnx9000.core.ops.torch_auto import feature_alpha_dropout as core_feature_alpha_dropout

    return core_feature_alpha_dropout(input_tensor, *args, **kwargs)


def feature_alpha_dropout_(input_tensor, *args, **kwargs):
    """
    Apply feature_alpha_dropout_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the feature_alpha_dropout_ operation.
    """
    from onnx9000.core.ops.torch_auto import feature_alpha_dropout_ as core_feature_alpha_dropout_

    return core_feature_alpha_dropout_(input_tensor, *args, **kwargs)


def feature_dropout(input_tensor, *args, **kwargs):
    """
    Apply feature_dropout to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the feature_dropout operation.
    """
    from onnx9000.core.ops.torch_auto import feature_dropout as core_feature_dropout

    return core_feature_dropout(input_tensor, *args, **kwargs)


def feature_dropout_(input_tensor, *args, **kwargs):
    """
    Apply feature_dropout_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the feature_dropout_ operation.
    """
    from onnx9000.core.ops.torch_auto import feature_dropout_ as core_feature_dropout_

    return core_feature_dropout_(input_tensor, *args, **kwargs)


def fill(input_tensor, *args, **kwargs):
    """
    Apply fill to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fill operation.
    """
    from onnx9000.core.ops.torch_auto import fill as core_fill

    return core_fill(input_tensor, *args, **kwargs)


def fill_(input_tensor, *args, **kwargs):
    """
    Apply fill_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fill_ operation.
    """
    from onnx9000.core.ops.torch_auto import fill_ as core_fill_

    return core_fill_(input_tensor, *args, **kwargs)


def fix(input_tensor, *args, **kwargs):
    """
    Apply fix to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fix operation.
    """
    from onnx9000.core.ops.torch_auto import fix as core_fix

    return core_fix(input_tensor, *args, **kwargs)


def fix_(input_tensor, *args, **kwargs):
    """
    Apply fix_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fix_ operation.
    """
    from onnx9000.core.ops.torch_auto import fix_ as core_fix_

    return core_fix_(input_tensor, *args, **kwargs)


def flatten(input_tensor, *args, **kwargs):
    """
    Apply flatten to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the flatten operation.
    """
    from onnx9000.core.ops.torch_auto import flatten as core_flatten

    return core_flatten(input_tensor, *args, **kwargs)


def flip(input_tensor, *args, **kwargs):
    """
    Apply flip to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the flip operation.
    """
    from onnx9000.core.ops.torch_auto import flip as core_flip

    return core_flip(input_tensor, *args, **kwargs)


def fliplr(input_tensor, *args, **kwargs):
    """
    Apply fliplr to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fliplr operation.
    """
    from onnx9000.core.ops.torch_auto import fliplr as core_fliplr

    return core_fliplr(input_tensor, *args, **kwargs)


def flipud(input_tensor, *args, **kwargs):
    """
    Apply flipud to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the flipud operation.
    """
    from onnx9000.core.ops.torch_auto import flipud as core_flipud

    return core_flipud(input_tensor, *args, **kwargs)


def float_power(input_tensor, *args, **kwargs):
    """
    Apply float_power to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the float_power operation.
    """
    from onnx9000.core.ops.torch_auto import float_power as core_float_power

    return core_float_power(input_tensor, *args, **kwargs)


def floor(input_tensor, *args, **kwargs):
    """
    Apply floor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the floor operation.
    """
    from onnx9000.core.ops import floor as core_floor

    return core_floor(input_tensor, *args, **kwargs)


def floor_(input_tensor, *args, **kwargs):
    """
    Apply floor_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the floor_ operation.
    """
    from onnx9000.core.ops.torch_auto import floor_ as core_floor_

    return core_floor_(input_tensor, *args, **kwargs)


def floor_divide(input_tensor, *args, **kwargs):
    """
    Apply floor_divide to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the floor_divide operation.
    """
    from onnx9000.core.ops.torch_auto import floor_divide as core_floor_divide

    return core_floor_divide(input_tensor, *args, **kwargs)


def fmax(input_tensor, *args, **kwargs):
    """
    Apply fmax to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fmax operation.
    """
    from onnx9000.core.ops.torch_auto import fmax as core_fmax

    return core_fmax(input_tensor, *args, **kwargs)


def fmin(input_tensor, *args, **kwargs):
    """
    Apply fmin to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fmin operation.
    """
    from onnx9000.core.ops.torch_auto import fmin as core_fmin

    return core_fmin(input_tensor, *args, **kwargs)


def fmod(input_tensor, *args, **kwargs):
    """
    Apply fmod to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fmod operation.
    """
    from onnx9000.core.ops.torch_auto import fmod as core_fmod

    return core_fmod(input_tensor, *args, **kwargs)


def frac(input_tensor, *args, **kwargs):
    """
    Apply frac to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the frac operation.
    """
    from onnx9000.core.ops.torch_auto import frac as core_frac

    return core_frac(input_tensor, *args, **kwargs)


def frac_(input_tensor, *args, **kwargs):
    """
    Apply frac_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the frac_ operation.
    """
    from onnx9000.core.ops.torch_auto import frac_ as core_frac_

    return core_frac_(input_tensor, *args, **kwargs)


def frexp(input_tensor, *args, **kwargs):
    """
    Apply frexp to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the frexp operation.
    """
    from onnx9000.core.ops.torch_auto import frexp as core_frexp

    return core_frexp(input_tensor, *args, **kwargs)


def frobenius_norm(input_tensor, *args, **kwargs):
    """
    Apply frobenius_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the frobenius_norm operation.
    """
    from onnx9000.core.ops.torch_auto import frobenius_norm as core_frobenius_norm

    return core_frobenius_norm(input_tensor, *args, **kwargs)


def from_file(input_tensor, *args, **kwargs):
    """
    Apply from_file to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the from_file operation.
    """
    from onnx9000.core.ops.torch_auto import from_file as core_from_file

    return core_from_file(input_tensor, *args, **kwargs)


def from_numpy(input_tensor, *args, **kwargs):
    """
    Apply from_numpy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the from_numpy operation.
    """
    from onnx9000.core.ops.torch_auto import from_numpy as core_from_numpy

    return core_from_numpy(input_tensor, *args, **kwargs)


def frombuffer(input_tensor, *args, **kwargs):
    """
    Apply frombuffer to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the frombuffer operation.
    """
    from onnx9000.core.ops.torch_auto import frombuffer as core_frombuffer

    return core_frombuffer(input_tensor, *args, **kwargs)


def full(input_tensor, *args, **kwargs):
    """
    Apply full to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the full operation.
    """
    from onnx9000.core.ops.torch_auto import full as core_full

    return core_full(input_tensor, *args, **kwargs)


def full_like(input_tensor, *args, **kwargs):
    """
    Apply full_like to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the full_like operation.
    """
    from onnx9000.core.ops.torch_auto import full_like as core_full_like

    return core_full_like(input_tensor, *args, **kwargs)


def fused_moving_avg_obs_fake_quant(input_tensor, *args, **kwargs):
    """
    Apply fused_moving_avg_obs_fake_quant to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the fused_moving_avg_obs_fake_quant operation.
    """
    from onnx9000.core.ops.torch_auto import (
        fused_moving_avg_obs_fake_quant as core_fused_moving_avg_obs_fake_quant,
    )

    return core_fused_moving_avg_obs_fake_quant(input_tensor, *args, **kwargs)


def gather(input_tensor, *args, **kwargs):
    """
    Apply gather to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the gather operation.
    """
    from onnx9000.core.ops import gather as core_gather

    return core_gather(input_tensor, *args, **kwargs)


def gcd(input_tensor, *args, **kwargs):
    """
    Apply gcd to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the gcd operation.
    """
    from onnx9000.core.ops.torch_auto import gcd as core_gcd

    return core_gcd(input_tensor, *args, **kwargs)


def gcd_(input_tensor, *args, **kwargs):
    """
    Apply gcd_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the gcd_ operation.
    """
    from onnx9000.core.ops.torch_auto import gcd_ as core_gcd_

    return core_gcd_(input_tensor, *args, **kwargs)


def ge(input_tensor, *args, **kwargs):
    """
    Apply ge to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ge operation.
    """
    from onnx9000.core.ops.torch_auto import ge as core_ge

    return core_ge(input_tensor, *args, **kwargs)


def geqrf(input_tensor, *args, **kwargs):
    """
    Apply geqrf to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the geqrf operation.
    """
    from onnx9000.core.ops.torch_auto import geqrf as core_geqrf

    return core_geqrf(input_tensor, *args, **kwargs)


def ger(input_tensor, *args, **kwargs):
    """
    Apply ger to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ger operation.
    """
    from onnx9000.core.ops.torch_auto import ger as core_ger

    return core_ger(input_tensor, *args, **kwargs)


def get_device(input_tensor, *args, **kwargs):
    """
    Apply get_device to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the get_device operation.
    """
    from onnx9000.core.ops.torch_auto import get_device as core_get_device

    return core_get_device(input_tensor, *args, **kwargs)


def gradient(input_tensor, *args, **kwargs):
    """
    Apply gradient to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the gradient operation.
    """
    from onnx9000.core.ops.torch_auto import gradient as core_gradient

    return core_gradient(input_tensor, *args, **kwargs)


def greater(input_tensor, *args, **kwargs):
    """
    Apply greater to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the greater operation.
    """
    from onnx9000.core.ops import greater as core_greater

    return core_greater(input_tensor, *args, **kwargs)


def greater_equal(input_tensor, *args, **kwargs):
    """
    Apply greater_equal to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the greater_equal operation.
    """
    from onnx9000.core.ops.torch_auto import greater_equal as core_greater_equal

    return core_greater_equal(input_tensor, *args, **kwargs)


def grid_sampler(input_tensor, *args, **kwargs):
    """
    Apply grid_sampler to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the grid_sampler operation.
    """
    from onnx9000.core.ops.torch_auto import grid_sampler as core_grid_sampler

    return core_grid_sampler(input_tensor, *args, **kwargs)


def grid_sampler_2d(input_tensor, *args, **kwargs):
    """
    Apply grid_sampler_2d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the grid_sampler_2d operation.
    """
    from onnx9000.core.ops.torch_auto import grid_sampler_2d as core_grid_sampler_2d

    return core_grid_sampler_2d(input_tensor, *args, **kwargs)


def grid_sampler_3d(input_tensor, *args, **kwargs):
    """
    Apply grid_sampler_3d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the grid_sampler_3d operation.
    """
    from onnx9000.core.ops.torch_auto import grid_sampler_3d as core_grid_sampler_3d

    return core_grid_sampler_3d(input_tensor, *args, **kwargs)


def group_norm(input_tensor, *args, **kwargs):
    """
    Apply group_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the group_norm operation.
    """
    from onnx9000.core.ops.torch_auto import group_norm as core_group_norm

    return core_group_norm(input_tensor, *args, **kwargs)


def gru(input_tensor, *args, **kwargs):
    """
    Apply gru to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the gru operation.
    """
    from onnx9000.core.ops import gru as core_gru

    return core_gru(input_tensor, *args, **kwargs)


def gru_cell(input_tensor, *args, **kwargs):
    """
    Apply gru_cell to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the gru_cell operation.
    """
    from onnx9000.core.ops.torch_auto import gru_cell as core_gru_cell

    return core_gru_cell(input_tensor, *args, **kwargs)


def gt(input_tensor, *args, **kwargs):
    """
    Apply gt to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the gt operation.
    """
    from onnx9000.core.ops.torch_auto import gt as core_gt

    return core_gt(input_tensor, *args, **kwargs)


def hamming_window(input_tensor, *args, **kwargs):
    """
    Apply hamming_window to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the hamming_window operation.
    """
    from onnx9000.core.ops.torch_auto import hamming_window as core_hamming_window

    return core_hamming_window(input_tensor, *args, **kwargs)


def hann_window(input_tensor, *args, **kwargs):
    """
    Apply hann_window to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the hann_window operation.
    """
    from onnx9000.core.ops.torch_auto import hann_window as core_hann_window

    return core_hann_window(input_tensor, *args, **kwargs)


def hardshrink(input_tensor, *args, **kwargs):
    """
    Apply hardshrink to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the hardshrink operation.
    """
    from onnx9000.core.ops.torch_auto import hardshrink as core_hardshrink

    return core_hardshrink(input_tensor, *args, **kwargs)


def hash_tensor(input_tensor, *args, **kwargs):
    """
    Apply hash_tensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the hash_tensor operation.
    """
    from onnx9000.core.ops.torch_auto import hash_tensor as core_hash_tensor

    return core_hash_tensor(input_tensor, *args, **kwargs)


def heaviside(input_tensor, *args, **kwargs):
    """
    Apply heaviside to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the heaviside operation.
    """
    from onnx9000.core.ops.torch_auto import heaviside as core_heaviside

    return core_heaviside(input_tensor, *args, **kwargs)


def hinge_embedding_loss(input_tensor, *args, **kwargs):
    """
    Apply hinge_embedding_loss to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the hinge_embedding_loss operation.
    """
    from onnx9000.core.ops.torch_auto import hinge_embedding_loss as core_hinge_embedding_loss

    return core_hinge_embedding_loss(input_tensor, *args, **kwargs)


def histc(input_tensor, *args, **kwargs):
    """
    Apply histc to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the histc operation.
    """
    from onnx9000.core.ops.torch_auto import histc as core_histc

    return core_histc(input_tensor, *args, **kwargs)


def histogram(input_tensor, *args, **kwargs):
    """
    Apply histogram to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the histogram operation.
    """
    from onnx9000.core.ops.torch_auto import histogram as core_histogram

    return core_histogram(input_tensor, *args, **kwargs)


def histogramdd(input_tensor, *args, **kwargs):
    """
    Apply histogramdd to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the histogramdd operation.
    """
    from onnx9000.core.ops.torch_auto import histogramdd as core_histogramdd

    return core_histogramdd(input_tensor, *args, **kwargs)


def hsmm(input_tensor, *args, **kwargs):
    """
    Apply hsmm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the hsmm operation.
    """
    from onnx9000.core.ops.torch_auto import hsmm as core_hsmm

    return core_hsmm(input_tensor, *args, **kwargs)


def hsplit(input_tensor, *args, **kwargs):
    """
    Apply hsplit to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the hsplit operation.
    """
    from onnx9000.core.ops.torch_auto import hsplit as core_hsplit

    return core_hsplit(input_tensor, *args, **kwargs)


def hspmm(input_tensor, *args, **kwargs):
    """
    Apply hspmm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the hspmm operation.
    """
    from onnx9000.core.ops.torch_auto import hspmm as core_hspmm

    return core_hspmm(input_tensor, *args, **kwargs)


def hstack(input_tensor, *args, **kwargs):
    """
    Apply hstack to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the hstack operation.
    """
    from onnx9000.core.ops.torch_auto import hstack as core_hstack

    return core_hstack(input_tensor, *args, **kwargs)


def hypot(input_tensor, *args, **kwargs):
    """
    Apply hypot to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the hypot operation.
    """
    from onnx9000.core.ops.torch_auto import hypot as core_hypot

    return core_hypot(input_tensor, *args, **kwargs)


def i0(input_tensor, *args, **kwargs):
    """
    Apply i0 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the i0 operation.
    """
    from onnx9000.core.ops.torch_auto import i0 as core_i0

    return core_i0(input_tensor, *args, **kwargs)


def i0_(input_tensor, *args, **kwargs):
    """
    Apply i0_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the i0_ operation.
    """
    from onnx9000.core.ops.torch_auto import i0_ as core_i0_

    return core_i0_(input_tensor, *args, **kwargs)


def igamma(input_tensor, *args, **kwargs):
    """
    Apply igamma to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the igamma operation.
    """
    from onnx9000.core.ops.torch_auto import igamma as core_igamma

    return core_igamma(input_tensor, *args, **kwargs)


def igammac(input_tensor, *args, **kwargs):
    """
    Apply igammac to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the igammac operation.
    """
    from onnx9000.core.ops.torch_auto import igammac as core_igammac

    return core_igammac(input_tensor, *args, **kwargs)


def imag(input_tensor, *args, **kwargs):
    """
    Apply imag to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the imag operation.
    """
    from onnx9000.core.ops.torch_auto import imag as core_imag

    return core_imag(input_tensor, *args, **kwargs)


def index_add(input_tensor, *args, **kwargs):
    """
    Apply index_add to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the index_add operation.
    """
    from onnx9000.core.ops.torch_auto import index_add as core_index_add

    return core_index_add(input_tensor, *args, **kwargs)


def index_copy(input_tensor, *args, **kwargs):
    """
    Apply index_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the index_copy operation.
    """
    from onnx9000.core.ops.torch_auto import index_copy as core_index_copy

    return core_index_copy(input_tensor, *args, **kwargs)


def index_fill(input_tensor, *args, **kwargs):
    """
    Apply index_fill to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the index_fill operation.
    """
    from onnx9000.core.ops.torch_auto import index_fill as core_index_fill

    return core_index_fill(input_tensor, *args, **kwargs)


def index_put(input_tensor, *args, **kwargs):
    """
    Apply index_put to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the index_put operation.
    """
    from onnx9000.core.ops.torch_auto import index_put as core_index_put

    return core_index_put(input_tensor, *args, **kwargs)


def index_put_(input_tensor, *args, **kwargs):
    """
    Apply index_put_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the index_put_ operation.
    """
    from onnx9000.core.ops.torch_auto import index_put_ as core_index_put_

    return core_index_put_(input_tensor, *args, **kwargs)


def index_reduce(input_tensor, *args, **kwargs):
    """
    Apply index_reduce to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the index_reduce operation.
    """
    from onnx9000.core.ops.torch_auto import index_reduce as core_index_reduce

    return core_index_reduce(input_tensor, *args, **kwargs)


def index_select(input_tensor, *args, **kwargs):
    """
    Apply index_select to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the index_select operation.
    """
    from onnx9000.core.ops.torch_auto import index_select as core_index_select

    return core_index_select(input_tensor, *args, **kwargs)


def indices_copy(input_tensor, *args, **kwargs):
    """
    Apply indices_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the indices_copy operation.
    """
    from onnx9000.core.ops.torch_auto import indices_copy as core_indices_copy

    return core_indices_copy(input_tensor, *args, **kwargs)


def inner(input_tensor, *args, **kwargs):
    """
    Apply inner to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the inner operation.
    """
    from onnx9000.core.ops.torch_auto import inner as core_inner

    return core_inner(input_tensor, *args, **kwargs)


def instance_norm(input_tensor, *args, **kwargs):
    """
    Apply instance_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the instance_norm operation.
    """
    from onnx9000.core.ops.torch_auto import instance_norm as core_instance_norm

    return core_instance_norm(input_tensor, *args, **kwargs)


def int_repr(input_tensor, *args, **kwargs):
    """
    Apply int_repr to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the int_repr operation.
    """
    from onnx9000.core.ops.torch_auto import int_repr as core_int_repr

    return core_int_repr(input_tensor, *args, **kwargs)


def inverse(input_tensor, *args, **kwargs):
    """
    Apply inverse to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the inverse operation.
    """
    from onnx9000.core.ops.torch_auto import inverse as core_inverse

    return core_inverse(input_tensor, *args, **kwargs)


def is_complex(input_tensor, *args, **kwargs):
    """
    Apply is_complex to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_complex operation.
    """
    from onnx9000.core.ops.torch_auto import is_complex as core_is_complex

    return core_is_complex(input_tensor, *args, **kwargs)


def is_conj(input_tensor, *args, **kwargs):
    """
    Apply is_conj to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_conj operation.
    """
    from onnx9000.core.ops.torch_auto import is_conj as core_is_conj

    return core_is_conj(input_tensor, *args, **kwargs)


def is_distributed(input_tensor, *args, **kwargs):
    """
    Apply is_distributed to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_distributed operation.
    """
    from onnx9000.core.ops.torch_auto import is_distributed as core_is_distributed

    return core_is_distributed(input_tensor, *args, **kwargs)


def is_floating_point(input_tensor, *args, **kwargs):
    """
    Apply is_floating_point to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_floating_point operation.
    """
    from onnx9000.core.ops.torch_auto import is_floating_point as core_is_floating_point

    return core_is_floating_point(input_tensor, *args, **kwargs)


def is_inference(input_tensor, *args, **kwargs):
    """
    Apply is_inference to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_inference operation.
    """
    from onnx9000.core.ops.torch_auto import is_inference as core_is_inference

    return core_is_inference(input_tensor, *args, **kwargs)


def is_neg(input_tensor, *args, **kwargs):
    """
    Apply is_neg to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_neg operation.
    """
    from onnx9000.core.ops.torch_auto import is_neg as core_is_neg

    return core_is_neg(input_tensor, *args, **kwargs)


def is_nonzero(input_tensor, *args, **kwargs):
    """
    Apply is_nonzero to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_nonzero operation.
    """
    from onnx9000.core.ops.torch_auto import is_nonzero as core_is_nonzero

    return core_is_nonzero(input_tensor, *args, **kwargs)


def is_same_size(input_tensor, *args, **kwargs):
    """
    Apply is_same_size to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_same_size operation.
    """
    from onnx9000.core.ops.torch_auto import is_same_size as core_is_same_size

    return core_is_same_size(input_tensor, *args, **kwargs)


def is_signed(input_tensor, *args, **kwargs):
    """
    Apply is_signed to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_signed operation.
    """
    from onnx9000.core.ops.torch_auto import is_signed as core_is_signed

    return core_is_signed(input_tensor, *args, **kwargs)


def is_vulkan_available(input_tensor, *args, **kwargs):
    """
    Apply is_vulkan_available to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the is_vulkan_available operation.
    """
    from onnx9000.core.ops.torch_auto import is_vulkan_available as core_is_vulkan_available

    return core_is_vulkan_available(input_tensor, *args, **kwargs)


def isclose(input_tensor, *args, **kwargs):
    """
    Apply isclose to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the isclose operation.
    """
    from onnx9000.core.ops.torch_auto import isclose as core_isclose

    return core_isclose(input_tensor, *args, **kwargs)


def isfinite(input_tensor, *args, **kwargs):
    """
    Apply isfinite to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the isfinite operation.
    """
    from onnx9000.core.ops.torch_auto import isfinite as core_isfinite

    return core_isfinite(input_tensor, *args, **kwargs)


def isin(input_tensor, *args, **kwargs):
    """
    Apply isin to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the isin operation.
    """
    from onnx9000.core.ops.torch_auto import isin as core_isin

    return core_isin(input_tensor, *args, **kwargs)


def isinf(input_tensor, *args, **kwargs):
    """
    Apply isinf to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the isinf operation.
    """
    from onnx9000.core.ops import isinf as core_isinf

    return core_isinf(input_tensor, *args, **kwargs)


def isnan(input_tensor, *args, **kwargs):
    """
    Apply isnan to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the isnan operation.
    """
    from onnx9000.core.ops import isnan as core_isnan

    return core_isnan(input_tensor, *args, **kwargs)


def isneginf(input_tensor, *args, **kwargs):
    """
    Apply isneginf to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the isneginf operation.
    """
    from onnx9000.core.ops.torch_auto import isneginf as core_isneginf

    return core_isneginf(input_tensor, *args, **kwargs)


def isposinf(input_tensor, *args, **kwargs):
    """
    Apply isposinf to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the isposinf operation.
    """
    from onnx9000.core.ops.torch_auto import isposinf as core_isposinf

    return core_isposinf(input_tensor, *args, **kwargs)


def isreal(input_tensor, *args, **kwargs):
    """
    Apply isreal to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the isreal operation.
    """
    from onnx9000.core.ops.torch_auto import isreal as core_isreal

    return core_isreal(input_tensor, *args, **kwargs)


def istft(input_tensor, *args, **kwargs):
    """
    Apply istft to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the istft operation.
    """
    from onnx9000.core.ops.torch_auto import istft as core_istft

    return core_istft(input_tensor, *args, **kwargs)


def kaiser_window(input_tensor, *args, **kwargs):
    """
    Apply kaiser_window to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the kaiser_window operation.
    """
    from onnx9000.core.ops.torch_auto import kaiser_window as core_kaiser_window

    return core_kaiser_window(input_tensor, *args, **kwargs)


def kl_div(input_tensor, *args, **kwargs):
    """
    Apply kl_div to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the kl_div operation.
    """
    from onnx9000.core.ops.torch_auto import kl_div as core_kl_div

    return core_kl_div(input_tensor, *args, **kwargs)


def kron(input_tensor, *args, **kwargs):
    """
    Apply kron to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the kron operation.
    """
    from onnx9000.core.ops.torch_auto import kron as core_kron

    return core_kron(input_tensor, *args, **kwargs)


def kthvalue(input_tensor, *args, **kwargs):
    """
    Apply kthvalue to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the kthvalue operation.
    """
    from onnx9000.core.ops.torch_auto import kthvalue as core_kthvalue

    return core_kthvalue(input_tensor, *args, **kwargs)


def layer_norm(input_tensor, *args, **kwargs):
    """
    Apply layer_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the layer_norm operation.
    """
    from onnx9000.core.ops.torch_auto import layer_norm as core_layer_norm

    return core_layer_norm(input_tensor, *args, **kwargs)


def lcm(input_tensor, *args, **kwargs):
    """
    Apply lcm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the lcm operation.
    """
    from onnx9000.core.ops.torch_auto import lcm as core_lcm

    return core_lcm(input_tensor, *args, **kwargs)


def lcm_(input_tensor, *args, **kwargs):
    """
    Apply lcm_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the lcm_ operation.
    """
    from onnx9000.core.ops.torch_auto import lcm_ as core_lcm_

    return core_lcm_(input_tensor, *args, **kwargs)


def ldexp(input_tensor, *args, **kwargs):
    """
    Apply ldexp to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ldexp operation.
    """
    from onnx9000.core.ops.torch_auto import ldexp as core_ldexp

    return core_ldexp(input_tensor, *args, **kwargs)


def ldexp_(input_tensor, *args, **kwargs):
    """
    Apply ldexp_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ldexp_ operation.
    """
    from onnx9000.core.ops.torch_auto import ldexp_ as core_ldexp_

    return core_ldexp_(input_tensor, *args, **kwargs)


def le(input_tensor, *args, **kwargs):
    """
    Apply le to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the le operation.
    """
    from onnx9000.core.ops.torch_auto import le as core_le

    return core_le(input_tensor, *args, **kwargs)


def lerp(input_tensor, *args, **kwargs):
    """
    Apply lerp to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the lerp operation.
    """
    from onnx9000.core.ops.torch_auto import lerp as core_lerp

    return core_lerp(input_tensor, *args, **kwargs)


def less(input_tensor, *args, **kwargs):
    """
    Apply less to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the less operation.
    """
    from onnx9000.core.ops import less as core_less

    return core_less(input_tensor, *args, **kwargs)


def less_equal(input_tensor, *args, **kwargs):
    """
    Apply less_equal to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the less_equal operation.
    """
    from onnx9000.core.ops.torch_auto import less_equal as core_less_equal

    return core_less_equal(input_tensor, *args, **kwargs)


def lgamma(input_tensor, *args, **kwargs):
    """
    Apply lgamma to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the lgamma operation.
    """
    from onnx9000.core.ops.torch_auto import lgamma as core_lgamma

    return core_lgamma(input_tensor, *args, **kwargs)


def linspace(input_tensor, *args, **kwargs):
    """
    Apply linspace to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the linspace operation.
    """
    from onnx9000.core.ops.torch_auto import linspace as core_linspace

    return core_linspace(input_tensor, *args, **kwargs)


def log(input_tensor, *args, **kwargs):
    """
    Apply log to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the log operation.
    """
    from onnx9000.core.ops.torch_auto import log as core_log

    return core_log(input_tensor, *args, **kwargs)


def log10(input_tensor, *args, **kwargs):
    """
    Apply log10 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the log10 operation.
    """
    from onnx9000.core.ops.torch_auto import log10 as core_log10

    return core_log10(input_tensor, *args, **kwargs)


def log10_(input_tensor, *args, **kwargs):
    """
    Apply log10_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the log10_ operation.
    """
    from onnx9000.core.ops.torch_auto import log10_ as core_log10_

    return core_log10_(input_tensor, *args, **kwargs)


def log1p(input_tensor, *args, **kwargs):
    """
    Apply log1p to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the log1p operation.
    """
    from onnx9000.core.ops.torch_auto import log1p as core_log1p

    return core_log1p(input_tensor, *args, **kwargs)


def log1p_(input_tensor, *args, **kwargs):
    """
    Apply log1p_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the log1p_ operation.
    """
    from onnx9000.core.ops.torch_auto import log1p_ as core_log1p_

    return core_log1p_(input_tensor, *args, **kwargs)


def log2(input_tensor, *args, **kwargs):
    """
    Apply log2 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the log2 operation.
    """
    from onnx9000.core.ops.torch_auto import log2 as core_log2

    return core_log2(input_tensor, *args, **kwargs)


def log2_(input_tensor, *args, **kwargs):
    """
    Apply log2_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the log2_ operation.
    """
    from onnx9000.core.ops.torch_auto import log2_ as core_log2_

    return core_log2_(input_tensor, *args, **kwargs)


def log_(input_tensor, *args, **kwargs):
    """
    Apply log_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the log_ operation.
    """
    from onnx9000.core.ops.torch_auto import log_ as core_log_

    return core_log_(input_tensor, *args, **kwargs)


def log_softmax(input_tensor, *args, **kwargs):
    """
    Apply log_softmax to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the log_softmax operation.
    """
    from onnx9000.core.ops import log_softmax as core_log_softmax

    return core_log_softmax(input_tensor, *args, **kwargs)


def logaddexp(input_tensor, *args, **kwargs):
    """
    Apply logaddexp to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logaddexp operation.
    """
    from onnx9000.core.ops.torch_auto import logaddexp as core_logaddexp

    return core_logaddexp(input_tensor, *args, **kwargs)


def logaddexp2(input_tensor, *args, **kwargs):
    """
    Apply logaddexp2 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logaddexp2 operation.
    """
    from onnx9000.core.ops.torch_auto import logaddexp2 as core_logaddexp2

    return core_logaddexp2(input_tensor, *args, **kwargs)


def logcumsumexp(input_tensor, *args, **kwargs):
    """
    Apply logcumsumexp to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logcumsumexp operation.
    """
    from onnx9000.core.ops.torch_auto import logcumsumexp as core_logcumsumexp

    return core_logcumsumexp(input_tensor, *args, **kwargs)


def logdet(input_tensor, *args, **kwargs):
    """
    Apply logdet to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logdet operation.
    """
    from onnx9000.core.ops.torch_auto import logdet as core_logdet

    return core_logdet(input_tensor, *args, **kwargs)


def logical_and(input_tensor, *args, **kwargs):
    """
    Apply logical_and to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logical_and operation.
    """
    from onnx9000.core.ops.torch_auto import logical_and as core_logical_and

    return core_logical_and(input_tensor, *args, **kwargs)


def logical_not(input_tensor, *args, **kwargs):
    """
    Apply logical_not to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logical_not operation.
    """
    from onnx9000.core.ops.torch_auto import logical_not as core_logical_not

    return core_logical_not(input_tensor, *args, **kwargs)


def logical_or(input_tensor, *args, **kwargs):
    """
    Apply logical_or to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logical_or operation.
    """
    from onnx9000.core.ops.torch_auto import logical_or as core_logical_or

    return core_logical_or(input_tensor, *args, **kwargs)


def logical_xor(input_tensor, *args, **kwargs):
    """
    Apply logical_xor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logical_xor operation.
    """
    from onnx9000.core.ops.torch_auto import logical_xor as core_logical_xor

    return core_logical_xor(input_tensor, *args, **kwargs)


def logit(input_tensor, *args, **kwargs):
    """
    Apply logit to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logit operation.
    """
    from onnx9000.core.ops.torch_auto import logit as core_logit

    return core_logit(input_tensor, *args, **kwargs)


def logit_(input_tensor, *args, **kwargs):
    """
    Apply logit_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logit_ operation.
    """
    from onnx9000.core.ops.torch_auto import logit_ as core_logit_

    return core_logit_(input_tensor, *args, **kwargs)


def logspace(input_tensor, *args, **kwargs):
    """
    Apply logspace to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logspace operation.
    """
    from onnx9000.core.ops.torch_auto import logspace as core_logspace

    return core_logspace(input_tensor, *args, **kwargs)


def logsumexp(input_tensor, *args, **kwargs):
    """
    Apply logsumexp to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the logsumexp operation.
    """
    from onnx9000.core.ops.torch_auto import logsumexp as core_logsumexp

    return core_logsumexp(input_tensor, *args, **kwargs)


def lstm(input_tensor, *args, **kwargs):
    """
    Apply lstm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the lstm operation.
    """
    from onnx9000.core.ops import lstm as core_lstm

    return core_lstm(input_tensor, *args, **kwargs)


def lstm_cell(input_tensor, *args, **kwargs):
    """
    Apply lstm_cell to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the lstm_cell operation.
    """
    from onnx9000.core.ops.torch_auto import lstm_cell as core_lstm_cell

    return core_lstm_cell(input_tensor, *args, **kwargs)


def lt(input_tensor, *args, **kwargs):
    """
    Apply lt to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the lt operation.
    """
    from onnx9000.core.ops.torch_auto import lt as core_lt

    return core_lt(input_tensor, *args, **kwargs)


def lu_solve(input_tensor, *args, **kwargs):
    """
    Apply lu_solve to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the lu_solve operation.
    """
    from onnx9000.core.ops.torch_auto import lu_solve as core_lu_solve

    return core_lu_solve(input_tensor, *args, **kwargs)


def lu_unpack(input_tensor, *args, **kwargs):
    """
    Apply lu_unpack to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the lu_unpack operation.
    """
    from onnx9000.core.ops.torch_auto import lu_unpack as core_lu_unpack

    return core_lu_unpack(input_tensor, *args, **kwargs)


def margin_ranking_loss(input_tensor, *args, **kwargs):
    """
    Apply margin_ranking_loss to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the margin_ranking_loss operation.
    """
    from onnx9000.core.ops.torch_auto import margin_ranking_loss as core_margin_ranking_loss

    return core_margin_ranking_loss(input_tensor, *args, **kwargs)


def masked_fill(input_tensor, *args, **kwargs):
    """
    Apply masked_fill to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the masked_fill operation.
    """
    from onnx9000.core.ops.torch_auto import masked_fill as core_masked_fill

    return core_masked_fill(input_tensor, *args, **kwargs)


def masked_scatter(input_tensor, *args, **kwargs):
    """
    Apply masked_scatter to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the masked_scatter operation.
    """
    from onnx9000.core.ops.torch_auto import masked_scatter as core_masked_scatter

    return core_masked_scatter(input_tensor, *args, **kwargs)


def masked_select(input_tensor, *args, **kwargs):
    """
    Apply masked_select to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the masked_select operation.
    """
    from onnx9000.core.ops.torch_auto import masked_select as core_masked_select

    return core_masked_select(input_tensor, *args, **kwargs)


def matrix_exp(input_tensor, *args, **kwargs):
    """
    Apply matrix_exp to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the matrix_exp operation.
    """
    from onnx9000.core.ops.torch_auto import matrix_exp as core_matrix_exp

    return core_matrix_exp(input_tensor, *args, **kwargs)


def matrix_power(input_tensor, *args, **kwargs):
    """
    Apply matrix_power to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the matrix_power operation.
    """
    from onnx9000.core.ops.torch_auto import matrix_power as core_matrix_power

    return core_matrix_power(input_tensor, *args, **kwargs)


def max(input_tensor, *args, **kwargs):
    """
    Apply max to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the max operation.
    """
    from onnx9000.core.ops import max as core_max

    return core_max(input_tensor, *args, **kwargs)


def max_pool1d(input_tensor, *args, **kwargs):
    """
    Apply max_pool1d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the max_pool1d operation.
    """
    from onnx9000.core.ops.torch_auto import max_pool1d as core_max_pool1d

    return core_max_pool1d(input_tensor, *args, **kwargs)


def max_pool1d_with_indices(input_tensor, *args, **kwargs):
    """
    Apply max_pool1d_with_indices to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the max_pool1d_with_indices operation.
    """
    from onnx9000.core.ops.torch_auto import max_pool1d_with_indices as core_max_pool1d_with_indices

    return core_max_pool1d_with_indices(input_tensor, *args, **kwargs)


def max_pool2d(input_tensor, *args, **kwargs):
    """
    Apply max_pool2d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the max_pool2d operation.
    """
    from onnx9000.core.ops.torch_auto import max_pool2d as core_max_pool2d

    return core_max_pool2d(input_tensor, *args, **kwargs)


def max_pool3d(input_tensor, *args, **kwargs):
    """
    Apply max_pool3d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the max_pool3d operation.
    """
    from onnx9000.core.ops.torch_auto import max_pool3d as core_max_pool3d

    return core_max_pool3d(input_tensor, *args, **kwargs)


def maximum(input_tensor, *args, **kwargs):
    """
    Apply maximum to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the maximum operation.
    """
    from onnx9000.core.ops.torch_auto import maximum as core_maximum

    return core_maximum(input_tensor, *args, **kwargs)


def mean(input_tensor, *args, **kwargs):
    """
    Apply mean to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mean operation.
    """
    from onnx9000.core.ops import mean as core_mean

    return core_mean(input_tensor, *args, **kwargs)


def median(input_tensor, *args, **kwargs):
    """
    Apply median to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the median operation.
    """
    from onnx9000.core.ops.torch_auto import median as core_median

    return core_median(input_tensor, *args, **kwargs)


def meshgrid(input_tensor, *args, **kwargs):
    """
    Apply meshgrid to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the meshgrid operation.
    """
    from onnx9000.core.ops.torch_auto import meshgrid as core_meshgrid

    return core_meshgrid(input_tensor, *args, **kwargs)


def min(input_tensor, *args, **kwargs):
    """
    Apply min to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the min operation.
    """
    from onnx9000.core.ops import min as core_min

    return core_min(input_tensor, *args, **kwargs)


def minimum(input_tensor, *args, **kwargs):
    """
    Apply minimum to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the minimum operation.
    """
    from onnx9000.core.ops.torch_auto import minimum as core_minimum

    return core_minimum(input_tensor, *args, **kwargs)


def miopen_batch_norm(input_tensor, *args, **kwargs):
    """
    Apply miopen_batch_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the miopen_batch_norm operation.
    """
    from onnx9000.core.ops.torch_auto import miopen_batch_norm as core_miopen_batch_norm

    return core_miopen_batch_norm(input_tensor, *args, **kwargs)


def miopen_convolution(input_tensor, *args, **kwargs):
    """
    Apply miopen_convolution to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the miopen_convolution operation.
    """
    from onnx9000.core.ops.torch_auto import miopen_convolution as core_miopen_convolution

    return core_miopen_convolution(input_tensor, *args, **kwargs)


def miopen_convolution_add_relu(input_tensor, *args, **kwargs):
    """
    Apply miopen_convolution_add_relu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the miopen_convolution_add_relu operation.
    """
    from onnx9000.core.ops.torch_auto import (
        miopen_convolution_add_relu as core_miopen_convolution_add_relu,
    )

    return core_miopen_convolution_add_relu(input_tensor, *args, **kwargs)


def miopen_convolution_relu(input_tensor, *args, **kwargs):
    """
    Apply miopen_convolution_relu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the miopen_convolution_relu operation.
    """
    from onnx9000.core.ops.torch_auto import miopen_convolution_relu as core_miopen_convolution_relu

    return core_miopen_convolution_relu(input_tensor, *args, **kwargs)


def miopen_convolution_transpose(input_tensor, *args, **kwargs):
    """
    Apply miopen_convolution_transpose to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the miopen_convolution_transpose operation.
    """
    from onnx9000.core.ops.torch_auto import (
        miopen_convolution_transpose as core_miopen_convolution_transpose,
    )

    return core_miopen_convolution_transpose(input_tensor, *args, **kwargs)


def miopen_ctc_loss(input_tensor, *args, **kwargs):
    """
    Apply miopen_ctc_loss to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the miopen_ctc_loss operation.
    """
    from onnx9000.core.ops.torch_auto import miopen_ctc_loss as core_miopen_ctc_loss

    return core_miopen_ctc_loss(input_tensor, *args, **kwargs)


def miopen_depthwise_convolution(input_tensor, *args, **kwargs):
    """
    Apply miopen_depthwise_convolution to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the miopen_depthwise_convolution operation.
    """
    from onnx9000.core.ops.torch_auto import (
        miopen_depthwise_convolution as core_miopen_depthwise_convolution,
    )

    return core_miopen_depthwise_convolution(input_tensor, *args, **kwargs)


def miopen_rnn(input_tensor, *args, **kwargs):
    """
    Apply miopen_rnn to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the miopen_rnn operation.
    """
    from onnx9000.core.ops.torch_auto import miopen_rnn as core_miopen_rnn

    return core_miopen_rnn(input_tensor, *args, **kwargs)


def mkldnn_adaptive_avg_pool2d(input_tensor, *args, **kwargs):
    """
    Apply mkldnn_adaptive_avg_pool2d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mkldnn_adaptive_avg_pool2d operation.
    """
    from onnx9000.core.ops.torch_auto import (
        mkldnn_adaptive_avg_pool2d as core_mkldnn_adaptive_avg_pool2d,
    )

    return core_mkldnn_adaptive_avg_pool2d(input_tensor, *args, **kwargs)


def mkldnn_convolution(input_tensor, *args, **kwargs):
    """
    Apply mkldnn_convolution to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mkldnn_convolution operation.
    """
    from onnx9000.core.ops.torch_auto import mkldnn_convolution as core_mkldnn_convolution

    return core_mkldnn_convolution(input_tensor, *args, **kwargs)


def mkldnn_linear_backward_weights(input_tensor, *args, **kwargs):
    """
    Apply mkldnn_linear_backward_weights to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mkldnn_linear_backward_weights operation.
    """
    from onnx9000.core.ops.torch_auto import (
        mkldnn_linear_backward_weights as core_mkldnn_linear_backward_weights,
    )

    return core_mkldnn_linear_backward_weights(input_tensor, *args, **kwargs)


def mkldnn_max_pool2d(input_tensor, *args, **kwargs):
    """
    Apply mkldnn_max_pool2d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mkldnn_max_pool2d operation.
    """
    from onnx9000.core.ops.torch_auto import mkldnn_max_pool2d as core_mkldnn_max_pool2d

    return core_mkldnn_max_pool2d(input_tensor, *args, **kwargs)


def mkldnn_max_pool3d(input_tensor, *args, **kwargs):
    """
    Apply mkldnn_max_pool3d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mkldnn_max_pool3d operation.
    """
    from onnx9000.core.ops.torch_auto import mkldnn_max_pool3d as core_mkldnn_max_pool3d

    return core_mkldnn_max_pool3d(input_tensor, *args, **kwargs)


def mkldnn_rnn_layer(input_tensor, *args, **kwargs):
    """
    Apply mkldnn_rnn_layer to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mkldnn_rnn_layer operation.
    """
    from onnx9000.core.ops.torch_auto import mkldnn_rnn_layer as core_mkldnn_rnn_layer

    return core_mkldnn_rnn_layer(input_tensor, *args, **kwargs)


def mm(input_tensor, *args, **kwargs):
    """
    Apply mm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mm operation.
    """
    from onnx9000.core.ops.torch_auto import mm as core_mm

    return core_mm(input_tensor, *args, **kwargs)


def mode(input_tensor, *args, **kwargs):
    """
    Apply mode to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mode operation.
    """
    from onnx9000.core.ops.torch_auto import mode as core_mode

    return core_mode(input_tensor, *args, **kwargs)


def moveaxis(input_tensor, *args, **kwargs):
    """
    Apply moveaxis to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the moveaxis operation.
    """
    from onnx9000.core.ops.torch_auto import moveaxis as core_moveaxis

    return core_moveaxis(input_tensor, *args, **kwargs)


def movedim(input_tensor, *args, **kwargs):
    """
    Apply movedim to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the movedim operation.
    """
    from onnx9000.core.ops.torch_auto import movedim as core_movedim

    return core_movedim(input_tensor, *args, **kwargs)


def msort(input_tensor, *args, **kwargs):
    """
    Apply msort to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the msort operation.
    """
    from onnx9000.core.ops.torch_auto import msort as core_msort

    return core_msort(input_tensor, *args, **kwargs)


def mul(input_tensor, *args, **kwargs):
    """
    Apply mul to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mul operation.
    """
    from onnx9000.core.ops import mul as core_mul

    return core_mul(input_tensor, *args, **kwargs)


def multinomial(input_tensor, *args, **kwargs):
    """
    Apply multinomial to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the multinomial operation.
    """
    from onnx9000.core.ops import multinomial as core_multinomial

    return core_multinomial(input_tensor, *args, **kwargs)


def multiply(input_tensor, *args, **kwargs):
    """
    Apply multiply to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the multiply operation.
    """
    from onnx9000.core.ops.torch_auto import multiply as core_multiply

    return core_multiply(input_tensor, *args, **kwargs)


def mv(input_tensor, *args, **kwargs):
    """
    Apply mv to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mv operation.
    """
    from onnx9000.core.ops.torch_auto import mv as core_mv

    return core_mv(input_tensor, *args, **kwargs)


def mvlgamma(input_tensor, *args, **kwargs):
    """
    Apply mvlgamma to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the mvlgamma operation.
    """
    from onnx9000.core.ops.torch_auto import mvlgamma as core_mvlgamma

    return core_mvlgamma(input_tensor, *args, **kwargs)


def nan_to_num(input_tensor, *args, **kwargs):
    """
    Apply nan_to_num to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the nan_to_num operation.
    """
    from onnx9000.core.ops.torch_auto import nan_to_num as core_nan_to_num

    return core_nan_to_num(input_tensor, *args, **kwargs)


def nan_to_num_(input_tensor, *args, **kwargs):
    """
    Apply nan_to_num_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the nan_to_num_ operation.
    """
    from onnx9000.core.ops.torch_auto import nan_to_num_ as core_nan_to_num_

    return core_nan_to_num_(input_tensor, *args, **kwargs)


def nanmean(input_tensor, *args, **kwargs):
    """
    Apply nanmean to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the nanmean operation.
    """
    from onnx9000.core.ops.torch_auto import nanmean as core_nanmean

    return core_nanmean(input_tensor, *args, **kwargs)


def nanmedian(input_tensor, *args, **kwargs):
    """
    Apply nanmedian to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the nanmedian operation.
    """
    from onnx9000.core.ops.torch_auto import nanmedian as core_nanmedian

    return core_nanmedian(input_tensor, *args, **kwargs)


def nanquantile(input_tensor, *args, **kwargs):
    """
    Apply nanquantile to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the nanquantile operation.
    """
    from onnx9000.core.ops.torch_auto import nanquantile as core_nanquantile

    return core_nanquantile(input_tensor, *args, **kwargs)


def nansum(input_tensor, *args, **kwargs):
    """
    Apply nansum to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the nansum operation.
    """
    from onnx9000.core.ops.torch_auto import nansum as core_nansum

    return core_nansum(input_tensor, *args, **kwargs)


def narrow(input_tensor, *args, **kwargs):
    """
    Apply narrow to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the narrow operation.
    """
    from onnx9000.core.ops.torch_auto import narrow as core_narrow

    return core_narrow(input_tensor, *args, **kwargs)


def narrow_copy(input_tensor, *args, **kwargs):
    """
    Apply narrow_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the narrow_copy operation.
    """
    from onnx9000.core.ops.torch_auto import narrow_copy as core_narrow_copy

    return core_narrow_copy(input_tensor, *args, **kwargs)


def native_batch_norm(input_tensor, *args, **kwargs):
    """
    Apply native_batch_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the native_batch_norm operation.
    """
    from onnx9000.core.ops.torch_auto import native_batch_norm as core_native_batch_norm

    return core_native_batch_norm(input_tensor, *args, **kwargs)


def native_channel_shuffle(input_tensor, *args, **kwargs):
    """
    Apply native_channel_shuffle to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the native_channel_shuffle operation.
    """
    from onnx9000.core.ops.torch_auto import native_channel_shuffle as core_native_channel_shuffle

    return core_native_channel_shuffle(input_tensor, *args, **kwargs)


def native_dropout(input_tensor, *args, **kwargs):
    """
    Apply native_dropout to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the native_dropout operation.
    """
    from onnx9000.core.ops.torch_auto import native_dropout as core_native_dropout

    return core_native_dropout(input_tensor, *args, **kwargs)


def native_group_norm(input_tensor, *args, **kwargs):
    """
    Apply native_group_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the native_group_norm operation.
    """
    from onnx9000.core.ops.torch_auto import native_group_norm as core_native_group_norm

    return core_native_group_norm(input_tensor, *args, **kwargs)


def native_layer_norm(input_tensor, *args, **kwargs):
    """
    Apply native_layer_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the native_layer_norm operation.
    """
    from onnx9000.core.ops.torch_auto import native_layer_norm as core_native_layer_norm

    return core_native_layer_norm(input_tensor, *args, **kwargs)


def native_norm(input_tensor, *args, **kwargs):
    """
    Apply native_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the native_norm operation.
    """
    from onnx9000.core.ops.torch_auto import native_norm as core_native_norm

    return core_native_norm(input_tensor, *args, **kwargs)


def ne(input_tensor, *args, **kwargs):
    """
    Apply ne to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ne operation.
    """
    from onnx9000.core.ops.torch_auto import ne as core_ne

    return core_ne(input_tensor, *args, **kwargs)


def neg(input_tensor, *args, **kwargs):
    """
    Apply neg to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the neg operation.
    """
    from onnx9000.core.ops import neg as core_neg

    return core_neg(input_tensor, *args, **kwargs)


def neg_(input_tensor, *args, **kwargs):
    """
    Apply neg_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the neg_ operation.
    """
    from onnx9000.core.ops.torch_auto import neg_ as core_neg_

    return core_neg_(input_tensor, *args, **kwargs)


def negative(input_tensor, *args, **kwargs):
    """
    Apply negative to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the negative operation.
    """
    from onnx9000.core.ops.torch_auto import negative as core_negative

    return core_negative(input_tensor, *args, **kwargs)


def negative_(input_tensor, *args, **kwargs):
    """
    Apply negative_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the negative_ operation.
    """
    from onnx9000.core.ops.torch_auto import negative_ as core_negative_

    return core_negative_(input_tensor, *args, **kwargs)


def nextafter(input_tensor, *args, **kwargs):
    """
    Apply nextafter to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the nextafter operation.
    """
    from onnx9000.core.ops.torch_auto import nextafter as core_nextafter

    return core_nextafter(input_tensor, *args, **kwargs)


def nonzero(input_tensor, *args, **kwargs):
    """
    Apply nonzero to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the nonzero operation.
    """
    from onnx9000.core.ops.torch_auto import nonzero as core_nonzero

    return core_nonzero(input_tensor, *args, **kwargs)


def nonzero_static(input_tensor, *args, **kwargs):
    """
    Apply nonzero_static to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the nonzero_static operation.
    """
    from onnx9000.core.ops.torch_auto import nonzero_static as core_nonzero_static

    return core_nonzero_static(input_tensor, *args, **kwargs)


def norm(input_tensor, *args, **kwargs):
    """
    Apply norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the norm operation.
    """
    from onnx9000.core.ops.torch_auto import norm as core_norm

    return core_norm(input_tensor, *args, **kwargs)


def norm_except_dim(input_tensor, *args, **kwargs):
    """
    Apply norm_except_dim to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the norm_except_dim operation.
    """
    from onnx9000.core.ops.torch_auto import norm_except_dim as core_norm_except_dim

    return core_norm_except_dim(input_tensor, *args, **kwargs)


def normal(input_tensor, *args, **kwargs):
    """
    Apply normal to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the normal operation.
    """
    from onnx9000.core.ops.torch_auto import normal as core_normal

    return core_normal(input_tensor, *args, **kwargs)


def not_equal(input_tensor, *args, **kwargs):
    """
    Apply not_equal to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the not_equal operation.
    """
    from onnx9000.core.ops.torch_auto import not_equal as core_not_equal

    return core_not_equal(input_tensor, *args, **kwargs)


def nuclear_norm(input_tensor, *args, **kwargs):
    """
    Apply nuclear_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the nuclear_norm operation.
    """
    from onnx9000.core.ops.torch_auto import nuclear_norm as core_nuclear_norm

    return core_nuclear_norm(input_tensor, *args, **kwargs)


def numel(input_tensor, *args, **kwargs):
    """
    Apply numel to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the numel operation.
    """
    from onnx9000.core.ops.torch_auto import numel as core_numel

    return core_numel(input_tensor, *args, **kwargs)


def ones_like(input_tensor, *args, **kwargs):
    """
    Apply ones_like to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ones_like operation.
    """
    from onnx9000.core.ops.torch_auto import ones_like as core_ones_like

    return core_ones_like(input_tensor, *args, **kwargs)


def orgqr(input_tensor, *args, **kwargs):
    """
    Apply orgqr to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the orgqr operation.
    """
    from onnx9000.core.ops.torch_auto import orgqr as core_orgqr

    return core_orgqr(input_tensor, *args, **kwargs)


def ormqr(input_tensor, *args, **kwargs):
    """
    Apply ormqr to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ormqr operation.
    """
    from onnx9000.core.ops.torch_auto import ormqr as core_ormqr

    return core_ormqr(input_tensor, *args, **kwargs)


def outer(input_tensor, *args, **kwargs):
    """
    Apply outer to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the outer operation.
    """
    from onnx9000.core.ops.torch_auto import outer as core_outer

    return core_outer(input_tensor, *args, **kwargs)


def pairwise_distance(input_tensor, *args, **kwargs):
    """
    Apply pairwise_distance to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the pairwise_distance operation.
    """
    from onnx9000.core.ops.torch_auto import pairwise_distance as core_pairwise_distance

    return core_pairwise_distance(input_tensor, *args, **kwargs)


def pdist(input_tensor, *args, **kwargs):
    """
    Apply pdist to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the pdist operation.
    """
    from onnx9000.core.ops.torch_auto import pdist as core_pdist

    return core_pdist(input_tensor, *args, **kwargs)


def permute(input_tensor, *args, **kwargs):
    """
    Apply permute to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the permute operation.
    """
    from onnx9000.core.ops.torch_auto import permute as core_permute

    return core_permute(input_tensor, *args, **kwargs)


def permute_copy(input_tensor, *args, **kwargs):
    """
    Apply permute_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the permute_copy operation.
    """
    from onnx9000.core.ops.torch_auto import permute_copy as core_permute_copy

    return core_permute_copy(input_tensor, *args, **kwargs)


def pinverse(input_tensor, *args, **kwargs):
    """
    Apply pinverse to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the pinverse operation.
    """
    from onnx9000.core.ops.torch_auto import pinverse as core_pinverse

    return core_pinverse(input_tensor, *args, **kwargs)


def pixel_shuffle(input_tensor, *args, **kwargs):
    """
    Apply pixel_shuffle to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the pixel_shuffle operation.
    """
    from onnx9000.core.ops.torch_auto import pixel_shuffle as core_pixel_shuffle

    return core_pixel_shuffle(input_tensor, *args, **kwargs)


def pixel_unshuffle(input_tensor, *args, **kwargs):
    """
    Apply pixel_unshuffle to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the pixel_unshuffle operation.
    """
    from onnx9000.core.ops.torch_auto import pixel_unshuffle as core_pixel_unshuffle

    return core_pixel_unshuffle(input_tensor, *args, **kwargs)


def poisson(input_tensor, *args, **kwargs):
    """
    Apply poisson to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the poisson operation.
    """
    from onnx9000.core.ops.torch_auto import poisson as core_poisson

    return core_poisson(input_tensor, *args, **kwargs)


def poisson_nll_loss(input_tensor, *args, **kwargs):
    """
    Apply poisson_nll_loss to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the poisson_nll_loss operation.
    """
    from onnx9000.core.ops.torch_auto import poisson_nll_loss as core_poisson_nll_loss

    return core_poisson_nll_loss(input_tensor, *args, **kwargs)


def polar(input_tensor, *args, **kwargs):
    """
    Apply polar to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the polar operation.
    """
    from onnx9000.core.ops.torch_auto import polar as core_polar

    return core_polar(input_tensor, *args, **kwargs)


def polygamma(input_tensor, *args, **kwargs):
    """
    Apply polygamma to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the polygamma operation.
    """
    from onnx9000.core.ops.torch_auto import polygamma as core_polygamma

    return core_polygamma(input_tensor, *args, **kwargs)


def positive(input_tensor, *args, **kwargs):
    """
    Apply positive to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the positive operation.
    """
    from onnx9000.core.ops.torch_auto import positive as core_positive

    return core_positive(input_tensor, *args, **kwargs)


def pow(input_tensor, *args, **kwargs):
    """
    Apply pow to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the pow operation.
    """
    from onnx9000.core.ops import pow as core_pow

    return core_pow(input_tensor, *args, **kwargs)


def prelu(input_tensor, *args, **kwargs):
    """
    Apply prelu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the prelu operation.
    """
    from onnx9000.core.ops import prelu as core_prelu

    return core_prelu(input_tensor, *args, **kwargs)


def prod(input_tensor, *args, **kwargs):
    """
    Apply prod to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the prod operation.
    """
    from onnx9000.core.ops.torch_auto import prod as core_prod

    return core_prod(input_tensor, *args, **kwargs)


def promote_types(input_tensor, *args, **kwargs):
    """
    Apply promote_types to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the promote_types operation.
    """
    from onnx9000.core.ops.torch_auto import promote_types as core_promote_types

    return core_promote_types(input_tensor, *args, **kwargs)


def put(input_tensor, *args, **kwargs):
    """
    Apply put to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the put operation.
    """
    from onnx9000.core.ops.torch_auto import put as core_put

    return core_put(input_tensor, *args, **kwargs)


def q_per_channel_axis(input_tensor, *args, **kwargs):
    """
    Apply q_per_channel_axis to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the q_per_channel_axis operation.
    """
    from onnx9000.core.ops.torch_auto import q_per_channel_axis as core_q_per_channel_axis

    return core_q_per_channel_axis(input_tensor, *args, **kwargs)


def q_per_channel_scales(input_tensor, *args, **kwargs):
    """
    Apply q_per_channel_scales to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the q_per_channel_scales operation.
    """
    from onnx9000.core.ops.torch_auto import q_per_channel_scales as core_q_per_channel_scales

    return core_q_per_channel_scales(input_tensor, *args, **kwargs)


def q_per_channel_zero_points(input_tensor, *args, **kwargs):
    """
    Apply q_per_channel_zero_points to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the q_per_channel_zero_points operation.
    """
    from onnx9000.core.ops.torch_auto import (
        q_per_channel_zero_points as core_q_per_channel_zero_points,
    )

    return core_q_per_channel_zero_points(input_tensor, *args, **kwargs)


def q_scale(input_tensor, *args, **kwargs):
    """
    Apply q_scale to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the q_scale operation.
    """
    from onnx9000.core.ops.torch_auto import q_scale as core_q_scale

    return core_q_scale(input_tensor, *args, **kwargs)


def q_zero_point(input_tensor, *args, **kwargs):
    """
    Apply q_zero_point to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the q_zero_point operation.
    """
    from onnx9000.core.ops.torch_auto import q_zero_point as core_q_zero_point

    return core_q_zero_point(input_tensor, *args, **kwargs)


def qr(input_tensor, *args, **kwargs):
    """
    Apply qr to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the qr operation.
    """
    from onnx9000.core.ops.torch_auto import qr as core_qr

    return core_qr(input_tensor, *args, **kwargs)


def quantile(input_tensor, *args, **kwargs):
    """
    Apply quantile to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantile operation.
    """
    from onnx9000.core.ops.torch_auto import quantile as core_quantile

    return core_quantile(input_tensor, *args, **kwargs)


def quantize_per_channel(input_tensor, *args, **kwargs):
    """
    Apply quantize_per_channel to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantize_per_channel operation.
    """
    from onnx9000.core.ops.torch_auto import quantize_per_channel as core_quantize_per_channel

    return core_quantize_per_channel(input_tensor, *args, **kwargs)


def quantize_per_tensor(input_tensor, *args, **kwargs):
    """
    Apply quantize_per_tensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantize_per_tensor operation.
    """
    from onnx9000.core.ops.torch_auto import quantize_per_tensor as core_quantize_per_tensor

    return core_quantize_per_tensor(input_tensor, *args, **kwargs)


def quantize_per_tensor_dynamic(input_tensor, *args, **kwargs):
    """
    Apply quantize_per_tensor_dynamic to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantize_per_tensor_dynamic operation.
    """
    from onnx9000.core.ops.torch_auto import (
        quantize_per_tensor_dynamic as core_quantize_per_tensor_dynamic,
    )

    return core_quantize_per_tensor_dynamic(input_tensor, *args, **kwargs)


def quantized_batch_norm(input_tensor, *args, **kwargs):
    """
    Apply quantized_batch_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantized_batch_norm operation.
    """
    from onnx9000.core.ops.torch_auto import quantized_batch_norm as core_quantized_batch_norm

    return core_quantized_batch_norm(input_tensor, *args, **kwargs)


def quantized_gru_cell(input_tensor, *args, **kwargs):
    """
    Apply quantized_gru_cell to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantized_gru_cell operation.
    """
    from onnx9000.core.ops.torch_auto import quantized_gru_cell as core_quantized_gru_cell

    return core_quantized_gru_cell(input_tensor, *args, **kwargs)


def quantized_lstm_cell(input_tensor, *args, **kwargs):
    """
    Apply quantized_lstm_cell to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantized_lstm_cell operation.
    """
    from onnx9000.core.ops.torch_auto import quantized_lstm_cell as core_quantized_lstm_cell

    return core_quantized_lstm_cell(input_tensor, *args, **kwargs)


def quantized_max_pool1d(input_tensor, *args, **kwargs):
    """
    Apply quantized_max_pool1d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantized_max_pool1d operation.
    """
    from onnx9000.core.ops.torch_auto import quantized_max_pool1d as core_quantized_max_pool1d

    return core_quantized_max_pool1d(input_tensor, *args, **kwargs)


def quantized_max_pool2d(input_tensor, *args, **kwargs):
    """
    Apply quantized_max_pool2d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantized_max_pool2d operation.
    """
    from onnx9000.core.ops.torch_auto import quantized_max_pool2d as core_quantized_max_pool2d

    return core_quantized_max_pool2d(input_tensor, *args, **kwargs)


def quantized_max_pool3d(input_tensor, *args, **kwargs):
    """
    Apply quantized_max_pool3d to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantized_max_pool3d operation.
    """
    from onnx9000.core.ops.torch_auto import quantized_max_pool3d as core_quantized_max_pool3d

    return core_quantized_max_pool3d(input_tensor, *args, **kwargs)


def quantized_rnn_relu_cell(input_tensor, *args, **kwargs):
    """
    Apply quantized_rnn_relu_cell to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantized_rnn_relu_cell operation.
    """
    from onnx9000.core.ops.torch_auto import quantized_rnn_relu_cell as core_quantized_rnn_relu_cell

    return core_quantized_rnn_relu_cell(input_tensor, *args, **kwargs)


def quantized_rnn_tanh_cell(input_tensor, *args, **kwargs):
    """
    Apply quantized_rnn_tanh_cell to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quantized_rnn_tanh_cell operation.
    """
    from onnx9000.core.ops.torch_auto import quantized_rnn_tanh_cell as core_quantized_rnn_tanh_cell

    return core_quantized_rnn_tanh_cell(input_tensor, *args, **kwargs)


def rad2deg(input_tensor, *args, **kwargs):
    """
    Apply rad2deg to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rad2deg operation.
    """
    from onnx9000.core.ops.torch_auto import rad2deg as core_rad2deg

    return core_rad2deg(input_tensor, *args, **kwargs)


def rad2deg_(input_tensor, *args, **kwargs):
    """
    Apply rad2deg_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rad2deg_ operation.
    """
    from onnx9000.core.ops.torch_auto import rad2deg_ as core_rad2deg_

    return core_rad2deg_(input_tensor, *args, **kwargs)


def rand_like(input_tensor, *args, **kwargs):
    """
    Apply rand_like to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rand_like operation.
    """
    from onnx9000.core.ops.torch_auto import rand_like as core_rand_like

    return core_rand_like(input_tensor, *args, **kwargs)


def randint(input_tensor, *args, **kwargs):
    """
    Apply randint to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the randint operation.
    """
    from onnx9000.core.ops.torch_auto import randint as core_randint

    return core_randint(input_tensor, *args, **kwargs)


def randint_like(input_tensor, *args, **kwargs):
    """
    Apply randint_like to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the randint_like operation.
    """
    from onnx9000.core.ops.torch_auto import randint_like as core_randint_like

    return core_randint_like(input_tensor, *args, **kwargs)


def randn_like(input_tensor, *args, **kwargs):
    """
    Apply randn_like to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the randn_like operation.
    """
    from onnx9000.core.ops.torch_auto import randn_like as core_randn_like

    return core_randn_like(input_tensor, *args, **kwargs)


def randperm(input_tensor, *args, **kwargs):
    """
    Apply randperm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the randperm operation.
    """
    from onnx9000.core.ops.torch_auto import randperm as core_randperm

    return core_randperm(input_tensor, *args, **kwargs)


def range(input_tensor, *args, **kwargs):
    """
    Apply range to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the range operation.
    """
    from onnx9000.core.ops.torch_auto import range as core_range

    return core_range(input_tensor, *args, **kwargs)


def ravel(input_tensor, *args, **kwargs):
    """
    Apply ravel to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the ravel operation.
    """
    from onnx9000.core.ops.torch_auto import ravel as core_ravel

    return core_ravel(input_tensor, *args, **kwargs)


def real(input_tensor, *args, **kwargs):
    """
    Apply real to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the real operation.
    """
    from onnx9000.core.ops.torch_auto import real as core_real

    return core_real(input_tensor, *args, **kwargs)


def reciprocal(input_tensor, *args, **kwargs):
    """
    Apply reciprocal to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the reciprocal operation.
    """
    from onnx9000.core.ops import reciprocal as core_reciprocal

    return core_reciprocal(input_tensor, *args, **kwargs)


def reciprocal_(input_tensor, *args, **kwargs):
    """
    Apply reciprocal_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the reciprocal_ operation.
    """
    from onnx9000.core.ops.torch_auto import reciprocal_ as core_reciprocal_

    return core_reciprocal_(input_tensor, *args, **kwargs)


def relu(input_tensor, *args, **kwargs):
    """
    Apply relu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the relu operation.
    """
    from onnx9000.core.ops import relu as core_relu

    return core_relu(input_tensor, *args, **kwargs)


def relu_(input_tensor, *args, **kwargs):
    """
    Apply relu_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the relu_ operation.
    """
    from onnx9000.core.ops.torch_auto import relu_ as core_relu_

    return core_relu_(input_tensor, *args, **kwargs)


def remainder(input_tensor, *args, **kwargs):
    """
    Apply remainder to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the remainder operation.
    """
    from onnx9000.core.ops.torch_auto import remainder as core_remainder

    return core_remainder(input_tensor, *args, **kwargs)


def renorm(input_tensor, *args, **kwargs):
    """
    Apply renorm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the renorm operation.
    """
    from onnx9000.core.ops.torch_auto import renorm as core_renorm

    return core_renorm(input_tensor, *args, **kwargs)


def repeat_interleave(input_tensor, *args, **kwargs):
    """
    Apply repeat_interleave to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the repeat_interleave operation.
    """
    from onnx9000.core.ops.torch_auto import repeat_interleave as core_repeat_interleave

    return core_repeat_interleave(input_tensor, *args, **kwargs)


def reshape(input_tensor, *args, **kwargs):
    """
    Apply reshape to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the reshape operation.
    """
    from onnx9000.core.ops import reshape as core_reshape

    return core_reshape(input_tensor, *args, **kwargs)


def resize_as_(input_tensor, *args, **kwargs):
    """
    Apply resize_as_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the resize_as_ operation.
    """
    from onnx9000.core.ops.torch_auto import resize_as_ as core_resize_as_

    return core_resize_as_(input_tensor, *args, **kwargs)


def resize_as_sparse_(input_tensor, *args, **kwargs):
    """
    Apply resize_as_sparse_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the resize_as_sparse_ operation.
    """
    from onnx9000.core.ops.torch_auto import resize_as_sparse_ as core_resize_as_sparse_

    return core_resize_as_sparse_(input_tensor, *args, **kwargs)


def resolve_conj(input_tensor, *args, **kwargs):
    """
    Apply resolve_conj to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the resolve_conj operation.
    """
    from onnx9000.core.ops.torch_auto import resolve_conj as core_resolve_conj

    return core_resolve_conj(input_tensor, *args, **kwargs)


def resolve_neg(input_tensor, *args, **kwargs):
    """
    Apply resolve_neg to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the resolve_neg operation.
    """
    from onnx9000.core.ops.torch_auto import resolve_neg as core_resolve_neg

    return core_resolve_neg(input_tensor, *args, **kwargs)


def result_type(input_tensor, *args, **kwargs):
    """
    Apply result_type to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the result_type operation.
    """
    from onnx9000.core.ops.torch_auto import result_type as core_result_type

    return core_result_type(input_tensor, *args, **kwargs)


def rms_norm(input_tensor, *args, **kwargs):
    """
    Apply rms_norm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rms_norm operation.
    """
    from onnx9000.core.ops.torch_auto import rms_norm as core_rms_norm

    return core_rms_norm(input_tensor, *args, **kwargs)


def rnn_relu(input_tensor, *args, **kwargs):
    """
    Apply rnn_relu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rnn_relu operation.
    """
    from onnx9000.core.ops.torch_auto import rnn_relu as core_rnn_relu

    return core_rnn_relu(input_tensor, *args, **kwargs)


def rnn_relu_cell(input_tensor, *args, **kwargs):
    """
    Apply rnn_relu_cell to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rnn_relu_cell operation.
    """
    from onnx9000.core.ops.torch_auto import rnn_relu_cell as core_rnn_relu_cell

    return core_rnn_relu_cell(input_tensor, *args, **kwargs)


def rnn_tanh(input_tensor, *args, **kwargs):
    """
    Apply rnn_tanh to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rnn_tanh operation.
    """
    from onnx9000.core.ops.torch_auto import rnn_tanh as core_rnn_tanh

    return core_rnn_tanh(input_tensor, *args, **kwargs)


def rnn_tanh_cell(input_tensor, *args, **kwargs):
    """
    Apply rnn_tanh_cell to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rnn_tanh_cell operation.
    """
    from onnx9000.core.ops.torch_auto import rnn_tanh_cell as core_rnn_tanh_cell

    return core_rnn_tanh_cell(input_tensor, *args, **kwargs)


def roll(input_tensor, *args, **kwargs):
    """
    Apply roll to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the roll operation.
    """
    from onnx9000.core.ops.torch_auto import roll as core_roll

    return core_roll(input_tensor, *args, **kwargs)


def rot90(input_tensor, *args, **kwargs):
    """
    Apply rot90 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rot90 operation.
    """
    from onnx9000.core.ops.torch_auto import rot90 as core_rot90

    return core_rot90(input_tensor, *args, **kwargs)


def round(input_tensor, *args, **kwargs):
    """
    Apply round to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the round operation.
    """
    from onnx9000.core.ops import round as core_round

    return core_round(input_tensor, *args, **kwargs)


def round_(input_tensor, *args, **kwargs):
    """
    Apply round_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the round_ operation.
    """
    from onnx9000.core.ops.torch_auto import round_ as core_round_

    return core_round_(input_tensor, *args, **kwargs)


def row_indices_copy(input_tensor, *args, **kwargs):
    """
    Apply row_indices_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the row_indices_copy operation.
    """
    from onnx9000.core.ops.torch_auto import row_indices_copy as core_row_indices_copy

    return core_row_indices_copy(input_tensor, *args, **kwargs)


def row_stack(input_tensor, *args, **kwargs):
    """
    Apply row_stack to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the row_stack operation.
    """
    from onnx9000.core.ops.torch_auto import row_stack as core_row_stack

    return core_row_stack(input_tensor, *args, **kwargs)


def rrelu(input_tensor, *args, **kwargs):
    """
    Apply rrelu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rrelu operation.
    """
    from onnx9000.core.ops.torch_auto import rrelu as core_rrelu

    return core_rrelu(input_tensor, *args, **kwargs)


def rrelu_(input_tensor, *args, **kwargs):
    """
    Apply rrelu_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rrelu_ operation.
    """
    from onnx9000.core.ops.torch_auto import rrelu_ as core_rrelu_

    return core_rrelu_(input_tensor, *args, **kwargs)


def rsqrt(input_tensor, *args, **kwargs):
    """
    Apply rsqrt to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rsqrt operation.
    """
    from onnx9000.core.ops.torch_auto import rsqrt as core_rsqrt

    return core_rsqrt(input_tensor, *args, **kwargs)


def rsqrt_(input_tensor, *args, **kwargs):
    """
    Apply rsqrt_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rsqrt_ operation.
    """
    from onnx9000.core.ops.torch_auto import rsqrt_ as core_rsqrt_

    return core_rsqrt_(input_tensor, *args, **kwargs)


def rsub(input_tensor, *args, **kwargs):
    """
    Apply rsub to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the rsub operation.
    """
    from onnx9000.core.ops.torch_auto import rsub as core_rsub

    return core_rsub(input_tensor, *args, **kwargs)


def saddmm(input_tensor, *args, **kwargs):
    """
    Apply saddmm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the saddmm operation.
    """
    from onnx9000.core.ops.torch_auto import saddmm as core_saddmm

    return core_saddmm(input_tensor, *args, **kwargs)


def scalar_tensor(input_tensor, *args, **kwargs):
    """
    Apply scalar_tensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the scalar_tensor operation.
    """
    from onnx9000.core.ops.torch_auto import scalar_tensor as core_scalar_tensor

    return core_scalar_tensor(input_tensor, *args, **kwargs)


def scatter(input_tensor, *args, **kwargs):
    """
    Apply scatter to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the scatter operation.
    """
    from onnx9000.core.ops import scatter as core_scatter

    return core_scatter(input_tensor, *args, **kwargs)


def scatter_add(input_tensor, *args, **kwargs):
    """
    Apply scatter_add to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the scatter_add operation.
    """
    from onnx9000.core.ops.torch_auto import scatter_add as core_scatter_add

    return core_scatter_add(input_tensor, *args, **kwargs)


def scatter_reduce(input_tensor, *args, **kwargs):
    """
    Apply scatter_reduce to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the scatter_reduce operation.
    """
    from onnx9000.core.ops.torch_auto import scatter_reduce as core_scatter_reduce

    return core_scatter_reduce(input_tensor, *args, **kwargs)


def searchsorted(input_tensor, *args, **kwargs):
    """
    Apply searchsorted to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the searchsorted operation.
    """
    from onnx9000.core.ops.torch_auto import searchsorted as core_searchsorted

    return core_searchsorted(input_tensor, *args, **kwargs)


def select(input_tensor, *args, **kwargs):
    """
    Apply select to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the select operation.
    """
    from onnx9000.core.ops.torch_auto import select as core_select

    return core_select(input_tensor, *args, **kwargs)


def select_copy(input_tensor, *args, **kwargs):
    """
    Apply select_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the select_copy operation.
    """
    from onnx9000.core.ops.torch_auto import select_copy as core_select_copy

    return core_select_copy(input_tensor, *args, **kwargs)


def select_scatter(input_tensor, *args, **kwargs):
    """
    Apply select_scatter to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the select_scatter operation.
    """
    from onnx9000.core.ops.torch_auto import select_scatter as core_select_scatter

    return core_select_scatter(input_tensor, *args, **kwargs)


def selu(input_tensor, *args, **kwargs):
    """
    Apply selu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the selu operation.
    """
    from onnx9000.core.ops import selu as core_selu

    return core_selu(input_tensor, *args, **kwargs)


def selu_(input_tensor, *args, **kwargs):
    """
    Apply selu_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the selu_ operation.
    """
    from onnx9000.core.ops.torch_auto import selu_ as core_selu_

    return core_selu_(input_tensor, *args, **kwargs)


def sgn(input_tensor, *args, **kwargs):
    """
    Apply sgn to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sgn operation.
    """
    from onnx9000.core.ops.torch_auto import sgn as core_sgn

    return core_sgn(input_tensor, *args, **kwargs)


def sigmoid(input_tensor, *args, **kwargs):
    """
    Apply sigmoid to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sigmoid operation.
    """
    from onnx9000.core.ops import sigmoid as core_sigmoid

    return core_sigmoid(input_tensor, *args, **kwargs)


def sigmoid_(input_tensor, *args, **kwargs):
    """
    Apply sigmoid_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sigmoid_ operation.
    """
    from onnx9000.core.ops.torch_auto import sigmoid_ as core_sigmoid_

    return core_sigmoid_(input_tensor, *args, **kwargs)


def sign(input_tensor, *args, **kwargs):
    """
    Apply sign to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sign operation.
    """
    from onnx9000.core.ops import sign as core_sign

    return core_sign(input_tensor, *args, **kwargs)


def signbit(input_tensor, *args, **kwargs):
    """
    Apply signbit to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the signbit operation.
    """
    from onnx9000.core.ops.torch_auto import signbit as core_signbit

    return core_signbit(input_tensor, *args, **kwargs)


def sin(input_tensor, *args, **kwargs):
    """
    Apply sin to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sin operation.
    """
    from onnx9000.core.ops import sin as core_sin

    return core_sin(input_tensor, *args, **kwargs)


def sin_(input_tensor, *args, **kwargs):
    """
    Apply sin_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sin_ operation.
    """
    from onnx9000.core.ops.torch_auto import sin_ as core_sin_

    return core_sin_(input_tensor, *args, **kwargs)


def sinc(input_tensor, *args, **kwargs):
    """
    Apply sinc to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sinc operation.
    """
    from onnx9000.core.ops.torch_auto import sinc as core_sinc

    return core_sinc(input_tensor, *args, **kwargs)


def sinc_(input_tensor, *args, **kwargs):
    """
    Apply sinc_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sinc_ operation.
    """
    from onnx9000.core.ops.torch_auto import sinc_ as core_sinc_

    return core_sinc_(input_tensor, *args, **kwargs)


def sinh(input_tensor, *args, **kwargs):
    """
    Apply sinh to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sinh operation.
    """
    from onnx9000.core.ops import sinh as core_sinh

    return core_sinh(input_tensor, *args, **kwargs)


def sinh_(input_tensor, *args, **kwargs):
    """
    Apply sinh_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sinh_ operation.
    """
    from onnx9000.core.ops.torch_auto import sinh_ as core_sinh_

    return core_sinh_(input_tensor, *args, **kwargs)


def slice_copy(input_tensor, *args, **kwargs):
    """
    Apply slice_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the slice_copy operation.
    """
    from onnx9000.core.ops.torch_auto import slice_copy as core_slice_copy

    return core_slice_copy(input_tensor, *args, **kwargs)


def slice_inverse(input_tensor, *args, **kwargs):
    """
    Apply slice_inverse to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the slice_inverse operation.
    """
    from onnx9000.core.ops.torch_auto import slice_inverse as core_slice_inverse

    return core_slice_inverse(input_tensor, *args, **kwargs)


def slice_scatter(input_tensor, *args, **kwargs):
    """
    Apply slice_scatter to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the slice_scatter operation.
    """
    from onnx9000.core.ops.torch_auto import slice_scatter as core_slice_scatter

    return core_slice_scatter(input_tensor, *args, **kwargs)


def slogdet(input_tensor, *args, **kwargs):
    """
    Apply slogdet to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the slogdet operation.
    """
    from onnx9000.core.ops.torch_auto import slogdet as core_slogdet

    return core_slogdet(input_tensor, *args, **kwargs)


def smm(input_tensor, *args, **kwargs):
    """
    Apply smm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the smm operation.
    """
    from onnx9000.core.ops.torch_auto import smm as core_smm

    return core_smm(input_tensor, *args, **kwargs)


def softmax(input_tensor, *args, **kwargs):
    """
    Apply softmax to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the softmax operation.
    """
    from onnx9000.core.ops import softmax as core_softmax

    return core_softmax(input_tensor, *args, **kwargs)


def sort(input_tensor, *args, **kwargs):
    """
    Apply sort to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sort operation.
    """
    from onnx9000.core.ops.torch_auto import sort as core_sort

    return core_sort(input_tensor, *args, **kwargs)


def sparse_bsc_tensor(input_tensor, *args, **kwargs):
    """
    Apply sparse_bsc_tensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sparse_bsc_tensor operation.
    """
    from onnx9000.core.ops.torch_auto import sparse_bsc_tensor as core_sparse_bsc_tensor

    return core_sparse_bsc_tensor(input_tensor, *args, **kwargs)


def sparse_bsr_tensor(input_tensor, *args, **kwargs):
    """
    Apply sparse_bsr_tensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sparse_bsr_tensor operation.
    """
    from onnx9000.core.ops.torch_auto import sparse_bsr_tensor as core_sparse_bsr_tensor

    return core_sparse_bsr_tensor(input_tensor, *args, **kwargs)


def sparse_compressed_tensor(input_tensor, *args, **kwargs):
    """
    Apply sparse_compressed_tensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sparse_compressed_tensor operation.
    """
    from onnx9000.core.ops.torch_auto import (
        sparse_compressed_tensor as core_sparse_compressed_tensor,
    )

    return core_sparse_compressed_tensor(input_tensor, *args, **kwargs)


def sparse_coo_tensor(input_tensor, *args, **kwargs):
    """
    Apply sparse_coo_tensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sparse_coo_tensor operation.
    """
    from onnx9000.core.ops.torch_auto import sparse_coo_tensor as core_sparse_coo_tensor

    return core_sparse_coo_tensor(input_tensor, *args, **kwargs)


def sparse_csc_tensor(input_tensor, *args, **kwargs):
    """
    Apply sparse_csc_tensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sparse_csc_tensor operation.
    """
    from onnx9000.core.ops.torch_auto import sparse_csc_tensor as core_sparse_csc_tensor

    return core_sparse_csc_tensor(input_tensor, *args, **kwargs)


def sparse_csr_tensor(input_tensor, *args, **kwargs):
    """
    Apply sparse_csr_tensor to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sparse_csr_tensor operation.
    """
    from onnx9000.core.ops.torch_auto import sparse_csr_tensor as core_sparse_csr_tensor

    return core_sparse_csr_tensor(input_tensor, *args, **kwargs)


def split_copy(input_tensor, *args, **kwargs):
    """
    Apply split_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the split_copy operation.
    """
    from onnx9000.core.ops.torch_auto import split_copy as core_split_copy

    return core_split_copy(input_tensor, *args, **kwargs)


def split_with_sizes(input_tensor, *args, **kwargs):
    """
    Apply split_with_sizes to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the split_with_sizes operation.
    """
    from onnx9000.core.ops.torch_auto import split_with_sizes as core_split_with_sizes

    return core_split_with_sizes(input_tensor, *args, **kwargs)


def split_with_sizes_copy(input_tensor, *args, **kwargs):
    """
    Apply split_with_sizes_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the split_with_sizes_copy operation.
    """
    from onnx9000.core.ops.torch_auto import split_with_sizes_copy as core_split_with_sizes_copy

    return core_split_with_sizes_copy(input_tensor, *args, **kwargs)


def spmm(input_tensor, *args, **kwargs):
    """
    Apply spmm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the spmm operation.
    """
    from onnx9000.core.ops.torch_auto import spmm as core_spmm

    return core_spmm(input_tensor, *args, **kwargs)


def sqrt(input_tensor, *args, **kwargs):
    """
    Apply sqrt to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sqrt operation.
    """
    from onnx9000.core.ops.torch_auto import sqrt as core_sqrt

    return core_sqrt(input_tensor, *args, **kwargs)


def sqrt_(input_tensor, *args, **kwargs):
    """
    Apply sqrt_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sqrt_ operation.
    """
    from onnx9000.core.ops.torch_auto import sqrt_ as core_sqrt_

    return core_sqrt_(input_tensor, *args, **kwargs)


def square(input_tensor, *args, **kwargs):
    """
    Apply square to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the square operation.
    """
    from onnx9000.core.ops.torch_auto import square as core_square

    return core_square(input_tensor, *args, **kwargs)


def square_(input_tensor, *args, **kwargs):
    """
    Apply square_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the square_ operation.
    """
    from onnx9000.core.ops.torch_auto import square_ as core_square_

    return core_square_(input_tensor, *args, **kwargs)


def squeeze(input_tensor, *args, **kwargs):
    """
    Apply squeeze to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the squeeze operation.
    """
    from onnx9000.core.ops.torch_auto import squeeze as core_squeeze

    return core_squeeze(input_tensor, *args, **kwargs)


def squeeze_copy(input_tensor, *args, **kwargs):
    """
    Apply squeeze_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the squeeze_copy operation.
    """
    from onnx9000.core.ops.torch_auto import squeeze_copy as core_squeeze_copy

    return core_squeeze_copy(input_tensor, *args, **kwargs)


def sspaddmm(input_tensor, *args, **kwargs):
    """
    Apply sspaddmm to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sspaddmm operation.
    """
    from onnx9000.core.ops.torch_auto import sspaddmm as core_sspaddmm

    return core_sspaddmm(input_tensor, *args, **kwargs)


def std(input_tensor, *args, **kwargs):
    """
    Apply std to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the std operation.
    """
    from onnx9000.core.ops.torch_auto import std as core_std

    return core_std(input_tensor, *args, **kwargs)


def std_mean(input_tensor, *args, **kwargs):
    """
    Apply std_mean to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the std_mean operation.
    """
    from onnx9000.core.ops.torch_auto import std_mean as core_std_mean

    return core_std_mean(input_tensor, *args, **kwargs)


def stft(input_tensor, *args, **kwargs):
    """
    Apply stft to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the stft operation.
    """
    from onnx9000.core.ops.torch_auto import stft as core_stft

    return core_stft(input_tensor, *args, **kwargs)


def sub(input_tensor, *args, **kwargs):
    """
    Apply sub to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sub operation.
    """
    from onnx9000.core.ops import sub as core_sub

    return core_sub(input_tensor, *args, **kwargs)


def subtract(input_tensor, *args, **kwargs):
    """
    Apply subtract to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the subtract operation.
    """
    from onnx9000.core.ops.torch_auto import subtract as core_subtract

    return core_subtract(input_tensor, *args, **kwargs)


def sum(input_tensor, *args, **kwargs):
    """
    Apply sum to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sum operation.
    """
    from onnx9000.core.ops import sum as core_sum

    return core_sum(input_tensor, *args, **kwargs)


def svd(input_tensor, *args, **kwargs):
    """
    Apply svd to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the svd operation.
    """
    from onnx9000.core.ops.torch_auto import svd as core_svd

    return core_svd(input_tensor, *args, **kwargs)


def swapaxes(input_tensor, *args, **kwargs):
    """
    Apply swapaxes to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the swapaxes operation.
    """
    from onnx9000.core.ops.torch_auto import swapaxes as core_swapaxes

    return core_swapaxes(input_tensor, *args, **kwargs)


def swapdims(input_tensor, *args, **kwargs):
    """
    Apply swapdims to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the swapdims operation.
    """
    from onnx9000.core.ops.torch_auto import swapdims as core_swapdims

    return core_swapdims(input_tensor, *args, **kwargs)


def sym_constrain_range(input_tensor, *args, **kwargs):
    """
    Apply sym_constrain_range to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sym_constrain_range operation.
    """
    from onnx9000.core.ops.torch_auto import sym_constrain_range as core_sym_constrain_range

    return core_sym_constrain_range(input_tensor, *args, **kwargs)


def sym_constrain_range_for_size(input_tensor, *args, **kwargs):
    """
    Apply sym_constrain_range_for_size to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the sym_constrain_range_for_size operation.
    """
    from onnx9000.core.ops.torch_auto import (
        sym_constrain_range_for_size as core_sym_constrain_range_for_size,
    )

    return core_sym_constrain_range_for_size(input_tensor, *args, **kwargs)


def t(input_tensor, *args, **kwargs):
    """
    Apply t to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the t operation.
    """
    from onnx9000.core.ops.torch_auto import t as core_t

    return core_t(input_tensor, *args, **kwargs)


def t_copy(input_tensor, *args, **kwargs):
    """
    Apply t_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the t_copy operation.
    """
    from onnx9000.core.ops.torch_auto import t_copy as core_t_copy

    return core_t_copy(input_tensor, *args, **kwargs)


def take(input_tensor, *args, **kwargs):
    """
    Apply take to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the take operation.
    """
    from onnx9000.core.ops.torch_auto import take as core_take

    return core_take(input_tensor, *args, **kwargs)


def take_along_dim(input_tensor, *args, **kwargs):
    """
    Apply take_along_dim to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the take_along_dim operation.
    """
    from onnx9000.core.ops.torch_auto import take_along_dim as core_take_along_dim

    return core_take_along_dim(input_tensor, *args, **kwargs)


def tan(input_tensor, *args, **kwargs):
    """
    Apply tan to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the tan operation.
    """
    from onnx9000.core.ops import tan as core_tan

    return core_tan(input_tensor, *args, **kwargs)


def tan_(input_tensor, *args, **kwargs):
    """
    Apply tan_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the tan_ operation.
    """
    from onnx9000.core.ops.torch_auto import tan_ as core_tan_

    return core_tan_(input_tensor, *args, **kwargs)


def tanh(input_tensor, *args, **kwargs):
    """
    Apply tanh to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the tanh operation.
    """
    from onnx9000.core.ops import tanh as core_tanh

    return core_tanh(input_tensor, *args, **kwargs)


def tanh_(input_tensor, *args, **kwargs):
    """
    Apply tanh_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the tanh_ operation.
    """
    from onnx9000.core.ops.torch_auto import tanh_ as core_tanh_

    return core_tanh_(input_tensor, *args, **kwargs)


def tensor_split(input_tensor, *args, **kwargs):
    """
    Apply tensor_split to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the tensor_split operation.
    """
    from onnx9000.core.ops.torch_auto import tensor_split as core_tensor_split

    return core_tensor_split(input_tensor, *args, **kwargs)


def tensordot(input_tensor, *args, **kwargs):
    """
    Apply tensordot to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the tensordot operation.
    """
    from onnx9000.core.ops.torch_auto import tensordot as core_tensordot

    return core_tensordot(input_tensor, *args, **kwargs)


def threshold(input_tensor, *args, **kwargs):
    """
    Apply threshold to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the threshold operation.
    """
    from onnx9000.core.ops.torch_auto import threshold as core_threshold

    return core_threshold(input_tensor, *args, **kwargs)


def threshold_(input_tensor, *args, **kwargs):
    """
    Apply threshold_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the threshold_ operation.
    """
    from onnx9000.core.ops.torch_auto import threshold_ as core_threshold_

    return core_threshold_(input_tensor, *args, **kwargs)


def tile(input_tensor, *args, **kwargs):
    """
    Apply tile to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the tile operation.
    """
    from onnx9000.core.ops import tile as core_tile

    return core_tile(input_tensor, *args, **kwargs)


def topk(input_tensor, *args, **kwargs):
    """
    Apply topk to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the topk operation.
    """
    from onnx9000.core.ops import topk as core_topk

    return core_topk(input_tensor, *args, **kwargs)


def transpose(input_tensor, *args, **kwargs):
    """
    Apply transpose to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the transpose operation.
    """
    from onnx9000.core.ops import transpose as core_transpose

    return core_transpose(input_tensor, *args, **kwargs)


def transpose_copy(input_tensor, *args, **kwargs):
    """
    Apply transpose_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the transpose_copy operation.
    """
    from onnx9000.core.ops.torch_auto import transpose_copy as core_transpose_copy

    return core_transpose_copy(input_tensor, *args, **kwargs)


def trapezoid(input_tensor, *args, **kwargs):
    """
    Apply trapezoid to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the trapezoid operation.
    """
    from onnx9000.core.ops.torch_auto import trapezoid as core_trapezoid

    return core_trapezoid(input_tensor, *args, **kwargs)


def trapz(input_tensor, *args, **kwargs):
    """
    Apply trapz to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the trapz operation.
    """
    from onnx9000.core.ops.torch_auto import trapz as core_trapz

    return core_trapz(input_tensor, *args, **kwargs)


def triangular_solve(input_tensor, *args, **kwargs):
    """
    Apply triangular_solve to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the triangular_solve operation.
    """
    from onnx9000.core.ops.torch_auto import triangular_solve as core_triangular_solve

    return core_triangular_solve(input_tensor, *args, **kwargs)


def tril(input_tensor, *args, **kwargs):
    """
    Apply tril to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the tril operation.
    """
    from onnx9000.core.ops.torch_auto import tril as core_tril

    return core_tril(input_tensor, *args, **kwargs)


def tril_indices(input_tensor, *args, **kwargs):
    """
    Apply tril_indices to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the tril_indices operation.
    """
    from onnx9000.core.ops.torch_auto import tril_indices as core_tril_indices

    return core_tril_indices(input_tensor, *args, **kwargs)


def triplet_margin_loss(input_tensor, *args, **kwargs):
    """
    Apply triplet_margin_loss to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the triplet_margin_loss operation.
    """
    from onnx9000.core.ops.torch_auto import triplet_margin_loss as core_triplet_margin_loss

    return core_triplet_margin_loss(input_tensor, *args, **kwargs)


def triu(input_tensor, *args, **kwargs):
    """
    Apply triu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the triu operation.
    """
    from onnx9000.core.ops.torch_auto import triu as core_triu

    return core_triu(input_tensor, *args, **kwargs)


def triu_indices(input_tensor, *args, **kwargs):
    """
    Apply triu_indices to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the triu_indices operation.
    """
    from onnx9000.core.ops.torch_auto import triu_indices as core_triu_indices

    return core_triu_indices(input_tensor, *args, **kwargs)


def true_divide(input_tensor, *args, **kwargs):
    """
    Apply true_divide to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the true_divide operation.
    """
    from onnx9000.core.ops.torch_auto import true_divide as core_true_divide

    return core_true_divide(input_tensor, *args, **kwargs)


def trunc(input_tensor, *args, **kwargs):
    """
    Apply trunc to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the trunc operation.
    """
    from onnx9000.core.ops.torch_auto import trunc as core_trunc

    return core_trunc(input_tensor, *args, **kwargs)


def trunc_(input_tensor, *args, **kwargs):
    """
    Apply trunc_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the trunc_ operation.
    """
    from onnx9000.core.ops.torch_auto import trunc_ as core_trunc_

    return core_trunc_(input_tensor, *args, **kwargs)


def unbind(input_tensor, *args, **kwargs):
    """
    Apply unbind to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unbind operation.
    """
    from onnx9000.core.ops.torch_auto import unbind as core_unbind

    return core_unbind(input_tensor, *args, **kwargs)


def unbind_copy(input_tensor, *args, **kwargs):
    """
    Apply unbind_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unbind_copy operation.
    """
    from onnx9000.core.ops.torch_auto import unbind_copy as core_unbind_copy

    return core_unbind_copy(input_tensor, *args, **kwargs)


def unflatten(input_tensor, *args, **kwargs):
    """
    Apply unflatten to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unflatten operation.
    """
    from onnx9000.core.ops.torch_auto import unflatten as core_unflatten

    return core_unflatten(input_tensor, *args, **kwargs)


def unfold_copy(input_tensor, *args, **kwargs):
    """
    Apply unfold_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unfold_copy operation.
    """
    from onnx9000.core.ops.torch_auto import unfold_copy as core_unfold_copy

    return core_unfold_copy(input_tensor, *args, **kwargs)


def unique_consecutive(input_tensor, *args, **kwargs):
    """
    Apply unique_consecutive to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unique_consecutive operation.
    """
    from onnx9000.core.ops.torch_auto import unique_consecutive as core_unique_consecutive

    return core_unique_consecutive(input_tensor, *args, **kwargs)


def unsafe_chunk(input_tensor, *args, **kwargs):
    """
    Apply unsafe_chunk to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unsafe_chunk operation.
    """
    from onnx9000.core.ops.torch_auto import unsafe_chunk as core_unsafe_chunk

    return core_unsafe_chunk(input_tensor, *args, **kwargs)


def unsafe_split(input_tensor, *args, **kwargs):
    """
    Apply unsafe_split to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unsafe_split operation.
    """
    from onnx9000.core.ops.torch_auto import unsafe_split as core_unsafe_split

    return core_unsafe_split(input_tensor, *args, **kwargs)


def unsafe_split_with_sizes(input_tensor, *args, **kwargs):
    """
    Apply unsafe_split_with_sizes to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unsafe_split_with_sizes operation.
    """
    from onnx9000.core.ops.torch_auto import unsafe_split_with_sizes as core_unsafe_split_with_sizes

    return core_unsafe_split_with_sizes(input_tensor, *args, **kwargs)


def unsqueeze(input_tensor, *args, **kwargs):
    """
    Apply unsqueeze to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unsqueeze operation.
    """
    from onnx9000.core.ops.torch_auto import unsqueeze as core_unsqueeze

    return core_unsqueeze(input_tensor, *args, **kwargs)


def unsqueeze_copy(input_tensor, *args, **kwargs):
    """
    Apply unsqueeze_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the unsqueeze_copy operation.
    """
    from onnx9000.core.ops.torch_auto import unsqueeze_copy as core_unsqueeze_copy

    return core_unsqueeze_copy(input_tensor, *args, **kwargs)


def values_copy(input_tensor, *args, **kwargs):
    """
    Apply values_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the values_copy operation.
    """
    from onnx9000.core.ops.torch_auto import values_copy as core_values_copy

    return core_values_copy(input_tensor, *args, **kwargs)


def vander(input_tensor, *args, **kwargs):
    """
    Apply vander to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the vander operation.
    """
    from onnx9000.core.ops.torch_auto import vander as core_vander

    return core_vander(input_tensor, *args, **kwargs)


def var(input_tensor, *args, **kwargs):
    """
    Apply var to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the var operation.
    """
    from onnx9000.core.ops.torch_auto import var as core_var

    return core_var(input_tensor, *args, **kwargs)


def var_mean(input_tensor, *args, **kwargs):
    """
    Apply var_mean to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the var_mean operation.
    """
    from onnx9000.core.ops.torch_auto import var_mean as core_var_mean

    return core_var_mean(input_tensor, *args, **kwargs)


def vdot(input_tensor, *args, **kwargs):
    """
    Apply vdot to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the vdot operation.
    """
    from onnx9000.core.ops.torch_auto import vdot as core_vdot

    return core_vdot(input_tensor, *args, **kwargs)


def view_as_complex(input_tensor, *args, **kwargs):
    """
    Apply view_as_complex to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the view_as_complex operation.
    """
    from onnx9000.core.ops.torch_auto import view_as_complex as core_view_as_complex

    return core_view_as_complex(input_tensor, *args, **kwargs)


def view_as_complex_copy(input_tensor, *args, **kwargs):
    """
    Apply view_as_complex_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the view_as_complex_copy operation.
    """
    from onnx9000.core.ops.torch_auto import view_as_complex_copy as core_view_as_complex_copy

    return core_view_as_complex_copy(input_tensor, *args, **kwargs)


def view_as_real(input_tensor, *args, **kwargs):
    """
    Apply view_as_real to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the view_as_real operation.
    """
    from onnx9000.core.ops.torch_auto import view_as_real as core_view_as_real

    return core_view_as_real(input_tensor, *args, **kwargs)


def view_as_real_copy(input_tensor, *args, **kwargs):
    """
    Apply view_as_real_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the view_as_real_copy operation.
    """
    from onnx9000.core.ops.torch_auto import view_as_real_copy as core_view_as_real_copy

    return core_view_as_real_copy(input_tensor, *args, **kwargs)


def view_copy(input_tensor, *args, **kwargs):
    """
    Apply view_copy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the view_copy operation.
    """
    from onnx9000.core.ops.torch_auto import view_copy as core_view_copy

    return core_view_copy(input_tensor, *args, **kwargs)


def vsplit(input_tensor, *args, **kwargs):
    """
    Apply vsplit to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the vsplit operation.
    """
    from onnx9000.core.ops.torch_auto import vsplit as core_vsplit

    return core_vsplit(input_tensor, *args, **kwargs)


def vstack(input_tensor, *args, **kwargs):
    """
    Apply vstack to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the vstack operation.
    """
    from onnx9000.core.ops.torch_auto import vstack as core_vstack

    return core_vstack(input_tensor, *args, **kwargs)


def where(input_tensor, *args, **kwargs):
    """
    Apply where to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the where operation.
    """
    from onnx9000.core.ops import where as core_where

    return core_where(input_tensor, *args, **kwargs)


def xlogy(input_tensor, *args, **kwargs):
    """
    Apply xlogy to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the xlogy operation.
    """
    from onnx9000.core.ops.torch_auto import xlogy as core_xlogy

    return core_xlogy(input_tensor, *args, **kwargs)


def xlogy_(input_tensor, *args, **kwargs):
    """
    Apply xlogy_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the xlogy_ operation.
    """
    from onnx9000.core.ops.torch_auto import xlogy_ as core_xlogy_

    return core_xlogy_(input_tensor, *args, **kwargs)


def zero_(input_tensor, *args, **kwargs):
    """
    Apply zero_ to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the zero_ operation.
    """
    from onnx9000.core.ops.torch_auto import zero_ as core_zero_

    return core_zero_(input_tensor, *args, **kwargs)


def zeros_like(input_tensor, *args, **kwargs):
    """
    Apply zeros_like to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the zeros_like operation.
    """
    from onnx9000.core.ops.torch_auto import zeros_like as core_zeros_like

    return core_zeros_like(input_tensor, *args, **kwargs)


def bfloat16(input_tensor, *args, **kwargs):
    """
    Apply bfloat16 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bfloat16 operation.
    """
    from onnx9000.core.ops.torch_auto import bfloat16 as core_bfloat16

    return core_bfloat16(input_tensor, *args, **kwargs)


def bit(input_tensor, *args, **kwargs):
    """
    Apply bit to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bit operation.
    """
    from onnx9000.core.ops.torch_auto import bit as core_bit

    return core_bit(input_tensor, *args, **kwargs)


def bits16(input_tensor, *args, **kwargs):
    """
    Apply bits16 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bits16 operation.
    """
    from onnx9000.core.ops.torch_auto import bits16 as core_bits16

    return core_bits16(input_tensor, *args, **kwargs)


def bits1x8(input_tensor, *args, **kwargs):
    """
    Apply bits1x8 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bits1x8 operation.
    """
    from onnx9000.core.ops.torch_auto import bits1x8 as core_bits1x8

    return core_bits1x8(input_tensor, *args, **kwargs)


def bits2x4(input_tensor, *args, **kwargs):
    """
    Apply bits2x4 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bits2x4 operation.
    """
    from onnx9000.core.ops.torch_auto import bits2x4 as core_bits2x4

    return core_bits2x4(input_tensor, *args, **kwargs)


def bits4x2(input_tensor, *args, **kwargs):
    """
    Apply bits4x2 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bits4x2 operation.
    """
    from onnx9000.core.ops.torch_auto import bits4x2 as core_bits4x2

    return core_bits4x2(input_tensor, *args, **kwargs)


def bits8(input_tensor, *args, **kwargs):
    """
    Apply bits8 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the bits8 operation.
    """
    from onnx9000.core.ops.torch_auto import bits8 as core_bits8

    return core_bits8(input_tensor, *args, **kwargs)


def cdouble(input_tensor, *args, **kwargs):
    """
    Apply cdouble to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cdouble operation.
    """
    from onnx9000.core.ops.torch_auto import cdouble as core_cdouble

    return core_cdouble(input_tensor, *args, **kwargs)


def cfloat(input_tensor, *args, **kwargs):
    """
    Apply cfloat to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the cfloat operation.
    """
    from onnx9000.core.ops.torch_auto import cfloat as core_cfloat

    return core_cfloat(input_tensor, *args, **kwargs)


def chalf(input_tensor, *args, **kwargs):
    """
    Apply chalf to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the chalf operation.
    """
    from onnx9000.core.ops.torch_auto import chalf as core_chalf

    return core_chalf(input_tensor, *args, **kwargs)


def complex128(input_tensor, *args, **kwargs):
    """
    Apply complex128 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the complex128 operation.
    """
    from onnx9000.core.ops.torch_auto import complex128 as core_complex128

    return core_complex128(input_tensor, *args, **kwargs)


def complex32(input_tensor, *args, **kwargs):
    """
    Apply complex32 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the complex32 operation.
    """
    from onnx9000.core.ops.torch_auto import complex32 as core_complex32

    return core_complex32(input_tensor, *args, **kwargs)


def complex64(input_tensor, *args, **kwargs):
    """
    Apply complex64 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the complex64 operation.
    """
    from onnx9000.core.ops.torch_auto import complex64 as core_complex64

    return core_complex64(input_tensor, *args, **kwargs)


def double(input_tensor, *args, **kwargs):
    """
    Apply double to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the double operation.
    """
    from onnx9000.core.ops.torch_auto import double as core_double

    return core_double(input_tensor, *args, **kwargs)


def float(input_tensor, *args, **kwargs):
    """
    Apply float to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the float operation.
    """
    from onnx9000.core.ops.torch_auto import float as core_float

    return core_float(input_tensor, *args, **kwargs)


def float16(input_tensor, *args, **kwargs):
    """
    Apply float16 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the float16 operation.
    """
    from onnx9000.core.ops.torch_auto import float16 as core_float16

    return core_float16(input_tensor, *args, **kwargs)


def float4_e2m1fn_x2(input_tensor, *args, **kwargs):
    """
    Apply float4_e2m1fn_x2 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the float4_e2m1fn_x2 operation.
    """
    from onnx9000.core.ops.torch_auto import float4_e2m1fn_x2 as core_float4_e2m1fn_x2

    return core_float4_e2m1fn_x2(input_tensor, *args, **kwargs)


def float8_e4m3fn(input_tensor, *args, **kwargs):
    """
    Apply float8_e4m3fn to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the float8_e4m3fn operation.
    """
    from onnx9000.core.ops.torch_auto import float8_e4m3fn as core_float8_e4m3fn

    return core_float8_e4m3fn(input_tensor, *args, **kwargs)


def float8_e4m3fnuz(input_tensor, *args, **kwargs):
    """
    Apply float8_e4m3fnuz to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the float8_e4m3fnuz operation.
    """
    from onnx9000.core.ops.torch_auto import float8_e4m3fnuz as core_float8_e4m3fnuz

    return core_float8_e4m3fnuz(input_tensor, *args, **kwargs)


def float8_e5m2(input_tensor, *args, **kwargs):
    """
    Apply float8_e5m2 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the float8_e5m2 operation.
    """
    from onnx9000.core.ops.torch_auto import float8_e5m2 as core_float8_e5m2

    return core_float8_e5m2(input_tensor, *args, **kwargs)


def float8_e5m2fnuz(input_tensor, *args, **kwargs):
    """
    Apply float8_e5m2fnuz to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the float8_e5m2fnuz operation.
    """
    from onnx9000.core.ops.torch_auto import float8_e5m2fnuz as core_float8_e5m2fnuz

    return core_float8_e5m2fnuz(input_tensor, *args, **kwargs)


def float8_e8m0fnu(input_tensor, *args, **kwargs):
    """
    Apply float8_e8m0fnu to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the float8_e8m0fnu operation.
    """
    from onnx9000.core.ops.torch_auto import float8_e8m0fnu as core_float8_e8m0fnu

    return core_float8_e8m0fnu(input_tensor, *args, **kwargs)


def half(input_tensor, *args, **kwargs):
    """
    Apply half to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the half operation.
    """
    from onnx9000.core.ops.torch_auto import half as core_half

    return core_half(input_tensor, *args, **kwargs)


def int(input_tensor, *args, **kwargs):
    """
    Apply int to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the int operation.
    """
    from onnx9000.core.ops.torch_auto import int as core_int

    return core_int(input_tensor, *args, **kwargs)


def int1(input_tensor, *args, **kwargs):
    """
    Apply int1 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the int1 operation.
    """
    from onnx9000.core.ops.torch_auto import int1 as core_int1

    return core_int1(input_tensor, *args, **kwargs)


def int16(input_tensor, *args, **kwargs):
    """
    Apply int16 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the int16 operation.
    """
    from onnx9000.core.ops.torch_auto import int16 as core_int16

    return core_int16(input_tensor, *args, **kwargs)


def int2(input_tensor, *args, **kwargs):
    """
    Apply int2 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the int2 operation.
    """
    from onnx9000.core.ops.torch_auto import int2 as core_int2

    return core_int2(input_tensor, *args, **kwargs)


def int3(input_tensor, *args, **kwargs):
    """
    Apply int3 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the int3 operation.
    """
    from onnx9000.core.ops.torch_auto import int3 as core_int3

    return core_int3(input_tensor, *args, **kwargs)


def int4(input_tensor, *args, **kwargs):
    """
    Apply int4 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the int4 operation.
    """
    from onnx9000.core.ops.torch_auto import int4 as core_int4

    return core_int4(input_tensor, *args, **kwargs)


def int5(input_tensor, *args, **kwargs):
    """
    Apply int5 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the int5 operation.
    """
    from onnx9000.core.ops.torch_auto import int5 as core_int5

    return core_int5(input_tensor, *args, **kwargs)


def int6(input_tensor, *args, **kwargs):
    """
    Apply int6 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the int6 operation.
    """
    from onnx9000.core.ops.torch_auto import int6 as core_int6

    return core_int6(input_tensor, *args, **kwargs)


def int7(input_tensor, *args, **kwargs):
    """
    Apply int7 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the int7 operation.
    """
    from onnx9000.core.ops.torch_auto import int7 as core_int7

    return core_int7(input_tensor, *args, **kwargs)


def int8(input_tensor, *args, **kwargs):
    """
    Apply int8 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the int8 operation.
    """
    from onnx9000.core.ops.torch_auto import int8 as core_int8

    return core_int8(input_tensor, *args, **kwargs)


def long(input_tensor, *args, **kwargs):
    """
    Apply long to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the long operation.
    """
    from onnx9000.core.ops.torch_auto import long as core_long

    return core_long(input_tensor, *args, **kwargs)


def qint32(input_tensor, *args, **kwargs):
    """
    Apply qint32 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the qint32 operation.
    """
    from onnx9000.core.ops.torch_auto import qint32 as core_qint32

    return core_qint32(input_tensor, *args, **kwargs)


def qint8(input_tensor, *args, **kwargs):
    """
    Apply qint8 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the qint8 operation.
    """
    from onnx9000.core.ops.torch_auto import qint8 as core_qint8

    return core_qint8(input_tensor, *args, **kwargs)


def quint2x4(input_tensor, *args, **kwargs):
    """
    Apply quint2x4 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quint2x4 operation.
    """
    from onnx9000.core.ops.torch_auto import quint2x4 as core_quint2x4

    return core_quint2x4(input_tensor, *args, **kwargs)


def quint4x2(input_tensor, *args, **kwargs):
    """
    Apply quint4x2 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quint4x2 operation.
    """
    from onnx9000.core.ops.torch_auto import quint4x2 as core_quint4x2

    return core_quint4x2(input_tensor, *args, **kwargs)


def quint8(input_tensor, *args, **kwargs):
    """
    Apply quint8 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the quint8 operation.
    """
    from onnx9000.core.ops.torch_auto import quint8 as core_quint8

    return core_quint8(input_tensor, *args, **kwargs)


def short(input_tensor, *args, **kwargs):
    """
    Apply short to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the short operation.
    """
    from onnx9000.core.ops.torch_auto import short as core_short

    return core_short(input_tensor, *args, **kwargs)


def uint1(input_tensor, *args, **kwargs):
    """
    Apply uint1 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the uint1 operation.
    """
    from onnx9000.core.ops.torch_auto import uint1 as core_uint1

    return core_uint1(input_tensor, *args, **kwargs)


def uint16(input_tensor, *args, **kwargs):
    """
    Apply uint16 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the uint16 operation.
    """
    from onnx9000.core.ops.torch_auto import uint16 as core_uint16

    return core_uint16(input_tensor, *args, **kwargs)


def uint2(input_tensor, *args, **kwargs):
    """
    Apply uint2 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the uint2 operation.
    """
    from onnx9000.core.ops.torch_auto import uint2 as core_uint2

    return core_uint2(input_tensor, *args, **kwargs)


def uint3(input_tensor, *args, **kwargs):
    """
    Apply uint3 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the uint3 operation.
    """
    from onnx9000.core.ops.torch_auto import uint3 as core_uint3

    return core_uint3(input_tensor, *args, **kwargs)


def uint32(input_tensor, *args, **kwargs):
    """
    Apply uint32 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the uint32 operation.
    """
    from onnx9000.core.ops.torch_auto import uint32 as core_uint32

    return core_uint32(input_tensor, *args, **kwargs)


def uint4(input_tensor, *args, **kwargs):
    """
    Apply uint4 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the uint4 operation.
    """
    from onnx9000.core.ops.torch_auto import uint4 as core_uint4

    return core_uint4(input_tensor, *args, **kwargs)


def uint5(input_tensor, *args, **kwargs):
    """
    Apply uint5 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the uint5 operation.
    """
    from onnx9000.core.ops.torch_auto import uint5 as core_uint5

    return core_uint5(input_tensor, *args, **kwargs)


def uint6(input_tensor, *args, **kwargs):
    """
    Apply uint6 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the uint6 operation.
    """
    from onnx9000.core.ops.torch_auto import uint6 as core_uint6

    return core_uint6(input_tensor, *args, **kwargs)


def uint64(input_tensor, *args, **kwargs):
    """
    Apply uint64 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the uint64 operation.
    """
    from onnx9000.core.ops.torch_auto import uint64 as core_uint64

    return core_uint64(input_tensor, *args, **kwargs)


def uint7(input_tensor, *args, **kwargs):
    """
    Apply uint7 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the uint7 operation.
    """
    from onnx9000.core.ops.torch_auto import uint7 as core_uint7

    return core_uint7(input_tensor, *args, **kwargs)


def uint8(input_tensor, *args, **kwargs):
    """
    Apply uint8 to the input tensor.

    Args:
        input_tensor: The input Tensor.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The result of the uint8 operation.
    """
    from onnx9000.core.ops.torch_auto import uint8 as core_uint8

    return core_uint8(input_tensor, *args, **kwargs)
