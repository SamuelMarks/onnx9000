"""TFLite FlatBuffer schema definitions and object creation helpers."""

from enum import IntEnum

from .builder import FlatBufferBuilder


class TensorType:
    """TFLite tensor data types."""

    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    UINT8 = 3
    INT64 = 4
    STRING = 5
    BOOL = 6
    INT16 = 7
    COMPLEX64 = 8
    INT8 = 9
    FLOAT64 = 10
    COMPLEX128 = 11
    UINT64 = 12
    RESOURCE = 13
    VARIANT = 14
    UINT32 = 15
    UINT16 = 16
    INT4 = 17


class Padding:
    """TFLite padding types."""

    SAME = 0
    VALID = 1


class BuiltinOperator(IntEnum):
    """TFLite builtin operators."""

    ADD = 0
    AVERAGE_POOL_2D = 1
    CONCATENATION = 2
    CONV_2D = 3
    DEPTHWISE_CONV_2D = 4
    DEPTH_TO_SPACE = 5
    DEQUANTIZE = 6
    EMBEDDING_LOOKUP = 7
    FLOOR = 8
    FULLY_CONNECTED = 9
    HASHTABLE_LOOKUP = 10
    L2_NORMALIZATION = 11
    L2_POOL_2D = 12
    LOCAL_RESPONSE_NORMALIZATION = 13
    LOGISTIC = 14
    LSH_PROJECTION = 15
    LSTM = 16
    MAX_POOL_2D = 17
    MUL = 18
    RELU = 19
    RELU_N1_TO_1 = 20
    RELU6 = 21
    RESHAPE = 22
    RESIZE_BILINEAR = 23
    RNN = 24
    SOFTMAX = 25
    SPACE_TO_DEPTH = 26
    SVDF = 27
    TANH = 28
    CONCAT_EMBEDDINGS = 29
    SKIP_GRAM = 30
    CALL = 31
    CUSTOM = 32
    EMBEDDING_LOOKUP_SPARSE = 33
    PAD = 34
    UNIDIRECTIONAL_SEQUENCE_RNN = 35
    GATHER = 36
    BATCH_TO_SPACE_ND = 37
    SPACE_TO_BATCH_ND = 38
    TRANSPOSE = 39
    MEAN = 40
    SUB = 41
    DIV = 42
    SQUEEZE = 43
    UNIDIRECTIONAL_SEQUENCE_LSTM = 44
    STRIDED_SLICE = 45
    BIDIRECTIONAL_SEQUENCE_RNN = 46
    EXP = 47
    TOPK_V2 = 48
    SPLIT = 49
    LOG_SOFTMAX = 50
    DELEGATE = 51
    BIDIRECTIONAL_SEQUENCE_LSTM = 52
    CAST = 53
    PRELU = 54
    MAXIMUM = 55
    ARG_MAX = 56
    MINIMUM = 57
    LESS = 58
    NEG = 59
    PADV2 = 60
    GREATER = 61
    GREATER_EQUAL = 62
    LESS_EQUAL = 63
    SELECT = 64
    SLICE = 65
    SIN = 66
    TRANSPOSE_CONV = 67
    SPARSE_TO_DENSE = 68
    TILE = 69
    EXPAND_DIMS = 70
    EQUAL = 71
    NOT_EQUAL = 72
    LOG = 73
    SUM = 74
    SQRT = 75
    RSQRT = 76
    SHAPE = 77
    POW = 78
    ARG_MIN = 79
    FAKE_QUANT = 80
    REDUCE_PROD = 81
    REDUCE_MAX = 82
    PACK = 83
    LOGICAL_OR = 84
    ONE_HOT = 85
    LOGICAL_AND = 86
    LOGICAL_NOT = 87
    UNPACK = 88
    REDUCE_MIN = 89
    FLOOR_DIV = 90
    REDUCE_ANY = 91
    SQUARE = 92
    ZEROS_LIKE = 93
    FILL = 94
    FLOOR_MOD = 95
    RANGE = 96
    RESIZE_NEAREST_NEIGHBOR = 97
    LEAKY_RELU = 98
    SQUARED_DIFFERENCE = 99
    MIRROR_PAD = 100
    ABS = 101
    SPLIT_V = 102
    UNIQUE = 103
    CEIL = 104
    REVERSE_V2 = 105
    ADD_N = 106
    GATHER_ND = 107
    COS = 108
    WHERE = 109
    RANK = 110
    ELU = 111
    REVERSE_SEQUENCE = 112
    MATRIX_DIAG = 113
    QUANTIZE = 114
    MATRIX_SET_DIAG = 115
    ROUND = 116
    HARD_SWISH = 117
    IF = 118
    WHILE = 119
    NON_MAX_SUPPRESSION_V4 = 120
    NON_MAX_SUPPRESSION_V5 = 121
    SCATTER_ND = 122
    SELECT_V2 = 123
    DENSIFY = 124
    SEGMENT_SUM = 125
    BATCH_MATMUL = 126
    PLACEHOLDER_FOR_GREATER_OP_CODES = 127
    CUMSUM = 128
    CALL_ONCE = 129
    BROADCAST_TO = 130
    RFFT2D = 131
    CONV_3D = 132
    IMAG = 133
    REAL = 134
    COMPLEX_ABS = 135
    HASHTABLE = 136
    HASHTABLE_FIND = 137
    HASHTABLE_IMPORT = 138
    HASHTABLE_SIZE = 139
    REDUCE_ALL = 140
    CONV_3D_TRANSPOSE = 141
    VAR_HANDLE = 142
    READ_VARIABLE = 143
    ASSIGN_VARIABLE = 144
    BROADCAST_ARGS = 145
    RANDOM_STANDARD_NORMAL = 146
    BUCKETIZE = 147
    RANDOM_UNIFORM = 148
    MULTINOMIAL = 149
    GELU = 150
    DYNAMIC_UPDATE_SLICE = 151
    RELU_0_TO_1 = 152
    UNSORTED_SEGMENT_PROD = 153
    UNSORTED_SEGMENT_MAX = 154
    UNSORTED_SEGMENT_SUM = 155
    ATAN2 = 156
    UNSORTED_SEGMENT_MIN = 157
    COMPLEX = 158
    SIGN = 159
    BITCAST = 160
    BITWISE_XOR = 161
    RIGHT_SHIFT = 162


class BuiltinOptions(IntEnum):
    """TFLite builtin operator options types."""

    NONE = 0
    Conv2DOptions = 1
    DepthwiseConv2DOptions = 2
    ConcatEmbeddingsOptions = 3
    LSHProjectionOptions = 4
    Pool2DOptions = 5
    SVDFOptions = 6
    RNNOptions = 7
    FullyConnectedOptions = 8
    SoftmaxOptions = 9
    ConcatenationOptions = 10
    AddOptions = 11
    L2NormOptions = 12
    LocalResponseNormOptions = 13
    LSTMOptions = 14
    ResizeBilinearOptions = 15
    CallOptions = 16
    ReshapeOptions = 17
    SkipGramOptions = 18
    SpaceToDepthOptions = 19
    EmbeddingLookupSparseOptions = 20
    MulOptions = 21
    PadOptions = 22
    GatherOptions = 23
    BatchToSpaceNDOptions = 24
    SpaceToBatchNDOptions = 25
    TransposeOptions = 26
    ReducerOptions = 27
    SubOptions = 28
    DivOptions = 29
    SqueezeOptions = 30
    SequenceRNNOptions = 31
    StridedSliceOptions = 32
    ExpOptions = 33
    TopKV2Options = 34
    SplitOptions = 35
    LogSoftmaxOptions = 36
    CastOptions = 37
    DequantizeOptions = 38
    MaximumMinimumOptions = 39
    ArgMaxOptions = 40
    LessOptions = 41
    NegOptions = 42
    PadV2Options = 43
    GreaterOptions = 44
    GreaterEqualOptions = 45
    LessEqualOptions = 46
    SelectOptions = 47
    SliceOptions = 48
    TransposeConvOptions = 49
    SparseToDenseOptions = 50
    TileOptions = 51
    ExpandDimsOptions = 52
    EqualOptions = 53
    NotEqualOptions = 54
    ShapeOptions = 55
    PowOptions = 56
    ArgMinOptions = 57
    FakeQuantOptions = 58
    PackOptions = 59
    LogicalOrOptions = 60
    OneHotOptions = 61
    LogicalAndOptions = 62
    LogicalNotOptions = 63
    UnpackOptions = 64
    FloorDivOptions = 65
    SquareOptions = 66
    ZerosLikeOptions = 67
    FillOptions = 68
    FloorModOptions = 69
    RangeOptions = 70
    ResizeNearestNeighborOptions = 71
    LeakyReluOptions = 72
    SquaredDifferenceOptions = 73
    MirrorPadOptions = 74
    AbsOptions = 75
    SplitVOptions = 76
    UniqueOptions = 77
    ReverseV2Options = 78
    AddNOptions = 79
    GatherNdOptions = 80
    CosOptions = 81
    WhereOptions = 82
    RankOptions = 83
    ReverseSequenceOptions = 84
    MatrixDiagOptions = 85
    QuantizeOptions = 86
    MatrixSetDiagOptions = 87
    HardSwishOptions = 88
    IfOptions = 89
    WhileOptions = 90
    DepthToSpaceOptions = 91
    NonMaxSuppressionV4Options = 92
    NonMaxSuppressionV5Options = 93
    ScatterNdOptions = 94
    SelectV2Options = 95
    DensifyOptions = 96
    SegmentSumOptions = 97
    BatchMatMulOptions = 98
    CumsumOptions = 99
    CallOnceOptions = 100
    BroadcastToOptions = 101
    RFFT2DOptions = 102
    Conv3DOptions = 103
    HashtableOptions = 104
    HashtableFindOptions = 105
    HashtableImportOptions = 106
    HashtableSizeOptions = 107
    VarHandleOptions = 108
    ReadVariableOptions = 109
    AssignVariableOptions = 110
    BroadcastArgsOptions = 111
    RandomOptions = 112
    BucketizeOptions = 113
    GeluOptions = 114
    DynamicUpdateSliceOptions = 115


class OperatorCode:
    """TFLite OperatorCode object helper."""

    @staticmethod
    def create(
        builder: FlatBufferBuilder, builtin_code: int, custom_code_offset: int, version: int
    ) -> int:
        """Create a TFLite OperatorCode object."""
        builder.start_object(4)
        builder.add_field_int8(0, builtin_code, 0)
        builder.add_field_offset(1, custom_code_offset, 0)
        builder.add_field_int32(2, version, 1)
        builder.add_field_int32(3, builtin_code, 0)
        return builder.end_object()


class QuantizationParameters:
    """TFLite QuantizationParameters object helper."""

    @staticmethod
    def create(
        builder: FlatBufferBuilder,
        min_offset: int,
        max_offset: int,
        scale_offset: int,
        zero_point_offset: int,
        details_type: int,
        details_offset: int,
        quantized_dimension: int,
    ) -> int:
        """Create a TFLite QuantizationParameters object."""
        builder.start_object(7)
        builder.add_field_offset(0, min_offset, 0)
        builder.add_field_offset(1, max_offset, 0)
        builder.add_field_offset(2, scale_offset, 0)
        builder.add_field_offset(3, zero_point_offset, 0)
        builder.add_field_int8(4, details_type, 0)
        builder.add_field_offset(5, details_offset, 0)
        builder.add_field_int32(6, quantized_dimension, 0)
        return builder.end_object()


class Tensor:
    """TFLite Tensor object helper."""

    @staticmethod
    def create(
        builder: FlatBufferBuilder,
        shape_offset: int,
        tensor_type: int,
        buffer: int,
        name_offset: int,
        quantization_offset: int,
        is_variable: bool,
        sparsity_offset: int,
        shape_signature_offset: int,
        has_rank: bool,
    ) -> int:
        """Create a TFLite Tensor object."""
        builder.start_object(10)
        builder.add_field_offset(0, shape_offset, 0)
        builder.add_field_int8(1, tensor_type, 0)
        builder.add_field_int32(2, buffer, 0)
        builder.add_field_offset(3, name_offset, 0)
        builder.add_field_offset(4, quantization_offset, 0)
        builder.add_field_int8(5, 1 if is_variable else 0, 0)
        builder.add_field_offset(6, sparsity_offset, 0)
        builder.add_field_offset(7, shape_signature_offset, 0)
        builder.add_field_int8(8, 1 if has_rank else 0, 0)
        return builder.end_object()


class Operator:
    """TFLite Operator object helper."""

    @staticmethod
    def create(
        builder: FlatBufferBuilder,
        opcode_index: int,
        inputs_offset: int,
        outputs_offset: int,
        builtin_options_type: int,
        builtin_options_offset: int,
        custom_options_offset: int,
        custom_options_format: int,
        mutating_variable_inputs: bool,
        intermediates_offset: int,
    ) -> int:
        """Create a TFLite Operator object."""
        builder.start_object(9)
        builder.add_field_int32(0, opcode_index, 0)
        builder.add_field_offset(1, inputs_offset, 0)
        builder.add_field_offset(2, outputs_offset, 0)
        builder.add_field_int8(3, builtin_options_type, 0)
        builder.add_field_offset(4, builtin_options_offset, 0)
        builder.add_field_offset(5, custom_options_offset, 0)
        builder.add_field_int8(6, custom_options_format, 0)
        builder.add_field_int8(7, 1 if mutating_variable_inputs else 0, 0)
        builder.add_field_offset(8, intermediates_offset, 0)
        return builder.end_object()


class SubGraph:
    """TFLite SubGraph object helper."""

    @staticmethod
    def create(
        builder: FlatBufferBuilder,
        tensors_offset: int,
        inputs_offset: int,
        outputs_offset: int,
        operators_offset: int,
        name_offset: int,
    ) -> int:
        """Create a TFLite SubGraph object."""
        builder.start_object(5)
        builder.add_field_offset(0, tensors_offset, 0)
        builder.add_field_offset(1, inputs_offset, 0)
        builder.add_field_offset(2, outputs_offset, 0)
        builder.add_field_offset(3, operators_offset, 0)
        builder.add_field_offset(4, name_offset, 0)
        return builder.end_object()


class Buffer:
    """TFLite Buffer object helper."""

    @staticmethod
    def create(builder: FlatBufferBuilder, data_offset: int) -> int:
        """Create a TFLite Buffer object."""
        builder.start_object(1)
        builder.add_field_offset(0, data_offset, 0)
        return builder.end_object()


class Metadata:
    """TFLite Metadata object helper."""

    @staticmethod
    def create(builder: FlatBufferBuilder, name_offset: int, buffer: int) -> int:
        """Create a TFLite Metadata object."""
        builder.start_object(2)
        builder.add_field_offset(0, name_offset, 0)
        builder.add_field_int32(1, buffer, 0)
        return builder.end_object()


class Model:
    """TFLite Model object helper."""

    @staticmethod
    def create(
        builder: FlatBufferBuilder,
        version: int,
        operator_codes_offset: int,
        subgraphs_offset: int,
        description_offset: int,
        buffers_offset: int,
        metadata_buffer_offset: int,
        metadata_offset: int,
        signature_defs_offset: int,
    ) -> int:
        """Create a TFLite Model object."""
        builder.start_object(8)
        builder.add_field_int32(0, version, 0)
        builder.add_field_offset(1, operator_codes_offset, 0)
        builder.add_field_offset(2, subgraphs_offset, 0)
        builder.add_field_offset(3, description_offset, 0)
        builder.add_field_offset(4, buffers_offset, 0)
        builder.add_field_offset(5, metadata_buffer_offset, 0)
        builder.add_field_offset(6, metadata_offset, 0)
        builder.add_field_offset(7, signature_defs_offset, 0)
        return builder.end_object()
