"""Operator mapping from ONNX to TFLite.

This module defines how ONNX operators are mapped to TFLite operators,
including the transformation of attributes into TFLite BuiltinOptions.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from onnx9000.core.ir import Node

from ..flatbuffer.schema import BuiltinOperator, BuiltinOptions, TensorType


@dataclass
class TFLiteOperatorMapping:
    """Represents a mapping from an ONNX operator to a TFLite operator.

    Attributes:
        builtin_code: The TFLite BuiltinOperator code.
        builtin_options_type: The type of BuiltinOptions for this operator.
        create_options: An optional callable to create the BuiltinOptions flatbuffer object.

    """

    builtin_code: int
    builtin_options_type: int
    create_options: Optional[Callable] = None


def _map_cast(b, n):
    """Map ONNX Cast attributes to TFLite CastOptions."""
    to_attr = n.attributes.get("to")
    to_val = to_attr.value if to_attr else 1
    out_type = TensorType.FLOAT32
    if to_val == 2:
        out_type = TensorType.UINT8
    elif to_val == 3:
        out_type = TensorType.INT8
    elif to_val == 6:
        out_type = TensorType.INT32
    elif to_val == 7:
        out_type = TensorType.INT64
    elif to_val == 9:
        out_type = TensorType.BOOL
    elif to_val == 10:
        out_type = TensorType.FLOAT16

    b.start_object(2)
    b.add_field_int8(0, 0, 0)
    b.add_field_int8(1, out_type, 0)
    return b.end_object()


def _map_fully_connected(b, n):
    """Map ONNX Gemm attributes to TFLite FullyConnectedOptions."""
    b.start_object(4)
    b.add_field_int8(0, 0, 0)
    b.add_field_int8(1, 0, 0)
    b.add_field_int8(2, 0, 0)
    b.add_field_int8(3, 1, 0)
    return b.end_object()


def _map_transpose_conv(b, n):
    """Map ONNX ConvTranspose attributes to TFLite TransposeConvOptions."""
    strides_attr = n.attributes.get("strides")
    pads_attr = n.attributes.get("pads")
    auto_pad_attr = n.attributes.get("auto_pad")

    strides = strides_attr.value if strides_attr else [1, 1]
    pads = pads_attr.value if pads_attr else [0, 0, 0, 0]
    auto_pad = auto_pad_attr.value if auto_pad_attr else b"NOTSET"

    stride_h = strides[0] if len(strides) > 0 else 1
    stride_w = strides[1] if len(strides) > 1 else 1

    padding = 1  # VALID by default
    if auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
        padding = 0  # SAME
    elif auto_pad == b"VALID":
        padding = 1  # VALID
    elif sum(pads) > 0:
        padding = 0  # SAME

    b.start_object(3)
    b.add_field_int8(0, padding, 0)
    b.add_field_int32(1, stride_w, 1)
    b.add_field_int32(2, stride_h, 1)
    return b.end_object()


def _map_scatter_elements(b, n):
    """Map ONNX ScatterElements to TFLite SCATTER_ND (with warning)."""
    import logging

    logging.warning(
        "[onnx2tf] Warning: ScatterElements mapped to SCATTER_ND. Layout mutations may be necessary. Ensure input structures match."
    )
    return 0


ELEMENTWISE_OPS = {
    "ConstantOfShape": TFLiteOperatorMapping(BuiltinOperator.FILL, BuiltinOptions.FillOptions),
    "QuantizeLinear": TFLiteOperatorMapping(
        BuiltinOperator.QUANTIZE, BuiltinOptions.QuantizeOptions
    ),
    "DequantizeLinear": TFLiteOperatorMapping(
        BuiltinOperator.DEQUANTIZE, BuiltinOptions.DequantizeOptions
    ),
    "Cast": TFLiteOperatorMapping(
        BuiltinOperator.CAST, BuiltinOptions.CastOptions, lambda b, n, g=None: _map_cast(b, n)
    ),
    "PRelu": TFLiteOperatorMapping(BuiltinOperator.PRELU, BuiltinOptions.NONE),
    "Expand": TFLiteOperatorMapping(
        BuiltinOperator.BROADCAST_TO, BuiltinOptions.BroadcastToOptions
    ),
    # 75. Emit ADD
    "Add": TFLiteOperatorMapping(
        BuiltinOperator.ADD, BuiltinOptions.AddOptions, lambda b, n, g=None: _map_math_fused(b, n)
    ),
    # 76. Emit SUB
    "Sub": TFLiteOperatorMapping(
        BuiltinOperator.SUB, BuiltinOptions.SubOptions, lambda b, n, g=None: _map_math_fused(b, n)
    ),
    # 77. Emit MUL
    "Mul": TFLiteOperatorMapping(
        BuiltinOperator.MUL, BuiltinOptions.MulOptions, lambda b, n, g=None: _map_math_fused(b, n)
    ),
    # 78. Emit DIV
    "Div": TFLiteOperatorMapping(
        BuiltinOperator.DIV, BuiltinOptions.DivOptions, lambda b, n, g=None: _map_math_fused(b, n)
    ),
    # 79. Emit FLOOR_DIV
    "FloorDiv": TFLiteOperatorMapping(BuiltinOperator.FLOOR_DIV, BuiltinOptions.FloorDivOptions),
    # 80. Emit MOD
    "Mod": TFLiteOperatorMapping(BuiltinOperator.FLOOR_MOD, BuiltinOptions.FloorModOptions),
    # 81. Emit MAXIMUM
    "Max": TFLiteOperatorMapping(BuiltinOperator.MAXIMUM, BuiltinOptions.MaximumMinimumOptions),
    # 82. Emit MINIMUM
    "Min": TFLiteOperatorMapping(BuiltinOperator.MINIMUM, BuiltinOptions.MaximumMinimumOptions),
    # 83. Emit POW
    "Pow": TFLiteOperatorMapping(BuiltinOperator.POW, BuiltinOptions.PowOptions),
    # 84. Emit ABS
    "Abs": TFLiteOperatorMapping(BuiltinOperator.ABS, BuiltinOptions.AbsOptions),
    # 85. Emit EXP
    "Exp": TFLiteOperatorMapping(BuiltinOperator.EXP, BuiltinOptions.ExpOptions),
    # 86. Emit LOG
    "Log": TFLiteOperatorMapping(BuiltinOperator.LOG, BuiltinOptions.NONE),
    # 87. Emit SQRT
    "Sqrt": TFLiteOperatorMapping(BuiltinOperator.SQRT, BuiltinOptions.NONE),
    # 88. Emit RSQRT
    "Rsqrt": TFLiteOperatorMapping(BuiltinOperator.RSQRT, BuiltinOptions.NONE),
    # 89. Emit SIN
    "Sin": TFLiteOperatorMapping(BuiltinOperator.SIN, BuiltinOptions.NONE),
    # 90. Emit COS
    "Cos": TFLiteOperatorMapping(BuiltinOperator.COS, BuiltinOptions.CosOptions),
    # 91. Emit NEG
    "Neg": TFLiteOperatorMapping(BuiltinOperator.NEG, BuiltinOptions.NegOptions),
    # 92. Emit CEIL
    "Ceil": TFLiteOperatorMapping(BuiltinOperator.CEIL, BuiltinOptions.NONE),
    # 93. Emit FLOOR
    "Floor": TFLiteOperatorMapping(BuiltinOperator.FLOOR, BuiltinOptions.NONE),
    # 94. Emit ROUND
    "Round": TFLiteOperatorMapping(BuiltinOperator.ROUND, BuiltinOptions.NONE),
    # 95. Emit SIGN
    "Sign": TFLiteOperatorMapping(BuiltinOperator.SIGN, BuiltinOptions.NONE),
    # 126. Emit RELU
    "Relu": TFLiteOperatorMapping(BuiltinOperator.RELU, BuiltinOptions.NONE),
    # 127. Emit RELU6
    "Relu6": TFLiteOperatorMapping(BuiltinOperator.RELU6, BuiltinOptions.NONE),
    # 128. Emit LEAKY_RELU
    "LeakyRelu": TFLiteOperatorMapping(
        BuiltinOperator.LEAKY_RELU,
        BuiltinOptions.LeakyReluOptions,
        lambda b, n, g=None: _map_leaky_relu(b, n),
    ),
    # 129. Emit ELU
    "Elu": TFLiteOperatorMapping(BuiltinOperator.ELU, BuiltinOptions.NONE),
    # 130. Emit LOGISTIC (Sigmoid)
    "Sigmoid": TFLiteOperatorMapping(BuiltinOperator.LOGISTIC, BuiltinOptions.NONE),
    # 131. Emit TANH
    "Tanh": TFLiteOperatorMapping(BuiltinOperator.TANH, BuiltinOptions.NONE),
    # 132. Emit SOFTMAX
    "Softmax": TFLiteOperatorMapping(
        BuiltinOperator.SOFTMAX,
        BuiltinOptions.SoftmaxOptions,
        lambda b, n, g=None: _map_softmax(b, n),
    ),
    # 134. Emit LOG_SOFTMAX
    "LogSoftmax": TFLiteOperatorMapping(
        BuiltinOperator.LOG_SOFTMAX, BuiltinOptions.LogSoftmaxOptions
    ),
    # 135. Emit HARD_SWISH
    "HardSwish": TFLiteOperatorMapping(BuiltinOperator.HARD_SWISH, BuiltinOptions.NONE),
    # 136. Emit GELU
    "Gelu": TFLiteOperatorMapping(
        BuiltinOperator.GELU, BuiltinOptions.GeluOptions, lambda b, n, g=None: _map_gelu(b, n)
    ),
    # 138. Emit PRelu
    "PRelu": TFLiteOperatorMapping(BuiltinOperator.PRELU, BuiltinOptions.NONE),
    # 142. Emit L2_NORMALIZATION
    "LpNormalization": TFLiteOperatorMapping(
        BuiltinOperator.L2_NORMALIZATION,
        BuiltinOptions.L2NormOptions,
        lambda b, n, g=None: _map_l2norm(b, n),
    ),
    # 143. Emit LOCAL_RESPONSE_NORMALIZATION
    "LRN": TFLiteOperatorMapping(
        BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION,
        BuiltinOptions.LocalResponseNormOptions,
        lambda b, n, g=None: _map_lrn(b, n),
    ),
    # 146. Emit RESHAPE
    "Reshape": TFLiteOperatorMapping(
        BuiltinOperator.RESHAPE,
        BuiltinOptions.ReshapeOptions,
        lambda b, n, g=None: _map_reshape(b, n, g),
    ),
    # 148. Emit TRANSPOSE
    "Transpose": TFLiteOperatorMapping(BuiltinOperator.TRANSPOSE, BuiltinOptions.TransposeOptions),
    # 149. Emit SQUEEZE
    "Squeeze": TFLiteOperatorMapping(
        BuiltinOperator.SQUEEZE,
        BuiltinOptions.SqueezeOptions,
        lambda b, n, g=None: _map_squeeze(b, n),
    ),
    # 150. Emit EXPAND_DIMS (Unsqueeze)
    "Unsqueeze": TFLiteOperatorMapping(
        BuiltinOperator.EXPAND_DIMS, BuiltinOptions.ExpandDimsOptions
    ),
    # 151. Emit CONCATENATION
    "Concat": TFLiteOperatorMapping(
        BuiltinOperator.CONCATENATION,
        BuiltinOptions.ConcatenationOptions,
        lambda b, n, g=None: _map_concat(b, n),
    ),
    # 153. Emit SPLIT
    "Split": TFLiteOperatorMapping(
        BuiltinOperator.SPLIT, BuiltinOptions.SplitOptions, lambda b, n, g=None: _map_split(b, n)
    ),
    # 154. Emit SPLIT_V
    "SplitV": TFLiteOperatorMapping(
        BuiltinOperator.SPLIT_V, BuiltinOptions.SplitVOptions, lambda b, n, g=None: _map_split(b, n)
    ),
    # 155. Emit SLICE
    "Slice": TFLiteOperatorMapping(BuiltinOperator.SLICE, BuiltinOptions.SliceOptions),
    # 156. Emit STRIDED_SLICE
    "StridedSlice": TFLiteOperatorMapping(
        BuiltinOperator.STRIDED_SLICE,
        BuiltinOptions.StridedSliceOptions,
        lambda b, n, g=None: _map_strided_slice(b, n),
    ),
    # 158. Emit GATHER
    "Gather": TFLiteOperatorMapping(
        BuiltinOperator.GATHER, BuiltinOptions.GatherOptions, lambda b, n, g=None: _map_gather(b, n)
    ),
    # 159. Emit GATHER_ND
    "GatherND": TFLiteOperatorMapping(BuiltinOperator.GATHER_ND, BuiltinOptions.GatherNdOptions),
    # 160. Emit SCATTER_ND
    "ScatterND": TFLiteOperatorMapping(BuiltinOperator.SCATTER_ND, BuiltinOptions.ScatterNdOptions),
    "ScatterElements": TFLiteOperatorMapping(
        BuiltinOperator.SCATTER_ND,
        BuiltinOptions.ScatterNdOptions,
        lambda b, n, g=None: _map_scatter_elements(b, n),
    ),
    # 162. Emit TILE
    "Tile": TFLiteOperatorMapping(BuiltinOperator.TILE, BuiltinOptions.TileOptions),
    # 163. Emit PAD
    "Pad": TFLiteOperatorMapping(BuiltinOperator.PAD, BuiltinOptions.PadOptions),
    # 164. Emit PADV2
    "PadV2": TFLiteOperatorMapping(BuiltinOperator.PADV2, BuiltinOptions.PadV2Options),
    # 165. Emit MIRROR_PAD
    "MirrorPad": TFLiteOperatorMapping(
        BuiltinOperator.MIRROR_PAD,
        BuiltinOptions.MirrorPadOptions,
        lambda b, n, g=None: _map_mirror_pad(b, n),
    ),
    # 166. Emit SHAPE
    "Shape": TFLiteOperatorMapping(BuiltinOperator.SHAPE, BuiltinOptions.ShapeOptions),
    # 167. Emit PACK
    "SequenceConstruct": TFLiteOperatorMapping(
        BuiltinOperator.PACK, BuiltinOptions.PackOptions, lambda b, n, g=None: _map_pack(b, n)
    ),
    # 168. Emit UNPACK
    "SplitToSequence": TFLiteOperatorMapping(
        BuiltinOperator.UNPACK, BuiltinOptions.UnpackOptions, lambda b, n, g=None: _map_unpack(b, n)
    ),
    # 113. Emit MAX_POOL_2D
    "MaxPool": TFLiteOperatorMapping(
        BuiltinOperator.MAX_POOL_2D,
        BuiltinOptions.Pool2DOptions,
        lambda b, n, g=None: map_pool2d_options(b, n),
    ),
    # 115. Emit AVERAGE_POOL_2D
    "AveragePool": TFLiteOperatorMapping(
        BuiltinOperator.AVERAGE_POOL_2D,
        BuiltinOptions.Pool2DOptions,
        lambda b, n, g=None: map_pool2d_options(b, n),
    ),
    "LpPool": TFLiteOperatorMapping(
        BuiltinOperator.L2_POOL_2D,
        BuiltinOptions.Pool2DOptions,
        lambda b, n, g=None: map_pool2d_options(b, n),
    ),
    # 116. Emit MEAN (GlobalAveragePool)
    "GlobalAveragePool": TFLiteOperatorMapping(
        BuiltinOperator.MEAN, BuiltinOptions.ReducerOptions, lambda b, n, g=None: _map_reducer(b, n)
    ),
    # 117. Emit REDUCE_MAX (GlobalMaxPool)
    "GlobalMaxPool": TFLiteOperatorMapping(
        BuiltinOperator.REDUCE_MAX,
        BuiltinOptions.ReducerOptions,
        lambda b, n, g=None: _map_reducer(b, n),
    ),
    # Phase 8: Matrix Multiplication
    # 171. Emit FULLY_CONNECTED
    "Gemm": TFLiteOperatorMapping(
        BuiltinOperator.FULLY_CONNECTED,
        BuiltinOptions.FullyConnectedOptions,
        lambda b, n, g=None: _map_fully_connected(b, n),
    ),
    # 176. Emit BATCH_MATMUL
    "MatMul": TFLiteOperatorMapping(
        BuiltinOperator.BATCH_MATMUL,
        BuiltinOptions.BatchMatMulOptions,
        lambda b, n, g=None: _map_matmul(b, n, g),
    ),
    # Phase 9: Logical, Reduction
    # 181. Emit EQUAL
    "Equal": TFLiteOperatorMapping(BuiltinOperator.EQUAL, BuiltinOptions.EqualOptions),
    # 182. Emit NOT_EQUAL
    "NotEqual": TFLiteOperatorMapping(BuiltinOperator.NOT_EQUAL, BuiltinOptions.NotEqualOptions),
    # 183. Emit LESS
    "Less": TFLiteOperatorMapping(BuiltinOperator.LESS, BuiltinOptions.LessOptions),
    # 184. Emit LESS_EQUAL
    "LessOrEqual": TFLiteOperatorMapping(
        BuiltinOperator.LESS_EQUAL, BuiltinOptions.LessEqualOptions
    ),
    # 185. Emit GREATER
    "Greater": TFLiteOperatorMapping(BuiltinOperator.GREATER, BuiltinOptions.GreaterOptions),
    # 186. Emit GREATER_EQUAL
    "GreaterOrEqual": TFLiteOperatorMapping(
        BuiltinOperator.GREATER_EQUAL, BuiltinOptions.GreaterEqualOptions
    ),
    # 187. Emit LOGICAL_AND
    "And": TFLiteOperatorMapping(BuiltinOperator.LOGICAL_AND, BuiltinOptions.LogicalAndOptions),
    # 188. Emit LOGICAL_OR
    "Or": TFLiteOperatorMapping(BuiltinOperator.LOGICAL_OR, BuiltinOptions.LogicalOrOptions),
    # 189. Emit LOGICAL_NOT
    "Not": TFLiteOperatorMapping(BuiltinOperator.LOGICAL_NOT, BuiltinOptions.LogicalNotOptions),
    # 190. Emit WHERE
    "Where": TFLiteOperatorMapping(BuiltinOperator.WHERE, BuiltinOptions.WhereOptions),
    # 191. Emit REDUCE_MEAN
    "ReduceMean": TFLiteOperatorMapping(
        BuiltinOperator.MEAN,
        BuiltinOptions.ReducerOptions,
        lambda b, n, g=None: _map_reducer_options(b, n),
    ),
    # 192. Emit REDUCE_MAX
    "ReduceMax": TFLiteOperatorMapping(
        BuiltinOperator.REDUCE_MAX,
        BuiltinOptions.ReducerOptions,
        lambda b, n, g=None: _map_reducer_options(b, n),
    ),
    # 193. Emit REDUCE_MIN
    "ReduceMin": TFLiteOperatorMapping(
        BuiltinOperator.REDUCE_MIN,
        BuiltinOptions.ReducerOptions,
        lambda b, n, g=None: _map_reducer_options(b, n),
    ),
    "ReduceProd": TFLiteOperatorMapping(
        BuiltinOperator.REDUCE_PROD,
        BuiltinOptions.ReducerOptions,
        lambda b, n, g=None: _map_reducer_options(b, n),
    ),
    "ReduceSum": TFLiteOperatorMapping(
        BuiltinOperator.SUM,
        BuiltinOptions.ReducerOptions,
        lambda b, n, g=None: _map_reducer_options(b, n),
    ),
    "ReduceAny": TFLiteOperatorMapping(
        BuiltinOperator.REDUCE_ANY,
        BuiltinOptions.ReducerOptions,
        lambda b, n, g=None: _map_reducer_options(b, n),
    ),
    "ReduceAll": TFLiteOperatorMapping(
        BuiltinOperator.REDUCE_ALL,
        BuiltinOptions.ReducerOptions,
        lambda b, n, g=None: _map_reducer_options(b, n),
    ),
    # Phase 10: Vision & Sorting
    # 201. Emit RESIZE_BILINEAR (Assume Resize ops go to Bilinear by default if mode matches)
    "Resize": TFLiteOperatorMapping(
        BuiltinOperator.RESIZE_BILINEAR,
        BuiltinOptions.ResizeBilinearOptions,
        lambda b, n, g=None: _map_resize(b, n),
    ),
    # 205. Emit SPACE_TO_DEPTH
    "SpaceToDepth": TFLiteOperatorMapping(
        BuiltinOperator.SPACE_TO_DEPTH,
        BuiltinOptions.SpaceToDepthOptions,
        lambda b, n, g=None: _map_space_depth(b, n),
    ),
    # 207. Emit DEPTH_TO_SPACE
    "DepthToSpace": TFLiteOperatorMapping(
        BuiltinOperator.DEPTH_TO_SPACE,
        BuiltinOptions.DepthToSpaceOptions,
        lambda b, n, g=None: _map_space_depth(b, n),
    ),
    "SpaceToBatchND": TFLiteOperatorMapping(
        BuiltinOperator.SPACE_TO_BATCH_ND,
        BuiltinOptions.SpaceToBatchNDOptions,
    ),
    "BatchToSpaceND": TFLiteOperatorMapping(
        BuiltinOperator.BATCH_TO_SPACE_ND,
        BuiltinOptions.BatchToSpaceNDOptions,
    ),
    "ConvTranspose": TFLiteOperatorMapping(
        BuiltinOperator.TRANSPOSE_CONV,
        BuiltinOptions.TransposeConvOptions,
        lambda b, n, g=None: _map_transpose_conv(b, n),
    ),
    # 210. Emit ARG_MAX
    "ArgMax": TFLiteOperatorMapping(
        BuiltinOperator.ARG_MAX, BuiltinOptions.ArgMaxOptions, lambda b, n, g=None: _map_arg(b, n)
    ),
    # 211. Emit ARG_MIN
    "ArgMin": TFLiteOperatorMapping(
        BuiltinOperator.ARG_MIN, BuiltinOptions.ArgMinOptions, lambda b, n, g=None: _map_arg(b, n)
    ),
    # 212. Emit TOPK_V2
    "TopK": TFLiteOperatorMapping(BuiltinOperator.TOPK_V2, BuiltinOptions.TopKV2Options),
    # 214. Emit REVERSE_V2
    "Reverse": TFLiteOperatorMapping(BuiltinOperator.REVERSE_V2, BuiltinOptions.ReverseV2Options),
    # 215. Emit CUMSUM
    "CumSum": TFLiteOperatorMapping(
        BuiltinOperator.CUMSUM, BuiltinOptions.CumsumOptions, lambda b, n, g=None: _map_cumsum(b, n)
    ),
    # 219. Emit SEGMENT_SUM
    "SegmentSum": TFLiteOperatorMapping(
        BuiltinOperator.SEGMENT_SUM, BuiltinOptions.SegmentSumOptions
    ),
    "LshProjection": TFLiteOperatorMapping(
        BuiltinOperator.LSH_PROJECTION, BuiltinOptions.LSHProjectionOptions
    ),
    # 213. Emit UNIQUE
    "Unique": TFLiteOperatorMapping(BuiltinOperator.UNIQUE, BuiltinOptions.UniqueOptions),
    # 218. Map ONNX GridSample to TFLite custom or math equivalents.
    "GridSample": TFLiteOperatorMapping(BuiltinOperator.CUSTOM, BuiltinOptions.NONE),
    # 179. Emit MATRIX_DIAG
    "MatrixDiag": TFLiteOperatorMapping(
        BuiltinOperator.MATRIX_DIAG, BuiltinOptions.MatrixDiagOptions
    ),
    # 180. Emit MATRIX_SET_DIAG
    "MatrixSetDiag": TFLiteOperatorMapping(
        BuiltinOperator.MATRIX_SET_DIAG, BuiltinOptions.MatrixSetDiagOptions
    ),
    # Phase 11: RNN, LSTM, Sequence
    # 221. Emit RNN
    "RNN": TFLiteOperatorMapping(
        BuiltinOperator.RNN, BuiltinOptions.RNNOptions, lambda b, n, g=None: _map_rnn(b, n)
    ),
    # 223. Emit LSTM
    "LSTM": TFLiteOperatorMapping(
        BuiltinOperator.LSTM, BuiltinOptions.LSTMOptions, lambda b, n, g=None: _map_lstm(b, n)
    ),
    # 222. Emit UNIDIRECTIONAL_SEQUENCE_RNN
    "UnidirectionalSequenceRNN": TFLiteOperatorMapping(
        BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_RNN,
        BuiltinOptions.SequenceRNNOptions,
        lambda b, n, g=None: _map_sequence_rnn(b, n),
    ),
    # 224. Emit UNIDIRECTIONAL_SEQUENCE_LSTM
    "UnidirectionalSequenceLSTM": TFLiteOperatorMapping(
        BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
        BuiltinOptions.SequenceRNNOptions,
        lambda b, n, g=None: _map_sequence_rnn(b, n),
    ),
    # 227. Emit BIDIRECTIONAL_SEQUENCE_LSTM
    "BidirectionalSequenceLSTM": TFLiteOperatorMapping(
        BuiltinOperator.BIDIRECTIONAL_SEQUENCE_LSTM,
        BuiltinOptions.SequenceRNNOptions,
        lambda b, n, g=None: _map_sequence_rnn(b, n),
    ),
    # 229. Emit GRU
    "GRU": TFLiteOperatorMapping(
        BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_RNN,
        BuiltinOptions.SequenceRNNOptions,
        lambda b, n, g=None: _map_sequence_rnn(b, n),
    ),
}


def _map_cumsum(b, n):
    """Map ONNX CumSum attributes to TFLite CumsumOptions."""
    exclusive = n.attributes.get("exclusive").value if n.attributes.get("exclusive") else 0
    reverse = n.attributes.get("reverse").value if n.attributes.get("reverse") else 0
    b.start_object(2)
    b.add_field_int8(0, exclusive, 0)
    b.add_field_int8(1, reverse, 0)
    return b.end_object()


def _map_rnn(b, n):
    """Map ONNX RNN attributes to TFLite RNNOptions."""
    b.start_object(2)
    b.add_field_int8(0, 0, 0)
    b.add_field_int8(1, 0, 0)
    return b.end_object()


def _map_lstm(b, n):
    """Map ONNX LSTM attributes to TFLite LSTMOptions."""
    b.start_object(5)
    b.add_field_int8(0, 0, 0)
    b.add_field_float32(1, 0.0, 0.0)
    b.add_field_float32(2, 0.0, 0.0)
    b.add_field_int8(3, 0, 0)
    b.add_field_int8(4, 0, 0)
    return b.end_object()


def _map_sequence_rnn(b, n):
    """Map ONNX Sequence RNN attributes to TFLite SequenceRNNOptions."""
    # 226. Support time_major flags natively.
    time_major = (
        1 if (n.attributes.get("time_major") and n.attributes.get("time_major").value == 1) else 0
    )
    b.start_object(3)
    b.add_field_int8(0, time_major, 0)
    b.add_field_int8(1, 0, 0)
    b.add_field_int8(2, 0, 0)
    return b.end_object()


def _map_matmul(b, n, graph=None):
    """Map ONNX MatMul attributes to TFLite BatchMatMulOptions."""
    adj_x_attr = n.attributes.get("adj_x")
    adj_y_attr = n.attributes.get("adj_y")

    adj_x = adj_x_attr.value if adj_x_attr else 0
    adj_y = adj_y_attr.value if adj_y_attr else 0

    # 177. Configure adj_x and adj_y natively based on ONNX transpose structures.
    # Evaluated structurally in Surgeon/Layout prior to emission.

    b.start_object(3)
    b.add_field_int8(0, adj_x, 0)
    b.add_field_int8(1, adj_y, 0)
    b.add_field_int8(2, 0, 0)
    return b.end_object()


def _map_resize(b, n):
    """Map ONNX Resize attributes to TFLite ResizeBilinear/NearestNeighborOptions."""
    coord_mode = n.attributes.get("coordinate_transformation_mode")
    coord_mode_val = coord_mode.value if coord_mode else ""
    align_corners = 1 if coord_mode_val == "align_corners" else 0
    half_pixel = 1 if coord_mode_val == "half_pixel" else 0

    mode = n.attributes.get("mode")
    mode_val = mode.value if mode else ""

    if mode_val == "nearest":
        b.start_object(2)
        b.add_field_int8(0, align_corners, 0)
        b.add_field_int8(1, half_pixel, 0)
        return b.end_object()

    b.start_object(3)
    b.add_field_int8(0, align_corners, 0)
    b.add_field_int8(1, half_pixel, 0)
    return b.end_object()


def _map_space_depth(b, n):
    """Map ONNX SpaceToDepth/DepthToSpace attributes to TFLite Options."""
    bs = n.attributes.get("blocksize").value if n.attributes.get("blocksize") else 1
    b.start_object(1)
    b.add_field_int32(0, bs, 0)
    return b.end_object()


def _map_arg(b, n):
    """Map ONNX ArgMax/ArgMin attributes to TFLite Options."""
    b.start_object(1)
    b.add_field_int8(0, 3, 0)
    return b.end_object()


def _map_reducer_options(b, n):
    """Map ONNX Reduction attributes to TFLite ReducerOptions."""
    keepdims = n.attributes.get("keepdims").value if n.attributes.get("keepdims") else 1
    b.start_object(2)
    b.add_field_int8(0, keepdims, 0)
    return b.end_object()


def _map_softmax(b, n):
    """Map ONNX Softmax attributes to TFLite SoftmaxOptions."""
    b.start_object(1)
    b.add_field_float32(0, 1.0, 1.0)
    return b.end_object()


def _map_l2norm(b, n):
    """Map ONNX LpNormalization attributes to TFLite L2NormOptions."""
    b.start_object(1)
    b.add_field_int8(0, 0, 0)
    return b.end_object()


def _map_lrn(b, n):
    """Map ONNX LRN attributes to TFLite LocalResponseNormOptions."""
    radius = n.attributes.get("size").value if n.attributes.get("size") else 1
    bias = n.attributes.get("bias").value if n.attributes.get("bias") else 1.0
    alpha = n.attributes.get("alpha").value if n.attributes.get("alpha") else 1.0
    beta = n.attributes.get("beta").value if n.attributes.get("beta") else 0.5
    b.start_object(4)
    b.add_field_int32(0, radius, 0)
    b.add_field_float32(1, bias, 0.0)
    b.add_field_float32(2, alpha, 0.0)
    b.add_field_float32(3, beta, 0.0)
    return b.end_object()


def _map_split(b, n):
    """Map ONNX Split attributes to TFLite SplitOptions."""
    num_splits = len(n.outputs)
    b.start_object(1)
    b.add_field_int32(0, num_splits, 0)
    return b.end_object()


def _map_strided_slice(b, n):
    """Map ONNX Slice/StridedSlice attributes to TFLite StridedSliceOptions."""
    bm_attr = n.attributes.get("begin_mask")
    em_attr = n.attributes.get("end_mask")
    sm_attr = n.attributes.get("shrink_axis_mask")
    el_attr = n.attributes.get("ellipsis_mask")
    na_attr = n.attributes.get("new_axis_mask")

    bm = bm_attr.value if bm_attr else 0
    em = em_attr.value if em_attr else 0
    sm = sm_attr.value if sm_attr else 0
    el = el_attr.value if el_attr else 0
    na = na_attr.value if na_attr else 0

    b.start_object(5)
    b.add_field_int32(0, bm, 0)
    b.add_field_int32(1, em, 0)
    b.add_field_int32(2, el, 0)
    b.add_field_int32(3, na, 0)
    b.add_field_int32(4, sm, 0)
    return b.end_object()


def _map_gather(b, n):
    """Map ONNX Gather attributes to TFLite GatherOptions."""
    axis = n.attributes.get("axis").value if n.attributes.get("axis") else 0
    b.start_object(2)
    b.add_field_int32(0, axis, 0)
    b.add_field_int32(1, 0, 0)
    return b.end_object()


def _map_mirror_pad(b, n):
    """Map ONNX MirrorPad attributes to TFLite MirrorPadOptions."""
    b.start_object(1)
    b.add_field_int8(0, 0, 0)
    return b.end_object()


def _map_pack(b, n):
    """Map ONNX SequenceConstruct to TFLite PackOptions."""
    b.start_object(2)
    b.add_field_int32(0, len(n.inputs), 0)
    b.add_field_int32(1, 0, 0)
    return b.end_object()


def _map_unpack(b, n):
    """Map ONNX SplitToSequence to TFLite UnpackOptions."""
    b.start_object(2)
    b.add_field_int32(0, len(n.outputs), 0)
    b.add_field_int32(1, 0, 0)
    return b.end_object()


def _map_math_fused(b, n):
    """Map fused activation attributes to TFLite math options."""
    act_attr = n.attributes.get("fused_activation")
    act = act_attr.value if act_attr else ""
    act_val = 1 if act == "Relu" else (3 if act == "Relu6" else 0)
    b.start_object(2)
    b.add_field_int8(0, act_val, 0)
    return b.end_object()


def _map_leaky_relu(b, n):
    """Map ONNX LeakyRelu attributes to TFLite LeakyReluOptions."""
    alpha = n.attributes.get("alpha").value if n.attributes.get("alpha") else 0.01
    b.start_object(1)
    b.add_field_float32(0, alpha, 0.0)
    return b.end_object()


def _map_gelu(b, n):
    """Map ONNX Gelu attributes to TFLite GeluOptions."""
    b.start_object(1)
    b.add_field_int8(0, 0, 0)
    return b.end_object()


def _map_reshape(b, n, graph=None):
    """Map ONNX Reshape attributes to TFLite ReshapeOptions, including static shape extraction."""
    new_shape_offset = 0
    if len(n.inputs) > 1 and graph and graph.tensors:
        shape_input = n.inputs[1]
        tensor = graph.tensors.get(shape_input)
        if tensor and tensor.is_initializer and tensor.data:
            import struct

            # It's INT64
            arr = struct.unpack(f"<{len(tensor.data) // 8}q", tensor.data)
            b.start_vector(4, len(arr), 4)
            for val in reversed(arr):
                b.add_int32(int(val))
            new_shape_offset = b.end_vector(len(arr))

    b.start_object(1)
    b.add_field_offset(0, new_shape_offset, 0)
    return b.end_object()


def _map_squeeze(b, n):
    """Map ONNX Squeeze attributes to TFLite SqueezeOptions."""
    b.start_object(1)
    b.add_field_offset(0, 0, 0)
    return b.end_object()


def _map_concat(b, n):
    """Map ONNX Concat attributes to TFLite ConcatenationOptions."""
    axis = n.attributes.get("axis").value if n.attributes.get("axis") else 0
    b.start_object(3)
    b.add_field_int32(0, axis, 0)
    b.add_field_int8(1, 0, 0)
    return b.end_object()


def _map_reducer(b, n):
    """Map ONNX Global Pooling to TFLite ReducerOptions."""
    b.start_object(2)
    b.add_field_int8(0, 1, 0)
    return b.end_object()


# 114. Extract pool filter_height, filter_width.
def map_pool2d_options(builder: any, node: Node, graph=None) -> int:
    """Map ONNX Pool attributes to TFLite Pool2DOptions."""
    strides_attr = node.attributes.get("strides")
    kernel_attr = node.attributes.get("kernel_shape")
    pads_attr = node.attributes.get("pads")
    auto_pad_attr = node.attributes.get("auto_pad")

    strides = strides_attr.value if strides_attr else [1, 1]
    kernel = kernel_attr.value if kernel_attr else [1, 1]
    pads = pads_attr.value if pads_attr else [0, 0, 0, 0]
    auto_pad = auto_pad_attr.value if auto_pad_attr else b"NOTSET"

    stride_h = strides[0] if len(strides) > 0 else 1
    stride_w = strides[1] if len(strides) > 1 else 1
    filter_h = kernel[0] if len(kernel) > 0 else 1
    filter_w = kernel[1] if len(kernel) > 1 else 1

    padding = 1  # VALID by default
    if auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
        padding = 0  # SAME
    elif auto_pad == b"VALID":
        padding = 1  # VALID
    elif sum(pads) > 0:
        # TFLite only supports SAME or VALID natively in options.
        # If explicit pads are symmetric and match SAME, we might map it to SAME.
        # For now, explicit non-zero pads require padding injection (handled outside or best effort mapping).
        # We will set SAME if pads are roughly symmetric and non-zero, though explicit PAD injection is better.
        padding = 0  # SAME (approximation)

    builder.start_object(6)
    builder.add_field_int8(0, padding, 0)  # PADDING
    builder.add_field_int32(1, stride_w, 1)
    builder.add_field_int32(2, stride_h, 1)
    builder.add_field_int32(3, filter_w, 1)
    builder.add_field_int32(4, filter_h, 1)
    builder.add_field_int8(5, 0, 0)  # Activation
    return builder.end_object()


# 101. Emit CONV_2D
def map_conv2d_options(builder: any, node: Node, graph=None) -> int:
    """Map ONNX Conv attributes to TFLite Conv2DOptions."""
    strides_attr = node.attributes.get("strides")
    dilations_attr = node.attributes.get("dilations")
    pads_attr = node.attributes.get("pads")
    auto_pad_attr = node.attributes.get("auto_pad")

    strides = strides_attr.value if strides_attr else [1, 1]
    dilations = dilations_attr.value if dilations_attr else [1, 1]
    pads = pads_attr.value if pads_attr else [0, 0, 0, 0]
    auto_pad = auto_pad_attr.value if auto_pad_attr else b"NOTSET"

    stride_h = strides[0] if len(strides) > 0 else 1
    stride_w = strides[1] if len(strides) > 1 else 1
    dilation_h = dilations[0] if len(dilations) > 0 else 1
    dilation_w = dilations[1] if len(dilations) > 1 else 1

    padding = 1  # VALID by default
    if auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
        padding = 0  # SAME
    elif auto_pad == b"VALID":
        padding = 1  # VALID
    elif sum(pads) > 0:
        padding = 0  # SAME (approximation)

    act_attr = node.attributes.get("fused_activation")
    act = act_attr.value if act_attr else ""
    activation = 0
    if act == "Relu":
        activation = 1
    elif act == "Relu6":
        activation = 3

    builder.start_object(6)
    builder.add_field_int8(0, padding, 0)
    builder.add_field_int32(1, stride_w, 1)
    builder.add_field_int32(2, stride_h, 1)
    builder.add_field_int8(3, activation, 0)
    builder.add_field_int32(4, dilation_w, 1)
    builder.add_field_int32(5, dilation_h, 1)
    return builder.end_object()


# 108. Emit DEPTHWISE_CONV_2D
def map_depthwise_conv2d_options(builder: any, node: Node, graph=None) -> int:
    """Map ONNX Depthwise Conv attributes to TFLite DepthwiseConv2DOptions."""
    strides_attr = node.attributes.get("strides")
    dilations_attr = node.attributes.get("dilations")
    pads_attr = node.attributes.get("pads")
    auto_pad_attr = node.attributes.get("auto_pad")

    strides = strides_attr.value if strides_attr else [1, 1]
    dilations = dilations_attr.value if dilations_attr else [1, 1]
    pads = pads_attr.value if pads_attr else [0, 0, 0, 0]
    auto_pad = auto_pad_attr.value if auto_pad_attr else b"NOTSET"

    stride_h = strides[0] if len(strides) > 0 else 1
    stride_w = strides[1] if len(strides) > 1 else 1
    dilation_h = dilations[0] if len(dilations) > 0 else 1
    dilation_w = dilations[1] if len(dilations) > 1 else 1

    padding = 1  # VALID by default
    if auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
        padding = 0  # SAME
    elif auto_pad == b"VALID":
        padding = 1  # VALID
    elif sum(pads) > 0:
        padding = 0  # SAME (approximation)

    depth_multiplier = 1

    builder.start_object(7)
    builder.add_field_int8(0, padding, 0)
    builder.add_field_int32(1, stride_w, 1)
    builder.add_field_int32(2, stride_h, 1)
    builder.add_field_int32(3, depth_multiplier, 1)
    builder.add_field_int8(4, 0, 0)
    builder.add_field_int32(5, dilation_w, 1)
    builder.add_field_int32(6, dilation_h, 1)
    return builder.end_object()


def map_onnx_node_to_tflite(node: Node) -> Optional[TFLiteOperatorMapping]:
    """Map an ONNX node to its TFLite operator mapping.

    Args:
        node: The ONNX node to map.

    Returns:
        The TFLiteOperatorMapping if successful, else None.

    """
    if node.op_type in ELEMENTWISE_OPS:
        mapping = ELEMENTWISE_OPS[node.op_type]
        if node.op_type == "Resize":
            mode = node.attributes.get("mode")
            mode_val = mode.value if mode else ""
            if mode_val == "nearest":
                return TFLiteOperatorMapping(
                    BuiltinOperator.RESIZE_NEAREST_NEIGHBOR,
                    BuiltinOptions.ResizeNearestNeighborOptions,
                    mapping.create_options,
                )
        return mapping

    if node.op_type in ("NonMaxSuppression", "InstanceNormalization", "LayerNormalization"):
        return TFLiteOperatorMapping(
            BuiltinOperator.CUSTOM, BuiltinOptions.NONE, lambda b, n, g=None: 0
        )

    if node.domain and node.domain != "":
        return TFLiteOperatorMapping(
            BuiltinOperator.CUSTOM, BuiltinOptions.NONE, lambda b, n, g=None: 0
        )

    if node.op_type == "Conv":
        group_attr = node.attributes.get("group")
        group_val = group_attr.value if group_attr else 1
        if group_val > 1:
            return TFLiteOperatorMapping(
                BuiltinOperator.DEPTHWISE_CONV_2D,
                BuiltinOptions.DepthwiseConv2DOptions,
                map_depthwise_conv2d_options,
            )
        return TFLiteOperatorMapping(
            BuiltinOperator.CONV_2D, BuiltinOptions.Conv2DOptions, map_conv2d_options
        )

    return None
