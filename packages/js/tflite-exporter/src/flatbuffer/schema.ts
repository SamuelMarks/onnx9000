/* eslint-disable */
import { FlatBufferBuilder } from './builder';

export enum TensorType {
  FLOAT32 = 0,
  FLOAT16 = 1,
  INT32 = 2,
  UINT8 = 3,
  INT64 = 4,
  STRING = 5,
  BOOL = 6,
  INT16 = 7,
  COMPLEX64 = 8,
  INT8 = 9,
  FLOAT64 = 10,
  COMPLEX128 = 11,
  UINT64 = 12,
  RESOURCE = 13,
  VARIANT = 14,
  UINT32 = 15,
  UINT16 = 16,
  INT4 = 17,
}

export enum Padding {
  SAME = 0,
  VALID = 1,
}

export enum BuiltinOperator {
  ADD = 0,
  AVERAGE_POOL_2D = 1,
  CONCATENATION = 2,
  CONV_2D = 3,
  DEPTHWISE_CONV_2D = 4,
  DEPTH_TO_SPACE = 5,
  DEQUANTIZE = 6,
  EMBEDDING_LOOKUP = 7,
  FLOOR = 8,
  FULLY_CONNECTED = 9,
  HASHTABLE_LOOKUP = 10,
  L2_NORMALIZATION = 11,
  L2_POOL_2D = 12,
  LOCAL_RESPONSE_NORMALIZATION = 13,
  LOGISTIC = 14,
  LSH_PROJECTION = 15,
  LSTM = 16,
  MAX_POOL_2D = 17,
  MUL = 18,
  RELU = 19,
  RELU_N1_TO_1 = 20,
  RELU6 = 21,
  RESHAPE = 22,
  RESIZE_BILINEAR = 23,
  RNN = 24,
  SOFTMAX = 25,
  SPACE_TO_DEPTH = 26,
  SVDF = 27,
  TANH = 28,
  CONCAT_EMBEDDINGS = 29,
  SKIP_GRAM = 30,
  CALL = 31,
  CUSTOM = 32,
  EMBEDDING_LOOKUP_SPARSE = 33,
  PAD = 34,
  UNIDIRECTIONAL_SEQUENCE_RNN = 35,
  GATHER = 36,
  BATCH_TO_SPACE_ND = 37,
  SPACE_TO_BATCH_ND = 38,
  TRANSPOSE = 39,
  MEAN = 40,
  SUB = 41,
  DIV = 42,
  SQUEEZE = 43,
  UNIDIRECTIONAL_SEQUENCE_LSTM = 44,
  STRIDED_SLICE = 45,
  BIDIRECTIONAL_SEQUENCE_RNN = 46,
  EXP = 47,
  TOPK_V2 = 48,
  SPLIT = 49,
  LOG_SOFTMAX = 50,
  DELEGATE = 51,
  BIDIRECTIONAL_SEQUENCE_LSTM = 52,
  CAST = 53,
  PRELU = 54,
  MAXIMUM = 55,
  ARG_MAX = 56,
  MINIMUM = 57,
  LESS = 58,
  NEG = 59,
  PADV2 = 60,
  GREATER = 61,
  GREATER_EQUAL = 62,
  LESS_EQUAL = 63,
  SELECT = 64,
  SLICE = 65,
  SIN = 66,
  TRANSPOSE_CONV = 67,
  SPARSE_TO_DENSE = 68,
  TILE = 69,
  EXPAND_DIMS = 70,
  EQUAL = 71,
  NOT_EQUAL = 72,
  LOG = 73,
  SUM = 74,
  SQRT = 75,
  RSQRT = 76,
  SHAPE = 77,
  POW = 78,
  ARG_MIN = 79,
  FAKE_QUANT = 80,
  REDUCE_PROD = 81,
  REDUCE_MAX = 82,
  PACK = 83,
  LOGICAL_OR = 84,
  ONE_HOT = 85,
  LOGICAL_AND = 86,
  LOGICAL_NOT = 87,
  UNPACK = 88,
  REDUCE_MIN = 89,
  FLOOR_DIV = 90,
  REDUCE_ANY = 91,
  SQUARE = 92,
  ZEROS_LIKE = 93,
  FILL = 94,
  FLOOR_MOD = 95,
  RANGE = 96,
  RESIZE_NEAREST_NEIGHBOR = 97,
  LEAKY_RELU = 98,
  SQUARED_DIFFERENCE = 99,
  MIRROR_PAD = 100,
  ABS = 101,
  SPLIT_V = 102,
  UNIQUE = 103,
  CEIL = 104,
  REVERSE_V2 = 105,
  ADD_N = 106,
  GATHER_ND = 107,
  COS = 108,
  WHERE = 109,
  RANK = 110,
  ELU = 111,
  REVERSE_SEQUENCE = 112,
  MATRIX_DIAG = 113,
  QUANTIZE = 114,
  MATRIX_SET_DIAG = 115,
  ROUND = 116,
  HARD_SWISH = 117,
  IF = 118,
  WHILE = 119,
  NON_MAX_SUPPRESSION_V4 = 120,
  NON_MAX_SUPPRESSION_V5 = 121,
  SCATTER_ND = 122,
  SELECT_V2 = 123,
  DENSIFY = 124,
  SEGMENT_SUM = 125,
  BATCH_MATMUL = 126,
  PLACEHOLDER_FOR_GREATER_OP_CODES = 127,
  CUMSUM = 128,
  CALL_ONCE = 129,
  BROADCAST_TO = 130,
  RFFT2D = 131,
  CONV_3D = 132,
  IMAG = 133,
  REAL = 134,
  COMPLEX_ABS = 135,
  HASHTABLE = 136,
  HASHTABLE_FIND = 137,
  HASHTABLE_IMPORT = 138,
  HASHTABLE_SIZE = 139,
  REDUCE_ALL = 140,
  CONV_3D_TRANSPOSE = 141,
  VAR_HANDLE = 142,
  READ_VARIABLE = 143,
  ASSIGN_VARIABLE = 144,
  BROADCAST_ARGS = 145,
  RANDOM_STANDARD_NORMAL = 146,
  BUCKETIZE = 147,
  RANDOM_UNIFORM = 148,
  MULTINOMIAL = 149,
  GELU = 150,
  DYNAMIC_UPDATE_SLICE = 151,
  RELU_0_TO_1 = 152,
  UNSORTED_SEGMENT_PROD = 153,
  UNSORTED_SEGMENT_MAX = 154,
  UNSORTED_SEGMENT_SUM = 155,
  ATAN2 = 156,
  UNSORTED_SEGMENT_MIN = 157,
  COMPLEX = 158,
  SIGN = 159,
  BITCAST = 160,
  BITWISE_XOR = 161,
  RIGHT_SHIFT = 162,
}

export enum BuiltinOptions {
  NONE = 0,
  Conv2DOptions = 1,
  DepthwiseConv2DOptions = 2,
  ConcatEmbeddingsOptions = 3,
  LSHProjectionOptions = 4,
  Pool2DOptions = 5,
  SVDFOptions = 6,
  RNNOptions = 7,
  FullyConnectedOptions = 8,
  SoftmaxOptions = 9,
  ConcatenationOptions = 10,
  AddOptions = 11,
  L2NormOptions = 12,
  LocalResponseNormOptions = 13,
  LSTMOptions = 14,
  ResizeBilinearOptions = 15,
  CallOptions = 16,
  ReshapeOptions = 17,
  SkipGramOptions = 18,
  SpaceToDepthOptions = 19,
  EmbeddingLookupSparseOptions = 20,
  MulOptions = 21,
  PadOptions = 22,
  GatherOptions = 23,
  BatchToSpaceNDOptions = 24,
  SpaceToBatchNDOptions = 25,
  TransposeOptions = 26,
  ReducerOptions = 27,
  SubOptions = 28,
  DivOptions = 29,
  SqueezeOptions = 30,
  SequenceRNNOptions = 31,
  StridedSliceOptions = 32,
  ExpOptions = 33,
  TopKV2Options = 34,
  SplitOptions = 35,
  LogSoftmaxOptions = 36,
  CastOptions = 37,
  DequantizeOptions = 38,
  MaximumMinimumOptions = 39,
  ArgMaxOptions = 40,
  LessOptions = 41,
  NegOptions = 42,
  PadV2Options = 43,
  GreaterOptions = 44,
  GreaterEqualOptions = 45,
  LessEqualOptions = 46,
  SelectOptions = 47,
  SliceOptions = 48,
  TransposeConvOptions = 49,
  SparseToDenseOptions = 50,
  TileOptions = 51,
  ExpandDimsOptions = 52,
  EqualOptions = 53,
  NotEqualOptions = 54,
  ShapeOptions = 55,
  PowOptions = 56,
  ArgMinOptions = 57,
  FakeQuantOptions = 58,
  PackOptions = 59,
  LogicalOrOptions = 60,
  OneHotOptions = 61,
  LogicalAndOptions = 62,
  LogicalNotOptions = 63,
  UnpackOptions = 64,
  FloorDivOptions = 65,
  SquareOptions = 66,
  ZerosLikeOptions = 67,
  FillOptions = 68,
  FloorModOptions = 69,
  RangeOptions = 70,
  ResizeNearestNeighborOptions = 71,
  LeakyReluOptions = 72,
  SquaredDifferenceOptions = 73,
  MirrorPadOptions = 74,
  AbsOptions = 75,
  SplitVOptions = 76,
  UniqueOptions = 77,
  ReverseV2Options = 78,
  AddNOptions = 79,
  GatherNdOptions = 80,
  CosOptions = 81,
  WhereOptions = 82,
  RankOptions = 83,
  ReverseSequenceOptions = 84,
  MatrixDiagOptions = 85,
  QuantizeOptions = 86,
  MatrixSetDiagOptions = 87,
  HardSwishOptions = 88,
  IfOptions = 89,
  WhileOptions = 90,
  DepthToSpaceOptions = 91,
  NonMaxSuppressionV4Options = 92,
  NonMaxSuppressionV5Options = 93,
  ScatterNdOptions = 94,
  SelectV2Options = 95,
  DensifyOptions = 96,
  SegmentSumOptions = 97,
  BatchMatMulOptions = 98,
  CumsumOptions = 99,
  CallOnceOptions = 100,
  BroadcastToOptions = 101,
  RFFT2DOptions = 102,
  Conv3DOptions = 103,
  HashtableOptions = 104,
  HashtableFindOptions = 105,
  HashtableImportOptions = 106,
  HashtableSizeOptions = 107,
  VarHandleOptions = 108,
  ReadVariableOptions = 109,
  AssignVariableOptions = 110,
  BroadcastArgsOptions = 111,
  RandomOptions = 112,
  BucketizeOptions = 113,
  GeluOptions = 114,
  DynamicUpdateSliceOptions = 115,
}

export class OperatorCode {
  static create(
    builder: FlatBufferBuilder,
    builtinCode: BuiltinOperator,
    customCodeOffset: number,
    version: number,
  ): number {
    builder.startObject(4);
    builder.addFieldInt8(0, builtinCode, 0);
    builder.addFieldOffset(1, customCodeOffset, 0);
    builder.addFieldInt32(2, version, 1);
    builder.addFieldInt32(3, builtinCode, 0); // builtin_code for extended codes
    return builder.endObject();
  }
}

export class QuantizationParameters {
  static create(
    builder: FlatBufferBuilder,
    minOffset: number,
    maxOffset: number,
    scaleOffset: number,
    zeroPointOffset: number,
    detailsType: number,
    detailsOffset: number,
    quantizedDimension: number,
  ): number {
    builder.startObject(7);
    builder.addFieldOffset(0, minOffset, 0);
    builder.addFieldOffset(1, maxOffset, 0);
    builder.addFieldOffset(2, scaleOffset, 0);
    builder.addFieldOffset(3, zeroPointOffset, 0);
    builder.addFieldInt8(4, detailsType, 0);
    builder.addFieldOffset(5, detailsOffset, 0);
    builder.addFieldInt32(6, quantizedDimension, 0);
    return builder.endObject();
  }
}

export class Tensor {
  static create(
    builder: FlatBufferBuilder,
    shapeOffset: number,
    type: TensorType,
    buffer: number,
    nameOffset: number,
    quantizationOffset: number,
    isVariable: boolean,
    sparsityOffset: number,
    shapeSignatureOffset: number,
    hasRank: boolean,
  ): number {
    builder.startObject(10);
    builder.addFieldOffset(0, shapeOffset, 0);
    builder.addFieldInt8(1, type, 0);
    builder.addFieldInt32(2, buffer, 0);
    builder.addFieldOffset(3, nameOffset, 0);
    builder.addFieldOffset(4, quantizationOffset, 0);
    builder.addFieldInt8(5, isVariable ? 1 : 0, 0);
    builder.addFieldOffset(6, sparsityOffset, 0);
    builder.addFieldOffset(7, shapeSignatureOffset, 0);
    builder.addFieldInt8(8, hasRank ? 1 : 0, 0);
    // variant_tensors omitted for simplicity
    return builder.endObject();
  }
}

export class Operator {
  static create(
    builder: FlatBufferBuilder,
    opcodeIndex: number,
    inputsOffset: number,
    outputsOffset: number,
    builtinOptionsType: BuiltinOptions,
    builtinOptionsOffset: number,
    customOptionsOffset: number,
    customOptionsFormat: number,
    mutatingVariableInputs: boolean,
    intermediatesOffset: number,
  ): number {
    builder.startObject(9);
    builder.addFieldInt32(0, opcodeIndex, 0);
    builder.addFieldOffset(1, inputsOffset, 0);
    builder.addFieldOffset(2, outputsOffset, 0);
    builder.addFieldInt8(3, builtinOptionsType, 0);
    builder.addFieldOffset(4, builtinOptionsOffset, 0);
    builder.addFieldOffset(5, customOptionsOffset, 0);
    builder.addFieldInt8(6, customOptionsFormat, 0);
    builder.addFieldInt8(7, mutatingVariableInputs ? 1 : 0, 0);
    builder.addFieldOffset(8, intermediatesOffset, 0);
    return builder.endObject();
  }
}

export class SubGraph {
  static create(
    builder: FlatBufferBuilder,
    tensorsOffset: number,
    inputsOffset: number,
    outputsOffset: number,
    operatorsOffset: number,
    nameOffset: number,
  ): number {
    builder.startObject(5);
    builder.addFieldOffset(0, tensorsOffset, 0);
    builder.addFieldOffset(1, inputsOffset, 0);
    builder.addFieldOffset(2, outputsOffset, 0);
    builder.addFieldOffset(3, operatorsOffset, 0);
    builder.addFieldOffset(4, nameOffset, 0);
    return builder.endObject();
  }
}

export class Buffer {
  static create(builder: FlatBufferBuilder, dataOffset: number): number {
    builder.startObject(1);
    builder.addFieldOffset(0, dataOffset, 0);
    return builder.endObject();
  }
}

export class Metadata {
  static create(builder: FlatBufferBuilder, nameOffset: number, buffer: number): number {
    builder.startObject(2);
    builder.addFieldOffset(0, nameOffset, 0);
    builder.addFieldInt32(1, buffer, 0);
    return builder.endObject();
  }
}

export class Model {
  static create(
    builder: FlatBufferBuilder,
    version: number,
    operatorCodesOffset: number,
    subgraphsOffset: number,
    descriptionOffset: number,
    buffersOffset: number,
    metadataBufferOffset: number,
    metadataOffset: number,
    signatureDefsOffset: number,
  ): number {
    builder.startObject(8);
    builder.addFieldInt32(0, version, 0);
    builder.addFieldOffset(1, operatorCodesOffset, 0);
    builder.addFieldOffset(2, subgraphsOffset, 0);
    builder.addFieldOffset(3, descriptionOffset, 0);
    builder.addFieldOffset(4, buffersOffset, 0);
    builder.addFieldOffset(5, metadataBufferOffset, 0);
    builder.addFieldOffset(6, metadataOffset, 0);
    builder.addFieldOffset(7, signatureDefsOffset, 0);
    return builder.endObject();
  }
}
