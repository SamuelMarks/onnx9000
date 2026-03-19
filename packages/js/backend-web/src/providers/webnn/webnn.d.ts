/// <reference lib="dom" />

declare interface Navigator {
  readonly ml?: ML;
}

declare interface ML {
  createContext(options?: MLContextOptions): Promise<MLContext>;
}

declare interface MLContextOptions {
  deviceType?: 'cpu' | 'gpu' | 'npu';
  powerPreference?: 'default' | 'high-performance' | 'low-power';
}

declare interface MLContext {
  compute(
    graph: MLGraph,
    inputs: Record<string, ArrayBufferView>,
    outputs: Record<string, ArrayBufferView>,
  ): Promise<MLComputeResult>;
  opSupportLimits?(): MLOpSupportLimits;
}

declare interface MLComputeResult {
  inputs: Record<string, ArrayBufferView>;
  outputs: Record<string, ArrayBufferView>;
}

declare interface MLGraph {
  destroy?(): void;
}

declare type MLOperandDataType =
  | 'float32'
  | 'float16'
  | 'int32'
  | 'uint32'
  | 'int8'
  | 'uint8'
  | 'int64'
  | 'uint64';

declare interface MLOperandDescriptor {
  dataType: MLOperandDataType;
  dimensions: number[];
}

declare interface MLOperand {
  dataType: MLOperandDataType;
  shape: number[];
}

declare interface MLClampOptions {
  minValue?: MLOperand;
  maxValue?: MLOperand;
}

declare interface MLLeakyReluOptions {
  alpha?: number;
}

declare interface MLEluOptions {
  alpha?: number;
}

declare interface MLHardSigmoidOptions {
  alpha?: number;
  beta?: number;
}

declare interface MLGemmOptions {
  c?: MLOperand;
  alpha?: number;
  beta?: number;
  aTranspose?: boolean;
  bTranspose?: boolean;
}

declare interface MLTransposeOptions {
  permutation?: number[];
}

declare interface MLSliceOptions {
  axes?: number[];
  strides?: number[];
}

declare interface MLPadOptions {
  mode?: 'constant' | 'edge' | 'reflection' | 'symmetric';
  value?: number;
}

declare interface MLGatherOptions {
  axis?: number;
}

declare interface MLSplitOptions {
  axis?: number;
}

declare interface MLConv2dOptions {
  padding?: number[];
  strides?: number[];
  dilations?: number[];
  autoPad?: 'explicit' | 'same-upper' | 'same-lower';
  groups?: number;
  inputLayout?: 'nchw' | 'nhwc';
  filterLayout?: 'oihw' | 'hwio' | 'ohwi' | 'ihwo';
  bias?: MLOperand;
}

declare interface MLConvTranspose2dOptions extends MLConv2dOptions {
  outputPadding?: number[];
  outputSizes?: number[];
}

declare interface MLPool2dOptions {
  windowDimensions?: number[];
  padding?: number[];
  strides?: number[];
  dilations?: number[];
  autoPad?: 'explicit' | 'same-upper' | 'same-lower';
  layout?: 'nchw' | 'nhwc';
  roundingType?: 'floor' | 'ceil';
  outputSizes?: number[];
}

declare interface MLReduceOptions {
  axes?: number[];
  keepDimensions?: boolean;
}

declare interface MLBatchNormalizationOptions {
  scale?: MLOperand;
  bias?: MLOperand;
  axis?: number;
  epsilon?: number;
}

declare interface MLInstanceNormalizationOptions {
  scale?: MLOperand;
  bias?: MLOperand;
  epsilon?: number;
  layout?: 'nchw' | 'nhwc';
}

declare interface MLLayerNormalizationOptions {
  scale?: MLOperand;
  bias?: MLOperand;
  axes?: number[];
  epsilon?: number;
}

declare class MLGraphBuilder {
  constructor(context: MLContext);
  input(name: string, descriptor: MLOperandDescriptor): MLOperand;
  constant(descriptor: MLOperandDescriptor, bufferView: ArrayBufferView): MLOperand;
  build(outputs: Record<string, MLOperand>): Promise<MLGraph>;

  // Binary Arithmetic
  add(a: MLOperand, b: MLOperand): MLOperand;
  sub(a: MLOperand, b: MLOperand): MLOperand;
  mul(a: MLOperand, b: MLOperand): MLOperand;
  div(a: MLOperand, b: MLOperand): MLOperand;
  max(a: MLOperand, b: MLOperand): MLOperand;
  min(a: MLOperand, b: MLOperand): MLOperand;
  pow(a: MLOperand, b: MLOperand): MLOperand;

  // Unary Arithmetic
  abs(a: MLOperand): MLOperand;
  ceil(a: MLOperand): MLOperand;
  floor(a: MLOperand): MLOperand;
  exp(a: MLOperand): MLOperand;
  log(a: MLOperand): MLOperand;
  cos(a: MLOperand): MLOperand;
  sin(a: MLOperand): MLOperand;
  tan(a: MLOperand): MLOperand;
  acos(a: MLOperand): MLOperand;
  asin(a: MLOperand): MLOperand;
  atan(a: MLOperand): MLOperand;
  sqrt(a: MLOperand): MLOperand;
  erf(a: MLOperand): MLOperand;
  sign(a: MLOperand): MLOperand;
  neg(a: MLOperand): MLOperand;

  // Activation
  relu(a: MLOperand): MLOperand;
  sigmoid(a: MLOperand): MLOperand;
  tanh(a: MLOperand): MLOperand;
  softmax(a: MLOperand, axis?: number): MLOperand;
  leakyRelu(a: MLOperand, options?: MLLeakyReluOptions): MLOperand;
  elu(a: MLOperand, options?: MLEluOptions): MLOperand;
  hardSigmoid(a: MLOperand, options?: MLHardSigmoidOptions): MLOperand;
  softplus(a: MLOperand): MLOperand;
  softsign(a: MLOperand): MLOperand;
  gelu(a: MLOperand): MLOperand;
  prelu(a: MLOperand, slope: MLOperand): MLOperand;
  clamp(a: MLOperand, options?: MLClampOptions): MLOperand;

  // Matrix Math
  matmul(a: MLOperand, b: MLOperand): MLOperand;
  gemm(a: MLOperand, b: MLOperand, options?: MLGemmOptions): MLOperand;

  // Tensor Manipulation
  reshape(a: MLOperand, newShape: number[]): MLOperand;
  transpose(a: MLOperand, options?: MLTransposeOptions): MLOperand;
  slice(a: MLOperand, starts: number[], sizes: number[], options?: MLSliceOptions): MLOperand;
  concat(inputs: MLOperand[], axis: number): MLOperand;
  split(a: MLOperand, splits: number | number[], options?: MLSplitOptions): MLOperand[];
  expand(a: MLOperand, newShape: number[]): MLOperand;
  gather(a: MLOperand, indices: MLOperand, options?: MLGatherOptions): MLOperand;
  pad(a: MLOperand, padding: number[], options?: MLPadOptions): MLOperand;
  cast(a: MLOperand, type: MLOperandDataType): MLOperand;

  // Convolution & Pooling
  conv2d(input: MLOperand, filter: MLOperand, options?: MLConv2dOptions): MLOperand;
  convTranspose2d(
    input: MLOperand,
    filter: MLOperand,
    options?: MLConvTranspose2dOptions,
  ): MLOperand;
  maxPool2d(input: MLOperand, options?: MLPool2dOptions): MLOperand;
  averagePool2d(input: MLOperand, options?: MLPool2dOptions): MLOperand;
  l2Pool2d(input: MLOperand, options?: MLPool2dOptions): MLOperand;

  // Reduction
  reduceMean(input: MLOperand, options?: MLReduceOptions): MLOperand;
  reduceSum(input: MLOperand, options?: MLReduceOptions): MLOperand;
  reduceMax(input: MLOperand, options?: MLReduceOptions): MLOperand;
  reduceMin(input: MLOperand, options?: MLReduceOptions): MLOperand;
  reduceProduct(input: MLOperand, options?: MLReduceOptions): MLOperand;
  reduceL1(input: MLOperand, options?: MLReduceOptions): MLOperand;
  reduceL2(input: MLOperand, options?: MLReduceOptions): MLOperand;
  reduceLogSumExp(input: MLOperand, options?: MLReduceOptions): MLOperand;
  argMax(input: MLOperand, options?: MLReduceOptions): MLOperand; // Might not be widely supported yet
  argMin(input: MLOperand, options?: MLReduceOptions): MLOperand;

  // Normalization
  batchNormalization(
    input: MLOperand,
    mean: MLOperand,
    variance: MLOperand,
    options?: MLBatchNormalizationOptions,
  ): MLOperand;
  instanceNormalization(input: MLOperand, options?: MLInstanceNormalizationOptions): MLOperand;
  layerNormalization(input: MLOperand, options?: MLLayerNormalizationOptions): MLOperand;
  l2Normalization(input: MLOperand, options?: MLReduceOptions): MLOperand;

  // Transformer & NLP Drafts (193, 282)
  triangular?(input: MLOperand, options?: { upper?: boolean; diagonal?: number }): MLOperand;
  scaledDotProductAttention?(
    query: MLOperand,
    key: MLOperand,
    value: MLOperand,
    options?: any,
  ): MLOperand;

  // Quantization (201, 202)
  quantizeLinear?(input: MLOperand, scale: MLOperand, zeroPoint: MLOperand): MLOperand;
  dequantizeLinear?(input: MLOperand, scale: MLOperand, zeroPoint: MLOperand): MLOperand;
  bitwiseAnd?(a: MLOperand, b: MLOperand): MLOperand;
  shiftRightLogical?(a: MLOperand, b: MLOperand): MLOperand;

  // Logical & Relational
  equal(a: MLOperand, b: MLOperand): MLOperand;
  greater(a: MLOperand, b: MLOperand): MLOperand;
  greaterOrEqual(a: MLOperand, b: MLOperand): MLOperand;
  lesser(a: MLOperand, b: MLOperand): MLOperand; // NOTE: WebNN usually calls this 'lesser'
  lesserOrEqual(a: MLOperand, b: MLOperand): MLOperand;
  logicalNot(a: MLOperand): MLOperand;
  logicalAnd(a: MLOperand, b: MLOperand): MLOperand;
  logicalOr(a: MLOperand, b: MLOperand): MLOperand;
  logicalXor(a: MLOperand, b: MLOperand): MLOperand;
  where(condition: MLOperand, trueValue: MLOperand, falseValue: MLOperand): MLOperand;
}

declare interface MLOpSupportLimits {
  input: {
    dataTypes: MLOperandDataType[];
  };
}
