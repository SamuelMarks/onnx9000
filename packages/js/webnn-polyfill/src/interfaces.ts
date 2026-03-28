/// <reference lib="dom" />

export interface Navigator {
  readonly ml?: ML;
}

export interface ML {
  createContext(options?: MLContextOptions): Promise<MLContext>;
}

export interface MLContextOptions {
  deviceType?: 'cpu' | 'gpu' | 'npu';
  powerPreference?: 'default' | 'high-performance' | 'low-power';
}

export interface MLContext {
  compute(
    graph: MLGraph,
    inputs: Record<string, ArrayBufferView>,
    outputs: Record<string, ArrayBufferView>,
  ): Promise<MLComputeResult>;
  opSupportLimits?(): MLOpSupportLimits;
}

export interface MLComputeResult {
  inputs: Record<string, ArrayBufferView>;
  outputs: Record<string, ArrayBufferView>;
}

export interface MLGraph {
  destroy?(): void;
}

export type MLOperandDataType =
  | 'float32'
  | 'float16'
  | 'int32'
  | 'uint32'
  | 'int8'
  | 'uint8'
  | 'int64'
  | 'uint64';

export interface MLOperandDescriptor {
  dataType: MLOperandDataType;
  dimensions: number[];
}

export interface MLOperand {
  dataType: MLOperandDataType;
  shape: number[];
}

export interface MLClampOptions {
  minValue?: MLOperand;
  maxValue?: MLOperand;
}

export interface MLLeakyReluOptions {
  alpha?: number;
}

export interface MLEluOptions {
  alpha?: number;
}

export interface MLHardSigmoidOptions {
  alpha?: number;
  beta?: number;
}

export interface MLGemmOptions {
  c?: MLOperand;
  alpha?: number;
  beta?: number;
  aTranspose?: boolean;
  bTranspose?: boolean;
}

export interface MLTransposeOptions {
  permutation?: number[];
}

export interface MLSliceOptions {
  axes?: number[];
  strides?: number[];
}

export interface MLPadOptions {
  mode?: 'constant' | 'edge' | 'reflection' | 'symmetric';
  value?: number;
}

export interface MLGatherOptions {
  axis?: number;
}

export interface MLSplitOptions {
  axis?: number;
}

export interface MLConv2dOptions {
  padding?: number[];
  strides?: number[];
  dilations?: number[];
  autoPad?: 'explicit' | 'same-upper' | 'same-lower';
  groups?: number;
  inputLayout?: 'nchw' | 'nhwc';
  filterLayout?: 'oihw' | 'hwio' | 'ohwi' | 'ihwo';
  bias?: MLOperand;
}

export interface MLConvTranspose2dOptions extends MLConv2dOptions {
  outputPadding?: number[];
  outputSizes?: number[];
}

export interface MLPool2dOptions {
  windowDimensions?: number[];
  padding?: number[];
  strides?: number[];
  dilations?: number[];
  autoPad?: 'explicit' | 'same-upper' | 'same-lower';
  layout?: 'nchw' | 'nhwc';
  roundingType?: 'floor' | 'ceil';
  outputSizes?: number[];
}

export interface MLReduceOptions {
  axes?: number[];
  keepDimensions?: boolean;
}

export interface MLBatchNormalizationOptions {
  scale?: MLOperand;
  bias?: MLOperand;
  axis?: number;
  epsilon?: number;
}

export interface MLInstanceNormalizationOptions {
  scale?: MLOperand;
  bias?: MLOperand;
  epsilon?: number;
  layout?: 'nchw' | 'nhwc';
}

export interface MLLayerNormalizationOptions {
  scale?: MLOperand;
  bias?: MLOperand;
  axes?: number[];
  epsilon?: number;
}

export interface MLOpSupportLimits {
  input: {
    dataTypes: MLOperandDataType[];
  };
}
