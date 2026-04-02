/// <reference lib="dom" />

/**
 * Extended Navigator interface to include WebNN ML property.
 */
export interface Navigator {
  /**
   * The ML interface for WebNN.
   */
  readonly ml?: ML;
}

/**
 * Main ML interface for creating contexts.
 */
export interface ML {
  /**
   * Creates an MLContext with the specified options.
   * @param options Configuration options for the context.
   * @returns A promise that resolves to an MLContext.
   */
  createContext(options?: MLContextOptions): Promise<MLContext>;
}

/**
 * Options for creating an MLContext.
 */
export interface MLContextOptions {
  /**
   * The preferred device type.
   */
  deviceType?: 'cpu' | 'gpu' | 'npu';
  /**
   * Power preference for the context.
   */
  powerPreference?: 'default' | 'high-performance' | 'low-power';
}

/**
 * Represents an execution context for ML operations.
 */
export interface MLContext {
  /**
   * Performs a computation on the specified graph with given inputs and outputs.
   * @param graph The MLGraph to execute.
   * @param inputs Map of input names to buffer views.
   * @param outputs Map of output names to buffer views.
   * @returns A promise that resolves to the computation results.
   */
  compute(
    graph: MLGraph,
    inputs: Record<string, ArrayBufferView>,
    outputs: Record<string, ArrayBufferView>,
  ): Promise<MLComputeResult>;
  /**
   * Returns the support limits for operations in this context.
   * @returns Support limits.
   */
  opSupportLimits?(): MLOpSupportLimits;
}

/**
 * Results of an ML computation.
 */
export interface MLComputeResult {
  /**
   * Map of input names to buffer views used in the computation.
   */
  inputs: Record<string, ArrayBufferView>;
  /**
   * Map of output names to buffer views containing the results.
   */
  outputs: Record<string, ArrayBufferView>;
}

/**
 * Represents a compiled ML graph.
 */
export interface MLGraph {
  /**
   * Destroys the graph and releases resources.
   */
  destroy?(): void;
}

/**
 * Valid data types for ML operands.
 */
export type MLOperandDataType =
  | 'float32'
  | 'float16'
  | 'int32'
  | 'uint32'
  | 'int8'
  | 'uint8'
  | 'int64'
  | 'uint64';

/**
 * Describes an ML operand's shape and data type.
 */
export interface MLOperandDescriptor {
  /**
   * The data type of the operand.
   */
  dataType: MLOperandDataType;
  /**
   * The dimensions of the operand.
   */
  dimensions: number[];
}

/**
 * Represents an operand in an ML graph.
 */
export interface MLOperand {
  /**
   * The data type of the operand.
   */
  dataType: MLOperandDataType;
  /**
   * The shape of the operand.
   */
  shape: number[];
}

/**
 * Options for the Clamp operation.
 */
export interface MLClampOptions {
  /**
   * The minimum value for clamping.
   */
  minValue?: MLOperand;
  /**
   * The maximum value for clamping.
   */
  maxValue?: MLOperand;
}

/**
 * Options for the LeakyRelu operation.
 */
export interface MLLeakyReluOptions {
  /**
   * The alpha value (slope for negative values).
   */
  alpha?: number;
}

/**
 * Options for the Elu operation.
 */
export interface MLEluOptions {
  /**
   * The alpha value.
   */
  alpha?: number;
}

/**
 * Options for the HardSigmoid operation.
 */
export interface MLHardSigmoidOptions {
  /**
   * The alpha value.
   */
  alpha?: number;
  /**
   * The beta value.
   */
  beta?: number;
}

/**
 * Options for the General Matrix Multiplication (Gemm) operation.
 */
export interface MLGemmOptions {
  /**
   * Optional third matrix to add.
   */
  c?: MLOperand;
  /**
   * Scalar multiplier for A * B.
   */
  alpha?: number;
  /**
   * Scalar multiplier for C.
   */
  beta?: number;
  /**
   * Whether to transpose matrix A.
   */
  aTranspose?: boolean;
  /**
   * Whether to transpose matrix B.
   */
  bTranspose?: boolean;
}

/**
 * Options for the Transpose operation.
 */
export interface MLTransposeOptions {
  /**
   * Permutation of the dimensions.
   */
  permutation?: number[];
}

/**
 * Options for the Slice operation.
 */
export interface MLSliceOptions {
  /**
   * Axes to slice along.
   */
  axes?: number[];
  /**
   * Strides for slicing.
   */
  strides?: number[];
}

/**
 * Options for the Pad operation.
 */
export interface MLPadOptions {
  /**
   * Padding mode.
   */
  mode?: 'constant' | 'edge' | 'reflection' | 'symmetric';
  /**
   * Constant value to use for padding.
   */
  value?: number;
}

/**
 * Options for the Gather operation.
 */
export interface MLGatherOptions {
  /**
   * Axis to gather along.
   */
  axis?: number;
}

/**
 * Options for the Split operation.
 */
export interface MLSplitOptions {
  /**
   * Axis to split along.
   */
  axis?: number;
}

/**
 * Options for the Conv2d operation.
 */
export interface MLConv2dOptions {
  /**
   * Padding for each dimension.
   */
  padding?: number[];
  /**
   * Strides for each dimension.
   */
  strides?: number[];
  /**
   * Dilations for each dimension.
   */
  dilations?: number[];
  /**
   * Automatic padding mode.
   */
  autoPad?: 'explicit' | 'same-upper' | 'same-lower';
  /**
   * Number of groups for grouped convolution.
   */
  groups?: number;
  /**
   * Layout of the input data.
   */
  inputLayout?: 'nchw' | 'nhwc';
  /**
   * Layout of the filter data.
   */
  filterLayout?: 'oihw' | 'hwio' | 'ohwi' | 'ihwo';
  /**
   * Optional bias operand.
   */
  bias?: MLOperand;
}

/**
 * Options for the ConvTranspose2d operation.
 */
export interface MLConvTranspose2dOptions extends MLConv2dOptions {
  /**
   * Additional padding added to the output.
   */
  outputPadding?: number[];
  /**
   * Explicit output sizes.
   */
  outputSizes?: number[];
}

/**
 * Options for pooling operations.
 */
export interface MLPool2dOptions {
  /**
   * Dimensions of the pooling window.
   */
  windowDimensions?: number[];
  /**
   * Padding for each dimension.
   */
  padding?: number[];
  /**
   * Strides for each dimension.
   */
  strides?: number[];
  /**
   * Dilations for each dimension.
   */
  dilations?: number[];
  /**
   * Automatic padding mode.
   */
  autoPad?: 'explicit' | 'same-upper' | 'same-lower';
  /**
   * Layout of the data.
   */
  layout?: 'nchw' | 'nhwc';
  /**
   * Rounding type for output dimensions.
   */
  roundingType?: 'floor' | 'ceil';
  /**
   * Explicit output sizes.
   */
  outputSizes?: number[];
}

/**
 * Options for reduction operations.
 */
export interface MLReduceOptions {
  /**
   * Axes to reduce along.
   */
  axes?: number[];
  /**
   * Whether to keep the reduced dimensions.
   */
  keepDimensions?: boolean;
}

/**
 * Options for the BatchNormalization operation.
 */
export interface MLBatchNormalizationOptions {
  /**
   * Scale operand.
   */
  scale?: MLOperand;
  /**
   * Bias operand.
   */
  bias?: MLOperand;
  /**
   * Axis to normalize along.
   */
  axis?: number;
  /**
   * Epsilon value for numerical stability.
   */
  epsilon?: number;
}

/**
 * Options for the InstanceNormalization operation.
 */
export interface MLInstanceNormalizationOptions {
  /**
   * Scale operand.
   */
  scale?: MLOperand;
  /**
   * Bias operand.
   */
  bias?: MLOperand;
  /**
   * Epsilon value for numerical stability.
   */
  epsilon?: number;
  /**
   * Layout of the data.
   */
  layout?: 'nchw' | 'nhwc';
}

/**
 * Options for the LayerNormalization operation.
 */
export interface MLLayerNormalizationOptions {
  /**
   * Scale operand.
   */
  scale?: MLOperand;
  /**
   * Bias operand.
   */
  bias?: MLOperand;
  /**
   * Axes to normalize along.
   */
  axes?: number[];
  /**
   * Epsilon value for numerical stability.
   */
  epsilon?: number;
}

/**
 * Describes support limits for operations.
 */
export interface MLOpSupportLimits {
  /**
   * Input support limits.
   */
  input: {
    /**
     * Supported data types.
     */
    dataTypes: MLOperandDataType[];
  };
}

/**
 * Profile data entry for flamegraphs.
 */
export interface MLProfileEntry {
  /**
   * Name of the operation.
   */
  name: string;
  /**
   * Time taken in milliseconds.
   */
  time: number;
}

/**
 * Information about a node in the graph for diagnostic purposes.
 */
export interface MLGraphNodeInfo {
  /**
   * Operation type (e.g., 'Add').
   */
  opType: string;
  /**
   * Name of the node.
   */
  name: string;
  /**
   * Input operand names.
   */
  inputs: string[];
  /**
   * Output operand names.
   */
  outputs: string[];
}

/**
 * Information about the whole graph for diagnostic purposes.
 */
export interface MLGraphInfo {
  /**
   * List of nodes in the graph.
   */
  nodes: MLGraphNodeInfo[];
}
