/// <reference path="./webnn.d.ts" />

/**
 * Options for building a fused Conv2D + BN + ReLU sequence in WebNN.
 */
export interface Conv2DBNReluOptions {
  /** Padding for convolution. */
  padding?: number[];
  /** Strides for convolution. */
  strides?: number[];
  /** Dilations for convolution. */
  dilations?: number[];
  /** Number of groups. */
  groups?: number;
  /** Epsilon for BatchNormalization. */
  epsilon?: number;
}

/**
 * Options for building a Separable Conv2D sequence in WebNN.
 */
export interface SeparableConv2DOptions {
  /** Padding for depthwise convolution. */
  padding?: number[];
  /** Strides for depthwise convolution. */
  strides?: number[];
  /** Dilations for depthwise convolution. */
  dilations?: number[];
  /** Number of input channels. */
  inChannels: number;
}

/**
 * Helper class for building Keras-specific fused WebNN graphs.
 */
export class KerasWebNNCompiler {
  private builder: MLGraphBuilder;
  private context: MLContext;

  /**
   * Initializes a new instance of the KerasWebNNCompiler.
   * @param builder The WebNN graph builder.
   * @param context The WebNN context.
   */
  constructor(builder: MLGraphBuilder, context: MLContext) {
    this.builder = builder;
    this.context = context;
  }

  /**
   * Builds a fused sequence of Conv2D, BatchNormalization, and ReLU.
   * @param input Input operand.
   * @param weights Convolution kernel weights.
   * @param bias Convolution bias.
   * @param bnGamma BatchNormalization scale (gamma).
   * @param bnBeta BatchNormalization bias (beta).
   * @param bnMean BatchNormalization moving mean.
   * @param bnVar BatchNormalization moving variance.
   * @param options Configuration for the nodes.
   * @returns The output operand of the ReLU.
   */
  public buildConv2DBNRelu(
    input: MLOperand,
    weights: MLOperand,
    bias: MLOperand | undefined,
    bnGamma: MLOperand,
    bnBeta: MLOperand,
    bnMean: MLOperand,
    bnVar: MLOperand,
    options: Conv2DBNReluOptions,
  ): MLOperand {
    const convOptions: MLConv2dOptions = {
      groups: options.groups || 1,
    };
    if (options.padding) convOptions.padding = options.padding;
    if (options.strides) convOptions.strides = options.strides;
    if (options.dilations) convOptions.dilations = options.dilations;
    if (bias) convOptions.bias = bias;

    const convOut = this.builder.conv2d(input, weights, convOptions);

    const bnOptions: MLBatchNormalizationOptions = {
      scale: bnGamma,
      bias: bnBeta,
      epsilon: options.epsilon || 1e-5,
    };

    const bnOut = this.builder.batchNormalization(convOut, bnMean, bnVar, bnOptions);

    return this.builder.relu(bnOut);
  }

  /**
   * Builds a separable convolution using two Conv2D operations.
   * @param input Input operand.
   * @param depthwiseWeights Weights for depthwise convolution.
   * @param pointwiseWeights Weights for pointwise (1x1) convolution.
   * @param bias Final bias to apply after pointwise.
   * @param options Configuration for the nodes.
   * @returns The output operand.
   */
  public buildSeparableConv2D(
    input: MLOperand,
    depthwiseWeights: MLOperand,
    pointwiseWeights: MLOperand,
    bias: MLOperand | undefined,
    options: SeparableConv2DOptions,
  ): MLOperand {
    const dwOptions: MLConv2dOptions = {
      groups: options.inChannels,
    };
    if (options.padding) dwOptions.padding = options.padding;
    if (options.strides) dwOptions.strides = options.strides;
    if (options.dilations) dwOptions.dilations = options.dilations;

    const depthOut = this.builder.conv2d(input, depthwiseWeights, dwOptions);

    const pwOptions: MLConv2dOptions = {};
    if (bias) pwOptions.bias = bias;

    return this.builder.conv2d(depthOut, pointwiseWeights, pwOptions);
  }

  /**
   * Compiles and executes a WebNN graph asynchronously.
   * @param outputs Map of output operands.
   * @param inputs Map of input buffers.
   * @returns Map of output buffers.
   */
  public async executeAsync(
    outputs: Record<string, MLOperand>,
    inputs: Record<string, ArrayBufferView>,
  ): Promise<Record<string, ArrayBufferView>> {
    const compiledGraph = await this.builder.build(outputs);
    const result = await this.context.compute(compiledGraph, inputs, {});
    return result.outputs;
  }
}
