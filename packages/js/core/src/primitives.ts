/**
 * @fileoverview primitives module for core.
 */
/**
 * Common Primitive Registry for ONNX9000.
 */

import { Tensor } from './ir/tensor.js';
import type { AttributeValue } from './ops/registry.js';

/**
 * Creates a generic dummy output tensor for ops.
 */
function recordOp(
  opType: string,
  inputs: Tensor[],
  _attributes?: Record<string, AttributeValue>,
): Tensor {
  // Basic shape inference could be added here, but for now we follow the Python pattern
  const dtype = inputs.length > 0 && inputs[0] ? inputs[0].dtype : 'float32';
  return new Tensor(`${opType}_out`, [], dtype, false, false, new Float32Array());
}

/**
 * Abstract base class handling axis reduction and epsilon addition.
 */
export abstract class BaseNorm {
  /** Property `epsilon` of `BaseNorm`. */
  public epsilon: number;

  /**
   * Initializes a new `BaseNorm` instance.
   * @param epsilon Parameter `epsilon`.
   */
  constructor(epsilon: number = 1e-5) {
    this.epsilon = epsilon;
  }

  /**
   * Applies normalization.
   */
  abstract call(...args: ReturnType<typeof JSON.parse>[]): Tensor;
}

/**
 * Inherits from BaseNorm. Implements running mean/var tracking.
 */
export class BatchNormalization extends BaseNorm {
  /** Property `numFeatures` of `BatchNormalization`. */
  public numFeatures: number;
  /** Property `momentum` of `BatchNormalization`. */
  public momentum: number;

  /**
   * Initializes a new `BatchNormalization` instance.
   * @param numFeatures Parameter `numFeatures`.
   * @param epsilon Parameter `epsilon`.
   * @param momentum Parameter `momentum`.
   */
  constructor(numFeatures: number, epsilon: number = 1e-5, momentum: number = 0.1) {
    super(epsilon);
    this.numFeatures = numFeatures;
    this.momentum = momentum;
  }

  /**
   * Applies Batch Normalization.
   */
  call(x: Tensor, scale: Tensor, b: Tensor, inputMean: Tensor, inputVar: Tensor): Tensor {
    return recordOp('BatchNormalization', [x, scale, b, inputMean, inputVar], {
      epsilon: this.epsilon,
      momentum: this.momentum,
    });
  }
}

/**
 * Inherits from BaseNorm. Parametrized by normalized_shape.
 */
export class LayerNormalization extends BaseNorm {
  /** Property `normalizedShape` of `LayerNormalization`. */
  public normalizedShape: number[];
  /** Property `axis` of `LayerNormalization`. */
  public axis: number;

  /**
   * Initializes a new `LayerNormalization` instance.
   * @param normalizedShape Parameter `normalizedShape`.
   * @param epsilon Parameter `epsilon`.
   */
  constructor(normalizedShape: number[], epsilon: number = 1e-5) {
    super(epsilon);
    this.normalizedShape = normalizedShape;
    this.axis = -normalizedShape.length;
  }

  /**
   * Applies Layer Normalization.
   */
  call(x: Tensor, scale: Tensor, b?: Tensor): Tensor {
    const inputs = b ? [x, scale, b] : [x, scale];
    return recordOp('LayerNormalization', inputs, {
      axis: this.axis,
      epsilon: this.epsilon,
    });
  }
}

/**
 * Inherits from BaseNorm. Standardizes LLaMA/Gemma variance-only scaling.
 */
export class RMSNorm extends BaseNorm {
  /** Property `normalizedShape` of `RMSNorm`. */
  public normalizedShape: number[];

  /**
   * Initializes a new `RMSNorm` instance.
   * @param normalizedShape Parameter `normalizedShape`.
   * @param epsilon Parameter `epsilon`.
   */
  constructor(normalizedShape: number[], epsilon: number = 1e-5) {
    super(epsilon);
    this.normalizedShape = normalizedShape;
  }

  /**
   * Applies RMS Normalization.
   */
  call(x: Tensor, scale: Tensor): Tensor {
    return recordOp('RMSNormalization', [x, scale], {});
  }
}

/**
 * Maps via generalized Reshape -> LayerNorm -> Reshape subgraphs.
 */
export class GroupNorm extends BaseNorm {
  /** Property `numGroups` of `GroupNorm`. */
  public numGroups: number;
  /** Property `numChannels` of `GroupNorm`. */
  public numChannels: number;

  /**
   * Initializes a new `GroupNorm` instance.
   * @param numGroups Parameter `numGroups`.
   * @param numChannels Parameter `numChannels`.
   * @param epsilon Parameter `epsilon`.
   */
  constructor(numGroups: number, numChannels: number, epsilon: number = 1e-5) {
    super(epsilon);
    this.numGroups = numGroups;
    this.numChannels = numChannels;
  }

  /**
   * Applies Group Normalization.
   */
  call(x: Tensor, scale: Tensor, b: Tensor): Tensor {
    return recordOp('GroupNormalization', [x, scale, b], {
      epsilon: this.epsilon,
      num_groups: this.numGroups,
    });
  }
}

/**
 * Instance Normalization.
 */
export class InstanceNorm extends BaseNorm {
  /** Property `numFeatures` of `InstanceNorm`. */
  public numFeatures: number;

  /**
   * Initializes a new `InstanceNorm` instance.
   * @param numFeatures Parameter `numFeatures`.
   * @param epsilon Parameter `epsilon`.
   */
  constructor(numFeatures: number, epsilon: number = 1e-5) {
    super(epsilon);
    this.numFeatures = numFeatures;
  }

  /**
   * Applies Instance Normalization.
   */
  call(x: Tensor, scale: Tensor, b: Tensor): Tensor {
    return recordOp('InstanceNormalization', [x, scale, b], {
      epsilon: this.epsilon,
    });
  }
}

/**
 * Abstract base class for element-wise non-linearities.
 */
export abstract class BaseActivation {
  /**
   * Executes `call` on `BaseActivation`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  abstract call(x: Tensor): Tensor;

  /**
   * Executes `generateLUT` on `BaseActivation`.
   * @param numPoints Parameter `numPoints`.
   * @param rangeMin Parameter `rangeMin`.
   * @param rangeMax Parameter `rangeMax`.
   * @returns Result of `generateLUT`.
   */
  generateLUT(numPoints: number = 256, rangeMin: number = -8.0, rangeMax: number = 8.0): Tensor {
    return recordOp('Constant', [], {
      lut_range: [rangeMin, rangeMax],
      lut_points: numPoints,
      activation: this.constructor.name,
    });
  }
}

/** Represents the `Relu` class. */
export class Relu extends BaseActivation {
  /**
   * Executes `call` on `Relu`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    return recordOp('Relu', [x]);
  }
}

/** Represents the `Sigmoid` class. */
export class Sigmoid extends BaseActivation {
  /**
   * Executes `call` on `Sigmoid`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    return recordOp('Sigmoid', [x]);
  }
}

/** Represents the `Tanh` class. */
export class Tanh extends BaseActivation {
  /**
   * Executes `call` on `Tanh`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    return recordOp('Tanh', [x]);
  }
}

/** Represents the `LeakyRelu` class. */
export class LeakyRelu extends BaseActivation {
  /** Property `alpha` of `LeakyRelu`. */
  public alpha: number;
  /**
   * Initializes a new `LeakyRelu` instance.
   * @param alpha Parameter `alpha`.
   */
  constructor(alpha: number = 0.01) {
    super();
    this.alpha = alpha;
  }
  /**
   * Executes `call` on `LeakyRelu`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    return recordOp('LeakyRelu', [x], { alpha: this.alpha });
  }
}

/** Represents the `Gelu` class. */
export class Gelu extends BaseActivation {
  /** Property `approximate` of `Gelu`. */
  public approximate: string;
  /**
   * Initializes a new `Gelu` instance.
   * @param approximate Parameter `approximate`.
   */
  constructor(approximate: string = 'none') {
    super();
    this.approximate = approximate;
  }
  /**
   * Executes `call` on `Gelu`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    return recordOp('Gelu', [x], { approximate: this.approximate });
  }
}

/** Represents the `Silu` class. */
export class Silu extends BaseActivation {
  /**
   * Executes `call` on `Silu`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    return recordOp('Swish', [x]);
  }
}

/** Represents the `Swish` class. */
export class Swish extends Silu {}

/** Represents the `Mish` class. */
export class Mish extends BaseActivation {
  /**
   * Executes `call` on `Mish`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    return recordOp('Mish', [x]);
  }
}

/** Represents the `ConvFamily` class. */
export abstract class ConvFamily {
  /** Property `inChannels` of `ConvFamily`. */
  public inChannels: number;
  /** Property `outChannels` of `ConvFamily`. */
  public outChannels: number;
  /** Property `kernelSize` of `ConvFamily`. */
  public kernelSize: number | number[];
  /** Property `stride` of `ConvFamily`. */
  public stride: number | number[];
  /** Property `padding` of `ConvFamily`. */
  public padding: number | number[];
  /** Property `dilation` of `ConvFamily`. */
  public dilation: number | number[];
  /** Property `groups` of `ConvFamily`. */
  public groups: number;
  /** Property `bias` of `ConvFamily`. */
  public bias: boolean;

  /**
   * Initializes a new `ConvFamily` instance.
   * @param inChannels Parameter `inChannels`.
   * @param outChannels Parameter `outChannels`.
   * @param kernelSize Parameter `kernelSize`.
   * @param stride Parameter `stride`.
   * @param padding Parameter `padding`.
   * @param dilation Parameter `dilation`.
   * @param groups Parameter `groups`.
   * @param bias Parameter `bias`.
   */
  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | number[],
    stride: number | number[] = 1,
    padding: number | number[] = 0,
    dilation: number | number[] = 1,
    groups: number = 1,
    bias: boolean = true,
  ) {
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.stride = stride;
    this.padding = padding;
    this.dilation = dilation;
    this.groups = groups;
    this.bias = bias;
  }

  /**
   * Executes `call` on `ConvFamily`.
   * @param x Parameter `x`.
   * @param w Parameter `w`.
   * @param b Parameter `b`.
   * @returns Result of `call`.
   */
  abstract call(x: Tensor, w: Tensor, b?: Tensor): Tensor;
}

/** Represents the `ConvND` class. */
export class ConvND extends ConvFamily {
  /** Property `dims` of `ConvND`. */
  public dims: number;
  /**
   * Initializes a new `ConvND` instance.
   * @param dims Parameter `dims`.
   * @param inChannels Parameter `inChannels`.
   * @param outChannels Parameter `outChannels`.
   * @param kernelSize Parameter `kernelSize`.
   * @param stride Parameter `stride`.
   * @param padding Parameter `padding`.
   * @param dilation Parameter `dilation`.
   * @param groups Parameter `groups`.
   * @param bias Parameter `bias`.
   */
  constructor(
    dims: number,
    inChannels: number,
    outChannels: number,
    kernelSize: number | number[],
    stride: number | number[] = 1,
    padding: number | number[] = 0,
    dilation: number | number[] = 1,
    groups: number = 1,
    bias: boolean = true,
  ) {
    super(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, bias);
    this.dims = dims;
  }

  /**
   * Executes `call` on `ConvND`.
   * @param x Parameter `x`.
   * @param w Parameter `w`.
   * @param b Parameter `b`.
   * @returns Result of `call`.
   */
  override call(x: Tensor, w: Tensor, b?: Tensor): Tensor {
    const ks = Array.isArray(this.kernelSize)
      ? this.kernelSize
      : Array(this.dims).fill(this.kernelSize);
    const st = Array.isArray(this.stride) ? this.stride : Array(this.dims).fill(this.stride);
    const pa = Array.isArray(this.padding) ? this.padding : Array(this.dims * 2).fill(this.padding);
    const di = Array.isArray(this.dilation) ? this.dilation : Array(this.dims).fill(this.dilation);

    const inputs = b ? [x, w, b] : [x, w];
    return recordOp('Conv', inputs, {
      kernel_shape: ks,
      strides: st,
      pads: pa,
      dilations: di,
      group: this.groups,
    });
  }
}

/** Represents the `DepthwiseConv` class. */
export class DepthwiseConv extends ConvND {
  /**
   * Initializes a new `DepthwiseConv` instance.
   * @param dims Parameter `dims`.
   * @param channels Parameter `channels`.
   * @param kernelSize Parameter `kernelSize`.
   * @param stride Parameter `stride`.
   * @param padding Parameter `padding`.
   * @param dilation Parameter `dilation`.
   * @param bias Parameter `bias`.
   */
  constructor(
    dims: number,
    channels: number,
    kernelSize: number | number[],
    stride: number | number[] = 1,
    padding: number | number[] = 0,
    dilation: number | number[] = 1,
    bias: boolean = true,
  ) {
    super(dims, channels, channels, kernelSize, stride, padding, dilation, channels, bias);
  }
}

/** Represents the `MatMul` class. */
export class MatMul {
  /**
   * Executes `call` on `MatMul`.
   * @param x Parameter `x`.
   * @param y Parameter `y`.
   * @returns Result of `call`.
   */
  call(x: Tensor, y: Tensor): Tensor {
    return recordOp('MatMul', [x, y]);
  }
}

/** Represents the `Gemm` class. */
export class Gemm {
  /** Property `alpha` of `Gemm`. */
  public alpha: number;
  /** Property `beta` of `Gemm`. */
  public beta: number;
  /** Property `transA` of `Gemm`. */
  public transA: number;
  /** Property `transB` of `Gemm`. */
  public transB: number;

  /**
   * Initializes a new `Gemm` instance.
   * @param alpha Parameter `alpha`.
   * @param beta Parameter `beta`.
   * @param transA Parameter `transA`.
   * @param transB Parameter `transB`.
   */
  constructor(alpha: number = 1.0, beta: number = 1.0, transA: number = 0, transB: number = 0) {
    this.alpha = alpha;
    this.beta = beta;
    this.transA = transA;
    this.transB = transB;
  }

  /**
   * Executes `call` on `Gemm`.
   * @param x Parameter `x`.
   * @param y Parameter `y`.
   * @param c Parameter `c`.
   * @returns Result of `call`.
   */
  call(x: Tensor, y: Tensor, c?: Tensor): Tensor {
    const inputs = c ? [x, y, c] : [x, y];
    return recordOp('Gemm', inputs, {
      alpha: this.alpha,
      beta: this.beta,
      transA: this.transA,
      transB: this.transB,
    });
  }
}

/** Represents the `MultiHeadAttention` class. */
export class MultiHeadAttention {
  /** Property `numHeads` of `MultiHeadAttention`. */
  public numHeads: number;
  /** Property `qkvBias` of `MultiHeadAttention`. */
  public qkvBias: boolean;
  /** Property `outBias` of `MultiHeadAttention`. */
  public outBias: boolean;

  /**
   * Initializes a new `MultiHeadAttention` instance.
   * @param numHeads Parameter `numHeads`.
   * @param qkvBias Parameter `qkvBias`.
   * @param outBias Parameter `outBias`.
   */
  constructor(numHeads: number, qkvBias: boolean = true, outBias: boolean = true) {
    this.numHeads = numHeads;
    this.qkvBias = qkvBias;
    this.outBias = outBias;
  }

  /**
   * Executes `call` on `MultiHeadAttention`.
   * @param q Parameter `q`.
   * @param k Parameter `k`.
   * @param v Parameter `v`.
   * @param mask Parameter `mask`.
   * @returns Result of `call`.
   */
  call(q: Tensor, k: Tensor, v: Tensor, mask?: Tensor): Tensor {
    const inputs = mask ? [q, k, v, mask] : [q, k, v];
    return recordOp('Attention', inputs, { num_heads: this.numHeads });
  }
}

/** Represents the `FlashAttention` class. */
export class FlashAttention extends MultiHeadAttention {
  /**
   * Executes `call` on `FlashAttention`.
   * @param q Parameter `q`.
   * @param k Parameter `k`.
   * @param v Parameter `v`.
   * @param mask Parameter `mask`.
   * @returns Result of `call`.
   */
  override call(q: Tensor, k: Tensor, v: Tensor, mask?: Tensor): Tensor {
    const inputs = mask ? [q, k, v, mask] : [q, k, v];
    return recordOp('FlashAttention', inputs, { num_heads: this.numHeads });
  }
}

/** Represents the `GroupedQueryAttention` class. */
export class GroupedQueryAttention extends MultiHeadAttention {
  /** Property `numKvHeads` of `GroupedQueryAttention`. */
  public numKvHeads: number;

  /**
   * Initializes a new `GroupedQueryAttention` instance.
   * @param numHeads Parameter `numHeads`.
   * @param numKvHeads Parameter `numKvHeads`.
   * @param qkvBias Parameter `qkvBias`.
   * @param outBias Parameter `outBias`.
   */
  constructor(
    numHeads: number,
    numKvHeads: number,
    qkvBias: boolean = false,
    outBias: boolean = false,
  ) {
    super(numHeads, qkvBias, outBias);
    this.numKvHeads = numKvHeads;
  }

  /**
   * Executes `call` on `GroupedQueryAttention`.
   * @param q Parameter `q`.
   * @param k Parameter `k`.
   * @param v Parameter `v`.
   * @param mask Parameter `mask`.
   * @returns Result of `call`.
   */
  override call(q: Tensor, k: Tensor, v: Tensor, mask?: Tensor): Tensor {
    const inputs = mask ? [q, k, v, mask] : [q, k, v];
    return recordOp('GroupedQueryAttention', inputs, {
      num_heads: this.numHeads,
      num_kv_heads: this.numKvHeads,
    });
  }
}

/** Represents the `RoPE` class. */
export class RoPE {
  /** Property `dim` of `RoPE`. */
  public dim: number;
  /** Property `base` of `RoPE`. */
  public base: number;
  /** Property `maxSeqLen` of `RoPE`. */
  public maxSeqLen: number;

  /**
   * Initializes a new `RoPE` instance.
   * @param dim Parameter `dim`.
   * @param base Parameter `base`.
   * @param maxSeqLen Parameter `maxSeqLen`.
   */
  constructor(dim: number, base: number = 10000.0, maxSeqLen: number = 2048) {
    this.dim = dim;
    this.base = base;
    this.maxSeqLen = maxSeqLen;
  }

  /**
   * Executes `call` on `RoPE`.
   * @param x Parameter `x`.
   * @param pos Parameter `pos`.
   * @returns Result of `call`.
   */
  call(x: Tensor, pos: Tensor): Tensor {
    return recordOp('RoPE', [x, pos], { dim: this.dim, base: this.base });
  }
}

/** Represents the `AlibiBias` class. */
export class AlibiBias {
  /** Property `numHeads` of `AlibiBias`. */
  public numHeads: number;

  /**
   * Initializes a new `AlibiBias` instance.
   * @param numHeads Parameter `numHeads`.
   */
  constructor(numHeads: number) {
    this.numHeads = numHeads;
  }

  /**
   * Executes `call` on `AlibiBias`.
   * @param mask Parameter `mask`.
   * @returns Result of `call`.
   */
  call(mask: Tensor): Tensor {
    return recordOp('AlibiBias', [mask], { num_heads: this.numHeads });
  }
}
