/**
 * Common Primitive Registry for ONNX9000.
 */
import { Tensor } from './ir/tensor.js';
import { Node } from './ir/node.js';
import { AttributeValue } from './ops/registry.js';

/**
 * Creates a generic dummy output tensor for ops.
 */
function recordOp(
  opType: string,
  inputs: Tensor[],
  attributes: Record<string, AttributeValue> | undefined = undefined,
): Tensor {
  // Basic shape inference could be added here, but for now we follow the Python pattern
  const dtype = inputs.length > 0 && inputs[0] ? inputs[0].dtype : 'float32';
  return new Tensor(`${opType}_out`, [], dtype, false, false, new Float32Array());
}

/**
 * Abstract base class handling axis reduction and epsilon addition.
 */
export abstract class BaseNorm {
  public epsilon: number;

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
  public numFeatures: number;
  public momentum: number;

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
  public normalizedShape: number[];
  public axis: number;

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
  public normalizedShape: number[];

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
  public numGroups: number;
  public numChannels: number;

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
  public numFeatures: number;

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
  abstract call(x: Tensor): Tensor;

  generateLUT(numPoints: number = 256, rangeMin: number = -8.0, rangeMax: number = 8.0): Tensor {
    return recordOp('Constant', [], {
      lut_range: [rangeMin, rangeMax],
      lut_points: numPoints,
      activation: this.constructor.name,
    });
  }
}

export class Relu extends BaseActivation {
  call(x: Tensor): Tensor {
    return recordOp('Relu', [x]);
  }
}

export class Sigmoid extends BaseActivation {
  call(x: Tensor): Tensor {
    return recordOp('Sigmoid', [x]);
  }
}

export class Tanh extends BaseActivation {
  call(x: Tensor): Tensor {
    return recordOp('Tanh', [x]);
  }
}

export class LeakyRelu extends BaseActivation {
  public alpha: number;
  constructor(alpha: number = 0.01) {
    super();
    this.alpha = alpha;
  }
  call(x: Tensor): Tensor {
    return recordOp('LeakyRelu', [x], { alpha: this.alpha });
  }
}

export class Gelu extends BaseActivation {
  public approximate: string;
  constructor(approximate: string = 'none') {
    super();
    this.approximate = approximate;
  }
  call(x: Tensor): Tensor {
    return recordOp('Gelu', [x], { approximate: this.approximate });
  }
}

export class Silu extends BaseActivation {
  call(x: Tensor): Tensor {
    return recordOp('Swish', [x]);
  }
}

export class Swish extends Silu {}

export class Mish extends BaseActivation {
  call(x: Tensor): Tensor {
    return recordOp('Mish', [x]);
  }
}

export abstract class ConvFamily {
  public inChannels: number;
  public outChannels: number;
  public kernelSize: number | number[];
  public stride: number | number[];
  public padding: number | number[];
  public dilation: number | number[];
  public groups: number;
  public bias: boolean;

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

  abstract call(x: Tensor, w: Tensor, b?: Tensor): Tensor;
}

export class ConvND extends ConvFamily {
  public dims: number;
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

export class DepthwiseConv extends ConvND {
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

export class MatMul {
  call(x: Tensor, y: Tensor): Tensor {
    return recordOp('MatMul', [x, y]);
  }
}

export class Gemm {
  public alpha: number;
  public beta: number;
  public transA: number;
  public transB: number;

  constructor(alpha: number = 1.0, beta: number = 1.0, transA: number = 0, transB: number = 0) {
    this.alpha = alpha;
    this.beta = beta;
    this.transA = transA;
    this.transB = transB;
  }

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

export class MultiHeadAttention {
  public numHeads: number;
  public qkvBias: boolean;
  public outBias: boolean;

  constructor(numHeads: number, qkvBias: boolean = true, outBias: boolean = true) {
    this.numHeads = numHeads;
    this.qkvBias = qkvBias;
    this.outBias = outBias;
  }

  call(q: Tensor, k: Tensor, v: Tensor, mask?: Tensor): Tensor {
    const inputs = mask ? [q, k, v, mask] : [q, k, v];
    return recordOp('Attention', inputs, { num_heads: this.numHeads });
  }
}

export class FlashAttention extends MultiHeadAttention {
  override call(q: Tensor, k: Tensor, v: Tensor, mask?: Tensor): Tensor {
    const inputs = mask ? [q, k, v, mask] : [q, k, v];
    return recordOp('FlashAttention', inputs, { num_heads: this.numHeads });
  }
}

export class GroupedQueryAttention extends MultiHeadAttention {
  public numKvHeads: number;

  constructor(
    numHeads: number,
    numKvHeads: number,
    qkvBias: boolean = false,
    outBias: boolean = false,
  ) {
    super(numHeads, qkvBias, outBias);
    this.numKvHeads = numKvHeads;
  }

  override call(q: Tensor, k: Tensor, v: Tensor, mask?: Tensor): Tensor {
    const inputs = mask ? [q, k, v, mask] : [q, k, v];
    return recordOp('GroupedQueryAttention', inputs, {
      num_heads: this.numHeads,
      num_kv_heads: this.numKvHeads,
    });
  }
}

export class RoPE {
  public dim: number;
  public base: number;
  public maxSeqLen: number;

  constructor(dim: number, base: number = 10000.0, maxSeqLen: number = 2048) {
    this.dim = dim;
    this.base = base;
    this.maxSeqLen = maxSeqLen;
  }

  call(x: Tensor, pos: Tensor): Tensor {
    return recordOp('RoPE', [x, pos], { dim: this.dim, base: this.base });
  }
}

export class AlibiBias {
  public numHeads: number;

  constructor(numHeads: number) {
    this.numHeads = numHeads;
  }

  call(mask: Tensor): Tensor {
    return recordOp('AlibiBias', [mask], { num_heads: this.numHeads });
  }
}
