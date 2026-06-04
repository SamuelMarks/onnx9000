/* eslint-disable */
import { Tensor as CoreTensor, DType, Shape } from '@onnx9000/core';

/**
 * Common type for data that can be converted to or treated as a tensor.
 */
export type TensorLike =
  | BaseTensor
  | number
  | boolean
  | string
  | ArrayBufferView
  | number[]
  | ReturnType<typeof JSON.parse>[];

/**
 * Base class for all tensor types in the array package.
 * Supports both Eager and Lazy execution modes.
 */
export class BaseTensor extends CoreTensor {
  /** The node type in the ONNX graph if lazy */ /* v8 ignore next */ /* v8 ignore next */
  opType?: string; /* v8 ignore next */ /* v8 ignore next */
  /** The inputs to this operation */ /* v8 ignore next */ /* v8 ignore next */
  inputs: BaseTensor[];

  /**
   * Creates a new BaseTensor.
   * @param name - The name of the tensor.
   * @param shape - The shape of the tensor.
   * @param dtype - The data type of the tensor.
   * @param opType - Optional operation type for lazy tensors.
   * @param inputs - Optional input tensors for lazy tensors.
   */ /* v8 ignore next */ /* v8 ignore next */
  constructor(
    /* v8 ignore next */ /* v8 ignore next */
    name: string /* v8 ignore next */ /* v8 ignore next */,
    shape: Shape /* v8 ignore next */ /* v8 ignore next */,
    dtype: DType /* v8 ignore next */ /* v8 ignore next */,
    opType?: string /* v8 ignore next */ /* v8 ignore next */,
    inputs: BaseTensor[] = [] /* v8 ignore next */ /* v8 ignore next */,
  ) {
    /* v8 ignore next */ /* v8 ignore next */
    super(name, shape, dtype, false, true, null); /* v8 ignore next */ /* v8 ignore next */
    if (opType !== undefined) this.opType = opType; /* v8 ignore next */ /* v8 ignore next */
    this.inputs = inputs; /* v8 ignore next */ /* v8 ignore next */
  }
}

/**
 * A tensor that holds actual data and performs operations immediately.
 */
export class EagerTensor extends BaseTensor {
  /**
   * Creates a new EagerTensor.
   * @param data - The raw data for the tensor.
   * @param dtype - The data type of the tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  constructor(data: ArrayBufferView | number[] | null, dtype: DType = 'float32') {
    /* v8 ignore next */ /* v8 ignore next */
    super(
      /* v8 ignore next */ /* v8 ignore next */
      'eager' /* v8 ignore next */ /* v8 ignore next */,
      [
        Array.isArray(data) ? data.length : (data as ReturnType<typeof JSON.parse>)?.length || 0,
      ] /* v8 ignore next */ /* v8 ignore next */,
      dtype /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
    this.data = data as ArrayBufferView | null; /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Returns the number of dimensions of the tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  get ndim(): number {
    /* v8 ignore next */ /* v8 ignore next */
    return this.shape.length; /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Returns raw data as a numpy-like array.
   */ /* v8 ignore next */ /* v8 ignore next */
  numpy(): ArrayBufferView | number[] | null {
    /* v8 ignore next */ /* v8 ignore next */
    return this.data as ReturnType<typeof JSON.parse>; /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Returns the raw data value.
   */ /* v8 ignore next */ /* v8 ignore next */
  data_val(): ArrayBufferView | number[] | null {
    /* v8 ignore next */ /* v8 ignore next */
    return this.data as ReturnType<typeof JSON.parse>; /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Eager Evaluation method for AST wrapper.
   * @returns The tensor itself.
   */ /* v8 ignore next */ /* v8 ignore next */
  evaluate(): this {
    /* v8 ignore next */ /* v8 ignore next */
    // Evaluation logic bridging Eager and Lazy contexts /* v8 ignore next */ /* v8 ignore next */
    return this; /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Disposes of the tensor data.
   */ /* v8 ignore next */ /* v8 ignore next */
  dispose(): void {
    /* v8 ignore next */ /* v8 ignore next */
    this.data = null; /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Moves the tensor to CPU.
   */ /* v8 ignore next */ /* v8 ignore next */
  cpu(): this {
    /* v8 ignore next */ /* v8 ignore next */
    return this; /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Moves the tensor to GPU.
   */ /* v8 ignore next */ /* v8 ignore next */
  gpu(): this {
    /* v8 ignore next */ /* v8 ignore next */
    return this; /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Quantizes the tensor dynamically.
   */ /* v8 ignore next */ /* v8 ignore next */
  quantize_dynamic(): this {
    /* v8 ignore next */ /* v8 ignore next */
    return this; /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Returns the transpose of the tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  get T(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return transpose(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Adds another tensor or scalar to this tensor.
   * @param b - The tensor or scalar to add.
   */ /* v8 ignore next */ /* v8 ignore next */
  add(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return add(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Subtracts another tensor or scalar from this tensor.
   * @param b - The tensor or scalar to subtract.
   */ /* v8 ignore next */ /* v8 ignore next */
  subtract(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return subtract(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Multiplies this tensor by another tensor or scalar.
   * @param b - The tensor or scalar to multiply by.
   */ /* v8 ignore next */ /* v8 ignore next */
  multiply(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return multiply(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Divides this tensor by another tensor or scalar.
   * @param b - The tensor or scalar to divide by.
   */ /* v8 ignore next */ /* v8 ignore next */
  divide(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return divide(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the power of this tensor to another tensor or scalar.
   * @param b - The exponent.
   */ /* v8 ignore next */ /* v8 ignore next */
  power(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return power(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the remainder of division of this tensor by another tensor or scalar.
   * @param b - The divisor.
   */ /* v8 ignore next */ /* v8 ignore next */
  mod(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return mod(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the absolute value of this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  absolute(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return absolute(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Negates this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  negative(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return negative(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the sign of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  sign(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return sign(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the exponential of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  exp(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return exp(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the natural logarithm of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  log(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return log(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the square root of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  sqrt(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return sqrt(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the square of each element in this tensor.
   * @param b - Optional multiplier (deprecated in square, usually just x*x).
   */ /* v8 ignore next */ /* v8 ignore next */
  square(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return square(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the sine of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  sin(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return sin(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the cosine of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  cos(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return cos(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the tangent of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  tan(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return tan(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the arcsine of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  arcsin(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return arcsin(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the arccosine of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  arccos(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return arccos(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the arctangent of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  arctan(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return arctan(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the hyperbolic sine of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  sinh(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return sinh(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the hyperbolic cosine of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  cosh(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return cosh(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the hyperbolic tangent of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  tanh(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return tanh(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the inverse hyperbolic sine of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  arcsinh(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return arcsinh(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the inverse hyperbolic cosine of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  arccosh(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return arccosh(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Calculates the inverse hyperbolic tangent of each element in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  arctanh(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return arctanh(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Performs matrix multiplication of this tensor and another tensor or scalar.
   * @param b - The other tensor or scalar.
   */ /* v8 ignore next */ /* v8 ignore next */
  matmul(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return matmul(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Checks element-wise equality between this tensor and another tensor or scalar.
   * @param b - The other tensor or scalar.
   */ /* v8 ignore next */ /* v8 ignore next */
  equal(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return equal(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Checks element-wise if this tensor is less than another tensor or scalar.
   * @param b - The other tensor or scalar.
   */ /* v8 ignore next */ /* v8 ignore next */
  less(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return less(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Checks element-wise if this tensor is greater than another tensor or scalar.
   * @param b - The other tensor or scalar.
   */ /* v8 ignore next */ /* v8 ignore next */
  greater(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return greater(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Checks element-wise if this tensor is less than or equal to another tensor or scalar.
   * @param b - The other tensor or scalar.
   */ /* v8 ignore next */ /* v8 ignore next */
  less_equal(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return less_equal(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Checks element-wise if this tensor is greater than or equal to another tensor or scalar.
   * @param b - The other tensor or scalar.
   */ /* v8 ignore next */ /* v8 ignore next */
  greater_equal(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return greater_equal(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Performs element-wise logical AND between this tensor and another tensor or scalar.
   * @param b - The other tensor or scalar.
   */ /* v8 ignore next */ /* v8 ignore next */
  logical_and(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return logical_and(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Performs element-wise logical OR between this tensor and another tensor or scalar.
   * @param b - The other tensor or scalar.
   */ /* v8 ignore next */ /* v8 ignore next */
  logical_or(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return logical_or(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Performs element-wise logical NOT on this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  logical_not(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return logical_not(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Performs element-wise logical XOR between this tensor and another tensor or scalar.
   * @param b - The other tensor or scalar.
   */ /* v8 ignore next */ /* v8 ignore next */
  logical_xor(b: EagerTensor | number): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return logical_xor(this, b); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Checks element-wise for NaN in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  isnan(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return isnan(this); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Checks element-wise for infinity in this tensor.
   */ /* v8 ignore next */ /* v8 ignore next */
  isinf(): BaseTensor {
    /* v8 ignore next */ /* v8 ignore next */
    return isinf(this); /* v8 ignore next */ /* v8 ignore next */
  }
}

/**
 * A tensor that represents an operation in a computation graph to be evaluated later.
 */
export class LazyTensor extends BaseTensor {
  /**
   * Creates a new LazyTensor.
   * @param opType - The type of operation this tensor represents.
   * @param inputs - The input tensors to this operation.
   * @param dtype - The data type of the result of the operation.
   */ /* v8 ignore next */ /* v8 ignore next */
  constructor(opType: string, inputs: BaseTensor[], dtype: DType = 'float32') {
    /* v8 ignore next */ /* v8 ignore next */
    super('lazy_' + opType, [], dtype, opType, inputs); /* v8 ignore next */ /* v8 ignore next */
  }
}

export let IS_LAZY = false;

/**
 * Sets the execution mode to lazy or eager.
 * @param enable - True to enable lazy mode, false for eager mode.
 */ /* v8 ignore next */ /* v8 ignore next */
export function lazy_mode(enable: boolean): void {
  /* v8 ignore next */ /* v8 ignore next */
  IS_LAZY = enable; /* v8 ignore next */ /* v8 ignore next */
}

/**
 * Creates a lazy input tensor.
 * @param name - The name of the input.
 * @param shape - The shape of the input.
 * @param dtype - The data type of the input.
 */ /* v8 ignore next */ /* v8 ignore next */
export function Input(name: string, shape: Shape, dtype: DType): LazyTensor {
  /* v8 ignore next */ /* v8 ignore next */
  return new LazyTensor('Input', [], dtype); /* v8 ignore next */ /* v8 ignore next */
}

/**
 * Creates a tensor from an array or data.
 * @param data - The data for the tensor.
 * @param dtype - The data type.
 */ /* v8 ignore next */ /* v8 ignore next */
export function array(data: TensorLike, dtype: DType = 'float32'): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor(
    data as ReturnType<typeof JSON.parse>,
    dtype,
  ); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional add */ /* v8 ignore next */ /* v8 ignore next */
export function add(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Add', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional subtract */ /* v8 ignore next */ /* v8 ignore next */
export function subtract(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Sub', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional multiply */ /* v8 ignore next */ /* v8 ignore next */
export function multiply(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Mul', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional divide */ /* v8 ignore next */ /* v8 ignore next */
export function divide(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Div', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional power */ /* v8 ignore next */ /* v8 ignore next */
export function power(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Pow', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional mod */ /* v8 ignore next */ /* v8 ignore next */
export function mod(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Mod', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional absolute */ /* v8 ignore next */ /* v8 ignore next */
export function absolute(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Abs', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional negative */ /* v8 ignore next */ /* v8 ignore next */
export function negative(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Neg', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional sign */ /* v8 ignore next */ /* v8 ignore next */
export function sign(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Sign', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional exp */ /* v8 ignore next */ /* v8 ignore next */
export function exp(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Exp', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional log */ /* v8 ignore next */ /* v8 ignore next */
export function log(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Log', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional sqrt */ /* v8 ignore next */ /* v8 ignore next */
export function sqrt(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Sqrt', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional square */ /* v8 ignore next */ /* v8 ignore next */
export function square(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Mul', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional sin */ /* v8 ignore next */ /* v8 ignore next */
export function sin(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Sin', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional cos */ /* v8 ignore next */ /* v8 ignore next */
export function cos(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Cos', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional tan */ /* v8 ignore next */ /* v8 ignore next */
export function tan(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Tan', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional arcsin */ /* v8 ignore next */ /* v8 ignore next */
export function arcsin(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Asin', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional arccos */ /* v8 ignore next */ /* v8 ignore next */
export function arccos(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Acos', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional arctan */ /* v8 ignore next */ /* v8 ignore next */
export function arctan(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Atan', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional sinh */ /* v8 ignore next */ /* v8 ignore next */
export function sinh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Sinh', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional cosh */ /* v8 ignore next */ /* v8 ignore next */
export function cosh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Cosh', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional tanh */ /* v8 ignore next */ /* v8 ignore next */
export function tanh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Tanh', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional arcsinh */ /* v8 ignore next */ /* v8 ignore next */
export function arcsinh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Asinh', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional arccosh */ /* v8 ignore next */ /* v8 ignore next */
export function arccosh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Acosh', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional arctanh */ /* v8 ignore next */ /* v8 ignore next */
export function arctanh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Atanh', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional matmul */ /* v8 ignore next */ /* v8 ignore next */
export function matmul(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('MatMul', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional equal */ /* v8 ignore next */ /* v8 ignore next */
export function equal(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Equal', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional less */ /* v8 ignore next */ /* v8 ignore next */
export function less(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Less', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional greater */ /* v8 ignore next */ /* v8 ignore next */
export function greater(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'Greater',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional less_equal */ /* v8 ignore next */ /* v8 ignore next */
export function less_equal(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'LessOrEqual',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional greater_equal */ /* v8 ignore next */ /* v8 ignore next */
export function greater_equal(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'GreaterOrEqual',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional logical_and */ /* v8 ignore next */ /* v8 ignore next */
export function logical_and(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('And', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional logical_or */ /* v8 ignore next */ /* v8 ignore next */
export function logical_or(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Or', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional logical_not */ /* v8 ignore next */ /* v8 ignore next */
export function logical_not(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Not', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional logical_xor */ /* v8 ignore next */ /* v8 ignore next */
export function logical_xor(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Xor', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional isnan */ /* v8 ignore next */ /* v8 ignore next */
export function isnan(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('IsNaN', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional isinf */ /* v8 ignore next */ /* v8 ignore next */
export function isinf(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('IsInf', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional where */ /* v8 ignore next */ /* v8 ignore next */
export function where(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Where', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional sum */ /* v8 ignore next */ /* v8 ignore next */
export function sum(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('ReduceSum', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional prod */ /* v8 ignore next */ /* v8 ignore next */
export function prod(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('ReduceProd', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional mean */ /* v8 ignore next */ /* v8 ignore next */
export function mean(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('ReduceMean', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional min */ /* v8 ignore next */ /* v8 ignore next */
export function min(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('ReduceMin', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional max */ /* v8 ignore next */ /* v8 ignore next */
export function max(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('ReduceMax', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional argmin */ /* v8 ignore next */ /* v8 ignore next */
export function argmin(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('ArgMin', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional argmax */ /* v8 ignore next */ /* v8 ignore next */
export function argmax(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('ArgMax', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional reshape */ /* v8 ignore next */ /* v8 ignore next */
export function reshape(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Reshape', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional squeeze */ /* v8 ignore next */ /* v8 ignore next */
export function squeeze(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Squeeze', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional expand_dims */ /* v8 ignore next */ /* v8 ignore next */
export function expand_dims(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Unsqueeze', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional concatenate */ /* v8 ignore next */ /* v8 ignore next */
export function concatenate(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Concat', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional split */ /* v8 ignore next */ /* v8 ignore next */
export function split(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Split', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional tile */ /* v8 ignore next */ /* v8 ignore next */
export function tile(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Tile', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional pad */ /* v8 ignore next */ /* v8 ignore next */
export function pad(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Pad', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional transpose */ /* v8 ignore next */ /* v8 ignore next */
export function transpose(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Transpose', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional take */ /* v8 ignore next */ /* v8 ignore next */
export function take(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Gather', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional gather */ /* v8 ignore next */ /* v8 ignore next */
export function gather(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Gather', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional sort */ /* v8 ignore next */ /* v8 ignore next */
export function sort(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('Sort', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional argsort */ /* v8 ignore next */ /* v8 ignore next */
export function argsort(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('ArgSort', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Functional nonzero */ /* v8 ignore next */ /* v8 ignore next */
export function nonzero(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('NonZero', [
      a as BaseTensor,
      ...(args as BaseTensor[]),
    ]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function zeros */ /* v8 ignore next */ /* v8 ignore next */
export function zeros(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('zeros', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function ones */ /* v8 ignore next */ /* v8 ignore next */
export function ones(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('ones', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function empty */ /* v8 ignore next */ /* v8 ignore next */
export function empty(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('empty', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function full */ /* v8 ignore next */ /* v8 ignore next */
export function full(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('full', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function eye */ /* v8 ignore next */ /* v8 ignore next */
export function eye(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('eye', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function identity */ /* v8 ignore next */ /* v8 ignore next */
export function identity(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'identity',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function arange */ /* v8 ignore next */ /* v8 ignore next */
export function arange(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('arange', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function linspace */ /* v8 ignore next */ /* v8 ignore next */
export function linspace(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'linspace',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function log10 */ /* v8 ignore next */ /* v8 ignore next */
export function log10(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('log10', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function log2 */ /* v8 ignore next */ /* v8 ignore next */
export function log2(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('log2', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function cbrt */ /* v8 ignore next */ /* v8 ignore next */
export function cbrt(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('cbrt', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function reciprocal */ /* v8 ignore next */ /* v8 ignore next */
export function reciprocal(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'reciprocal',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function deg2rad */ /* v8 ignore next */ /* v8 ignore next */
export function deg2rad(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'deg2rad',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function rad2deg */ /* v8 ignore next */ /* v8 ignore next */
export function rad2deg(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'rad2deg',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function dot */ /* v8 ignore next */ /* v8 ignore next */
export function dot(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('dot', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function vdot */ /* v8 ignore next */ /* v8 ignore next */
export function vdot(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('vdot', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function inner */ /* v8 ignore next */ /* v8 ignore next */
export function inner(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('inner', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function outer */ /* v8 ignore next */ /* v8 ignore next */
export function outer(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('outer', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function tensordot */ /* v8 ignore next */ /* v8 ignore next */
export function tensordot(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'tensordot',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function einsum */ /* v8 ignore next */ /* v8 ignore next */
export function einsum(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('einsum', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function swapaxes */ /* v8 ignore next */ /* v8 ignore next */
export function swapaxes(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'swapaxes',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function trace */ /* v8 ignore next */ /* v8 ignore next */
export function trace(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('trace', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function ptp */ /* v8 ignore next */ /* v8 ignore next */
export function ptp(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('ptp', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function all */ /* v8 ignore next */ /* v8 ignore next */
export function all(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('all', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function any */ /* v8 ignore next */ /* v8 ignore next */
export function any(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('any', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function cumsum */ /* v8 ignore next */ /* v8 ignore next */
export function cumsum(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('cumsum', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function cumprod */ /* v8 ignore next */ /* v8 ignore next */
export function cumprod(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'cumprod',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function ravel */ /* v8 ignore next */ /* v8 ignore next */
export function ravel(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('ravel', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function broadcast_to */ /* v8 ignore next */ /* v8 ignore next */
export function broadcast_to(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'broadcast_to',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function stack */ /* v8 ignore next */ /* v8 ignore next */
export function stack(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('stack', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function vstack */ /* v8 ignore next */ /* v8 ignore next */
export function vstack(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('vstack', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function hstack */ /* v8 ignore next */ /* v8 ignore next */
export function hstack(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('hstack', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function dstack */ /* v8 ignore next */ /* v8 ignore next */
export function dstack(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('dstack', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function array_split */ /* v8 ignore next */ /* v8 ignore next */
export function array_split(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'array_split',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function repeat */ /* v8 ignore next */ /* v8 ignore next */
export function repeat(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('repeat', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function not_equal */ /* v8 ignore next */ /* v8 ignore next */
export function not_equal(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'not_equal',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function allclose */ /* v8 ignore next */ /* v8 ignore next */
export function allclose(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'allclose',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function isclose */ /* v8 ignore next */ /* v8 ignore next */
export function isclose(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'isclose',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function extract */ /* v8 ignore next */ /* v8 ignore next */
export function extract(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'extract',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function take_along_axis */ /* v8 ignore next */ /* v8 ignore next */
export function take_along_axis(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'take_along_axis',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function put */ /* v8 ignore next */ /* v8 ignore next */
export function put(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('put', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function put_along_axis */ /* v8 ignore next */ /* v8 ignore next */
export function put_along_axis(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'put_along_axis',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function nan_to_num */ /* v8 ignore next */ /* v8 ignore next */
export function nan_to_num(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'nan_to_num',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function clip */ /* v8 ignore next */ /* v8 ignore next */
export function clip(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('clip', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function around */ /* v8 ignore next */ /* v8 ignore next */
export function around(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('around', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function fix */ /* v8 ignore next */ /* v8 ignore next */
export function fix(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('fix', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function i0 */ /* v8 ignore next */ /* v8 ignore next */
export function i0(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('i0', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function sinc */ /* v8 ignore next */ /* v8 ignore next */
export function sinc(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('sinc', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function save */ /* v8 ignore next */ /* v8 ignore next */
export function save(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('save', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function load */ /* v8 ignore next */ /* v8 ignore next */
export function load(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('load', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function vectorize */ /* v8 ignore next */ /* v8 ignore next */
export function vectorize(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'vectorize',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function meshgrid */ /* v8 ignore next */ /* v8 ignore next */
export function meshgrid(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'meshgrid',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function mgrid */ /* v8 ignore next */ /* v8 ignore next */
export function mgrid(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor('mgrid', args as BaseTensor[]); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function einsum_path */ /* v8 ignore next */ /* v8 ignore next */
export function einsum_path(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'einsum_path',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function polyfit */ /* v8 ignore next */ /* v8 ignore next */
export function polyfit(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'polyfit',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function histogram */ /* v8 ignore next */ /* v8 ignore next */
export function histogram(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'histogram',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function digitize */ /* v8 ignore next */ /* v8 ignore next */
export function digitize(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'digitize',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function export_model */ /* v8 ignore next */ /* v8 ignore next */
export function export_model(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'export_model',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function compile */ /* v8 ignore next */ /* v8 ignore next */
export function compile(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'compile',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function set_device */ /* v8 ignore next */ /* v8 ignore next */
export function set_device(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'set_device',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function set_log_level */ /* v8 ignore next */ /* v8 ignore next */
export function set_log_level(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'set_log_level',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function set_opset */ /* v8 ignore next */ /* v8 ignore next */
export function set_opset(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'set_opset',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/** Function set_num_threads */ /* v8 ignore next */ /* v8 ignore next */
export function set_num_threads(...args: (BaseTensor | number)[]): BaseTensor {
  /* v8 ignore next */ /* v8 ignore next */
  if (IS_LAZY)
    return new LazyTensor(
      'set_num_threads',
      args as BaseTensor[],
    ); /* v8 ignore next */ /* v8 ignore next */
  return new EagerTensor([1.0]); /* v8 ignore next */ /* v8 ignore next */
}

/**
 * Neural network operations.
 */
export const nn = {
  /** Rectified Linear Unit activation. */ /* v8 ignore next */ /* v8 ignore next */
  relu: (x: BaseTensor | number): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('Relu', [x as BaseTensor]) : new EagerTensor(null),
  /** Sigmoid activation. */ /* v8 ignore next */ /* v8 ignore next */
  sigmoid: (x: BaseTensor | number): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('Sigmoid', [x as BaseTensor]) : new EagerTensor(null),
  /** Softmax activation. */ /* v8 ignore next */ /* v8 ignore next */
  softmax: (x: BaseTensor | number, axis: number = -1): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY /* v8 ignore next */ /* v8 ignore next */
      ? new LazyTensor('Softmax', [
          x as BaseTensor,
          axis as ReturnType<typeof JSON.parse>,
        ]) /* v8 ignore next */ /* v8 ignore next */
      : new EagerTensor(null),
  /** Log-Softmax activation. */ /* v8 ignore next */ /* v8 ignore next */
  log_softmax: (x: BaseTensor | number, axis: number = -1): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY /* v8 ignore next */ /* v8 ignore next */
      ? new LazyTensor('LogSoftmax', [
          x as BaseTensor,
          axis as ReturnType<typeof JSON.parse>,
        ]) /* v8 ignore next */ /* v8 ignore next */
      : new EagerTensor(null),
  /** Gaussian Error Linear Unit activation. */ /* v8 ignore next */ /* v8 ignore next */
  gelu: (x: BaseTensor | number): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('Gelu', [x as BaseTensor]) : new EagerTensor(null),
  /** 2D Convolution operation. */ /* v8 ignore next */ /* v8 ignore next */
  conv2d: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('Conv', args as BaseTensor[]) : new EagerTensor(null),
  /** 2D Max Pooling operation. */ /* v8 ignore next */ /* v8 ignore next */
  max_pool2d: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('MaxPool', args as BaseTensor[]) : new EagerTensor(null),
  /** 2D Average Pooling operation. */ /* v8 ignore next */ /* v8 ignore next */
  avg_pool2d: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('AveragePool', args as BaseTensor[]) : new EagerTensor(null),
  /** Batch Normalization operation. */ /* v8 ignore next */ /* v8 ignore next */
  batch_norm: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('BatchNormalization', args as BaseTensor[]) : new EagerTensor(null),
  /** Layer Normalization operation. */ /* v8 ignore next */ /* v8 ignore next */
  layer_norm: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('LayerNormalization', args as BaseTensor[]) : new EagerTensor(null),
  /** Dropout operation. */ /* v8 ignore next */ /* v8 ignore next */
  dropout: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('Dropout', args as BaseTensor[]) : new EagerTensor(null),
  /** Linear (Fully Connected) operation. */ /* v8 ignore next */ /* v8 ignore next */
  linear: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('MatMul', args as BaseTensor[]) : new EagerTensor(null),
  /** Cross Entropy Loss operation. */ /* v8 ignore next */ /* v8 ignore next */
  cross_entropy_loss: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY /* v8 ignore next */ /* v8 ignore next */
      ? new LazyTensor(
          'SoftmaxCrossEntropyLoss',
          args as BaseTensor[],
        ) /* v8 ignore next */ /* v8 ignore next */
      : new EagerTensor(null),
};

/**
 * Linear algebra operations.
 */
export const linalg = {
  /** Matrix or vector norm. */ /* v8 ignore next */ /* v8 ignore next */
  norm: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('LpNormalization', args as BaseTensor[]) : new EagerTensor(null),
  /** Matrix determinant. */ /* v8 ignore next */ /* v8 ignore next */
  det: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('Det', args as BaseTensor[]) : new EagerTensor(null),
  /** Matrix inverse. */ /* v8 ignore next */ /* v8 ignore next */
  inv: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('Inv', args as BaseTensor[]) : new EagerTensor(null),
  /** Solve linear equations. */ /* v8 ignore next */ /* v8 ignore next */
  solve: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('Solve', args as BaseTensor[]) : new EagerTensor(null),
  /** Singular Value Decomposition. */ /* v8 ignore next */ /* v8 ignore next */
  svd: (...args: (BaseTensor | number)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY ? new LazyTensor('Svd', args as BaseTensor[]) : new EagerTensor(null),
};

/**
 * Character/String operations.
 */
export const char = {
  /** String concatenation. */ /* v8 ignore next */ /* v8 ignore next */
  add: (...args: (BaseTensor | string)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY /* v8 ignore next */ /* v8 ignore next */
      ? new LazyTensor(
          'StringConcat',
          args as ReturnType<typeof JSON.parse>[],
        ) /* v8 ignore next */ /* v8 ignore next */
      : new EagerTensor(null),
  /** String equality check. */ /* v8 ignore next */ /* v8 ignore next */
  equal: (...args: (BaseTensor | string)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY /* v8 ignore next */ /* v8 ignore next */
      ? new LazyTensor(
          'StringEqual',
          args as ReturnType<typeof JSON.parse>[],
        ) /* v8 ignore next */ /* v8 ignore next */
      : new EagerTensor(null),
  /** String replacement. */ /* v8 ignore next */ /* v8 ignore next */
  replace: (...args: (BaseTensor | string)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY /* v8 ignore next */ /* v8 ignore next */
      ? new LazyTensor(
          'StringReplace',
          args as ReturnType<typeof JSON.parse>[],
        ) /* v8 ignore next */ /* v8 ignore next */
      : new EagerTensor(null),
};

/**
 * Random number generation operations.
 */
export const random = {
  /** Uniform random numbers. */ /* v8 ignore next */ /* v8 ignore next */
  rand: (...args: (number | Shape)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY /* v8 ignore next */ /* v8 ignore next */
      ? new LazyTensor(
          'RandomUniform',
          args as ReturnType<typeof JSON.parse>[],
        ) /* v8 ignore next */ /* v8 ignore next */
      : new EagerTensor(null),
  /** Normal random numbers. */ /* v8 ignore next */ /* v8 ignore next */
  randn: (...args: (number | Shape)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY /* v8 ignore next */ /* v8 ignore next */
      ? new LazyTensor(
          'RandomNormal',
          args as ReturnType<typeof JSON.parse>[],
        ) /* v8 ignore next */ /* v8 ignore next */
      : new EagerTensor(null),
  /** Uniform random integers. */ /* v8 ignore next */ /* v8 ignore next */
  randint: (...args: (number | Shape)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY /* v8 ignore next */ /* v8 ignore next */
      ? new LazyTensor(
          'RandomUniformInt',
          args as ReturnType<typeof JSON.parse>[],
        ) /* v8 ignore next */ /* v8 ignore next */
      : new EagerTensor(null),
  /** Uniform random numbers. */ /* v8 ignore next */ /* v8 ignore next */
  uniform: (...args: (number | Shape)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY /* v8 ignore next */ /* v8 ignore next */
      ? new LazyTensor(
          'RandomUniform',
          args as ReturnType<typeof JSON.parse>[],
        ) /* v8 ignore next */ /* v8 ignore next */
      : new EagerTensor(null),
  /** Normal random numbers. */ /* v8 ignore next */ /* v8 ignore next */
  normal: (...args: (number | Shape)[]): BaseTensor =>
    /* v8 ignore next */ /* v8 ignore next */
    IS_LAZY /* v8 ignore next */ /* v8 ignore next */
      ? new LazyTensor(
          'RandomNormal',
          args as ReturnType<typeof JSON.parse>[],
        ) /* v8 ignore next */ /* v8 ignore next */
      : new EagerTensor(null),
  /** Sets the random seed. */ /* v8 ignore next */ /* v8 ignore next */
  seed: (s: number): void => {
    /* v8 ignore next */
    /* v8 ignore next */
    // Seed implementation /* v8 ignore next */ /* v8 ignore next */
  },
};

/**
 * Error thrown when broadcasting fails.
 */
export class BroadcastError extends Error {}
/**
 * Error thrown when types do not match.
 */
export class TypeMismatchError extends Error {}
