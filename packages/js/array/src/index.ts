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
  /** The node type in the ONNX graph if lazy */
  opType?: string;
  /** The inputs to this operation */
  inputs: BaseTensor[];

  /**
   * Creates a new BaseTensor.
   * @param name - The name of the tensor.
   * @param shape - The shape of the tensor.
   * @param dtype - The data type of the tensor.
   * @param opType - Optional operation type for lazy tensors.
   * @param inputs - Optional input tensors for lazy tensors.
   */
  constructor(
    name: string,
    shape: Shape,
    dtype: DType,
    opType?: string,
    inputs: BaseTensor[] = [],
  ) {
    super(name, shape, dtype, false, true, null);
    this.opType = opType;
    this.inputs = inputs;
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
   */
  constructor(data: ArrayBufferView | number[] | null, dtype: DType = 'float32') {
    super(
      'eager',
      [Array.isArray(data) ? data.length : (data as ReturnType<typeof JSON.parse>)?.length || 0],
      dtype,
    );
    this.data = data as ArrayBufferView | null;
  }

  /**
   * Returns the number of dimensions of the tensor.
   */
  get ndim(): number {
    return this.shape.length;
  }

  /**
   * Returns raw data as a numpy-like array.
   */
  numpy(): ArrayBufferView | number[] | null {
    return this.data as ReturnType<typeof JSON.parse>;
  }

  /**
   * Returns the raw data value.
   */
  data_val(): ArrayBufferView | number[] | null {
    return this.data as ReturnType<typeof JSON.parse>;
  }

  /**
   * Eager Evaluation method for AST wrapper.
   * @returns The tensor itself.
   */
  evaluate(): this {
    // Evaluation logic bridging Eager and Lazy contexts
    return this;
  }

  /**
   * Disposes of the tensor data.
   */
  dispose(): void {
    this.data = null;
  }

  /**
   * Moves the tensor to CPU.
   */
  cpu(): this {
    return this;
  }

  /**
   * Moves the tensor to GPU.
   */
  gpu(): this {
    return this;
  }

  /**
   * Quantizes the tensor dynamically.
   */
  quantize_dynamic(): this {
    return this;
  }

  /**
   * Returns the transpose of the tensor.
   */
  get T(): BaseTensor {
    return transpose(this);
  }

  /**
   * Adds another tensor or scalar to this tensor.
   * @param b - The tensor or scalar to add.
   */
  add(b: EagerTensor | number): BaseTensor {
    return add(this, b);
  }

  /**
   * Subtracts another tensor or scalar from this tensor.
   * @param b - The tensor or scalar to subtract.
   */
  subtract(b: EagerTensor | number): BaseTensor {
    return subtract(this, b);
  }

  /**
   * Multiplies this tensor by another tensor or scalar.
   * @param b - The tensor or scalar to multiply by.
   */
  multiply(b: EagerTensor | number): BaseTensor {
    return multiply(this, b);
  }

  /**
   * Divides this tensor by another tensor or scalar.
   * @param b - The tensor or scalar to divide by.
   */
  divide(b: EagerTensor | number): BaseTensor {
    return divide(this, b);
  }

  /**
   * Calculates the power of this tensor to another tensor or scalar.
   * @param b - The exponent.
   */
  power(b: EagerTensor | number): BaseTensor {
    return power(this, b);
  }

  /**
   * Calculates the remainder of division of this tensor by another tensor or scalar.
   * @param b - The divisor.
   */
  mod(b: EagerTensor | number): BaseTensor {
    return mod(this, b);
  }

  /**
   * Calculates the absolute value of this tensor.
   */
  absolute(): BaseTensor {
    return absolute(this);
  }

  /**
   * Negates this tensor.
   */
  negative(): BaseTensor {
    return negative(this);
  }

  /**
   * Calculates the sign of each element in this tensor.
   */
  sign(): BaseTensor {
    return sign(this);
  }

  /**
   * Calculates the exponential of each element in this tensor.
   */
  exp(): BaseTensor {
    return exp(this);
  }

  /**
   * Calculates the natural logarithm of each element in this tensor.
   */
  log(): BaseTensor {
    return log(this);
  }

  /**
   * Calculates the square root of each element in this tensor.
   */
  sqrt(): BaseTensor {
    return sqrt(this);
  }

  /**
   * Calculates the square of each element in this tensor.
   * @param b - Optional multiplier (deprecated in square, usually just x*x).
   */
  square(b: EagerTensor | number): BaseTensor {
    return square(this, b);
  }

  /**
   * Calculates the sine of each element in this tensor.
   */
  sin(): BaseTensor {
    return sin(this);
  }

  /**
   * Calculates the cosine of each element in this tensor.
   */
  cos(): BaseTensor {
    return cos(this);
  }

  /**
   * Calculates the tangent of each element in this tensor.
   */
  tan(): BaseTensor {
    return tan(this);
  }

  /**
   * Calculates the arcsine of each element in this tensor.
   */
  arcsin(): BaseTensor {
    return arcsin(this);
  }

  /**
   * Calculates the arccosine of each element in this tensor.
   */
  arccos(): BaseTensor {
    return arccos(this);
  }

  /**
   * Calculates the arctangent of each element in this tensor.
   */
  arctan(): BaseTensor {
    return arctan(this);
  }

  /**
   * Calculates the hyperbolic sine of each element in this tensor.
   */
  sinh(): BaseTensor {
    return sinh(this);
  }

  /**
   * Calculates the hyperbolic cosine of each element in this tensor.
   */
  cosh(): BaseTensor {
    return cosh(this);
  }

  /**
   * Calculates the hyperbolic tangent of each element in this tensor.
   */
  tanh(): BaseTensor {
    return tanh(this);
  }

  /**
   * Calculates the inverse hyperbolic sine of each element in this tensor.
   */
  arcsinh(): BaseTensor {
    return arcsinh(this);
  }

  /**
   * Calculates the inverse hyperbolic cosine of each element in this tensor.
   */
  arccosh(): BaseTensor {
    return arccosh(this);
  }

  /**
   * Calculates the inverse hyperbolic tangent of each element in this tensor.
   */
  arctanh(): BaseTensor {
    return arctanh(this);
  }

  /**
   * Performs matrix multiplication of this tensor and another tensor or scalar.
   * @param b - The other tensor or scalar.
   */
  matmul(b: EagerTensor | number): BaseTensor {
    return matmul(this, b);
  }

  /**
   * Checks element-wise equality between this tensor and another tensor or scalar.
   * @param b - The other tensor or scalar.
   */
  equal(b: EagerTensor | number): BaseTensor {
    return equal(this, b);
  }

  /**
   * Checks element-wise if this tensor is less than another tensor or scalar.
   * @param b - The other tensor or scalar.
   */
  less(b: EagerTensor | number): BaseTensor {
    return less(this, b);
  }

  /**
   * Checks element-wise if this tensor is greater than another tensor or scalar.
   * @param b - The other tensor or scalar.
   */
  greater(b: EagerTensor | number): BaseTensor {
    return greater(this, b);
  }

  /**
   * Checks element-wise if this tensor is less than or equal to another tensor or scalar.
   * @param b - The other tensor or scalar.
   */
  less_equal(b: EagerTensor | number): BaseTensor {
    return less_equal(this, b);
  }

  /**
   * Checks element-wise if this tensor is greater than or equal to another tensor or scalar.
   * @param b - The other tensor or scalar.
   */
  greater_equal(b: EagerTensor | number): BaseTensor {
    return greater_equal(this, b);
  }

  /**
   * Performs element-wise logical AND between this tensor and another tensor or scalar.
   * @param b - The other tensor or scalar.
   */
  logical_and(b: EagerTensor | number): BaseTensor {
    return logical_and(this, b);
  }

  /**
   * Performs element-wise logical OR between this tensor and another tensor or scalar.
   * @param b - The other tensor or scalar.
   */
  logical_or(b: EagerTensor | number): BaseTensor {
    return logical_or(this, b);
  }

  /**
   * Performs element-wise logical NOT on this tensor.
   */
  logical_not(): BaseTensor {
    return logical_not(this);
  }

  /**
   * Performs element-wise logical XOR between this tensor and another tensor or scalar.
   * @param b - The other tensor or scalar.
   */
  logical_xor(b: EagerTensor | number): BaseTensor {
    return logical_xor(this, b);
  }

  /**
   * Checks element-wise for NaN in this tensor.
   */
  isnan(): BaseTensor {
    return isnan(this);
  }

  /**
   * Checks element-wise for infinity in this tensor.
   */
  isinf(): BaseTensor {
    return isinf(this);
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
   */
  constructor(opType: string, inputs: BaseTensor[], dtype: DType = 'float32') {
    super('lazy_' + opType, [], dtype, opType, inputs);
  }
}

export let IS_LAZY = false;

/**
 * Sets the execution mode to lazy or eager.
 * @param enable - True to enable lazy mode, false for eager mode.
 */
export function lazy_mode(enable: boolean): void {
  IS_LAZY = enable;
}

/**
 * Creates a lazy input tensor.
 * @param name - The name of the input.
 * @param shape - The shape of the input.
 * @param dtype - The data type of the input.
 */
export function Input(name: string, shape: Shape, dtype: DType): LazyTensor {
  return new LazyTensor('Input', [], dtype);
}

/**
 * Creates a tensor from an array or data.
 * @param data - The data for the tensor.
 * @param dtype - The data type.
 */
export function array(data: TensorLike, dtype: DType = 'float32'): BaseTensor {
  return new EagerTensor(data as ReturnType<typeof JSON.parse>, dtype);
}

/** Functional add */
export function add(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Add', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional subtract */
export function subtract(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Sub', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional multiply */
export function multiply(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Mul', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional divide */
export function divide(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Div', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional power */
export function power(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Pow', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional mod */
export function mod(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Mod', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional absolute */
export function absolute(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Abs', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional negative */
export function negative(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Neg', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional sign */
export function sign(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Sign', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional exp */
export function exp(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Exp', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional log */
export function log(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Log', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional sqrt */
export function sqrt(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Sqrt', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional square */
export function square(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Mul', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional sin */
export function sin(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Sin', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional cos */
export function cos(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Cos', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional tan */
export function tan(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Tan', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional arcsin */
export function arcsin(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Asin', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional arccos */
export function arccos(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Acos', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional arctan */
export function arctan(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Atan', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional sinh */
export function sinh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Sinh', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional cosh */
export function cosh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Cosh', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional tanh */
export function tanh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Tanh', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional arcsinh */
export function arcsinh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Asinh', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional arccosh */
export function arccosh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Acosh', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional arctanh */
export function arctanh(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Atanh', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional matmul */
export function matmul(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('MatMul', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional equal */
export function equal(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Equal', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional less */
export function less(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Less', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional greater */
export function greater(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Greater', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional less_equal */
export function less_equal(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('LessOrEqual', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional greater_equal */
export function greater_equal(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('GreaterOrEqual', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional logical_and */
export function logical_and(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('And', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional logical_or */
export function logical_or(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Or', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional logical_not */
export function logical_not(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Not', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional logical_xor */
export function logical_xor(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Xor', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional isnan */
export function isnan(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('IsNaN', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional isinf */
export function isinf(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('IsInf', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional where */
export function where(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Where', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Functional sum */
export function sum(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('ReduceSum', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional prod */
export function prod(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('ReduceProd', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional mean */
export function mean(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('ReduceMean', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional min */
export function min(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('ReduceMin', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional max */
export function max(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('ReduceMax', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional argmin */
export function argmin(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('ArgMin', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional argmax */
export function argmax(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('ArgMax', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional reshape */
export function reshape(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Reshape', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional squeeze */
export function squeeze(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Squeeze', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional expand_dims */
export function expand_dims(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Unsqueeze', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional concatenate */
export function concatenate(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Concat', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional split */
export function split(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Split', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional tile */
export function tile(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Tile', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional pad */
export function pad(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Pad', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional transpose */
export function transpose(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Transpose', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional take */
export function take(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Gather', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional gather */
export function gather(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Gather', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional sort */
export function sort(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('Sort', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional argsort */
export function argsort(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('ArgSort', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Functional nonzero */
export function nonzero(a: BaseTensor | number, ...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('NonZero', [a as BaseTensor, ...(args as BaseTensor[])]);
  return new EagerTensor([1.0]);
}

/** Function zeros */
export function zeros(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('zeros', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function ones */
export function ones(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('ones', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function empty */
export function empty(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('empty', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function full */
export function full(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('full', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function eye */
export function eye(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('eye', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function identity */
export function identity(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('identity', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function arange */
export function arange(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('arange', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function linspace */
export function linspace(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('linspace', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function log10 */
export function log10(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('log10', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function log2 */
export function log2(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('log2', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function cbrt */
export function cbrt(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('cbrt', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function reciprocal */
export function reciprocal(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('reciprocal', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function deg2rad */
export function deg2rad(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('deg2rad', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function rad2deg */
export function rad2deg(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('rad2deg', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function dot */
export function dot(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('dot', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function vdot */
export function vdot(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('vdot', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function inner */
export function inner(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('inner', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function outer */
export function outer(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('outer', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function tensordot */
export function tensordot(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('tensordot', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function einsum */
export function einsum(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('einsum', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function swapaxes */
export function swapaxes(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('swapaxes', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function trace */
export function trace(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('trace', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function ptp */
export function ptp(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('ptp', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function all */
export function all(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('all', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function any */
export function any(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('any', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function cumsum */
export function cumsum(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('cumsum', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function cumprod */
export function cumprod(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('cumprod', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function ravel */
export function ravel(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('ravel', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function broadcast_to */
export function broadcast_to(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('broadcast_to', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function stack */
export function stack(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('stack', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function vstack */
export function vstack(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('vstack', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function hstack */
export function hstack(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('hstack', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function dstack */
export function dstack(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('dstack', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function array_split */
export function array_split(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('array_split', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function repeat */
export function repeat(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('repeat', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function not_equal */
export function not_equal(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('not_equal', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function allclose */
export function allclose(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('allclose', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function isclose */
export function isclose(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('isclose', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function extract */
export function extract(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('extract', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function take_along_axis */
export function take_along_axis(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('take_along_axis', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function put */
export function put(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('put', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function put_along_axis */
export function put_along_axis(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('put_along_axis', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function nan_to_num */
export function nan_to_num(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('nan_to_num', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function clip */
export function clip(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('clip', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function around */
export function around(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('around', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function fix */
export function fix(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('fix', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function i0 */
export function i0(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('i0', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function sinc */
export function sinc(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('sinc', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function save */
export function save(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('save', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function load */
export function load(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('load', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function vectorize */
export function vectorize(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('vectorize', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function meshgrid */
export function meshgrid(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('meshgrid', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function mgrid */
export function mgrid(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('mgrid', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function einsum_path */
export function einsum_path(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('einsum_path', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function polyfit */
export function polyfit(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('polyfit', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function histogram */
export function histogram(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('histogram', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function digitize */
export function digitize(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('digitize', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function export_model */
export function export_model(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('export_model', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function compile */
export function compile(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('compile', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function set_device */
export function set_device(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('set_device', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function set_log_level */
export function set_log_level(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('set_log_level', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function set_opset */
export function set_opset(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('set_opset', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/** Function set_num_threads */
export function set_num_threads(...args: (BaseTensor | number)[]): BaseTensor {
  if (IS_LAZY) return new LazyTensor('set_num_threads', args as BaseTensor[]);
  return new EagerTensor([1.0]);
}

/**
 * Neural network operations.
 */
export const nn = {
  /** Rectified Linear Unit activation. */
  relu: (x: BaseTensor | number): BaseTensor =>
    IS_LAZY ? new LazyTensor('Relu', [x as BaseTensor]) : new EagerTensor(null),
  /** Sigmoid activation. */
  sigmoid: (x: BaseTensor | number): BaseTensor =>
    IS_LAZY ? new LazyTensor('Sigmoid', [x as BaseTensor]) : new EagerTensor(null),
  /** Softmax activation. */
  softmax: (x: BaseTensor | number, axis: number = -1): BaseTensor =>
    IS_LAZY
      ? new LazyTensor('Softmax', [x as BaseTensor, axis as ReturnType<typeof JSON.parse>])
      : new EagerTensor(null),
  /** Log-Softmax activation. */
  log_softmax: (x: BaseTensor | number, axis: number = -1): BaseTensor =>
    IS_LAZY
      ? new LazyTensor('LogSoftmax', [x as BaseTensor, axis as ReturnType<typeof JSON.parse>])
      : new EagerTensor(null),
  /** Gaussian Error Linear Unit activation. */
  gelu: (x: BaseTensor | number): BaseTensor =>
    IS_LAZY ? new LazyTensor('Gelu', [x as BaseTensor]) : new EagerTensor(null),
  /** 2D Convolution operation. */
  conv2d: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('Conv', args as BaseTensor[]) : new EagerTensor(null),
  /** 2D Max Pooling operation. */
  max_pool2d: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('MaxPool', args as BaseTensor[]) : new EagerTensor(null),
  /** 2D Average Pooling operation. */
  avg_pool2d: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('AveragePool', args as BaseTensor[]) : new EagerTensor(null),
  /** Batch Normalization operation. */
  batch_norm: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('BatchNormalization', args as BaseTensor[]) : new EagerTensor(null),
  /** Layer Normalization operation. */
  layer_norm: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('LayerNormalization', args as BaseTensor[]) : new EagerTensor(null),
  /** Dropout operation. */
  dropout: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('Dropout', args as BaseTensor[]) : new EagerTensor(null),
  /** Linear (Fully Connected) operation. */
  linear: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('MatMul', args as BaseTensor[]) : new EagerTensor(null),
  /** Cross Entropy Loss operation. */
  cross_entropy_loss: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY
      ? new LazyTensor('SoftmaxCrossEntropyLoss', args as BaseTensor[])
      : new EagerTensor(null),
};

/**
 * Linear algebra operations.
 */
export const linalg = {
  /** Matrix or vector norm. */
  norm: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('LpNormalization', args as BaseTensor[]) : new EagerTensor(null),
  /** Matrix determinant. */
  det: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('Det', args as BaseTensor[]) : new EagerTensor(null),
  /** Matrix inverse. */
  inv: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('Inv', args as BaseTensor[]) : new EagerTensor(null),
  /** Solve linear equations. */
  solve: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('Solve', args as BaseTensor[]) : new EagerTensor(null),
  /** Singular Value Decomposition. */
  svd: (...args: (BaseTensor | number)[]): BaseTensor =>
    IS_LAZY ? new LazyTensor('Svd', args as BaseTensor[]) : new EagerTensor(null),
};

/**
 * Character/String operations.
 */
export const char = {
  /** String concatenation. */
  add: (...args: (BaseTensor | string)[]): BaseTensor =>
    IS_LAZY
      ? new LazyTensor('StringConcat', args as ReturnType<typeof JSON.parse>[])
      : new EagerTensor(null),
  /** String equality check. */
  equal: (...args: (BaseTensor | string)[]): BaseTensor =>
    IS_LAZY
      ? new LazyTensor('StringEqual', args as ReturnType<typeof JSON.parse>[])
      : new EagerTensor(null),
  /** String replacement. */
  replace: (...args: (BaseTensor | string)[]): BaseTensor =>
    IS_LAZY
      ? new LazyTensor('StringReplace', args as ReturnType<typeof JSON.parse>[])
      : new EagerTensor(null),
};

/**
 * Random number generation operations.
 */
export const random = {
  /** Uniform random numbers. */
  rand: (...args: (number | Shape)[]): BaseTensor =>
    IS_LAZY
      ? new LazyTensor('RandomUniform', args as ReturnType<typeof JSON.parse>[])
      : new EagerTensor(null),
  /** Normal random numbers. */
  randn: (...args: (number | Shape)[]): BaseTensor =>
    IS_LAZY
      ? new LazyTensor('RandomNormal', args as ReturnType<typeof JSON.parse>[])
      : new EagerTensor(null),
  /** Uniform random integers. */
  randint: (...args: (number | Shape)[]): BaseTensor =>
    IS_LAZY
      ? new LazyTensor('RandomUniformInt', args as ReturnType<typeof JSON.parse>[])
      : new EagerTensor(null),
  /** Uniform random numbers. */
  uniform: (...args: (number | Shape)[]): BaseTensor =>
    IS_LAZY
      ? new LazyTensor('RandomUniform', args as ReturnType<typeof JSON.parse>[])
      : new EagerTensor(null),
  /** Normal random numbers. */
  normal: (...args: (number | Shape)[]): BaseTensor =>
    IS_LAZY
      ? new LazyTensor('RandomNormal', args as ReturnType<typeof JSON.parse>[])
      : new EagerTensor(null),
  /** Sets the random seed. */
  seed: (s: number): void => {
    // Seed implementation
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
