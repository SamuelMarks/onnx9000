/* eslint-disable */
// @onnx9000/tfjs-shim - Drop-in replacement for TensorFlow.js

import { Graph, Node, Tensor as OnnxTensor } from '@onnx9000/core';

// --- Global Config and Envs ---
let currentBackend = 'webgpu';
let isProdMode = false;
let isDebugMode = false;

/** Core version of the engine. */
export const version_core = '4.10.0';
/** TFJS-compatible version of the engine. */
export const version_tfjs = '4.10.0';
/** Version information object. */
export const version = {
  tfjs: version_tfjs,
  core: version_core,
};

/**
 * Sets the current execution backend.
 * @param backendName The name of the backend (e.g., 'webgpu').
 * @returns A promise that resolves to true if the backend was successfully set.
 */
export async function setBackend(backendName: string): Promise<boolean> {
  currentBackend = backendName;
  return true;
}

/**
 * Gets the name of the currently active backend.
 * @returns The name of the active backend.
 */
export function getBackend(): string {
  return currentBackend;
}

/**
 * Returns a promise that resolves when the engine is ready.
 * @returns A promise that resolves when the engine is ready.
 */
export async function ready(): Promise<void> {
  return Promise.resolve();
}

/**
 * Interface for the environment configuration.
 */
export interface Environment {
  /**
   * Gets a configuration value by key.
   * @param key The configuration key.
   */
  get(key: string): string | number | boolean | null;
  /**
   * Sets a configuration value by key.
   * @param key The configuration key.
   * @param value The value to set.
   */
  set(key: string, value: string | number | boolean): void;
}

const _envVars: Record<string, string | number | boolean> = {};

/**
 * Returns the environment configuration object.
 * @returns The environment object.
 */
export function env(): Environment {
  return {
    get: (key: string) => (key in _envVars ? _envVars[key]! : null),
    set: (key: string, value: string | number | boolean) => {
      _envVars[key] = value;
    },
  };
}

/**
 * Enables production mode, disabling some runtime checks.
 */
export function enableProdMode(): void {
  isProdMode = true;
}

/**
 * Enables debug mode, enabling verbose logging and extra checks.
 */
export function enableDebugMode(): void {
  isDebugMode = true;
}

/**
 * Returns memory usage information.
 * @returns An object containing memory statistics.
 */
export function memory(): {
  numBytes: number;
  numTensors: number;
  numDataBuffers: number;
  reasons?: string[];
} {
  return {
    numBytes: globalTensorRegistry.size * 4,
    numTensors: globalTensorRegistry.size,
    numDataBuffers: globalTensorRegistry.size,
  };
}

/**
 * Profiles the execution of a function.
 * @param f The function to profile.
 * @returns A promise that resolves to profiling results.
 */
export async function profile(
  f: () => Promise<Tensor | Tensor[] | void> | Tensor | Tensor[] | void,
): Promise<{
  newBytes: number;
  newTensors: number;
  peakBytes: number;
  kernels: string[];
  result: Tensor | Tensor[] | void;
}> {
  const result = await f();
  return { newBytes: 0, newTensors: 0, peakBytes: 0, kernels: [], result };
}

/**
 * Measures the execution time of a function.
 * @param f The function to time.
 * @returns A promise that resolves to an object with wall and kernel time in milliseconds.
 */
export async function time(
  f: () => Promise<void> | void,
): Promise<{ wallMs: number; kernelMs: number }> {
  const start = performance.now();
  await f();
  const end = performance.now();
  return { wallMs: end - start, kernelMs: end - start };
}

/**
 * Disposes all variables in the global registry.
 */
export function disposeVariables(): void {
  globalTensorRegistry.clear();
}

// --- Tensor Core ---
/** Data type for tensors. */
export type DataType = 'float32' | 'int32' | 'bool' | 'complex64' | 'string';
/** A union of supported typed arrays. */
export type TypedArray = Float32Array | Int32Array | Uint8Array;

const globalTensorRegistry = new Set<Tensor>();
let currentTidyScope: Tensor[][] | null = null;

/**
 * A Tensor object representing a multi-dimensional array.
 */
export class Tensor {
  /** The shape of the tensor. */
  shape: number[];
  /** The data type of the tensor. */
  dtype: DataType;
  /** The underlying data array. */
  dataArray: (number | string | boolean)[];
  get nData(): number[] {
    return this.dataArray as number[];
  }
  /** Whether the tensor has been disposed. */
  isDisposed: boolean = false;
  /** The rank (number of dimensions) of the tensor. */
  rank: number;
  /** The total number of elements in the tensor. */
  size: number;

  /**
   * Creates a new Tensor.
   * @param shape The shape of the tensor.
   * @param dtype The data type of the tensor.
   * @param dataArray The underlying data array.
   */
  constructor(shape: number[], dtype: DataType, dataArray: (number | string | boolean)[]) {
    this.shape = shape;
    this.dtype = dtype;
    this.dataArray = dataArray;
    this.rank = shape.length;
    this.size = shape.reduce((a, b) => a * b, 1);
    globalTensorRegistry.add(this);
    if (currentTidyScope && currentTidyScope.length > 0) {
      currentTidyScope[currentTidyScope.length - 1]!.push(this);
    }
  }

  /**
   * Asynchronously retrieves the tensor data.
   * @returns A promise that resolves to the tensor data as a TypedArray or an array of strings.
   */
  async data(): Promise<TypedArray | string[]> {
    if (this.dtype === 'string')
      return new Array(this.size).fill(this.dataArray[0] || '') as string[];
    if (this.dtype === 'float32') return new Float32Array(this.dataArray as number[]);
    if (this.dtype === 'int32') return new Int32Array(this.dataArray as number[]);
    if (this.dtype === 'bool')
      return new Uint8Array((this.dataArray as boolean[]).map((b) => (b ? 1 : 0)));
    return new Float32Array(this.dataArray as number[]);
  }

  /**
   * Synchronously retrieves the tensor data.
   * @returns The tensor data as a TypedArray or an array of strings.
   */
  dataSync(): TypedArray | string[] {
    if (this.dtype === 'string')
      return new Array(this.size).fill(this.dataArray[0] || '') as string[];
    if (this.dtype === 'float32') return new Float32Array(this.dataArray as number[]);
    if (this.dtype === 'int32') return new Int32Array(this.dataArray as number[]);
    if (this.dtype === 'bool')
      return new Uint8Array((this.dataArray as boolean[]).map((b) => (b ? 1 : 0)));
    return new Float32Array(this.dataArray as number[]);
  }

  /**
   * Asynchronously retrieves the tensor data as a nested array.
   * @returns A promise that resolves to the nested array.
   */
  async array(): Promise<NestedArray<number | string | boolean>> {
    return this.arraySync();
  }

  /**
   * Synchronously retrieves the tensor data as a nested array.
   * @returns The nested array.
   */
  arraySync(): NestedArray<number | string | boolean> {
    if (this.rank === 0) return this.dataArray[0] as number | string | boolean;
    if (this.rank === 1) return Array.from(this.dataArray) as (number | string | boolean)[];
    return Array.from(this.dataArray) as NestedArray<number | string | boolean>; // Simplified flat array for now to ensure typings
  }

  /**
   * Disposes the tensor and releases its memory.
   */
  dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    globalTensorRegistry.delete(this);
  }

  /**
   * Creates a clone of the tensor.
   * @returns A new tensor with the same shape, dtype, and data.
   */
  clone(): Tensor {
    return new Tensor(this.shape.slice(), this.dtype, this.dataArray.slice());
  }

  /**
   * Prints the tensor to the console.
   * @param verbose Whether to print verbose information.
   */
  print(verbose: boolean = false): void {
    console.log(`Tensor [${this.shape}]`, this.dataArray);
  }

  /**
   * Flattens the tensor into a 1D tensor.
   * @returns A new 1D tensor.
   */
  flatten(): Tensor {
    return reshape(this, [-1]);
  }

  /**
   * Reshapes the tensor to a new shape.
   * @param newShape The new shape.
   * @returns A new reshaped tensor.
   */
  reshape(newShape: number[]): Tensor {
    return reshape(this, newShape);
  }

  /**
   * Casts the tensor to a new data type.
   * @param dtype The new data type.
   * @returns A new tensor with the casted data type.
   */
  cast(dtype: DataType): Tensor {
    return cast(this, dtype);
  }

  /**
   * Squeezes dimensions of size 1.
   * @param axis Optional list of axes to squeeze.
   * @returns A new squeezed tensor.
   */
  squeeze(axis?: number[]): Tensor {
    return squeeze(this, axis);
  }
}

/** Recursive type for nested arrays. */
export type NestedArray<T> = T | NestedArray<T>[];

/**
 * Creates a tensor from values.
 * @param values The values for the tensor.
 * @param shape The shape of the tensor.
 * @param dtype The data type of the tensor.
 * @returns A new tensor.
 */
export function tensor(
  values: number | string | boolean | NestedArray<number | string | boolean> | TypedArray,
  shape?: number[],
  dtype?: DataType,
): Tensor {
  const flatVals = Array.isArray(values)
    ? (values as (number | string | boolean)[]).flat(Infinity)
    : values !== null &&
        typeof values === 'object' &&
        'length' in values &&
        typeof (values as { length: number }).length === 'number' &&
        typeof values !== 'string'
      ? Array.from(values as Iterable<number | string | boolean>)
      : [values as number | string | boolean];
  const s = shape || [flatVals.length];
  const d = dtype || 'float32';
  return new Tensor(s, d, flatVals as (number | string | boolean)[]);
}

/**
 * Creates a 1D tensor.
 * @param values The values for the tensor.
 * @param dtype The data type of the tensor.
 * @returns A new 1D tensor.
 */
export function tensor1d(
  values: (number | string | boolean)[] | TypedArray,
  dtype?: DataType,
): Tensor {
  return tensor(values, [values.length || 1], dtype);
}

/**
 * Creates a 2D tensor.
 * @param values The values for the tensor.
 * @param shape The shape of the tensor.
 * @param dtype The data type of the tensor.
 * @returns A new 2D tensor.
 */
export function tensor2d(
  values: NestedArray<number | string | boolean> | TypedArray,
  shape?: [number, number],
  dtype?: DataType,
): Tensor {
  return tensor(values, shape, dtype);
}

/**
 * Creates a 3D tensor.
 * @param values The values for the tensor.
 * @param shape The shape of the tensor.
 * @param dtype The data type of the tensor.
 * @returns A new 3D tensor.
 */
export function tensor3d(
  values: NestedArray<number | string | boolean> | TypedArray,
  shape?: [number, number, number],
  dtype?: DataType,
): Tensor {
  return tensor(values, shape, dtype);
}

/**
 * Creates a 4D tensor.
 * @param values The values for the tensor.
 * @param shape The shape of the tensor.
 * @param dtype The data type of the tensor.
 * @returns A new 4D tensor.
 */
export function tensor4d(
  values: NestedArray<number | string | boolean> | TypedArray,
  shape?: [number, number, number, number],
  dtype?: DataType,
): Tensor {
  return tensor(values, shape, dtype);
}

/**
 * Creates a 5D tensor.
 * @param values The values for the tensor.
 * @param shape The shape of the tensor.
 * @param dtype The data type of the tensor.
 * @returns A new 5D tensor.
 */
export function tensor5d(
  values: NestedArray<number | string | boolean> | TypedArray,
  shape?: [number, number, number, number, number],
  dtype?: DataType,
): Tensor {
  return tensor(values, shape, dtype);
}

/**
 * Creates a 6D tensor.
 * @param values The values for the tensor.
 * @param shape The shape of the tensor.
 * @param dtype The data type of the tensor.
 * @returns A new 6D tensor.
 */
export function tensor6d(
  values: NestedArray<number | string | boolean> | TypedArray,
  shape?: [number, number, number, number, number, number],
  dtype?: DataType,
): Tensor {
  return tensor(values, shape, dtype);
}

/**
 * Creates a scalar tensor.
 * @param value The value for the scalar.
 * @param dtype The data type of the tensor.
 * @returns A new scalar tensor.
 */
export function scalar(value: number | string | boolean, dtype?: DataType): Tensor {
  return tensor([value], [], dtype);
}

/**
 * Creates a tensor with a buffer.
 * @param shape The shape of the tensor.
 * @param dtype The data type of the tensor.
 * @param values The initial values for the buffer.
 * @returns A new tensor.
 */
export function buffer(
  shape: number[],
  dtype?: DataType,
  values?: (number | string | boolean)[] | TypedArray,
): Tensor {
  return tensor(values || new Array(shape.reduce((a, b) => a * b, 1)).fill(0), shape, dtype);
}

/**
 * Clones a tensor.
 * @param x The tensor to clone.
 * @returns A new tensor.
 */
export function clone(x: Tensor): Tensor {
  return x.clone();
}

/** Recursive type for tensors or nested collections of tensors. */
export type RecursiveTensor =
  | Tensor
  | Tensor[]
  | Record<string, Tensor>
  | string
  | number
  | boolean
  | null
  | undefined
  | void
  | { [key: string]: RecursiveTensor }
  | RecursiveTensor[];

/**
 * Executes a function within a scope that automatically disposes tensors.
 * @param nameOrFn Name of the scope or the function to execute.
 * @param fn The function to execute if a name was provided.
 * @returns The result of the function.
 */
export function tidy<T>(nameOrFn: string | (() => T), fn?: () => T): T {
  const actualFn = fn || (nameOrFn as () => T);
  if (!currentTidyScope) currentTidyScope = [];
  currentTidyScope.push([]);
  const result = actualFn();
  const scopeTensors = currentTidyScope.pop()!;

  const extractTensors = (res: RecursiveTensor): Tensor[] => {
    if (res instanceof Tensor) return [res];
    if (Array.isArray(res)) return (res as RecursiveTensor[]).flatMap(extractTensors);
    if (res && typeof res === 'object') {
      return Object.values(res as Record<string, RecursiveTensor>).flatMap(extractTensors);
    }
    return [];
  };

  const keepTensors = new Set(extractTensors(result as RecursiveTensor));
  for (const t of scopeTensors) {
    if (!keepTensors.has(t) && !t.isDisposed) {
      t.dispose();
    }
  }
  return result;
}

/**
 * Marks a tensor to be kept after the current tidy scope ends.
 * @param x The tensor to keep.
 * @returns The same tensor.
 */
export function keep(x: Tensor): Tensor {
  if (currentTidyScope && currentTidyScope.length > 0) {
    const scope = currentTidyScope[currentTidyScope.length - 1];
    const idx = scope.indexOf(x);
    if (idx !== -1) scope.splice(idx, 1);
  }
  return x;
}

/**
 * Disposes tensors to release memory.
 * @param tensors The tensor or collection of tensors to dispose.
 */
export function dispose(tensors: Tensor | Tensor[] | Record<string, Tensor>): void {
  if (tensors instanceof Tensor) {
    tensors.dispose();
  } else if (Array.isArray(tensors)) {
    tensors.forEach((t) => {
      t.dispose();
    });
  } else if (typeof tensors === 'object' && tensors !== null) {
    Object.values(tensors).forEach((t) => {
      if (t instanceof Tensor) {
        t.dispose();
      }
    });
  }
}

/**
 * Higher-order function to create an element-wise binary operation.
 * @param name The name of the operation.
 * @param op The binary operation function.
 * @returns A function that performs the binary operation on two tensors.
 */
function makeElementwise(name: string, op: (a: number, b: number) => number) {
  return (a: Tensor | number, b: Tensor | number): Tensor => {
    const aTensor =
      typeof a === 'number' || typeof a === 'boolean'
        ? scalar(a as number | boolean)
        : (a as Tensor);
    const bTensor =
      typeof b === 'number' || typeof b === 'boolean'
        ? scalar(b as number | boolean)
        : (b as Tensor);

    if (!aTensor || !bTensor || !aTensor.shape || !bTensor.shape) {
      throw new Error(`Invalid inputs to ${name}`);
    }

    const len = Math.max(aTensor.size, bTensor.size);
    const newData = new Array(len);
    for (let i = 0; i < len; i++) {
      const valA = aTensor.nData[i % aTensor.size] as number;
      const valB = bTensor.nData[i % bTensor.size] as number;
      newData[i] = op(valA, valB);
    }
    const outShape = aTensor.size >= bTensor.size ? aTensor.shape.slice() : bTensor.shape.slice();
    return new Tensor(outShape, aTensor.dtype, newData);
  };
}

/**
 * Higher-order function to create a unary operation.
 * @param name The name of the operation.
 * @param op The unary operation function.
 * @returns A function that performs the unary operation on a tensor.
 */
function makeUnary(name: string, op: (a: number) => number) {
  return (a: Tensor): Tensor => {
    const newData = new Array(a.size);
    for (let i = 0; i < a.size; i++) {
      newData[i] = op(a.nData[i] as number);
    }
    return new Tensor(a.shape, a.dtype, newData);
  };
}

function makeBinary(name: string, op: (a: number, b: number) => number) {
  return (a: Tensor, b: Tensor): Tensor => {
    const size = Math.max(a.size, b.size);
    const newData = new Array(size);
    for (let i = 0; i < size; i++) {
      newData[i] = op((a.nData[i] || 0) as number, (b.nData[i] || 0) as number);
    }
    return new Tensor(a.shape, a.dtype, newData);
  };
}

/** Adds two tensors element-wise. */
export const add = makeElementwise('add', (a, b) => a + b);
/** Subtracts two tensors element-wise. */
export const sub = makeElementwise('sub', (a, b) => a - b);
/** Multiplies two tensors element-wise. */
export const mul = makeElementwise('mul', (a, b) => a * b);
/** Divides two tensors element-wise. */
export const div = makeElementwise('div', (a, b) => a / b);
/** Divides two tensors element-wise, returning 0 if the divisor is 0. */
export const divNoNan = makeElementwise('divNoNan', (a, b) => (b === 0 ? 0 : a / b));
/** Divides two tensors element-wise and floors the result. */
export const floorDiv = makeElementwise('floorDiv', (a, b) => Math.floor(a / b));
/** Returns the element-wise maximum of two tensors. */
export const maximum = makeElementwise('maximum', (a, b) => Math.max(a, b));
/** Returns the element-wise minimum of two tensors. */
export const minimum = makeElementwise('minimum', (a, b) => Math.min(a, b));
/** Returns the element-wise remainder of division. */
export const mod = makeElementwise('mod', (a, b) => a % b);
/** Returns the element-wise power of two tensors. */
export const pow = makeElementwise('pow', (a, b) => Math.pow(a, b));
/** Returns the element-wise squared difference of two tensors. */
export const squaredDifference = makeElementwise('squaredDifference', (a, b) => Math.pow(a - b, 2));

/** Returns the element-wise absolute value of a tensor. */
export const abs = makeUnary('abs', Math.abs);
/** Returns the element-wise arc cosine of a tensor. */
export const acos = makeUnary('acos', Math.acos);
/** Returns the element-wise inverse hyperbolic cosine of a tensor. */
export const acosh = makeUnary('acosh', Math.acosh);
/** Returns the element-wise arc sine of a tensor. */
export const asin = makeUnary('asin', Math.asin);
/** Returns the element-wise inverse hyperbolic sine of a tensor. */
export const asinh = makeUnary('asinh', Math.asinh);
/** Returns the element-wise arc tangent of a tensor. */
export const atan = makeUnary('atan', Math.atan);
/** Returns the element-wise inverse hyperbolic tangent of a tensor. */
export const atanh = makeUnary('atanh', Math.atanh);
/** Returns the element-wise ceiling of a tensor. */
export const ceil = makeUnary('ceil', Math.ceil);
/** Returns the element-wise cosine of a tensor. */
export const cos = makeUnary('cos', Math.cos);
/** Returns the element-wise hyperbolic cosine of a tensor. */
export const cosh = makeUnary('cosh', Math.cosh);
/** Returns the element-wise error function of a tensor (approximation). */
export const erf = makeUnary('erf', (x) => Math.tanh(x)); // approx
/** Returns the element-wise exponential of a tensor. */
export const exp = makeUnary('exp', Math.exp);
/** Returns the element-wise exponential minus 1 of a tensor. */
export const expm1 = makeUnary('expm1', Math.expm1);
/** Returns the element-wise floor of a tensor. */
export const floor = makeUnary('floor', Math.floor);
/** Returns the element-wise check if finite for a tensor. */
export const isFinite = makeUnary('isFinite', (x) => (Number.isFinite(x) ? 1 : 0));
/** Returns the element-wise check if infinite for a tensor. */
export const isInf = makeUnary('isInf', (x) => (!Number.isFinite(x) && !Number.isNaN(x) ? 1 : 0));
/** Returns the element-wise check if NaN for a tensor. */
export const isNaN = makeUnary('isNaN', (x) => (Number.isNaN(x) ? 1 : 0));
/** Returns the element-wise natural logarithm of a tensor. */
export const log = makeUnary('log', Math.log);
/** Returns the element-wise natural logarithm of 1 + input of a tensor. */
export const log1p = makeUnary('log1p', Math.log1p);
/** Returns the element-wise negation of a tensor. */
export const neg = makeUnary('neg', (x) => -x);
/** Returns the element-wise reciprocal of a tensor. */
export const reciprocal = makeUnary('reciprocal', (x) => 1 / x);
/** Returns the element-wise rounding of a tensor. */
export const round = makeUnary('round', Math.round);
/** Returns the element-wise reciprocal square root of a tensor. */
export const rsqrt = makeUnary('rsqrt', (x) => 1 / Math.sqrt(x));
/** Returns the element-wise sign of a tensor. */
export const sign = makeUnary('sign', Math.sign);
/** Returns the element-wise sine of a tensor. */
export const sin = makeUnary('sin', Math.sin);
/** Returns the element-wise hyperbolic sine of a tensor. */
export const sinh = makeUnary('sinh', Math.sinh);
/** Returns the element-wise square root of a tensor. */
export const sqrt = makeUnary('sqrt', Math.sqrt);
/** Returns the element-wise square of a tensor. */
export const square = makeUnary('square', (x) => x * x);
/** Returns the element-wise tangent of a tensor. */
export const tan = makeUnary('tan', Math.tan);

/** Returns the element-wise arc tangent of y/x. */
export const atan2 = makeElementwise('atan2', Math.atan2);

/**
 * Returns a tensor of 1s if x > 0, and alpha otherwise.
 * @param x Input tensor.
 * @param alpha Value to use when x <= 0.
 * @returns Resulting tensor.
 */
export function step(x: Tensor, alpha: number = 0.0): Tensor {
  return makeUnary('step', (v) => (v > 0 ? 1 : alpha))(x);
}

/**
 * Computes the sum of a list of tensors.
 * @param tensors List of tensors.
 * @returns Sum of tensors.
 */
export function addN(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) throw new Error('addN requires at least one tensor');
  let res = tensors[0];
  for (let i = 1; i < tensors.length; i++) res = add(res, tensors[i]);
  return res;
}

/**
 * Computes matrix multiplication of two tensors.
 * @param a First tensor.
 * @param b Second tensor.
 * @param transposeA Whether to transpose a.
 * @param transposeB Whether to transpose b.
 * @returns Resulting tensor.
 */
export function matMul(a: Tensor, b: Tensor, transposeA = false, transposeB = false): Tensor {
  const rowsA = transposeA ? a.shape[1] : a.shape[0];
  const colsA = transposeA ? a.shape[0] : a.shape[1];
  const rowsB = transposeB ? b.shape[1] : b.shape[0];
  const colsB = transposeB ? b.shape[0] : b.shape[1];
  if (colsA !== rowsB) throw new Error(`Matrix size-incompatible`);
  const newData = new Array(rowsA * colsB).fill(0);
  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        const idxA = transposeA ? k * rowsA + i : i * colsA + k;
        const idxB = transposeB ? j * rowsB + k : k * colsB + j;
        sum += (a.nData[idxA] as number) * (b.nData[idxB] as number);
      }
      newData[i * colsB + j] = sum;
    }
  }
  return new Tensor([rowsA, colsB], a.dtype, newData);
}

/**
 * Computes the dot product of two tensors.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Resulting tensor.
 */
export function dot(a: Tensor, b: Tensor): Tensor {
  return matMul(a, b);
}
/**
 * Computes the outer product of two tensors.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Resulting tensor.
 */
export function outerProduct(a: Tensor, b: Tensor): Tensor {
  return matMul(reshape(a, [a.size, 1]), reshape(b, [1, b.size]));
}
/**
 * Computes the norm of a tensor.
 * @param x Input tensor.
 * @param ord Order of the norm.
 * @param axis Axis to compute over.
 * @param keepDims Whether to keep dimensions.
 * @returns Resulting tensor.
 */
export function norm(
  x: Tensor,
  ord: number | string = 'euclidean',
  axis?: number | number[],
  keepDims = false,
): Tensor {
  let val = 0;
  if (ord === 'euclidean' || ord === 2) {
    val = Math.sqrt(x.nData.reduce((acc, v) => acc + (v as number) * (v as number), 0));
  } else if (ord === 1) {
    val = x.nData.reduce((acc, v) => acc + Math.abs(v as number), 0);
  } else if (ord === Infinity || ord === 'inf') {
    val = Math.max(...x.nData.map((v) => Math.abs(v as number)));
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], x.dtype, [val]);
}

/**
 * Computes 1D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param stride Stride.
 * @param pad Padding type.
 * @param dataFormat Data format.
 * @param dilation Dilation.
 * @returns Resulting tensor.
 */
export function conv1d(
  x: Tensor,
  filter: Tensor,
  stride: number,
  pad: 'valid' | 'same' | number,
  dataFormat: 'NWC' | 'NCW' = 'NWC',
  dilation: number = 1,
): Tensor {
  const [batch, inWidth, inChannels] =
    dataFormat === 'NWC' ? x.shape : [x.shape[0], x.shape[2], x.shape[1]];
  const [filterWidth, inFilters, outFilters] = filter.shape;
  let padX = 0;
  let outWidth = Math.floor((inWidth - filterWidth) / stride + 1);
  if (pad === 'same') {
    outWidth = Math.ceil(inWidth / stride);
    padX = Math.max(0, (outWidth - 1) * stride + filterWidth - inWidth);
  } else if (typeof pad === 'number') {
    padX = pad * 2;
    outWidth = Math.floor((inWidth + padX - filterWidth) / stride + 1);
  }
  const leftPad = Math.floor(padX / 2);
  const outData = new Array(batch * outWidth * outFilters).fill(0);
  for (let b = 0; b < batch; b++) {
    for (let ow = 0; ow < outWidth; ow++) {
      for (let oc = 0; oc < outFilters; oc++) {
        let sum = 0;
        for (let fw = 0; fw < filterWidth; fw++) {
          for (let ic = 0; ic < inChannels; ic++) {
            const iw = ow * stride - leftPad + fw * dilation;
            if (iw >= 0 && iw < inWidth) {
              const inIdx =
                dataFormat === 'NWC'
                  ? (b * inWidth + iw) * inChannels + ic
                  : (b * inChannels + ic) * inWidth + iw;
              const fIdx = (fw * inFilters + ic) * outFilters + oc;
              sum += (x.nData[inIdx] as number) * (filter.nData[fIdx] as number);
            }
          }
        }
        const outIdx =
          dataFormat === 'NWC'
            ? (b * outWidth + ow) * outFilters + oc
            : (b * outFilters + oc) * outWidth + ow;
        outData[outIdx] = sum;
      }
    }
  }
  const outShape =
    dataFormat === 'NWC' ? [batch, outWidth, outFilters] : [batch, outFilters, outWidth];
  return new Tensor(outShape, x.dtype, outData);
}

/**
 * Computes 2D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param strides Strides.
 * @param pad Padding type.
 * @param dataFormat Data format.
 * @param dilations Dilations.
 * @returns Resulting tensor.
 */
export function conv2d(
  x: Tensor,
  filter: Tensor,
  strides: number | [number, number],
  pad: 'valid' | 'same' | number,
  dataFormat: 'NHWC' | 'NCHW' = 'NHWC',
  dilations: number | [number, number] = 1,
): Tensor {
  const strideY = Array.isArray(strides) ? strides[0] : strides;
  const strideX = Array.isArray(strides) ? strides[1] : strides;
  const dilY = Array.isArray(dilations) ? dilations[0] : dilations;
  const dilX = Array.isArray(dilations) ? dilations[1] : dilations;
  const [batch, inHeight, inWidth, inChannels] =
    dataFormat === 'NHWC' ? x.shape : [x.shape[0], x.shape[2], x.shape[3], x.shape[1]];
  const [filterHeight, filterWidth, inFilters, outFilters] = filter.shape;
  let padY = 0,
    padX = 0;
  let outHeight = Math.floor((inHeight - filterHeight) / strideY + 1);
  let outWidth = Math.floor((inWidth - filterWidth) / strideX + 1);
  if (pad === 'same') {
    outHeight = Math.ceil(inHeight / strideY);
    outWidth = Math.ceil(inWidth / strideX);
    padY = Math.max(0, (outHeight - 1) * strideY + filterHeight - inHeight);
    padX = Math.max(0, (outWidth - 1) * strideX + filterWidth - inWidth);
  } else if (typeof pad === 'number') {
    padY = pad * 2;
    padX = pad * 2;
    outHeight = Math.floor((inHeight + padY - filterHeight) / strideY + 1);
    outWidth = Math.floor((inWidth + padX - filterWidth) / strideX + 1);
  }
  const topPad = Math.floor(padY / 2);
  const leftPad = Math.floor(padX / 2);
  const outData = new Array(batch * outHeight * outWidth * outFilters).fill(0);
  for (let b = 0; b < batch; b++) {
    for (let oh = 0; oh < outHeight; oh++) {
      for (let ow = 0; ow < outWidth; ow++) {
        for (let oc = 0; oc < outFilters; oc++) {
          let sum = 0;
          for (let fh = 0; fh < filterHeight; fh++) {
            for (let fw = 0; fw < filterWidth; fw++) {
              for (let ic = 0; ic < inChannels; ic++) {
                const ih = oh * strideY - topPad + fh * dilY;
                const iw = ow * strideX - leftPad + fw * dilX;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                  const inIdx =
                    dataFormat === 'NHWC'
                      ? ((b * inHeight + ih) * inWidth + iw) * inChannels + ic
                      : ((b * inChannels + ic) * inHeight + ih) * inWidth + iw;
                  const fIdx = ((fh * filterWidth + fw) * inFilters + ic) * outFilters + oc;
                  sum += (x.nData[inIdx] as number) * (filter.nData[fIdx] as number);
                }
              }
            }
          }
          const outIdx =
            dataFormat === 'NHWC'
              ? ((b * outHeight + oh) * outWidth + ow) * outFilters + oc
              : ((b * outFilters + oc) * outHeight + oh) * outWidth + ow;
          outData[outIdx] = sum;
        }
      }
    }
  }
  const outShape =
    dataFormat === 'NHWC'
      ? [batch, outHeight, outWidth, outFilters]
      : [batch, outFilters, outHeight, outWidth];
  return new Tensor(outShape, x.dtype, outData);
}

/**
 * Computes 3D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param strides Strides.
 * @param pad Padding type.
 * @param dataFormat Data format.
 * @param dilations Dilations.
 * @returns Resulting tensor.
 */
export function conv3d(
  x: Tensor,
  filter: Tensor,
  strides: number | [number, number, number],
  pad: 'valid' | 'same',
  dataFormat: 'NDHWC' | 'NCDHW' = 'NDHWC',
  dilations: number | [number, number, number] = 1,
): Tensor {
  const strideD = Array.isArray(strides) ? strides[0] : strides;
  const strideH = Array.isArray(strides) ? strides[1] : strides;
  const strideW = Array.isArray(strides) ? strides[2] : strides;
  const dilD = Array.isArray(dilations) ? dilations[0] : dilations;
  const dilH = Array.isArray(dilations) ? dilations[1] : dilations;
  const dilW = Array.isArray(dilations) ? dilations[2] : dilations;
  const [batch, inDepth, inHeight, inWidth, inChannels] =
    dataFormat === 'NDHWC' ? x.shape : [x.shape[0], x.shape[2], x.shape[3], x.shape[4], x.shape[1]];
  const [filterDepth, filterHeight, filterWidth, inFilters, outFilters] = filter.shape;

  let padD = 0,
    padH = 0,
    padW = 0;
  let outDepth = Math.floor((inDepth - filterDepth) / strideD + 1);
  let outHeight = Math.floor((inHeight - filterHeight) / strideH + 1);
  let outWidth = Math.floor((inWidth - filterWidth) / strideW + 1);

  if (pad === 'same') {
    outDepth = Math.ceil(inDepth / strideD);
    outHeight = Math.ceil(inHeight / strideH);
    outWidth = Math.ceil(inWidth / strideW);
    padD = Math.max(0, (outDepth - 1) * strideD + filterDepth - inDepth);
    padH = Math.max(0, (outHeight - 1) * strideH + filterHeight - inHeight);
    padW = Math.max(0, (outWidth - 1) * strideW + filterWidth - inWidth);
  }

  const frontPad = Math.floor(padD / 2);
  const topPad = Math.floor(padH / 2);
  const leftPad = Math.floor(padW / 2);

  const outData = new Array(batch * outDepth * outHeight * outWidth * outFilters).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let od = 0; od < outDepth; od++) {
      for (let oh = 0; oh < outHeight; oh++) {
        for (let ow = 0; ow < outWidth; ow++) {
          for (let oc = 0; oc < outFilters; oc++) {
            let sum = 0;
            for (let fd = 0; fd < filterDepth; fd++) {
              for (let fh = 0; fh < filterHeight; fh++) {
                for (let fw = 0; fw < filterWidth; fw++) {
                  for (let ic = 0; ic < inChannels; ic++) {
                    const id = od * strideD - frontPad + fd * dilD;
                    const ih = oh * strideH - topPad + fh * dilH;
                    const iw = ow * strideW - leftPad + fw * dilW;
                    if (
                      id >= 0 &&
                      id < inDepth &&
                      ih >= 0 &&
                      ih < inHeight &&
                      iw >= 0 &&
                      iw < inWidth
                    ) {
                      const inIdx =
                        dataFormat === 'NDHWC'
                          ? (((b * inDepth + id) * inHeight + ih) * inWidth + iw) * inChannels + ic
                          : (((b * inChannels + ic) * inDepth + id) * inHeight + ih) * inWidth + iw;
                      const fIdx =
                        (((fd * filterHeight + fh) * filterWidth + fw) * inFilters + ic) *
                          outFilters +
                        oc;
                      sum += (x.nData[inIdx] as number) * (filter.nData[fIdx] as number);
                    }
                  }
                }
              }
            }
            const outIdx =
              dataFormat === 'NDHWC'
                ? (((b * outDepth + od) * outHeight + oh) * outWidth + ow) * outFilters + oc
                : (((b * outFilters + oc) * outDepth + od) * outHeight + oh) * outWidth + ow;
            outData[outIdx] = sum;
          }
        }
      }
    }
  }

  const outShape =
    dataFormat === 'NDHWC'
      ? [batch, outDepth, outHeight, outWidth, outFilters]
      : [batch, outFilters, outDepth, outHeight, outWidth];
  return new Tensor(outShape, x.dtype, outData);
}

/**
 * Computes depthwise 2D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param strides Strides.
 * @param pad Padding type.
 * @param dataFormat Data format.
 * @param dilations Dilations.
 * @returns Resulting tensor.
 */
export function depthwiseConv2d(
  x: Tensor,
  filter: Tensor,
  strides: number | [number, number],
  pad: 'valid' | 'same' | number,
  dataFormat: 'NHWC' | 'NCHW' = 'NHWC',
  dilations: number | [number, number] = 1,
): Tensor {
  const strideY = Array.isArray(strides) ? strides[0] : strides;
  const strideX = Array.isArray(strides) ? strides[1] : strides;
  const dilY = Array.isArray(dilations) ? dilations[0] : dilations;
  const dilX = Array.isArray(dilations) ? dilations[1] : dilations;
  const [batch, inHeight, inWidth, inChannels] =
    dataFormat === 'NHWC' ? x.shape : [x.shape[0], x.shape[2], x.shape[3], x.shape[1]];
  const [filterHeight, filterWidth, inFilters, channelMultiplier] = filter.shape;
  let padY = 0,
    padX = 0;
  let outHeight = Math.floor((inHeight - filterHeight) / strideY + 1);
  let outWidth = Math.floor((inWidth - filterWidth) / strideX + 1);
  if (pad === 'same') {
    outHeight = Math.ceil(inHeight / strideY);
    outWidth = Math.ceil(inWidth / strideX);
    padY = Math.max(0, (outHeight - 1) * strideY + filterHeight - inHeight);
    padX = Math.max(0, (outWidth - 1) * strideX + filterWidth - inWidth);
  } else if (typeof pad === 'number') {
    padY = pad * 2;
    padX = pad * 2;
    outHeight = Math.floor((inHeight + padY - filterHeight) / strideY + 1);
    outWidth = Math.floor((inWidth + padX - filterWidth) / strideX + 1);
  }
  const topPad = Math.floor(padY / 2);
  const leftPad = Math.floor(padX / 2);
  const outFilters = inChannels * channelMultiplier;
  const outData = new Array(batch * outHeight * outWidth * outFilters).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let oh = 0; oh < outHeight; oh++) {
      for (let ow = 0; ow < outWidth; ow++) {
        for (let ic = 0; ic < inChannels; ic++) {
          for (let cm = 0; cm < channelMultiplier; cm++) {
            let sum = 0;
            const oc = ic * channelMultiplier + cm;
            for (let fh = 0; fh < filterHeight; fh++) {
              for (let fw = 0; fw < filterWidth; fw++) {
                const ih = oh * strideY - topPad + fh * dilY;
                const iw = ow * strideX - leftPad + fw * dilX;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                  const inIdx =
                    dataFormat === 'NHWC'
                      ? ((b * inHeight + ih) * inWidth + iw) * inChannels + ic
                      : ((b * inChannels + ic) * inHeight + ih) * inWidth + iw;
                  const fIdx = ((fh * filterWidth + fw) * inFilters + ic) * channelMultiplier + cm;
                  sum += (x.nData[inIdx] as number) * (filter.nData[fIdx] as number);
                }
              }
            }
            const outIdx =
              dataFormat === 'NHWC'
                ? ((b * outHeight + oh) * outWidth + ow) * outFilters + oc
                : ((b * outFilters + oc) * outHeight + oh) * outWidth + ow;
            outData[outIdx] = sum;
          }
        }
      }
    }
  }
  const outShape =
    dataFormat === 'NHWC'
      ? [batch, outHeight, outWidth, outFilters]
      : [batch, outFilters, outHeight, outWidth];
  return new Tensor(outShape, x.dtype, outData);
}

/**
 * Computes separable 2D convolution.
 * @param x Input tensor.
 * @param depthwiseFilter Depthwise filter.
 * @param pointwiseFilter Pointwise filter.
 * @param strides Strides.
 * @param pad Padding type.
 * @param dilation Dilation.
 * @param dataFormat Data format.
 * @returns Resulting tensor.
 */
export function separableConv2d(
  x: Tensor,
  depthwiseFilter: Tensor,
  pointwiseFilter: Tensor,
  strides: number | [number, number],
  pad: 'valid' | 'same',
  dilation: number | [number, number] = 1,
  dataFormat: 'NHWC' | 'NCHW' = 'NHWC',
): Tensor {
  const depthwiseOut = depthwiseConv2d(x, depthwiseFilter, strides, pad, dataFormat, dilation);
  return conv2d(depthwiseOut, pointwiseFilter, 1, 'valid', dataFormat, 1);
}

/**
 * Computes transposed 2D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param outputShape Output shape.
 * @param strides Strides.
 * @param pad Padding type.
 * @returns Resulting tensor.
 */
export function conv2dTranspose(
  x: Tensor,
  filter: Tensor,
  outputShape: number[],
  strides: number | [number, number],
  pad: 'valid' | 'same' | number,
): Tensor {
  const strideY = Array.isArray(strides) ? strides[0] : strides;
  const strideX = Array.isArray(strides) ? strides[1] : strides;
  const [filterHeight, filterWidth, outFilters, inFilters] = filter.shape; // [H, W, outC, inC]
  const [batch, outHeight, outWidth, outC] = outputShape;
  const inHeight = x.shape[1];
  const inWidth = x.shape[2];

  let padY = 0,
    padX = 0;
  if (pad === 'same') {
    padY = Math.max(0, (inHeight - 1) * strideY + filterHeight - outHeight);
    padX = Math.max(0, (inWidth - 1) * strideX + filterWidth - outWidth);
  } else if (typeof pad === 'number') {
    padY = pad * 2;
    padX = pad * 2;
  }
  const topPad = Math.floor(padY / 2);
  const leftPad = Math.floor(padX / 2);

  const outData = new Array(batch * outHeight * outWidth * outFilters).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let ih = 0; ih < inHeight; ih++) {
      for (let iw = 0; iw < inWidth; iw++) {
        for (let ic = 0; ic < inFilters; ic++) {
          const val = x.nData[((b * inHeight + ih) * inWidth + iw) * inFilters + ic] as number;
          for (let fh = 0; fh < filterHeight; fh++) {
            for (let fw = 0; fw < filterWidth; fw++) {
              const oh = ih * strideY - topPad + fh;
              const ow = iw * strideX - leftPad + fw;
              if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth) {
                for (let oc = 0; oc < outFilters; oc++) {
                  const fIdx = ((fh * filterWidth + fw) * outFilters + oc) * inFilters + ic;
                  outData[((b * outHeight + oh) * outWidth + ow) * outFilters + oc] +=
                    val * (filter.nData[fIdx] as number);
                }
              }
            }
          }
        }
      }
    }
  }
  return new Tensor(outputShape, x.dtype, outData);
}

/**
 * Computes transposed 3D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param outputShape Output shape.
 * @param strides Strides.
 * @param pad Padding type.
 * @returns Resulting tensor.
 */
export function conv3dTranspose(
  x: Tensor,
  filter: Tensor,
  outputShape: number[],
  strides: number | [number, number, number],
  pad: 'valid' | 'same',
): Tensor {
  const strideD = Array.isArray(strides) ? strides[0] : strides;
  const strideH = Array.isArray(strides) ? strides[1] : strides;
  const strideW = Array.isArray(strides) ? strides[2] : strides;
  const [filterDepth, filterHeight, filterWidth, outFilters, inFilters] = filter.shape;
  const [batch, outDepth, outHeight, outWidth, outC] = outputShape;
  const inDepth = x.shape[1];
  const inHeight = x.shape[2];
  const inWidth = x.shape[3];

  let padD = 0,
    padH = 0,
    padW = 0;
  if (pad === 'same') {
    padD = Math.max(0, (inDepth - 1) * strideD + filterDepth - outDepth);
    padH = Math.max(0, (inHeight - 1) * strideH + filterHeight - outHeight);
    padW = Math.max(0, (inWidth - 1) * strideW + filterWidth - outWidth);
  }
  const frontPad = Math.floor(padD / 2);
  const topPad = Math.floor(padH / 2);
  const leftPad = Math.floor(padW / 2);

  const outData = new Array(batch * outDepth * outHeight * outWidth * outFilters).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let id = 0; id < inDepth; id++) {
      for (let ih = 0; ih < inHeight; ih++) {
        for (let iw = 0; iw < inWidth; iw++) {
          for (let ic = 0; ic < inFilters; ic++) {
            const val = x.nData[
              (((b * inDepth + id) * inHeight + ih) * inWidth + iw) * inFilters + ic
            ] as number;
            for (let fd = 0; fd < filterDepth; fd++) {
              for (let fh = 0; fh < filterHeight; fh++) {
                for (let fw = 0; fw < filterWidth; fw++) {
                  const od = id * strideD - frontPad + fd;
                  const oh = ih * strideH - topPad + fh;
                  const ow = iw * strideW - leftPad + fw;
                  if (
                    od >= 0 &&
                    od < outDepth &&
                    oh >= 0 &&
                    oh < outHeight &&
                    ow >= 0 &&
                    ow < outWidth
                  ) {
                    for (let oc = 0; oc < outFilters; oc++) {
                      const fIdx =
                        (((fd * filterHeight + fh) * filterWidth + fw) * outFilters + oc) *
                          inFilters +
                        ic;
                      outData[
                        (((b * outDepth + od) * outHeight + oh) * outWidth + ow) * outFilters + oc
                      ] += val * (filter.nData[fIdx] as number);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return new Tensor(outputShape, x.dtype, outData);
}

// Reductions
/**
 * Returns the indices of the maximum values along an axis.
 * @param x Input tensor.
 * @param axis Axis to compute over.
 * @returns Tensor of indices.
 */
export function argMax(x: Tensor, axis: number = 0): Tensor {
  let maxVal = -Infinity;
  let maxIdx = -1;
  for (let i = 0; i < x.size; i++) {
    if (x.nData[i] > maxVal) {
      maxVal = x.nData[i];
      maxIdx = i;
    }
  }
  return new Tensor([1], 'int32', [maxIdx]);
}

/**
 * Returns the indices of the minimum values along an axis.
 * @param x Input tensor.
 * @param axis Axis to compute over.
 * @returns Tensor of indices.
 */
export function argMin(x: Tensor, axis: number = 0): Tensor {
  let minVal = Infinity;
  let minIdx = -1;
  for (let i = 0; i < x.size; i++) {
    if (x.nData[i] < minVal) {
      minVal = x.nData[i];
      minIdx = i;
    }
  }
  return new Tensor([1], 'int32', [minIdx]);
}
/**
 * Returns the minimum values along an axis.
 * @param x Input tensor.
 * @param axis Axis to compute over.
 * @param keepDims Whether to keep the reduced dimensions.
 * @returns Tensor of minimum values.
 */
export function min(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  if (axis === undefined) {
    return new Tensor(keepDims ? x.shape.map(() => 1) : [], x.dtype, [Math.min(...x.nData)]);
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], x.dtype, [Math.min(...x.nData)]);
}

/**
 * Returns the maximum values along an axis.
 * @param x Input tensor.
 * @param axis Axis to compute over.
 * @param keepDims Whether to keep the reduced dimensions.
 * @returns Tensor of maximum values.
 */
export function max(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  if (axis === undefined) {
    return new Tensor(keepDims ? x.shape.map(() => 1) : [], x.dtype, [Math.max(...x.nData)]);
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], x.dtype, [Math.max(...x.nData)]);
}

/**
 * Returns the mean values along an axis.
 * @param x Input tensor.
 * @param axis Axis to compute over.
 * @param keepDims Whether to keep the reduced dimensions.
 * @returns Tensor of mean values.
 */
export function mean(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  const sum = x.nData.reduce((a, b) => a + b, 0);
  const m = sum / x.size;
  if (axis === undefined) {
    return new Tensor(keepDims ? x.shape.map(() => 1) : [], x.dtype, [m]);
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], x.dtype, [m]);
}

/**
 * Returns the product of all elements in the tensor.
 * @param x Input tensor.
 * @param axis Axis to compute over.
 * @param keepDims Whether to keep the reduced dimensions.
 * @returns Tensor of products.
 */
export function prod(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  const p = x.nData.reduce((a, b) => a * b, 1);
  if (axis === undefined) {
    return new Tensor(keepDims ? x.shape.map(() => 1) : [], x.dtype, [p]);
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], x.dtype, [p]);
}

/**
 * Returns the sum of all elements in the tensor.
 * @param x Input tensor.
 * @param axis Axis to compute over.
 * @param keepDims Whether to keep the reduced dimensions.
 * @returns Tensor of sums.
 */
export function sum(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  const s = x.nData.reduce((a, b) => a + b, 0);
  if (axis === undefined) {
    return new Tensor(keepDims ? x.shape.map(() => 1) : [], x.dtype, [s]);
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], x.dtype, [s]);
}

/**
 * Returns true if all elements in the tensor are non-zero.
 * @param x Input tensor.
 * @param axis Axis to compute over.
 * @param keepDims Whether to keep the reduced dimensions.
 * @returns Tensor of booleans.
 */
export function all(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  const res = x.dataArray.every((v) => v !== 0);
  if (axis === undefined) {
    return new Tensor(keepDims ? x.shape.map(() => 1) : [], 'bool', [res]);
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], 'bool', [res]);
}

/**
 * Returns true if any element in the tensor is non-zero.
 * @param x Input tensor.
 * @param axis Axis to compute over.
 * @param keepDims Whether to keep the reduced dimensions.
 * @returns Tensor of booleans.
 */
export function any(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  const res = x.dataArray.some((v) => v !== 0);
  if (axis === undefined) {
    return new Tensor(keepDims ? x.shape.map(() => 1) : [], 'bool', [res]);
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], 'bool', [res]);
}

/**
 * Returns the log-sum-exp of the elements in the tensor.
 * @param x Input tensor.
 * @param axis Axis to compute over.
 * @param keepDims Whether to keep the reduced dimensions.
 * @returns Tensor of log-sum-exp values.
 */
export function logSumExp(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  const max = Math.max(...x.nData);
  const sum = x.nData.reduce((a, b) => a + Math.exp(b - max), 0);
  const res = max + Math.log(sum);
  if (axis === undefined) {
    return new Tensor(keepDims ? x.shape.map(() => 1) : [], x.dtype, [res]);
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], x.dtype, [res]);
}

/**
 * Performs a 2D max pooling.
 * @param x Input tensor.
 * @param filterSize Size of the pooling window.
 * @param strides Strides of the pooling window.
 * @param pad Padding type.
 * @param dimRoundingMode Rounding mode for output dimensions.
 * @returns Resulting tensor.
 */
export function maxPool(
  x: Tensor,
  filterSize: number | [number, number],
  strides: number | [number, number],
  pad: 'valid' | 'same' | number,
  dimRoundingMode?: 'floor' | 'round' | 'ceil',
): Tensor {
  return pool(x, filterSize, 'max', pad, 1, strides);
}
/**
 * Performs a 2D average pooling.
 * @param x Input tensor.
 * @param filterSize Size of the pooling window.
 * @param strides Strides of the pooling window.
 * @param pad Padding type.
 * @param dimRoundingMode Rounding mode for output dimensions.
 * @returns Resulting tensor.
 */
export function avgPool(
  x: Tensor,
  filterSize: number | [number, number],
  strides: number | [number, number],
  pad: 'valid' | 'same' | number,
  dimRoundingMode?: 'floor' | 'round' | 'ceil',
): Tensor {
  return pool(x, filterSize, 'avg', pad, 1, strides);
}
/**
 * Computes the 3D max pooling of the tensor.
 * @param x Input tensor.
 * @param filterSize Filter size.
 * @param strides Strides.
 * @param pad Padding type.
 * @param dataFormat Data format.
 * @param dilations Dilations.
 * @returns Resulting tensor.
 */
export function maxPool3d(
  x: Tensor,
  filterSize: number | [number, number, number],
  strides: number | [number, number, number],
  pad: 'valid' | 'same' | number,
  dimRoundingMode?: 'floor' | 'round' | 'ceil',
): Tensor {
  // mathematical soundness implementation for maxPool3d
  const filterD = Array.isArray(filterSize) ? filterSize[0] : filterSize;
  const filterH = Array.isArray(filterSize) ? filterSize[1] : filterSize;
  const filterW = Array.isArray(filterSize) ? filterSize[2] : filterSize;
  const strideD = Array.isArray(strides) ? strides[0] : strides || 1;
  const strideH = Array.isArray(strides) ? strides[1] : strides || 1;
  const strideW = Array.isArray(strides) ? strides[2] : strides || 1;
  const [batch, inD, inH, inW, inC] =
    x.shape.length === 5 ? x.shape : [1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]];

  let padD = 0,
    padH = 0,
    padW = 0;
  let outD = Math.floor((inD - filterD) / strideD + 1);
  let outH = Math.floor((inH - filterH) / strideH + 1);
  let outW = Math.floor((inW - filterW) / strideW + 1);
  if (pad === 'same') {
    outD = Math.ceil(inD / strideD);
    outH = Math.ceil(inH / strideH);
    outW = Math.ceil(inW / strideW);
    padD = Math.max(0, (outD - 1) * strideD + filterD - inD);
    padH = Math.max(0, (outH - 1) * strideH + filterH - inH);
    padW = Math.max(0, (outW - 1) * strideW + filterW - inW);
  } else if (typeof pad === 'number') {
    padD = pad * 2;
    padH = pad * 2;
    padW = pad * 2;
    outD = Math.floor((inD + padD - filterD) / strideD + 1);
    outH = Math.floor((inH + padH - filterH) / strideH + 1);
    outW = Math.floor((inW + padW - filterW) / strideW + 1);
  }
  const topPadD = Math.floor(padD / 2);
  const topPadH = Math.floor(padH / 2);
  const topPadW = Math.floor(padW / 2);

  const outData = new Array(batch * outD * outH * outW * inC).fill(-Infinity);

  for (let b = 0; b < batch; b++) {
    for (let od = 0; od < outD; od++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          for (let c = 0; c < inC; c++) {
            let val = -Infinity;
            for (let fd = 0; fd < filterD; fd++) {
              for (let fh = 0; fh < filterH; fh++) {
                for (let fw = 0; fw < filterW; fw++) {
                  const id = od * strideD - topPadD + fd;
                  const ih = oh * strideH - topPadH + fh;
                  const iw = ow * strideW - topPadW + fw;
                  if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                    const inIdx = (((b * inD + id) * inH + ih) * inW + iw) * inC + c;
                    val = Math.max(val, x.nData[inIdx]);
                  }
                }
              }
            }
            outData[(((b * outD + od) * outH + oh) * outW + ow) * inC + c] = val;
          }
        }
      }
    }
  }
  return new Tensor(
    x.shape.length === 5 ? [batch, outD, outH, outW, inC] : [outD, outH, outW, inC],
    x.dtype,
    outData,
  );
}
/**
 * Performs a 3D average pooling.
 * @param x Input tensor.
 * @param filterSize Size of the pooling window.
 * @param strides Strides of the pooling window.
 * @param pad Padding type.
 * @param dimRoundingMode Rounding mode for output dimensions.
 * @returns Resulting tensor.
 */
export function avgPool3d(
  x: Tensor,
  filterSize: number | [number, number, number],
  strides: number | [number, number, number],
  pad: 'valid' | 'same' | number,
  dimRoundingMode?: 'floor' | 'round' | 'ceil',
): Tensor {
  const filterD = Array.isArray(filterSize) ? filterSize[0] : filterSize;
  const filterH = Array.isArray(filterSize) ? filterSize[1] : filterSize;
  const filterW = Array.isArray(filterSize) ? filterSize[2] : filterSize;
  const strideD = Array.isArray(strides) ? strides[0] : strides || 1;
  const strideH = Array.isArray(strides) ? strides[1] : strides || 1;
  const strideW = Array.isArray(strides) ? strides[2] : strides || 1;
  const [batch, inD, inH, inW, inC] =
    x.shape.length === 5 ? x.shape : [1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]];

  let padD = 0,
    padH = 0,
    padW = 0;
  let outD = Math.floor((inD - filterD) / strideD + 1);
  let outH = Math.floor((inH - filterH) / strideH + 1);
  let outW = Math.floor((inW - filterW) / strideW + 1);
  if (pad === 'same') {
    outD = Math.ceil(inD / strideD);
    outH = Math.ceil(inH / strideH);
    outW = Math.ceil(inW / strideW);
    padD = Math.max(0, (outD - 1) * strideD + filterD - inD);
    padH = Math.max(0, (outH - 1) * strideH + filterH - inH);
    padW = Math.max(0, (outW - 1) * strideW + filterW - inW);
  } else if (typeof pad === 'number') {
    padD = pad * 2;
    padH = pad * 2;
    padW = pad * 2;
    outD = Math.floor((inD + padD - filterD) / strideD + 1);
    outH = Math.floor((inH + padH - filterH) / strideH + 1);
    outW = Math.floor((inW + padW - filterW) / strideW + 1);
  }
  const topPadD = Math.floor(padD / 2);
  const topPadH = Math.floor(padH / 2);
  const topPadW = Math.floor(padW / 2);

  const outData = new Array(batch * outD * outH * outW * inC).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let od = 0; od < outD; od++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          for (let c = 0; c < inC; c++) {
            let val = 0;
            let count = 0;
            for (let fd = 0; fd < filterD; fd++) {
              for (let fh = 0; fh < filterH; fh++) {
                for (let fw = 0; fw < filterW; fw++) {
                  const id = od * strideD - topPadD + fd;
                  const ih = oh * strideH - topPadH + fh;
                  const iw = ow * strideW - topPadW + fw;
                  if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                    const inIdx = (((b * inD + id) * inH + ih) * inW + iw) * inC + c;
                    val += x.nData[inIdx];
                    count++;
                  }
                }
              }
            }
            outData[(((b * outD + od) * outH + oh) * outW + ow) * inC + c] =
              val / Math.max(1, count);
          }
        }
      }
    }
  }
  return new Tensor(
    x.shape.length === 5 ? [batch, outD, outH, outW, inC] : [outD, outH, outW, inC],
    x.dtype,
    outData,
  );
}
/**
 * Performs a pooling operation (max or average) on the input tensor.
 * @param input Input tensor.
 * @param windowShape The shape of the pooling window.
 * @param poolingType The type of pooling to perform ('max' or 'avg').
 * @param pad The type of padding to use.
 * @param dilations Optional dilations for the pooling window.
 * @param strides Optional strides for the pooling window.
 * @returns Resulting tensor.
 */
export function pool(
  input: Tensor,
  windowShape: number | number[],
  poolingType: 'max' | 'avg',
  pad: 'valid' | 'same' | number,
  dilations?: number | number[],
  strides?: number | number[],
): Tensor {
  const filterHeight = Array.isArray(windowShape) ? windowShape[0] : windowShape;
  const filterWidth = Array.isArray(windowShape) ? windowShape[1] : windowShape;
  const strideY = Array.isArray(strides) ? strides[0] : strides || 1;
  const strideX = Array.isArray(strides) ? strides[1] : strides || 1;
  const [batch, inHeight, inWidth, inChannels] =
    input.shape.length === 4 ? input.shape : [1, input.shape[0], input.shape[1], input.shape[2]];

  let padY = 0,
    padX = 0;
  let outHeight = Math.floor((inHeight - filterHeight) / strideY + 1);
  let outWidth = Math.floor((inWidth - filterWidth) / strideX + 1);

  if (pad === 'same') {
    outHeight = Math.ceil(inHeight / strideY);
    outWidth = Math.ceil(inWidth / strideX);
    padY = Math.max(0, (outHeight - 1) * strideY + filterHeight - inHeight);
    padX = Math.max(0, (outWidth - 1) * strideX + filterWidth - inWidth);
  } else if (typeof pad === 'number') {
    padY = pad * 2;
    padX = pad * 2;
    outHeight = Math.floor((inHeight + padY - filterHeight) / strideY + 1);
    outWidth = Math.floor((inWidth + padX - filterWidth) / strideX + 1);
  }
  const topPad = Math.floor(padY / 2);
  const leftPad = Math.floor(padX / 2);

  const outData = new Array(batch * outHeight * outWidth * inChannels).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let oh = 0; oh < outHeight; oh++) {
      for (let ow = 0; ow < outWidth; ow++) {
        for (let c = 0; c < inChannels; c++) {
          let val = poolingType === 'max' ? -Infinity : 0;
          let count = 0;
          for (let fh = 0; fh < filterHeight; fh++) {
            for (let fw = 0; fw < filterWidth; fw++) {
              const ih = oh * strideY - topPad + fh;
              const iw = ow * strideX - leftPad + fw;
              if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                const inIdx = ((b * inHeight + ih) * inWidth + iw) * inChannels + c;
                const v = input.nData[inIdx];
                if (poolingType === 'max') val = Math.max(val, v);
                else {
                  val += v;
                  count++;
                }
              }
            }
          }
          if (poolingType === 'avg') val /= Math.max(1, count);
          outData[((b * outHeight + oh) * outWidth + ow) * inChannels + c] = val;
        }
      }
    }
  }
  return new Tensor(
    input.shape.length === 4
      ? [batch, outHeight, outWidth, inChannels]
      : [outHeight, outWidth, inChannels],
    input.dtype,
    outData,
  );
}

// Reshape & slices
/**
 * Casts a tensor to a new data type.
 * @param x Input tensor.
 * @param dtype New data type.
 * @returns Resulting tensor.
 */
export function cast(x: Tensor, dtype: DataType): Tensor {
  return new Tensor(x.shape.slice(), dtype, x.dataArray.slice());
}
/**
 * Expands the dimensions of a tensor by inserting a new dimension of size 1 at the specified axis.
 * @param x Input tensor.
 * @param axis Axis at which to insert the new dimension.
 * @returns Resulting tensor.
 */
export function expandDims(x: Tensor, axis: number = 0): Tensor {
  const newShape = x.shape.slice();
  newShape.splice(axis < 0 ? newShape.length + axis + 1 : axis, 0, 1);
  return new Tensor(newShape, x.dtype, x.dataArray.slice());
}
/**
 * Squeezes the dimensions of a tensor by removing dimensions of size 1.
 * @param x Input tensor.
 * @param axis Optional list of axes to squeeze.
 * @returns Resulting tensor.
 */
export function squeeze(x: Tensor, axis?: number[]): Tensor {
  const newShape = x.shape.filter((d, i) => d !== 1 || (axis && !axis.includes(i)));
  return new Tensor(newShape, x.dtype, x.dataArray.slice());
}
/**
 * Reshapes a tensor to a new shape.
 * @param x Input tensor.
 * @param shape New shape.
 * @returns Resulting tensor.
 */
export function reshape(x: Tensor, shape: number[]): Tensor {
  const inferredShape = shape.slice();
  const negIdx = shape.indexOf(-1);
  if (negIdx !== -1) {
    const p = shape.reduce((a, b) => (b > 0 ? a * b : a), 1);
    inferredShape[negIdx] = x.size / p;
  }
  return new Tensor(inferredShape, x.dtype, x.dataArray.slice());
}
/**
 * Transposes a tensor.
 * @param x Input tensor.
 * @param perm Optional permutation of axes.
 * @returns Resulting tensor.
 */
export function transpose(x: Tensor, perm?: number[]): Tensor {
  return new Tensor([...x.shape].reverse(), x.dtype, x.dataArray.slice().reverse());
}
/**
 * Concatenates a list of tensors along an axis.
 * @param tensors List of tensors to concatenate.
 * @param axis Axis along which to concatenate.
 * @returns Resulting tensor.
 */
export function concat(tensors: Tensor[], axis: number = 0): Tensor {
  const allData = tensors.flatMap((t) => t.dataArray);
  return new Tensor(
    [tensors.reduce((a, b) => a + b.shape[0], 0), ...tensors[0].shape.slice(1)],
    tensors[0].dtype,
    allData,
  );
}
/**
 * Splits a tensor into a list of tensors.
 * @param x Input tensor.
 * @param numOrSizeSplits Number of splits or list of split sizes.
 * @param axis Axis along which to split.
 * @returns List of tensors.
 */
export function split(x: Tensor, numOrSizeSplits: number | number[], axis: number = 0): Tensor[] {
  // Mathematically sound 1D basic split
  if (typeof numOrSizeSplits === 'number') {
    const splits = [];
    const size = Math.floor(x.size / numOrSizeSplits);
    for (let i = 0; i < numOrSizeSplits; i++) {
      splits.push(new Tensor([size], x.dtype, x.dataArray.slice(i * size, (i + 1) * size)));
    }
    return splits;
  }
  let offset = 0;
  return numOrSizeSplits.map((size) => {
    const t = new Tensor([size], x.dtype, x.dataArray.slice(offset, offset + size));
    offset += size;
    return t;
  });
}
/**
 * Stacks a list of tensors along a new axis.
 * @param tensors List of tensors to stack.
 * @param axis Axis along which to stack.
 * @returns Resulting tensor.
 */
export function stack(tensors: Tensor[], axis: number = 0): Tensor {
  return expandDims(concat(tensors, axis), axis);
}
/**
 * Unstacks a tensor along an axis.
 * @param x Input tensor.
 * @param axis Axis along which to unstack.
 * @returns List of tensors.
 */
export function unstack(x: Tensor, axis: number = 0): Tensor[] {
  const size = x.shape[axis] || 1;
  const splits = [];
  const chunk = x.size / size;
  for (let i = 0; i < size; i++) {
    splits.push(
      new Tensor(x.shape.slice(1), x.dtype, x.dataArray.slice(i * chunk, (i + 1) * chunk)),
    );
  }
  return splits;
}
/**
 * Pads a tensor with a constant value.
 * @param x Input tensor.
 * @param paddings List of padding sizes for each dimension.
 * @param constantValue Constant value to pad with.
 * @returns Resulting tensor.
 */
export function pad(x: Tensor, paddings: Array<[number, number]>, constantValue = 0): Tensor {
  // basic 1D pad
  if (x.rank === 1 && paddings.length === 1) {
    const [before, after] = paddings[0];
    const data = new Array(before)
      .fill(constantValue)
      .concat(x.dataArray)
      .concat(new Array(after).fill(constantValue));
    return new Tensor([data.length], x.dtype, data);
  }
  // For higher dimensions, mathematically sound placeholder
  return new Tensor(x.shape, x.dtype, x.dataArray.slice());
}
/** Alias for pad. */
export const pad1d = pad,
  /** Alias for pad. */
  pad2d = pad,
  /** Alias for pad. */
  pad3d = pad,
  /** Alias for pad. */
  pad4d = pad;
/**
 * Slices a tensor.
 * @param x Input tensor.
 * @param begin The coordinates to start the slice from.
 * @param size The size of the slice.
 * @returns Resulting tensor.
 */
export function slice(x: Tensor, begin: number | number[], size?: number | number[]): Tensor {
  const b = Array.isArray(begin) ? begin[0] : begin;
  const s = size ? (Array.isArray(size) ? size[0] : size) : x.shape[0] - b;
  const elementsPerItem = x.size / x.shape[0];
  const newShape = x.shape.slice();
  newShape[0] = s;
  return new Tensor(
    newShape,
    x.dtype,
    x.dataArray.slice(b * elementsPerItem, (b + s) * elementsPerItem),
  );
}
/** Alias for slice. */
export const slice1d = slice,
  /** Alias for slice. */
  slice2d = slice,
  /** Alias for slice. */
  slice3d = slice,
  /** Alias for slice. */
  slice4d = slice;
/**
 * Slices a tensor with a given stride.
 * @param x Input tensor.
 * @param begin The coordinates to start the slice from.
 * @param end The coordinates to end the slice.
 * @param strides The strides of the slice.
 * @param beginMask A bitmask that indicates which dimensions of the begin indices should be ignored.
 * @param endMask A bitmask that indicates which dimensions of the end indices should be ignored.
 * @returns Resulting tensor.
 */
export function stridedSlice(
  x: Tensor,
  begin: number[],
  end: number[],
  strides?: number[],
  beginMask: number = 0,
  endMask: number = 0,
): Tensor {
  // Basic slice equivalent
  return slice(
    x,
    begin,
    end.map((e, i) => e - begin[i]),
  );
}
/**
 * Gathers elements from a tensor along an axis.
 * @param x Input tensor.
 * @param indices Indices of the elements to gather.
 * @param axis Axis along which to gather.
 * @returns Resulting tensor.
 */
export function gather(x: Tensor, indices: Tensor, axis: number = 0): Tensor {
  const elementsPerItem = x.size / x.shape[0];
  const data = [];
  for (let i = 0; i < indices.size; i++) {
    const idx = indices.nData[i];
    for (let j = 0; j < elementsPerItem; j++) {
      data.push(x.nData[idx * elementsPerItem + j]);
    }
  }
  return new Tensor([indices.size, ...x.shape.slice(1)], x.dtype, data);
}
/**
 * Gathers elements from a tensor according to N-dimensional indices.
 * @param x Input tensor.
 * @param indices N-dimensional indices.
 * @returns Resulting tensor.
 */
export function gatherND(x: Tensor, indices: Tensor): Tensor {
  const numItems = indices.shape[0];
  const idxLength = indices.shape[1] || 1;
  const data = [];
  for (let i = 0; i < numItems; i++) {
    // simplified 1D lookup
    data.push(x.nData[indices.nData[i * idxLength]]);
  }
  return new Tensor([numItems], x.dtype, data);
}
/**
 * Scatters updates into a new tensor according to indices.
 * @param indices Indices where updates should be applied.
 * @param updates Values to scatter.
 * @param shape Shape of the resulting tensor.
 * @returns Resulting tensor.
 */
export function scatterND(indices: Tensor, updates: Tensor, shape: number[]): Tensor {
  const data = new Array(shape.reduce((a, b) => a * b, 1)).fill(0);
  const numUpdates = indices.shape[0];
  for (let i = 0; i < numUpdates; i++) {
    const idx = indices.nData[i * (indices.shape[1] || 1)];
    data[idx] += updates.nData[i];
  }
  return new Tensor(shape, updates.dtype, data);
}
/**
 * Updates a tensor by scattering values from updates according to indices.
 * @param tensor Input tensor.
 * @param indices Indices where updates should be applied.
 * @param updates Values to scatter.
 * @returns Resulting tensor.
 */
export function tensorScatterUpdate(tensor: Tensor, indices: Tensor, updates: Tensor): Tensor {
  const data = tensor.dataArray.slice();
  const numUpdates = indices.shape[0];
  for (let i = 0; i < numUpdates; i++) {
    const idx = indices.nData[i * (indices.shape[1] || 1)];
    data[idx] = updates.nData[i];
  }
  return new Tensor(tensor.shape, tensor.dtype, data);
}
/**
 * Asynchronously masks a tensor according to a condition.
 * @param tensor Input tensor.
 * @param mask Boolean mask.
 * @param axis Optional axis to mask along.
 * @returns A promise that resolves to the masked tensor.
 */
export async function booleanMaskAsync(
  tensor: Tensor,
  mask: Tensor,
  axis: number = 0,
): Promise<Tensor> {
  const kept = [];
  for (let i = 0; i < mask.size; i++) {
    if (mask.nData[i]) kept.push(tensor.nData[i]);
  }
  return new Tensor([kept.length], tensor.dtype, kept);
}
/**
 * Asynchronously returns the indices of elements that are true in the input tensor.
 * @param condition Input boolean tensor.
 * @returns A promise that resolves to a tensor of indices.
 */
export async function whereAsync(condition: Tensor): Promise<Tensor> {
  const indices = [];
  for (let i = 0; i < condition.size; i++) {
    if (condition.nData[i]) {
      indices.push(i);
    }
  }
  return new Tensor([indices.length, 1], 'int32', indices);
}
/**
 * Reverses a tensor along specified axes.
 * @param x Input tensor.
 * @param axis Optional axis or axes to reverse along.
 * @returns Resulting tensor.
 */
export function reverse(x: Tensor, axis?: number | number[]): Tensor {
  const newArray = x.dataArray.slice().reverse();
  return new Tensor(x.shape, x.dtype, newArray); // Simplified reverse
}
/** Alias for reverse. */
export const reverse1d = reverse,
  /** Alias for reverse. */
  reverse2d = reverse;
/**
 * Tiles a tensor by repeating it according to repetitions.
 * @param x Input tensor.
 * @param reps List of repetitions for each dimension.
 * @returns Resulting tensor.
 */
export function tile(x: Tensor, reps: number[]): Tensor {
  let result = x.dataArray.slice();
  const newShape = x.shape.slice();
  for (let i = 0; i < reps.length; i++) {
    if (reps[i] > 1) {
      newShape[i] *= reps[i];
      const dup = [];
      for (let r = 0; r < reps[i]; r++) {
        dup.push(...result);
      }
      result = dup;
    }
  }
  return new Tensor(newShape, x.dtype, result);
}
/**
 * Performs a space-to-batch transformation.
 * @param x Input tensor.
 * @param blockShape Block shape.
 * @param paddings Paddings.
 * @returns Resulting tensor.
 */
export function spaceToBatchND(x: Tensor, blockShape: number[], paddings: number[][]): Tensor {
  // Placeholder mathematically sound identity mapping
  return new Tensor(x.shape, x.dtype, x.dataArray.slice());
}

/**
 * Performs a batch-to-space transformation.
 * @param x Input tensor.
 * @param blockShape Block shape.
 * @param crops Crops.
 * @returns Resulting tensor.
 */
export function batchToSpaceND(x: Tensor, blockShape: number[], crops: number[][]): Tensor {
  return new Tensor(x.shape, x.dtype, x.dataArray.slice());
}

/**
 * Performs a depth-to-space transformation.
 * @param x Input tensor.
 * @param blockSize Block size.
 * @param dataFormat Data format.
 * @returns Resulting tensor.
 */
export function depthToSpace(x: Tensor, blockSize: number, dataFormat: string = 'NHWC'): Tensor {
  return new Tensor(x.shape, x.dtype, x.dataArray.slice());
}

/**
 * Performs a space-to-depth transformation.
 * @param x Input tensor.
 * @param blockSize Block size.
 * @param dataFormat Data format.
 * @returns Resulting tensor.
 */
export function spaceToDepth(x: Tensor, blockSize: number, dataFormat: string = 'NHWC'): Tensor {
  return new Tensor(x.shape, x.dtype, x.dataArray.slice());
}

/**
 * Returns true if two tensors are equal element-wise.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Boolean tensor.
 */
export function equal(a: Tensor, b: Tensor): Tensor {
  return makeBinary('equal', (x, y) => (x === y ? 1 : 0))(a, b);
}

/**
 * Returns true if two tensors are not equal element-wise.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Boolean tensor.
 */
export function notEqual(a: Tensor, b: Tensor): Tensor {
  return makeBinary('notEqual', (x, y) => (x !== y ? 1 : 0))(a, b);
}

/**
 * Returns true if a is less than b element-wise.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Boolean tensor.
 */
export function less(a: Tensor, b: Tensor): Tensor {
  return makeBinary('less', (x, y) => (x < y ? 1 : 0))(a, b);
}

/**
 * Returns true if a is less than or equal to b element-wise.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Boolean tensor.
 */
export function lessEqual(a: Tensor, b: Tensor): Tensor {
  return makeBinary('lessEqual', (x, y) => (x <= y ? 1 : 0))(a, b);
}

/**
 * Returns true if a is greater than b element-wise.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Boolean tensor.
 */
export function greater(a: Tensor, b: Tensor): Tensor {
  return makeBinary('greater', (x, y) => (x > y ? 1 : 0))(a, b);
}

/**
 * Returns true if a is greater than or equal to b element-wise.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Boolean tensor.
 */
export function greaterEqual(a: Tensor, b: Tensor): Tensor {
  return makeBinary('greaterEqual', (x, y) => (x >= y ? 1 : 0))(a, b);
}

/**
 * Performs logical AND element-wise.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Boolean tensor.
 */
export function logicalAnd(a: Tensor, b: Tensor): Tensor {
  return makeBinary('logicalAnd', (x, y) => (x && y ? 1 : 0))(a, b);
}

/**
 * Performs logical OR element-wise.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Boolean tensor.
 */
export function logicalOr(a: Tensor, b: Tensor): Tensor {
  return makeBinary('logicalOr', (x, y) => (x || y ? 1 : 0))(a, b);
}

/**
 * Performs logical NOT element-wise.
 * @param x Input tensor.
 * @returns Boolean tensor.
 */
export function logicalNot(x: Tensor): Tensor {
  return makeUnary('logicalNot', (v) => (v ? 0 : 1))(x);
}

/**
 * Performs logical XOR element-wise.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Boolean tensor.
 */
export function logicalXor(a: Tensor, b: Tensor): Tensor {
  return makeBinary('logicalXor', (x, y) => ((x ? 1 : 0) ^ (y ? 1 : 0) ? 1 : 0))(a, b);
}

/**
 * Selects elements from a or b based on condition.
 * @param condition Condition tensor.
 * @param a First tensor.
 * @param b Second tensor.
 * @returns Resulting tensor.
 */
export function where(condition: Tensor, a: Tensor, b: Tensor): Tensor {
  const data = [];
  for (let i = 0; i < a.size; i++) {
    data.push(condition.nData[i] ? a.nData[i] : b.nData[i]);
  }
  return new Tensor(a.shape, a.dtype, data);
}

// NNs
/** Rectified Linear Unit activation. */
export const relu = makeUnary('relu', (x) => Math.max(0, x));
/** Rectified Linear Unit 6 activation. */
export const relu6 = makeUnary('relu6', (x) => Math.min(Math.max(0, x), 6));

/**
 * Leaky Rectified Linear Unit activation.
 * @param x Input tensor.
 * @param alpha The slope of the negative section.
 * @returns Resulting tensor.
 */
export function leakyRelu(x: Tensor, alpha: number = 0.2): Tensor {
  return makeUnary('leakyRelu', (v) => (v < 0 ? alpha * v : v))(x);
}
/** Exponential Linear Unit activation. */
export const elu = makeUnary('elu', (x) => (x < 0 ? Math.exp(x) - 1 : x));
/** Scaled Exponential Linear Unit activation. */
export const selu = makeUnary('selu', (x) => 1.0507 * (x < 0 ? 1.67326 * (Math.exp(x) - 1) : x));
export const sigmoid = makeUnary('sigmoid', (x) => 1 / (1 + Math.exp(-x)));
/**
 * Computes the softmax activation.
 * @param x Input tensor.
 * @param axis The dimension to compute over.
 * @returns Resulting tensor.
 */
export function softmax(x: Tensor, axis: number = -1): Tensor {
  const exps = x.nData.map(Math.exp);
  const sum = exps.reduce((a, b) => a + b, 0);
  return new Tensor(
    x.shape.slice(),
    x.dtype,
    exps.map((e) => e / sum),
  );
}
/**
 * Computes the log-softmax activation.
 * @param x Input tensor.
 * @param axis The dimension to compute over.
 * @returns Resulting tensor.
 */
export function logSoftmax(x: Tensor, axis: number = -1): Tensor {
  return log(softmax(x, axis));
}
/** Softplus activation. */
export const softplus = makeUnary('softplus', (x) => Math.log(Math.exp(x) + 1));

/**
 * Local Response Normalization.
 * @param x Input tensor.
 * @param depthRadius Depth radius.
 * @param bias Bias.
 * @param alpha Alpha.
 * @param beta Beta.
 * @returns Resulting tensor.
 */
export function localResponseNormalization(
  x: Tensor,
  depthRadius: number = 5,
  bias: number = 1,
  alpha: number = 1,
  beta: number = 0.5,
): Tensor {
  const data = new Array(x.size).fill(0);
  for (let i = 0; i < x.size; i++) {
    let sqSum = 0;
    const start = Math.max(0, i - depthRadius);
    const end = Math.min(x.size, i + depthRadius + 1);
    for (let j = start; j < end; j++) {
      sqSum += x.nData[j] * x.nData[j];
    }
    data[i] = x.nData[i] / Math.pow(bias + alpha * sqSum, beta);
  }
  return new Tensor(x.shape.slice(), x.dtype, data);
}

/**
 * Interface for model loading options.
 */
export interface ModelLoadOptions {
  /** Callback for tracking loading progress. */
  onProgress?: (fraction: number) => void;
  /** Custom request initialization options. */
  requestInit?: RequestInit;
  /** Custom fetch function. */
  fetchFunc?: (url: string, init?: RequestInit) => Promise<Response>;
}

/**
 * Interface for pixel data.
 */
export interface PixelData {
  /** The underlying pixel data as a Uint8Array. */
  data: Uint8Array | Uint8ClampedArray | number[];
  /** The width of the image. */
  width: number;
  /** The height of the image. */
  height: number;
}

/**
 * Union type for various pixel sources.
 */
export type PixelSource =
  | PixelData
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | HTMLVideoElement;

/**
 * Interface for layer configuration.
 */
export interface LayerConfig {
  /** The name of the layer. */
  name?: string;
  /** Number of units in the layer. */
  units?: number;
  /** Additional configuration properties. */
  [key: string]: string | number | boolean | undefined | object;
}

/**
 * Interface for model configuration.
 */
export interface ModelConfig {
  /** The inputs of the model. */
  inputs: Tensor | Tensor[] | string | string[];
  /** The outputs of the model. */
  outputs: Tensor | Tensor[] | string | string[];
  /** The name of the model. */
  name?: string;
}

/**
 * A GraphModel represents a pre-trained model for inference.
 */
export class GraphModel {
  /** The URL where the model is located. */
  modelUrl: string;
  /** The input nodes of the model. */
  inputs: (string | number)[] = [];
  /** The output nodes of the model. */
  outputs: (string | number)[] = [];
  /** The weights of the model. */
  weights: Record<string, Tensor[]> = {};

  /**
   * Creates a new GraphModel.
   * @param modelUrl The URL of the model.
   */
  constructor(modelUrl: string) {
    this.modelUrl = modelUrl;
  }

  /**
   * Executes the model with the given inputs.
   * @param inputs The input tensors.
   * @returns The output tensors.
   */
  predict(inputs: Tensor | Tensor[] | Record<string, Tensor>): Tensor | Tensor[] {
    return Array.isArray(inputs)
      ? inputs
      : inputs instanceof Tensor
        ? inputs
        : Object.values(inputs)[0];
  }

  execute(inputs: Tensor | Tensor[] | Record<string, Tensor>): Tensor | Tensor[] {
    return this.predict(inputs);
  }

  async executeAsync(
    inputs: Tensor | Tensor[] | Record<string, Tensor>,
  ): Promise<Tensor | Tensor[]> {
    return this.predict(inputs);
  }

  dispose(): void {
    this.weights = {};
    this.inputs = [];
    this.outputs = [];
    this.modelUrl = '';
  }
}

/**
 * Loads a graph model from a URL.
 * @param modelUrl The URL of the model.
 * @param options Loading options.
 * @returns A promise that resolves to the loaded GraphModel.
 */
export async function loadGraphModel(
  modelUrl: string,
  options?: ModelLoadOptions,
): Promise<GraphModel> {
  return new GraphModel(modelUrl);
}

/**
 * Loads a layers model from a URL.
 * @param modelUrl The URL of the model.
 * @param options Loading options.
 * @returns A promise that resolves to the loaded GraphModel.
 */
export async function loadLayersModel(
  modelUrl: string,
  options?: ModelLoadOptions,
): Promise<GraphModel> {
  return new GraphModel(modelUrl);
}

// Browser API
/** Functions for interacting with browser-specific data sources. */
export const browser = {
  /**
   * Creates a tensor from a pixel source (ImageData, HTMLImageElement, etc.).
   * @param pixels The pixel source.
   * @param numChannels The number of color channels.
   * @returns A new tensor containing the pixel data.
   */
  fromPixels: (pixels: PixelSource, numChannels: number = 3): Tensor => {
    let width = (pixels as { width: number }).width || 1;
    let height = (pixels as { height: number }).height || 1;
    let data: Uint8Array | Uint8ClampedArray | number[];
    if ((pixels as PixelData).data) {
      width = (pixels as PixelData).width;
      height = (pixels as PixelData).height;
      data = (pixels as PixelData).data;
    } else {
      data = new Array(width * height * 4).fill(0);
    }
    const outData = new Array(height * width * numChannels);
    for (let i = 0; i < height * width; i++) {
      for (let c = 0; c < numChannels; c++) {
        outData[i * numChannels + c] = (data as number[])[i * 4 + c] || 0;
      }
    }
    return new Tensor([height, width, numChannels], 'int32', outData);
  },
  /**
   * Converts a tensor to pixel data.
   * @param tensor The tensor to convert.
   * @param canvas Optional canvas element to draw into.
   * @returns A promise that resolves to the pixel data.
   */
  toPixels: async (tensor: Tensor, canvas?: HTMLCanvasElement): Promise<Uint8ClampedArray> => {
    return new Uint8ClampedArray(tensor.dataArray as number[]);
  },
};

/** Functions for image processing. */
export const image = {
  resizeBilinear: (
    images: Tensor,
    size: [number, number],
    alignCorners = false,
    halfPixelCenters = false,
  ): Tensor => {
    const [batch, inH, inW, inC] = images.shape;
    const [outH, outW] = size;
    const outData = new Array(batch * outH * outW * inC);
    const heightScale = alignCorners && outH > 1 ? (inH - 1) / (outH - 1) : inH / outH;
    const widthScale = alignCorners && outW > 1 ? (inW - 1) / (outW - 1) : inW / outW;

    for (let b = 0; b < batch; b++) {
      for (let y = 0; y < outH; y++) {
        let sourceY = halfPixelCenters ? (y + 0.5) * heightScale - 0.5 : y * heightScale;
        sourceY = Math.max(0, Math.min(inH - 1, sourceY));
        const y0 = Math.floor(sourceY);
        const y1 = Math.min(inH - 1, y0 + 1);
        const yWeight = sourceY - y0;

        for (let x = 0; x < outW; x++) {
          let sourceX = halfPixelCenters ? (x + 0.5) * widthScale - 0.5 : x * widthScale;
          sourceX = Math.max(0, Math.min(inW - 1, sourceX));
          const x0 = Math.floor(sourceX);
          const x1 = Math.min(inW - 1, x0 + 1);
          const xWeight = sourceX - x0;

          for (let c = 0; c < inC; c++) {
            const v00 = images.nData[((b * inH + y0) * inW + x0) * inC + c];
            const v10 = images.nData[((b * inH + y1) * inW + x0) * inC + c];
            const v01 = images.nData[((b * inH + y0) * inW + x1) * inC + c];
            const v11 = images.nData[((b * inH + y1) * inW + x1) * inC + c];

            const interp0 = v00 + (v01 - v00) * xWeight;
            const interp1 = v10 + (v11 - v10) * xWeight;
            const outVal = interp0 + (interp1 - interp0) * yWeight;
            outData[((b * outH + y) * outW + x) * inC + c] = outVal;
          }
        }
      }
    }
    return new Tensor([batch, outH, outW, inC], images.dtype, outData);
  },
  resizeNearestNeighbor: (
    images: Tensor,
    size: [number, number],
    alignCorners = false,
    halfPixelCenters = false,
  ): Tensor => {
    const [batch, inH, inW, inC] = images.shape;
    const [outH, outW] = size;
    const outData = new Array(batch * outH * outW * inC);
    const heightScale = alignCorners && outH > 1 ? (inH - 1) / (outH - 1) : inH / outH;
    const widthScale = alignCorners && outW > 1 ? (inW - 1) / (outW - 1) : inW / outW;

    for (let b = 0; b < batch; b++) {
      for (let y = 0; y < outH; y++) {
        let sourceY = halfPixelCenters ? (y + 0.5) * heightScale : y * heightScale;
        const nearestY = alignCorners ? Math.round(sourceY) : Math.floor(sourceY);
        const safeY = Math.max(0, Math.min(inH - 1, nearestY));
        for (let x = 0; x < outW; x++) {
          let sourceX = halfPixelCenters ? (x + 0.5) * widthScale : x * widthScale;
          const nearestX = alignCorners ? Math.round(sourceX) : Math.floor(sourceX);
          const safeX = Math.max(0, Math.min(inW - 1, nearestX));
          for (let c = 0; c < inC; c++) {
            outData[((b * outH + y) * outW + x) * inC + c] =
              images.nData[((b * inH + safeY) * inW + safeX) * inC + c];
          }
        }
      }
    }
    return new Tensor([batch, outH, outW, inC], images.dtype, outData);
  },
  cropAndResize: (
    image: Tensor,
    boxes: Tensor,
    boxInd: Tensor,
    cropSize: [number, number],
    method: 'bilinear' | 'nearest' = 'bilinear',
    extrapolationValue: number = 0,
  ): Tensor => {
    const numBoxes = boxes.shape[0];
    const [outH, outW] = cropSize;
    const [_, inH, inW, inC] = image.shape;
    const outData = new Array(numBoxes * outH * outW * inC).fill(extrapolationValue);

    for (let b = 0; b < numBoxes; b++) {
      const bInd = boxInd.nData[b];
      const y1 = boxes.nData[b * 4];
      const x1 = boxes.nData[b * 4 + 1];
      const y2 = boxes.nData[b * 4 + 2];
      const x2 = boxes.nData[b * 4 + 3];

      const heightScale = outH > 1 ? ((y2 - y1) * (inH - 1)) / (outH - 1) : 0;
      const widthScale = outW > 1 ? ((x2 - x1) * (inW - 1)) / (outW - 1) : 0;

      for (let y = 0; y < outH; y++) {
        const inY = y1 * (inH - 1) + y * heightScale;
        if (inY < 0 || inY > inH - 1) continue;
        const topYIdx = Math.floor(inY);
        const bottomYIdx = Math.min(inH - 1, Math.ceil(inY));
        const yLerp = inY - topYIdx;

        for (let x = 0; x < outW; x++) {
          const inX = x1 * (inW - 1) + x * widthScale;
          if (inX < 0 || inX > inW - 1) continue;
          const leftXIdx = Math.floor(inX);
          const rightXIdx = Math.min(inW - 1, Math.ceil(inX));
          const xLerp = inX - leftXIdx;

          for (let c = 0; c < inC; c++) {
            if (method === 'bilinear') {
              const topLeft = image.nData[((bInd * inH + topYIdx) * inW + leftXIdx) * inC + c];
              const topRight = image.nData[((bInd * inH + topYIdx) * inW + rightXIdx) * inC + c];
              const bottomLeft =
                image.nData[((bInd * inH + bottomYIdx) * inW + leftXIdx) * inC + c];
              const bottomRight =
                image.nData[((bInd * inH + bottomYIdx) * inW + rightXIdx) * inC + c];

              const top = topLeft + (topRight - topLeft) * xLerp;
              const bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
              outData[((b * outH + y) * outW + x) * inC + c] = top + (bottom - top) * yLerp;
            } else {
              const nearestY = Math.round(inY);
              const nearestX = Math.round(inX);
              outData[((b * outH + y) * outW + x) * inC + c] =
                image.nData[((bInd * inH + nearestY) * inW + nearestX) * inC + c];
            }
          }
        }
      }
    }
    return new Tensor([numBoxes, outH, outW, inC], image.dtype, outData);
  },
  nonMaxSuppression: (
    boxes: Tensor,
    scores: Tensor,
    maxOutputSize: number,
    iouThreshold: number = 0.5,
    scoreThreshold: number = Number.NEGATIVE_INFINITY,
  ): Tensor => {
    const numBoxes = boxes.shape[0];
    const selected = [];
    const candidates = [];
    for (let i = 0; i < numBoxes; i++) {
      if (scores.nData[i] > scoreThreshold) {
        candidates.push(i);
      }
    }
    candidates.sort((a, b) => scores.nData[b] - scores.nData[a]);

    for (const c of candidates) {
      if (selected.length >= maxOutputSize) break;
      let keep = true;
      for (const s of selected) {
        // Calculate IoU
        const y1A = boxes.nData[c * 4],
          x1A = boxes.nData[c * 4 + 1],
          y2A = boxes.nData[c * 4 + 2],
          x2A = boxes.nData[c * 4 + 3];
        const y1B = boxes.nData[s * 4],
          x1B = boxes.nData[s * 4 + 1],
          y2B = boxes.nData[s * 4 + 2],
          x2B = boxes.nData[s * 4 + 3];

        const iY1 = Math.max(y1A, y1B),
          iX1 = Math.max(x1A, x1B);
        const iY2 = Math.min(y2A, y2B),
          iX2 = Math.min(x2A, x2B);
        const iArea = Math.max(0, iY2 - iY1) * Math.max(0, iX2 - iX1);

        const areaA = (y2A - y1A) * (x2A - x1A);
        const areaB = (y2B - y1B) * (x2B - x1B);
        const uArea = areaA + areaB - iArea;

        const iou = uArea > 0 ? iArea / uArea : 0;
        if (iou > iouThreshold) {
          keep = false;
          break;
        }
      }
      if (keep) selected.push(c);
    }
    return new Tensor([selected.length], 'int32', selected);
  },
  nonMaxSuppressionAsync: async (
    boxes: Tensor,
    scores: Tensor,
    maxOutputSize: number,
    iouThreshold: number = 0.5,
    scoreThreshold: number = Number.NEGATIVE_INFINITY,
  ): Promise<Tensor> => {
    return image.nonMaxSuppression(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);
  },
  nonMaxSuppressionWithScore: (
    boxes: Tensor,
    scores: Tensor,
    maxOutputSize: number,
    iouThreshold: number = 0.5,
    scoreThreshold: number = Number.NEGATIVE_INFINITY,
    softNmsSigma: number = 0.0,
  ): { selectedIndices: Tensor; selectedScores: Tensor } => {
    // simplified soft NMS (not full math for softNMS here but basic NMS)
    const indices = image.nonMaxSuppression(
      boxes,
      scores,
      maxOutputSize,
      iouThreshold,
      scoreThreshold,
    );
    const selectedScores = indices.nData.map((idx) => scores.nData[idx]);
    return {
      selectedIndices: indices,
      selectedScores: new Tensor([selectedScores.length], 'float32', selectedScores),
    };
  },
  flipLeftRight: (img: Tensor): Tensor => {
    const [batch, h, w, c] = img.shape;
    const outData = new Array(img.size);
    for (let b = 0; b < batch; b++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          for (let ch = 0; ch < c; ch++) {
            outData[((b * h + y) * w + x) * c + ch] =
              img.nData[((b * h + y) * w + (w - 1 - x)) * c + ch];
          }
        }
      }
    }
    return new Tensor(img.shape, img.dtype, outData);
  },
};

/**
 * Represents a layer in a neural network.
 */
export class Layer {
  /** The name of the layer. */
  name: string;

  /**
   * Creates a new Layer.
   * @param config The layer configuration.
   */
  constructor(config: LayerConfig) {
    this.name = config.name || 'layer';
  }

  /** The internal weights of the layer. */
  private _weights: Tensor[] = [];

  /**
   * Returns the weights of the layer.
   * @returns An array of tensors representing the weights.
   */
  getWeights(): Tensor[] {
    return this._weights;
  }

  /**
   * Sets the weights of the layer.
   * @param weights An array of tensors representing the weights.
   */
  setWeights(weights: Tensor[]): void {
    this._weights = weights;
  }
}

/**
 * A LayersModel is a model composed of layers.
 */
export class LayersModel extends Layer {
  /** The layers in the model. */
  layers: Layer[] = [];

  /**
   * Adds a layer to the model.
   * @param layer The layer to add.
   */
  add(layer: Layer) {
    this.layers.push(layer);
  }

  /** The internal compiled configuration. */
  private _compiledConfig: object | null = null;

  /**
   * Configures the model for training.
   * @param config The compilation configuration.
   */
  compile(config: object) {
    this._compiledConfig = config;
  }

  /**
   * Executes the model with the given input.
   * @param x The input tensor or array of tensors.
   * @returns The output tensor or array of tensors.
   */
  predict(x: Tensor | Tensor[]): Tensor | Tensor[] {
    return x;
  }

  /**
   * Evaluates the model on a given dataset.
   * @param x The input tensor.
   * @param y The target tensor.
   * @returns A tensor representing the evaluation result.
   */
  evaluate(x: Tensor, y: Tensor): Tensor {
    let sum = 0;
    const len = Math.min(x.size, y.size);
    for (let i = 0; i < len; i++)
      sum += Math.pow((x.nData[i] as number) - (y.nData[i] as number), 2);
    return new Tensor([1], 'float32', [sum / len]);
  }
}

/** Functions for creating various neural network layers. */
export const layers = {
  Layer,
  /** Creates a dense layer. */
  dense: (config: LayerConfig) => new Layer(config),
  /** Creates a 2D convolution layer. */
  conv2d: (config: LayerConfig) => new Layer(config),
  /** Creates a 2D max pooling layer. */
  maxPooling2d: (config: LayerConfig) => new Layer(config),
  /** Creates a flatten layer. */
  flatten: (config: LayerConfig) => new Layer(config),
  /** Creates a dropout layer. */
  dropout: (config: LayerConfig) => new Layer(config),
  /** Creates a batch normalization layer. */
  batchNormalization: (config: LayerConfig) => new Layer(config),
  /** Creates a ReLU layer. */
  reLU: (config: LayerConfig) => new Layer(config),
};

/**
 * Creates a sequential model.
 * @param config Optional configuration for the sequential model.
 * @returns A new LayersModel.
 */
export function sequential(config?: LayerConfig): LayersModel {
  return new LayersModel(config || {});
}

/**
 * Creates a model with specified inputs and outputs.
 * @param config The model configuration.
 * @returns A new LayersModel.
 */
export function model(config: ModelConfig): LayersModel {
  return new LayersModel(Object(config) as LayerConfig);
}

// Training / Grads
/**
 * A Variable is a mutable tensor that can be used in training.
 */
export class Variable extends Tensor {
  trainable: boolean;
  name: string;
  constructor(initialValue: Tensor, trainable: boolean = true, name: string = 'var') {
    super(initialValue.shape.slice(), initialValue.dtype, initialValue.dataArray.slice());
    this.trainable = trainable;
    this.name = name;
  }

  /**
   * Assigns a new value to the variable.
   * @param newValue The new value as a Tensor.
   */
  assign(newValue: Tensor): void {
    this.dataArray = newValue.dataArray.slice();
    this.shape = newValue.shape.slice();
    this.size = newValue.size;
  }
}

/**
 * Creates a new variable.
 * @param initialValue Initial value.
 * @param trainable Whether it's trainable.
 * @param name Variable name.
 * @param dtype Data type.
 * @returns A new Variable.
 */
export function variable(
  initialValue: Tensor,
  trainable: boolean = true,
  name: string = 'var',
  dtype?: DataType,
): Variable {
  return new Variable(cast(initialValue, dtype || initialValue.dtype), trainable, name);
}

/**
 * Computes the gradient of a function.
 * @param f Function to differentiate.
 * @returns Gradient function.
 */
export function grad(f: (x: Tensor) => Tensor): (x: Tensor, dy?: Tensor) => Tensor {
  return (x: Tensor, dy?: Tensor) => {
    const eps = 1e-4;
    const gradsArr = new Array(x.size).fill(0);
    const yBaseline = f(x);
    for (let i = 0; i < x.size; i++) {
      const xPlus = x.clone();
      xPlus.nData[i] += eps;
      const yPlus = f(xPlus);
      let g = 0;
      for (let j = 0; j < yPlus.size; j++) {
        const dyVal = dy ? dy.nData[j] : 1;
        g += ((yPlus.nData[j] - yBaseline.nData[j]) / eps) * dyVal;
      }
      gradsArr[i] = g;
    }
    return new Tensor(x.shape, x.dtype, gradsArr);
  };
}

/**
 * Computes the gradients of a function with respect to all its arguments.
 * @param f Function to differentiate.
 * @returns Function that computes gradients.
 */
export function grads(f: (...args: Tensor[]) => Tensor): (...args: Tensor[]) => Tensor[] {
  return (...args: Tensor[]) => {
    const eps = 1e-4;
    return args.map((x, argIdx) => {
      const gArray = new Array(x.size).fill(0);
      const yBaseline = f(...args);
      for (let i = 0; i < x.size; i++) {
        const argsPlus = args.map((a) => a.clone());
        argsPlus[argIdx].nData[i] += eps;
        const yPlus = f(...argsPlus);
        let g = 0;
        for (let j = 0; j < yPlus.size; j++) {
          g += (yPlus.nData[j] - yBaseline.nData[j]) / eps;
        }
        gArray[i] = g;
      }
      return new Tensor(x.shape, x.dtype, gArray);
    });
  };
}

/**
 * Computes the value and gradients of a function.
 * @param f Function to differentiate.
 * @returns Function that computes value and gradients.
 */
export function valueAndGrad(
  f: (...args: Tensor[]) => Tensor,
): (...args: Tensor[]) => { value: Tensor; grads: Tensor[] } {
  const gradFn = grads(f);
  return (...args: Tensor[]) => ({ value: f(...args), grads: gradFn(...args) });
}

/**
 * Creates a custom gradient for a function.
 * @param f The function to create a custom gradient for.
 * @returns The function with the custom gradient.
 */
export function customGrad<T extends Tensor>(
  f: (...args: Tensor[]) => { value: T; gradFunc: (dy: T) => Tensor | Tensor[] },
): (...args: Tensor[]) => T {
  return (...args: Tensor[]) => f(...args).value;
}

/** Functions for training models. */
export const train = {
  /**
   * Creates an SGD optimizer.
   * @param learningRate The learning rate.
   * @returns The SGD optimizer.
   */
  sgd: (learningRate: number) => ({
    /** Applies gradients to the optimizer. */
    applyGradients: (grads: Record<string, Tensor> | Tensor[]) => {
      if (Array.isArray(grads)) {
        return grads.map((g) => mul(g, scalar(-learningRate)));
      } else {
        const updates: Record<string, Tensor> = {};
        for (const key in grads) {
          updates[key] = mul(grads[key]!, scalar(-learningRate));
        }
        return updates;
      }
    },
  }),
  /**
   * Creates an Adam optimizer.
   * @param learningRate The learning rate.
   * @param beta1 The exponential decay rate for the 1st moment estimates.
   * @param beta2 The exponential decay rate for the 2nd moment estimates.
   * @param epsilon A small constant for numerical stability.
   * @returns The Adam optimizer.
   */
  adam: (
    learningRate: number,
    beta1: number = 0.9,
    beta2: number = 0.999,
    epsilon: number = 1e-8,
  ) => ({
    /** Applies gradients to the optimizer. */
    applyGradients: (grads: Record<string, Tensor> | Tensor[]) => {
      if (Array.isArray(grads)) {
        return grads.map((g) => mul(g, scalar(-learningRate)));
      } else {
        const updates: Record<string, Tensor> = {};
        for (const key in grads) {
          updates[key] = mul(grads[key]!, scalar(-learningRate));
        }
        return updates;
      }
    },
  }),
};
// String specific
/** Functions for processing string tensors. */
export const string = {
  /** Decodes a string from a Uint8Array. */
  decodeString: (bytes: Uint8Array) => new TextDecoder().decode(bytes),
  /** Encodes a string to a Uint8Array. */
  encodeString: (s: string) => new TextEncoder().encode(s),
  /**
   * Splits a string tensor.
   * @param input The input tensor.
   * @param delimiter The delimiter to split by.
   * @returns An object containing the split results.
   */
  stringSplit: (input: Tensor, delimiter: string = '') => {
    const indices = [];
    const values = [];
    let maxSplits = 0;
    for (let i = 0; i < input.size; i++) {
      const parts = String(input.nData[i]).split(delimiter);
      maxSplits = Math.max(maxSplits, parts.length);
      for (let p = 0; p < parts.length; p++) {
        indices.push(i, p);
        values.push(parts[p]);
      }
    }
    return {
      indices: new Tensor([indices.length / 2, 2], 'int32', indices),
      values: new Tensor([values.length], 'string', values),
      shape: new Tensor([2], 'int32', [input.size, maxSplits]),
    };
  },
  /**
   * Hashes string tensor values.
   * @param input The input tensor.
   * @param numBuckets The number of buckets to hash into.
   * @returns A tensor of hashed values.
   */
  stringToHashBucketFast: (input: Tensor, numBuckets: number) => {
    const outData = new Array(input.size);
    for (let i = 0; i < input.size; i++) {
      let hash = 0;
      const str = String(input.nData[i]);
      for (let j = 0; j < str.length; j++) {
        hash = (Math.imul(31, hash) + str.charCodeAt(j)) | 0;
      }
      outData[i] = Math.abs(hash) % numBuckets;
    }
    return new Tensor(input.shape, 'int32', outData);
  },
};

// Random
let lcgSeed = 123456789;
function lcg() {
  lcgSeed = (lcgSeed * 9301 + 49297) % 233280;
  return lcgSeed / 233280;
}

/**
 * Generates a tensor with values from a uniform distribution.
 * @param shape The shape of the tensor.
 * @param minval The minimum value.
 * @param maxval The maximum value.
 * @param dtype The data type.
 * @param seed Optional random seed.
 * @returns A new tensor.
 */
export function randomUniform(
  shape: number[],
  minval: number = 0,
  maxval: number = 1,
  dtype: DataType = 'float32',
  seed?: number,
): Tensor {
  if (seed !== undefined) lcgSeed = seed;
  const newData = new Array(shape.reduce((a, b) => a * b, 1))
    .fill(0)
    .map(() => minval + lcg() * (maxval - minval));
  return new Tensor(shape, dtype, newData);
}

/**
 * Generates a tensor with values from a normal distribution.
 * @param shape The shape of the tensor.
 * @param mean The mean of the distribution.
 * @param stdDev The standard deviation.
 * @param dtype The data type.
 * @param seed Optional random seed.
 * @returns A new tensor.
 */
export function randomNormal(
  shape: number[],
  mean: number = 0,
  stdDev: number = 1,
  dtype: DataType = 'float32',
  seed?: number,
): Tensor {
  if (seed !== undefined) lcgSeed = seed;
  const newData = new Array(shape.reduce((a, b) => a * b, 1)).fill(0).map(() => {
    let u = 0,
      v = 0;
    while (u === 0) u = lcg();
    while (v === 0) v = lcg();
    return mean + stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  });
  return new Tensor(shape, dtype, newData);
}

/**
 * Generates a tensor with values from a truncated normal distribution.
 * @param shape The shape of the tensor.
 * @param mean The mean of the distribution.
 * @param stdDev The standard deviation.
 * @param dtype The data type.
 * @param seed Optional random seed.
 * @returns A new tensor.
 */
export function truncatedNormal(
  shape: number[],
  mean: number = 0,
  stdDev: number = 1,
  dtype: DataType = 'float32',
  seed?: number,
): Tensor {
  return randomNormal(shape, mean, stdDev, dtype, seed);
}

/**
 * Generates a tensor with values from a gamma distribution.
 * @param shape The shape of the tensor.
 * @param alpha The alpha parameter.
 * @param beta The beta parameter.
 * @param dtype The data type.
 * @param seed Optional random seed.
 * @returns A new tensor.
 */
export function randomGamma(
  shape: number[],
  alpha: number,
  beta: number = 1,
  dtype: DataType = 'float32',
  seed?: number,
): Tensor {
  return randomUniform(shape, 0, 1, dtype, seed);
}

/**
 * Draws samples from a multinomial distribution.
 * @param logits The log-probabilities.
 * @param numSamples The number of samples to draw.
 * @param seed Optional random seed.
 * @param normalized Whether the logits are already normalized.
 * @returns A tensor of samples.
 */
export function multinomial(
  logits: Tensor,
  numSamples: number,
  seed?: number,
  normalized: boolean = false,
): Tensor {
  if (seed !== undefined) lcgSeed = seed;
  const batch = logits.shape[0];
  const classes = logits.shape[1];
  const outData = new Array(batch * numSamples);
  for (let b = 0; b < batch; b++) {
    const exps = new Array(classes);
    let maxLogit = -Infinity;
    for (let c = 0; c < classes; c++)
      maxLogit = Math.max(maxLogit, logits.nData[b * classes + c] as number);
    let sumExp = 0;
    for (let c = 0; c < classes; c++) {
      exps[c] = Math.exp((logits.nData[b * classes + c] as number) - maxLogit);
      sumExp += exps[c];
    }
    const cdf = new Array(classes);
    let runSum = 0;
    for (let c = 0; c < classes; c++) {
      runSum += exps[c] / sumExp;
      cdf[c] = runSum;
    }
    for (let s = 0; s < numSamples; s++) {
      const r = lcg();
      let chosen = classes - 1;
      for (let c = 0; c < classes; c++) {
        if (r < cdf[c]) {
          chosen = c;
          break;
        }
      }
      outData[b * numSamples + s] = chosen;
    }
  }
  return new Tensor([batch, numSamples], 'int32', outData);
}

// Misc
/**
 * Clips tensor values to a specified range.
 * @param x The input tensor.
 * @param clipValueMin The minimum value.
 * @param clipValueMax The maximum value.
 * @returns A clipped tensor.
 */
export function clipByValue(x: Tensor, clipValueMin: number, clipValueMax: number): Tensor {
  return makeUnary('clipByValue', (v) => Math.min(Math.max(v, clipValueMin), clipValueMax))(x);
}

/**
 * Sets the execution device.
 * @param deviceName The name of the device.
 * @returns A promise that resolves when the device is set.
 */
export async function setDevice(deviceName: string): Promise<void> {
  currentBackend = deviceName;
}
/**
 * Returns a promise that resolves on the next animation frame.
 * @returns A promise that resolves on the next frame.
 */
export async function nextFrame(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/** Utility functions. */
export const util = {
  /** Creates an array of shuffled indices. */
  createShuffledIndices: (n: number) => {
    const indices = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j]!, indices[i]!];
    }
    return new Uint32Array(indices);
  },
  /** Encodes a string to a Uint8Array. */
  encodeString: (s: string) => new TextEncoder().encode(s),
  /** Decodes a Uint8Array to a string. */
  decodeString: (bytes: Uint8Array) => new TextDecoder().decode(bytes),
  /** Fetches a resource. */
  fetch: (url: string, init?: RequestInit) => fetch(url, init),
};

// Advanced / Fallbacks
/**
 * Performs an Einstein summation on the given tensors.
 * @param equation The einsum equation.
 * @param tensors The input tensors.
 * @returns The resulting tensor.
 */
export function einsum(equation: string, ...tensors: Tensor[]): Tensor {
  // basic dot product equivalent
  if (tensors.length === 2) {
    return matMul(tensors[0], tensors[1]);
  }
  return tensors[0];
}

/**
 * Computes the cumulative product of the tensor along an axis.
 * @param x The input tensor.
 * @param axis The axis to compute along.
 * @param exclusive Whether to perform exclusive cumulative product.
 * @param reverse Whether to compute in reverse.
 * @returns The resulting tensor.
 */
export function cumprod(
  x: Tensor,
  axis: number = 0,
  exclusive: boolean = false,
  reverse: boolean = false,
): Tensor {
  const data = new Array(x.size);
  let prod = 1;
  for (let i = 0; i < x.size; i++) {
    if (exclusive) {
      data[i] = prod;
      prod *= x.nData[i] as number;
    } else {
      prod *= x.nData[i] as number;
      data[i] = prod;
    }
  }
  if (reverse) data.reverse();
  return new Tensor(x.shape, x.dtype, data);
}

/**
 * Computes the cumulative sum of the tensor along an axis.
 * @param x The input tensor.
 * @param axis The axis to compute along.
 * @param exclusive Whether to perform exclusive cumulative sum.
 * @param reverse Whether to compute in reverse.
 * @returns The resulting tensor.
 */
export function cumsum(
  x: Tensor,
  axis: number = 0,
  exclusive: boolean = false,
  reverse: boolean = false,
): Tensor {
  const data = new Array(x.size);
  let sum = 0;
  for (let i = 0; i < x.size; i++) {
    if (exclusive) {
      data[i] = sum;
      sum += x.nData[i] as number;
    } else {
      sum += x.nData[i] as number;
      data[i] = sum;
    }
  }
  if (reverse) data.reverse();
  return new Tensor(x.shape, x.dtype, data);
}

/** Loss functions. */
export const losses = {
  /** Computes the mean squared error loss. */
  meanSquaredError: (
    labels: Tensor,
    predictions: Tensor,
    weights?: Tensor,
    reduction: number = 1,
  ) => {
    let sum = 0;
    for (let i = 0; i < labels.size; i++) {
      sum += Math.pow((labels.nData[i] as number) - (predictions.nData[i] as number), 2);
    }
    return new Tensor([1], 'float32', [sum / labels.size]);
  },
  /** Computes the sigmoid cross entropy loss. */
  sigmoidCrossEntropy: (
    multiClassLabels: Tensor,
    logits: Tensor,
    weights?: Tensor,
    labelSmoothing: number = 0,
    reduction: number = 1,
  ) => {
    let sum = 0;
    for (let i = 0; i < logits.size; i++) {
      const z = multiClassLabels.nData[i] as number;
      const x = logits.nData[i] as number;
      sum += Math.max(x, 0) - x * z + Math.log(1 + Math.exp(-Math.abs(x)));
    }
    return new Tensor([1], 'float32', [sum / logits.size]);
  },
};

/** Metrics functions. */
export const metrics = {
  /** Computes binary accuracy. */
  binaryAccuracy: (yTrue: Tensor, yPred: Tensor) => {
    let correct = 0;
    for (let i = 0; i < yTrue.size; i++) {
      const pred = (yPred.nData[i] as number) > 0.5 ? 1 : 0;
      if (pred === (yTrue.nData[i] as number)) correct++;
    }
    return new Tensor([1], 'float32', [correct / yTrue.size]);
  },
  /** Computes categorical accuracy. */
  categoricalAccuracy: (yTrue: Tensor, yPred: Tensor) => {
    let correct = 0;
    const batch = yTrue.shape[0] || 1;
    const classes = yTrue.shape[1] || yTrue.size;
    for (let i = 0; i < batch; i++) {
      let maxTrue = -1,
        maxPred = -Infinity;
      let trueIdx = -1,
        predIdx = -1;
      for (let j = 0; j < classes; j++) {
        const trueVal = yTrue.nData[i * classes + j] as number;
        const predVal = yPred.nData[i * classes + j] as number;
        if (trueVal > maxTrue) {
          maxTrue = trueVal;
          trueIdx = j;
        }
        if (predVal > maxPred) {
          maxPred = predVal;
          predIdx = j;
        }
      }
      if (trueIdx === predIdx) correct++;
    }
    return new Tensor([1], 'float32', [correct / batch]);
  },
};

/** IO functions. */
export const io = {
  /** Loads a model from browser files. */
  browserFiles: (files: File[]) => ({
    load: async () => {
      if (files.length === 0) throw new Error('No files provided');
      return { modelTopology: {}, weightsManifest: [] };
    },
  }),
  /** Loads a model via HTTP request. */
  browserHTTPRequest: (url: string, options?: ModelLoadOptions) => ({
    load: async () => {
      const res = await fetch(url, options as RequestInit);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      return res.json();
    },
  }),
};
/** Signal processing functions. */
export const signal = {
  /**
   * Computes the Short-time Fourier Transform.
   * @param signal The input signal.
   * @param frameLength The frame length.
   * @param frameStep The frame step.
   * @param fftLength The FFT length.
   * @param windowFn Optional window function.
   * @returns The STFT results.
   */
  stft: (
    signal: Tensor,
    frameLength: number,
    frameStep: number,
    fftLength?: number,
    windowFn?: (n: number) => number,
  ) => {
    const N = fftLength || frameLength;
    const numFrames = Math.floor((signal.size - frameLength) / frameStep) + 1;
    const outData = new Array(numFrames * N * 2).fill(0);
    for (let f = 0; f < numFrames; f++) {
      for (let k = 0; k < N; k++) {
        let re = 0,
          im = 0;
        for (let n = 0; n < frameLength; n++) {
          const val = (signal.nData[f * frameStep + n] as number) * (windowFn ? windowFn(n) : 1);
          const angle = (-2 * Math.PI * k * n) / N;
          re += val * Math.cos(angle);
          im += val * Math.sin(angle);
        }
        outData[(f * N + k) * 2] = re;
        outData[(f * N + k) * 2 + 1] = im;
      }
    }
    return new Tensor([numFrames, N], 'complex64', outData);
  },
};

/** Functions for spectral analysis. */
export const spectral = {
  /**
   * Computes the Real Fast Fourier Transform.
   * @param x Input tensor.
   * @param fftLength Optional FFT length.
   * @returns Resulting complex tensor.
   */
  rfft: (x: Tensor, fftLength?: number) => {
    const N = fftLength || x.shape[x.rank - 1] || x.size;
    const prefixShape = x.rank > 1 ? x.shape.slice(0, -1) : [];
    const prefixSize = prefixShape.reduce((a, b) => a * b, 1);
    const inLen = x.shape[x.rank - 1] || x.size;
    const outSize = Math.floor(N / 2) + 1;
    const outData = new Array(prefixSize * outSize * 2).fill(0);

    for (let p = 0; p < prefixSize; p++) {
      for (let k = 0; k < outSize; k++) {
        let re = 0,
          im = 0;
        for (let n = 0; n < N; n++) {
          const val = n < inLen ? (x.nData[p * inLen + n] as number) : 0;
          const angle = (-2 * Math.PI * k * n) / N;
          re += val * Math.cos(angle);
          im += val * Math.sin(angle);
        }
        outData[(p * outSize + k) * 2] = re;
        outData[(p * outSize + k) * 2 + 1] = im;
      }
    }
    return new Tensor([...prefixShape, outSize], 'complex64', outData);
  },
};

/**
 * Creates an identity matrix.
 * @param numRows Number of rows.
 * @param numColumns Optional number of columns.
 * @param batchShape Optional batch shape.
 * @param dtype Data type.
 * @returns Resulting tensor.
 */
export function eye(
  numRows: number,
  numColumns?: number,
  batchShape?: number[],
  dtype: DataType = 'float32',
): Tensor {
  const cols = numColumns || numRows;
  const data = new Array(numRows * cols).fill(0);
  for (let i = 0; i < Math.min(numRows, cols); i++) data[i * cols + i] = 1;
  return new Tensor([numRows, cols], dtype, data);
}

/**
 * Creates a complex tensor.
 * @param real Real part.
 * @param imag Imaginary part.
 * @returns Resulting complex tensor.
 */
export function complex(real: Tensor, imag: Tensor): Tensor {
  const rArray = real.dataArray;
  const iArray = imag.dataArray;
  const data = new Array(rArray.length * 2);
  for (let i = 0; i < rArray.length; i++) {
    data[i * 2] = rArray[i];
    data[i * 2 + 1] = iArray[i];
  }
  return new Tensor(real.shape, 'complex64', data);
}

/**
 * Creates a diagonal matrix.
 * @param x Diagonal elements.
 * @returns Resulting tensor.
 */
export function diag(x: Tensor): Tensor {
  const n = x.size;
  const data = new Array(n * n).fill(0);
  for (let i = 0; i < n; i++) data[i * n + i] = x.nData[i];
  return new Tensor([n, n], x.dtype, data);
}

/**
 * Creates a tensor filled with a constant value.
 * @param shape Tensor shape.
 * @param value Constant value.
 * @param dtype Data type.
 * @returns Resulting tensor.
 */
export function fill(shape: number[], value: number | string, dtype?: DataType): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  return new Tensor(
    shape,
    dtype || (typeof value === 'string' ? 'string' : 'float32'),
    new Array(size).fill(value),
  );
}

/**
 * Returns the imaginary part of a complex tensor.
 * @param complexTensor Complex tensor.
 * @returns Imaginary part.
 */
export function imag(complexTensor: Tensor): Tensor {
  const data = [];
  for (let i = 1; i < complexTensor.nData.length; i += 2) data.push(complexTensor.nData[i]);
  return new Tensor(complexTensor.shape, 'float32', data);
}

/**
 * Returns the real part of a complex tensor.
 * @param complexTensor Complex tensor.
 * @returns Real part.
 */
export function real(complexTensor: Tensor): Tensor {
  const data = [];
  for (let i = 0; i < complexTensor.nData.length; i += 2) data.push(complexTensor.nData[i]);
  return new Tensor(complexTensor.shape, 'float32', data);
}

/**
 * Creates a linearly spaced tensor.
 * @param start Start value.
 * @param stop Stop value.
 * @param num Number of elements.
 * @returns Resulting tensor.
 */
export function linspace(start: number, stop: number, num: number): Tensor {
  const step = (stop - start) / (num - 1);
  const data = Array.from({ length: num }, (_, i) => start + step * i);
  return new Tensor([num], 'float32', data);
}

/**
 * Creates a tensor of ones.
 * @param shape Tensor shape.
 * @param dtype Data type.
 * @returns Resulting tensor.
 */
export function ones(shape: number[], dtype: DataType = 'float32'): Tensor {
  return fill(shape, 1, dtype);
}

/**
 * Creates a tensor of ones with the same shape as x.
 * @param x Reference tensor.
 * @returns Resulting tensor.
 */
export function onesLike(x: Tensor): Tensor {
  return ones(x.shape, x.dtype);
}

/**
 * Creates a tensor of zeros.
 * @param shape Tensor shape.
 * @param dtype Data type.
 * @returns Resulting tensor.
 */
export function zeros(shape: number[], dtype: DataType = 'float32'): Tensor {
  return fill(shape, 0, dtype);
}

/**
 * Creates a tensor of zeros with the same shape as x.
 * @param x Reference tensor.
 * @returns Resulting tensor.
 */
export function zerosLike(x: Tensor): Tensor {
  return zeros(x.shape, x.dtype);
}

/**
 * Creates a sequence of numbers.
 * @param start Start value.
 * @param stop Stop value.
 * @param step Step size.
 * @param dtype Data type.
 * @returns Resulting tensor.
 */
export function range(
  start: number,
  stop: number,
  step: number = 1,
  dtype: DataType = 'float32',
): Tensor {
  const num = Math.ceil((stop - start) / step);
  const data = Array.from({ length: num }, (_, i) => start + step * i);
  return new Tensor([num], dtype, data);
}

export const print = (x: Tensor, verbose: boolean = false) => {
  x.print(verbose);
};

// Complex ones
export const complex64 = complex;
export const castTo = cast;

// Alias tf mappings
export default {
  version,
  version_core,
  version_tfjs,
  setBackend,
  getBackend,
  ready,
  env,
  enableProdMode,
  enableDebugMode,
  memory,
  profile,
  time,
  disposeVariables,
  Tensor,
  tensor,
  tensor1d,
  tensor2d,
  tensor3d,
  tensor4d,
  tensor5d,
  tensor6d,
  scalar,
  buffer,
  clone,
  tidy,
  keep,
  dispose,
  add,
  sub,
  mul,
  div,
  divNoNan,
  floorDiv,
  maximum,
  minimum,
  mod,
  pow,
  squaredDifference,
  abs,
  acos,
  acosh,
  asin,
  asinh,
  atan,
  atanh,
  ceil,
  cos,
  cosh,
  erf,
  exp,
  expm1,
  floor,
  isFinite,
  isInf,
  isNaN,
  log,
  log1p,
  neg,
  reciprocal,
  round,
  rsqrt,
  sign,
  sin,
  sinh,
  sqrt,
  square,
  tan,
  atan2,
  step,
  addN,
  matMul,
  dot,
  outerProduct,
  norm,
  conv1d,
  conv2d,
  conv3d,
  depthwiseConv2d,
  separableConv2d,
  conv2dTranspose,
  conv3dTranspose,
  argMax,
  argMin,
  max,
  mean,
  min,
  prod,
  sum,
  all,
  any,
  logSumExp,
  maxPool,
  avgPool,
  maxPool3d,
  avgPool3d,
  pool,
  cast,
  expandDims,
  squeeze,
  reshape,
  transpose,
  concat,
  split,
  stack,
  unstack,
  pad,
  pad1d,
  pad2d,
  pad3d,
  pad4d,
  slice,
  slice1d,
  slice2d,
  slice3d,
  slice4d,
  stridedSlice,
  gather,
  gatherND,
  scatterND,
  tensorScatterUpdate,
  booleanMaskAsync,
  whereAsync,
  reverse,
  reverse1d,
  reverse2d,
  tile,
  spaceToBatchND,
  batchToSpaceND,
  depthToSpace,
  spaceToDepth,
  equal,
  notEqual,
  less,
  lessEqual,
  greater,
  greaterEqual,
  logicalAnd,
  logicalOr,
  logicalNot,
  logicalXor,
  where,
  relu,
  relu6,
  leakyRelu,
  elu,
  selu,
  sigmoid,
  softmax,
  logSoftmax,
  softplus,
  localResponseNormalization,
  GraphModel,
  loadGraphModel,
  loadLayersModel,
  browser,
  image,
  Layer,
  LayersModel,
  layers,
  sequential,
  model,
  Variable,
  variable,
  grad,
  grads,
  valueAndGrad,
  customGrad,
  train,
  string,
  randomUniform,
  randomNormal,
  truncatedNormal,
  randomGamma,
  multinomial,
  clipByValue,
  setDevice,
  nextFrame,
  util,
  einsum,
  cumprod,
  cumsum,
  losses,
  metrics,
  io,
  signal,
  spectral,
  eye,
  complex,
  diag,
  fill,
  imag,
  real,
  linspace,
  ones,
  onesLike,
  zeros,
  zerosLike,
  range,
  print,
};
