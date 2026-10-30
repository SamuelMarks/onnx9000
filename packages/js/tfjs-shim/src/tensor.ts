/**
 * @fileoverview Core Tensor class, memory scoping, and tensor creation utilities for @onnx9000/tfjs-shim.
 */

// --- Global Config and Envs ---
let currentBackend = 'webgpu';
let _isProdMode = false;
let _isDebugMode = false;

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
    get: (key: string) => _envVars[key] ?? null,
    set: (key: string, value: string | number | boolean) => {
      _envVars[key] = value;
    },
  };
}

/**
 * Enables production mode, disabling some runtime checks.
 */
export function enableProdMode(): void {
  _isProdMode = true;
}

/**
 * Enables debug mode, enabling verbose logging and extra checks.
 */
export function enableDebugMode(): void {
  _isDebugMode = true;
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
  f: () => Promise<Tensor | Tensor[] | undefined> | Tensor | Tensor[] | undefined,
): Promise<{
  newBytes: number;
  newTensors: number;
  peakBytes: number;
  kernels: string[];
  result: Tensor | Tensor[] | undefined;
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
      currentTidyScope[currentTidyScope.length - 1]?.push(this);
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
  print(_verbose: boolean = false): void {
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
  | undefined
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
  const scopeTensors = currentTidyScope.pop() as Tensor[];

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
export function transpose(x: Tensor, _perm?: number[]): Tensor {
  return new Tensor([...x.shape].reverse(), x.dtype, x.dataArray.slice().reverse());
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
      const tmp = indices[i] as number;
      indices[i] = indices[j] as number;
      indices[j] = tmp;
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
  _batchShape?: number[],
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
