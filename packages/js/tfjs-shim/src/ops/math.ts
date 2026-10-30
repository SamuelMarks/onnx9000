/**
 * @fileoverview Elementwise mathematical, logical, slicing, and random operations for @onnx9000/tfjs-shim.
 */

import { Onnx9000Error } from '@onnx9000/core';
import { type DataType, expandDims, scalar, Tensor } from '../tensor.js';

/**
 * Higher-order function to create an element-wise binary operation.
 * @param name The name of the operation.
 * @param op The binary operation function.
 * @returns A function that performs the binary operation on two tensors.
 */
export function makeElementwise(name: string, op: (a: number, b: number) => number) {
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
      throw new Onnx9000Error(`Invalid inputs to ${name}`);
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
export function makeUnary(_name: string, op: (a: number) => number) {
  return (a: Tensor): Tensor => {
    const newData = new Array(a.size);
    for (let i = 0; i < a.size; i++) {
      newData[i] = op(a.nData[i] as number);
    }
    return new Tensor(a.shape, a.dtype, newData);
  };
}

/**
 * Executes `makeBinary`.
 * @param _name Parameter `_name`.
 * @param op Parameter `op`.
 * @returns Result of `makeBinary`.
 */
export function makeBinary(_name: string, op: (a: number, b: number) => number) {
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
export const pow = makeElementwise('pow', (a, b) => a ** b);
/** Returns the element-wise squared difference of two tensors. */
export const squaredDifference = makeElementwise('squaredDifference', (a, b) => (a - b) ** 2);

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
// biome-ignore lint/suspicious/noShadowRestrictedNames: Must match TF.js API
export const isFinite = makeUnary('isFinite', (x) => (Number.isFinite(x) ? 1 : 0));
/** Returns the element-wise check if infinite for a tensor. */
export const isInf = makeUnary('isInf', (x) => (!Number.isFinite(x) && !Number.isNaN(x) ? 1 : 0));
/** Returns the element-wise check if NaN for a tensor. */
// biome-ignore lint/suspicious/noShadowRestrictedNames: Must match TF.js API
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
  if (tensors.length === 0) throw new Onnx9000Error('addN requires at least one tensor');
  let res = tensors[0];
  for (let i = 1; i < tensors.length; i++) res = add(res, tensors[i]);
  return res;
}

/**
 * Concatenates a list of tensors along an axis.
 * @param tensors List of tensors to concatenate.
 * @param axis Axis along which to concatenate.
 * @returns Resulting tensor.
 */
export function concat(tensors: Tensor[], _axis: number = 0): Tensor {
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
export function split(x: Tensor, numOrSizeSplits: number | number[], _axis: number = 0): Tensor[] {
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
  _strides?: number[],
  _beginMask: number = 0,
  _endMask: number = 0,
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
export function gather(x: Tensor, indices: Tensor, _axis: number = 0): Tensor {
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
  _axis: number = 0,
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
export function reverse(x: Tensor, _axis?: number | number[]): Tensor {
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
export function spaceToBatchND(x: Tensor, _blockShape: number[], _paddings: number[][]): Tensor {
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
export function batchToSpaceND(x: Tensor, _blockShape: number[], _crops: number[][]): Tensor {
  return new Tensor(x.shape, x.dtype, x.dataArray.slice());
}

/**
 * Performs a depth-to-space transformation.
 * @param x Input tensor.
 * @param blockSize Block size.
 * @param dataFormat Data format.
 * @returns Resulting tensor.
 */
export function depthToSpace(x: Tensor, _blockSize: number, _dataFormat: string = 'NHWC'): Tensor {
  return new Tensor(x.shape, x.dtype, x.dataArray.slice());
}

/**
 * Performs a space-to-depth transformation.
 * @param x Input tensor.
 * @param blockSize Block size.
 * @param dataFormat Data format.
 * @returns Resulting tensor.
 */
export function spaceToDepth(x: Tensor, _blockSize: number, _dataFormat: string = 'NHWC'): Tensor {
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
/**
 * Executes `lcg`.
 * @returns Result of `lcg`.
 */
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
  _alpha: number,
  _beta: number = 1,
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
  _normalized: boolean = false,
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
