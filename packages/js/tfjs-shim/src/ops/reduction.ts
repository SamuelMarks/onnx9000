/**
 * @fileoverview Reduction and cumulative aggregation operations for @onnx9000/tfjs-shim.
 */

import { Tensor } from '../tensor.js';

// Reductions
/**
 * Returns the indices of the maximum values along an axis.
 * @param x Input tensor.
 * @param axis Axis to compute over.
 * @returns Tensor of indices.
 */
export function argMax(x: Tensor, _axis: number = 0): Tensor {
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
export function argMin(x: Tensor, _axis: number = 0): Tensor {
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
 * Computes the cumulative product of the tensor along an axis.
 * @param x The input tensor.
 * @param axis The axis to compute along.
 * @param exclusive Whether to perform exclusive cumulative product.
 * @param reverse Whether to compute in reverse.
 * @returns The resulting tensor.
 */
export function cumprod(
  x: Tensor,
  _axis: number = 0,
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
  _axis: number = 0,
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
