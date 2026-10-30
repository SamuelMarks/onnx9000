/**
 * @fileoverview linalg.ts
 * Provides linalg functionality for the iree-compiler package.
 */
import { Operation, type Region, type Type, type Value } from '../../ir/core.js';

// 31. AffineMap class
/** Represents the `AffineExpr` class. */
export class AffineExpr {
  // Abstract base for affine expressions
}

/** Represents the `AffineDimExpr` class. */
export class AffineDimExpr extends AffineExpr {
  /**
   * Initializes a new `AffineDimExpr` instance.
   * @param position Parameter `position`.
   */
  constructor(public readonly position: number) {
    super();
  }
}

/** Represents the `AffineMap` class. */
export class AffineMap {
  /**
   * Initializes a new `AffineMap` instance.
   * @param numDims Parameter `numDims`.
   * @param numSymbols Parameter `numSymbols`.
   * @param results Parameter `results`.
   */
  constructor(
    public readonly numDims: number,
    public readonly numSymbols: number,
    public readonly results: AffineExpr[],
  ) {}

  /**
   * Executes `getMinorIdentity` on `AffineMap`.
   * @param numDims Parameter `numDims`.
   * @returns Result of `getMinorIdentity`.
   */
  static getMinorIdentity(numDims: number): AffineMap {
    const results = [];
    for (let i = 0; i < numDims; i++) {
      results.push(new AffineDimExpr(i));
    }
    return new AffineMap(numDims, 0, results);
  }
}

// 32, 33, 34. web.linalg.generic
/**
 * Executes `generic`.
 * @param inputs Parameter `inputs`.
 * @param outputs Parameter `outputs`.
 * @param indexingMaps Parameter `indexingMaps`.
 * @param iteratorTypes Parameter `iteratorTypes`.
 * @param body Parameter `body`.
 * @param resultTypes Parameter `resultTypes`.
 * @returns Result of `generic`.
 */
export function generic(
  inputs: Value[],
  outputs: Value[],
  indexingMaps: AffineMap[],
  iteratorTypes: string[],
  body: Region,
  resultTypes: Type[],
): Operation {
  return new Operation(
    'web.linalg.generic',
    [...inputs, ...outputs],
    resultTypes,
    {
      indexing_maps: indexingMaps,
      iterator_types: iteratorTypes,
      operand_segment_sizes: [inputs.length, outputs.length],
    },
    [body],
  );
}

// 35. web.linalg.matmul
/**
 * Executes `matmul`.
 * @param lhs Parameter `lhs`.
 * @param rhs Parameter `rhs`.
 * @param out Parameter `out`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `matmul`.
 */
export function matmul(lhs: Value, rhs: Value, out: Value, resultType: Type): Operation {
  return new Operation('web.linalg.matmul', [lhs, rhs, out], [resultType]);
}

// 36. web.linalg.batch_matmul
/**
 * Executes `batchMatmul`.
 * @param lhs Parameter `lhs`.
 * @param rhs Parameter `rhs`.
 * @param out Parameter `out`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `batchMatmul`.
 */
export function batchMatmul(lhs: Value, rhs: Value, out: Value, resultType: Type): Operation {
  return new Operation('web.linalg.batch_matmul', [lhs, rhs, out], [resultType]);
}

// 37. web.linalg.conv_2d_nhwc_hwcf
/**
 * Executes `conv2dNhwcHwcf`.
 * @param input Parameter `input`.
 * @param filter Parameter `filter`.
 * @param out Parameter `out`.
 * @param strides Parameter `strides`.
 * @param dilations Parameter `dilations`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `conv2dNhwcHwcf`.
 */
export function conv2dNhwcHwcf(
  input: Value,
  filter: Value,
  out: Value,
  strides: number[],
  dilations: number[],
  resultType: Type,
): Operation {
  return new Operation('web.linalg.conv_2d_nhwc_hwcf', [input, filter, out], [resultType], {
    strides,
    dilations,
  });
}

// 38. web.linalg.pooling_nhwc_max
/**
 * Executes `poolingNhwcMax`.
 * @param input Parameter `input`.
 * @param filter Parameter `filter`.
 * @param out Parameter `out`.
 * @param strides Parameter `strides`.
 * @param dilations Parameter `dilations`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `poolingNhwcMax`.
 */
export function poolingNhwcMax(
  input: Value,
  filter: Value,
  out: Value,
  strides: number[],
  dilations: number[],
  resultType: Type,
): Operation {
  return new Operation('web.linalg.pooling_nhwc_max', [input, filter, out], [resultType], {
    strides,
    dilations,
  });
}

// 39. web.linalg.fill
/**
 * Executes `fill`.
 * @param value Parameter `value`.
 * @param out Parameter `out`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `fill`.
 */
export function fill(value: Value, out: Value, resultType: Type): Operation {
  return new Operation('web.linalg.fill', [value, out], [resultType]);
}

// 40. web.linalg.yield
/**
 * Executes `yieldOp`.
 * @param values Parameter `values`.
 * @returns Result of `yieldOp`.
 */
export function yieldOp(values: Value[]): Operation {
  return new Operation('web.linalg.yield', values, []);
}
