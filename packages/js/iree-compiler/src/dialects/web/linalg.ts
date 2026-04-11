/* eslint-disable */
import { Operation, Type, Value, Region } from '../../ir/core.js';

// 31. AffineMap class
export class AffineExpr {
  // Abstract base for affine expressions
}

export class AffineDimExpr extends AffineExpr {
  constructor(public readonly position: number) {
    super();
  }
}

export class AffineMap {
  constructor(
    public readonly numDims: number,
    public readonly numSymbols: number,
    public readonly results: AffineExpr[],
  ) {}

  static getMinorIdentity(numDims: number): AffineMap {
    const results = [];
    for (let i = 0; i < numDims; i++) {
      results.push(new AffineDimExpr(i));
    }
    return new AffineMap(numDims, 0, results);
  }
}

// 32, 33, 34. web.linalg.generic
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
export function matmul(lhs: Value, rhs: Value, out: Value, resultType: Type): Operation {
  return new Operation('web.linalg.matmul', [lhs, rhs, out], [resultType]);
}

// 36. web.linalg.batch_matmul
export function batchMatmul(lhs: Value, rhs: Value, out: Value, resultType: Type): Operation {
  return new Operation('web.linalg.batch_matmul', [lhs, rhs, out], [resultType]);
}

// 37. web.linalg.conv_2d_nhwc_hwcf
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
export function fill(value: Value, out: Value, resultType: Type): Operation {
  return new Operation('web.linalg.fill', [value, out], [resultType]);
}

// 40. web.linalg.yield
export function yieldOp(values: Value[]): Operation {
  return new Operation('web.linalg.yield', values, []);
}
