/**
 * @fileoverview mhlo.ts
 * Provides mhlo functionality for the iree-compiler package.
 */
import { Operation, type Region, type Type, type Value } from '../../ir/core.js';

/**
 * Executes `createElementwiseOp`.
 * @param opcode Parameter `opcode`.
 * @returns Result of `createElementwiseOp`.
 */
function createElementwiseOp(opcode: string) {
  return (lhs: Value, rhs: Value, resultType: Type): Operation =>
    new Operation(opcode, [lhs, rhs], [resultType]);
}

/**
 * Executes `createUnaryOp`.
 * @param opcode Parameter `opcode`.
 * @returns Result of `createUnaryOp`.
 */
function createUnaryOp(opcode: string) {
  return (operand: Value, resultType: Type): Operation =>
    new Operation(opcode, [operand], [resultType]);
}

export const add = createElementwiseOp('web.mhlo.add');
export const subtract = createElementwiseOp('web.mhlo.subtract');
export const multiply = createElementwiseOp('web.mhlo.multiply');
export const divide = createElementwiseOp('web.mhlo.divide');
export const maximum = createElementwiseOp('web.mhlo.maximum');
export const minimum = createElementwiseOp('web.mhlo.minimum');

export const exponential = createUnaryOp('web.mhlo.exponential');
export const log = createUnaryOp('web.mhlo.log');
export const cosine = createUnaryOp('web.mhlo.cosine');
export const sine = createUnaryOp('web.mhlo.sine');

/**
 * Executes `dot`.
 * @param lhs Parameter `lhs`.
 * @param rhs Parameter `rhs`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `dot`.
 */
export function dot(lhs: Value, rhs: Value, resultType: Type): Operation {
  return new Operation('web.mhlo.dot', [lhs, rhs], [resultType]);
}

/**
 * Executes `convolution`.
 * @param lhs Parameter `lhs`.
 * @param rhs Parameter `rhs`.
 * @param windowStrides Parameter `windowStrides`.
 * @param padding Parameter `padding`.
 * @param lhsDilation Parameter `lhsDilation`.
 * @param rhsDilation Parameter `rhsDilation`.
 * @param windowReversal Parameter `windowReversal`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `convolution`.
 */
export function convolution(
  lhs: Value,
  rhs: Value,
  windowStrides: number[],
  padding: number[][],
  lhsDilation: number[],
  rhsDilation: number[],
  windowReversal: boolean[],
  resultType: Type,
): Operation {
  return new Operation('web.mhlo.convolution', [lhs, rhs], [resultType], {
    windowStrides,
    padding,
    lhsDilation,
    rhsDilation,
    windowReversal,
  });
}

/**
 * Executes `reduce`.
 * @param operands Parameter `operands`.
 * @param initValues Parameter `initValues`.
 * @param dimensions Parameter `dimensions`.
 * @param body Parameter `body`.
 * @param resultTypes Parameter `resultTypes`.
 * @returns Result of `reduce`.
 */
export function reduce(
  operands: Value[],
  initValues: Value[],
  dimensions: number[],
  body: Region,
  resultTypes: Type[],
): Operation {
  return new Operation(
    'web.mhlo.reduce',
    [...operands, ...initValues],
    resultTypes,
    { dimensions },
    [body],
  );
}

/**
 * Executes `reduceWindow`.
 * @param operands Parameter `operands`.
 * @param initValues Parameter `initValues`.
 * @param windowDimensions Parameter `windowDimensions`.
 * @param windowStrides Parameter `windowStrides`.
 * @param baseDilations Parameter `baseDilations`.
 * @param windowDilations Parameter `windowDilations`.
 * @param padding Parameter `padding`.
 * @param body Parameter `body`.
 * @param resultTypes Parameter `resultTypes`.
 * @returns Result of `reduceWindow`.
 */
export function reduceWindow(
  operands: Value[],
  initValues: Value[],
  windowDimensions: number[],
  windowStrides: number[],
  baseDilations: number[],
  windowDilations: number[],
  padding: number[][],
  body: Region,
  resultTypes: Type[],
): Operation {
  return new Operation(
    'web.mhlo.reduce_window',
    [...operands, ...initValues],
    resultTypes,
    {
      windowDimensions,
      windowStrides,
      baseDilations,
      windowDilations,
      padding,
    },
    [body],
  );
}

/**
 * Executes `select`.
 * @param pred Parameter `pred`.
 * @param onTrue Parameter `onTrue`.
 * @param onFalse Parameter `onFalse`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `select`.
 */
export function select(pred: Value, onTrue: Value, onFalse: Value, resultType: Type): Operation {
  return new Operation('web.mhlo.select', [pred, onTrue, onFalse], [resultType]);
}

/**
 * Executes `broadcastInDim`.
 * @param operand Parameter `operand`.
 * @param broadcastDimensions Parameter `broadcastDimensions`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `broadcastInDim`.
 */
export function broadcastInDim(
  operand: Value,
  broadcastDimensions: number[],
  resultType: Type,
): Operation {
  return new Operation('web.mhlo.broadcast_in_dim', [operand], [resultType], {
    broadcastDimensions,
  });
}

/**
 * Executes `reshape`.
 * @param operand Parameter `operand`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `reshape`.
 */
export function reshape(operand: Value, resultType: Type): Operation {
  return new Operation('web.mhlo.reshape', [operand], [resultType]);
}

/**
 * Executes `transpose`.
 * @param operand Parameter `operand`.
 * @param permutation Parameter `permutation`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `transpose`.
 */
export function transpose(operand: Value, permutation: number[], resultType: Type): Operation {
  return new Operation('web.mhlo.transpose', [operand], [resultType], {
    permutation,
  });
}

/**
 * Executes `concatenate`.
 * @param operands Parameter `operands`.
 * @param dimension Parameter `dimension`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `concatenate`.
 */
export function concatenate(operands: Value[], dimension: number, resultType: Type): Operation {
  return new Operation('web.mhlo.concatenate', operands, [resultType], {
    dimension,
  });
}

/**
 * Executes `slice`.
 * @param operand Parameter `operand`.
 * @param startIndices Parameter `startIndices`.
 * @param limitIndices Parameter `limitIndices`.
 * @param strides Parameter `strides`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `slice`.
 */
export function slice(
  operand: Value,
  startIndices: number[],
  limitIndices: number[],
  strides: number[],
  resultType: Type,
): Operation {
  return new Operation('web.mhlo.slice', [operand], [resultType], {
    startIndices,
    limitIndices,
    strides,
  });
}

/**
 * Executes `dynamicSlice`.
 * @param operand Parameter `operand`.
 * @param startIndices Parameter `startIndices`.
 * @param sliceSizes Parameter `sliceSizes`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `dynamicSlice`.
 */
export function dynamicSlice(
  operand: Value,
  startIndices: Value[],
  sliceSizes: number[],
  resultType: Type,
): Operation {
  return new Operation('web.mhlo.dynamic_slice', [operand, ...startIndices], [resultType], {
    sliceSizes,
  });
}

/**
 * Executes `gather`.
 * @param operand Parameter `operand`.
 * @param startIndices Parameter `startIndices`.
 * @param dimensionNumbers Parameter `dimensionNumbers`.
 * @param sliceSizes Parameter `sliceSizes`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `gather`.
 */
export function gather(
  operand: Value,
  startIndices: Value,
  dimensionNumbers: Record<string, ReturnType<typeof JSON.parse>>,
  sliceSizes: number[],
  resultType: Type,
): Operation {
  return new Operation('web.mhlo.gather', [operand, startIndices], [resultType], {
    dimensionNumbers,
    sliceSizes,
  });
}

/**
 * Executes `scatter`.
 * @param operand Parameter `operand`.
 * @param scatterIndices Parameter `scatterIndices`.
 * @param updates Parameter `updates`.
 * @param updateComputation Parameter `updateComputation`.
 * @param dimensionNumbers Parameter `dimensionNumbers`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `scatter`.
 */
export function scatter(
  operand: Value,
  scatterIndices: Value,
  updates: Value,
  updateComputation: Region,
  dimensionNumbers: Record<string, ReturnType<typeof JSON.parse>>,
  resultType: Type,
): Operation {
  return new Operation(
    'web.mhlo.scatter',
    [operand, scatterIndices, updates],
    [resultType],
    {
      dimensionNumbers,
    },
    [updateComputation],
  );
}
