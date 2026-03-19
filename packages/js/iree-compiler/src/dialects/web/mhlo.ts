import { Operation, Type, Value, Region } from '../../ir/core.js';

function createElementwiseOp(opcode: string) {
  return function (lhs: Value, rhs: Value, resultType: Type): Operation {
    return new Operation(opcode, [lhs, rhs], [resultType]);
  };
}

function createUnaryOp(opcode: string) {
  return function (operand: Value, resultType: Type): Operation {
    return new Operation(opcode, [operand], [resultType]);
  };
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

export function dot(lhs: Value, rhs: Value, resultType: Type): Operation {
  return new Operation('web.mhlo.dot', [lhs, rhs], [resultType]);
}

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

export function select(pred: Value, onTrue: Value, onFalse: Value, resultType: Type): Operation {
  return new Operation('web.mhlo.select', [pred, onTrue, onFalse], [resultType]);
}

export function broadcastInDim(
  operand: Value,
  broadcastDimensions: number[],
  resultType: Type,
): Operation {
  return new Operation('web.mhlo.broadcast_in_dim', [operand], [resultType], {
    broadcastDimensions,
  });
}

export function reshape(operand: Value, resultType: Type): Operation {
  return new Operation('web.mhlo.reshape', [operand], [resultType]);
}

export function transpose(operand: Value, permutation: number[], resultType: Type): Operation {
  return new Operation('web.mhlo.transpose', [operand], [resultType], { permutation });
}

export function concatenate(operands: Value[], dimension: number, resultType: Type): Operation {
  return new Operation('web.mhlo.concatenate', operands, [resultType], { dimension });
}

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

export function gather(
  operand: Value,
  startIndices: Value,
  dimensionNumbers: Record<string, any>,
  sliceSizes: number[],
  resultType: Type,
): Operation {
  return new Operation('web.mhlo.gather', [operand, startIndices], [resultType], {
    dimensionNumbers,
    sliceSizes,
  });
}

export function scatter(
  operand: Value,
  scatterIndices: Value,
  updates: Value,
  updateComputation: Region,
  dimensionNumbers: Record<string, any>,
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
