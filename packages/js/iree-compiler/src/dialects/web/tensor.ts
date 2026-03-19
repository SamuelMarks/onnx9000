import { Operation, Type, Value } from '../../ir/core.js';

export class TensorType implements Type {
  readonly id = 'tensor';
  readonly shape: number[];
  readonly elementType: string;

  constructor(shape: number[], elementType: string) {
    this.shape = [...shape];
    this.elementType = elementType;
  }
}

export function extract(tensor: Value, indices: Value[], resultType: Type): Operation {
  return new Operation('web.tensor.extract', [tensor, ...indices], [resultType]);
}

export function insert(
  tensor: Value,
  scalar: Value,
  indices: Value[],
  resultType: Type,
): Operation {
  return new Operation('web.tensor.insert', [tensor, scalar, ...indices], [resultType]);
}

export function splat(scalar: Value, resultType: Type): Operation {
  return new Operation('web.tensor.splat', [scalar], [resultType]);
}

export function pad(
  tensor: Value,
  padValue: Value,
  edgePaddingLow: number[],
  edgePaddingHigh: number[],
  interiorPadding: number[],
  resultType: Type,
): Operation {
  return new Operation('web.tensor.pad', [tensor, padValue], [resultType], {
    edgePaddingLow,
    edgePaddingHigh,
    interiorPadding,
  });
}
