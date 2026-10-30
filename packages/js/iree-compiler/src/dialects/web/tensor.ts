/**
 * @fileoverview tensor.ts
 * Provides tensor functionality for the iree-compiler package.
 */
import { Operation, type Type, type Value } from '../../ir/core.js';

/** Represents the `TensorType` class. */
export class TensorType implements Type {
  /** Property `id` of `TensorType`. */
  readonly id = 'tensor';
  /** Property `shape` of `TensorType`. */
  readonly shape: number[];
  /** Property `elementType` of `TensorType`. */
  readonly elementType: string;

  /**
   * Initializes a new `TensorType` instance.
   * @param shape Parameter `shape`.
   * @param elementType Parameter `elementType`.
   */
  constructor(shape: number[], elementType: string) {
    this.shape = [...shape];
    this.elementType = elementType;
  }
}

/**
 * Executes `extract`.
 * @param tensor Parameter `tensor`.
 * @param indices Parameter `indices`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `extract`.
 */
export function extract(tensor: Value, indices: Value[], resultType: Type): Operation {
  return new Operation('web.tensor.extract', [tensor, ...indices], [resultType]);
}

/**
 * Executes `insert`.
 * @param tensor Parameter `tensor`.
 * @param scalar Parameter `scalar`.
 * @param indices Parameter `indices`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `insert`.
 */
export function insert(
  tensor: Value,
  scalar: Value,
  indices: Value[],
  resultType: Type,
): Operation {
  return new Operation('web.tensor.insert', [tensor, scalar, ...indices], [resultType]);
}

/**
 * Executes `splat`.
 * @param scalar Parameter `scalar`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `splat`.
 */
export function splat(scalar: Value, resultType: Type): Operation {
  return new Operation('web.tensor.splat', [scalar], [resultType]);
}

/**
 * Executes `pad`.
 * @param tensor Parameter `tensor`.
 * @param padValue Parameter `padValue`.
 * @param edgePaddingLow Parameter `edgePaddingLow`.
 * @param edgePaddingHigh Parameter `edgePaddingHigh`.
 * @param interiorPadding Parameter `interiorPadding`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `pad`.
 */
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
