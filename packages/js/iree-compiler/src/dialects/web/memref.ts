/**
 * @fileoverview memref.ts
 * Provides memref functionality for the iree-compiler package.
 */
import { Operation, type Type, type Value } from '../../ir/core.js';

/** Represents the `MemRefType` class. */
export class MemRefType implements Type {
  /** Property `id` of `MemRefType`. */
  readonly id = 'memref';
  /** Property `shape` of `MemRefType`. */
  readonly shape: number[];
  /** Property `elementType` of `MemRefType`. */
  readonly elementType: string;

  /**
   * Initializes a new `MemRefType` instance.
   * @param shape Parameter `shape`.
   * @param elementType Parameter `elementType`.
   */
  constructor(shape: number[], elementType: string) {
    this.shape = [...shape];
    this.elementType = elementType;
  }
}

// 48. web.memref.alloc
/**
 * Executes `alloc`.
 * @param type Parameter `type`.
 * @returns Result of `alloc`.
 */
export function alloc(type: MemRefType): Operation {
  return new Operation('web.memref.alloc', [], [type]);
}

// 49. web.memref.dealloc
/**
 * Executes `dealloc`.
 * @param memref Parameter `memref`.
 * @returns Result of `dealloc`.
 */
export function dealloc(memref: Value): Operation {
  return new Operation('web.memref.dealloc', [memref], []);
}

// 50. web.memref.load
/**
 * Executes `load`.
 * @param memref Parameter `memref`.
 * @param indices Parameter `indices`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `load`.
 */
export function load(memref: Value, indices: Value[], resultType: Type): Operation {
  return new Operation('web.memref.load', [memref, ...indices], [resultType]);
}

// 50. web.memref.store
/**
 * Executes `store`.
 * @param value Parameter `value`.
 * @param memref Parameter `memref`.
 * @param indices Parameter `indices`.
 * @returns Result of `store`.
 */
export function store(value: Value, memref: Value, indices: Value[]): Operation {
  return new Operation('web.memref.store', [value, memref, ...indices], []);
}
