/**
 * @fileoverview index.ts
 * Provides index functionality for the compiler package.
 */

import type { Graph } from '@onnx9000/core';
import { Onnx9000Error } from '@onnx9000/core';

/**
 * Executes `emitWVM`.
 * @param graph Parameter `graph`.
 * @returns Result of `emitWVM`.
 */
export function emitWVM(graph: Graph): Uint8Array {
  if (graph.nodes.length === 0) {
    throw new Onnx9000Error('Graph is empty');
  }
  return new Uint8Array([0x57, 0x56, 0x4d]); // 'WVM' mock
}

/** Represents the `WVMInterpreter` class. */
export class WVMInterpreter {
  /** Property `bytecode` of `WVMInterpreter`. */
  private bytecode: Uint8Array;
  /**
   * Initializes a new `WVMInterpreter` instance.
   * @param bytecode Parameter `bytecode`.
   */
  constructor(bytecode: Uint8Array) {
    this.bytecode = bytecode;
  }
  /**
   * Executes `execute` on `WVMInterpreter`.
   * @returns Result of `execute`.
   */
  execute(): boolean {
    if (this.bytecode.length === 0) {
      throw new Onnx9000Error('Empty bytecode');
    }
    return true;
  }
}
