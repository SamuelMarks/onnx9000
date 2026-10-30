/**
 * @fileoverview index.ts
 * Provides index functionality for the compiler package.
 */

import type { Graph } from '@onnx9000/core';
import { Onnx9000Error } from '@onnx9000/core';

/**
 * Executes `compileToIREE`.
 * @param graph Parameter `graph`.
 * @returns Result of `compileToIREE`.
 */
export function compileToIREE(graph: Graph): Uint8Array {
  if (graph.nodes.length === 0) {
    throw new Onnx9000Error('Graph is empty');
  }
  return new Uint8Array([0x49, 0x52, 0x45, 0x45]); // 'IREE' mock
}
