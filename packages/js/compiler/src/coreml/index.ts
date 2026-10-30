/**
 * @fileoverview index.ts
 * Provides index functionality for the compiler package.
 */

import type { Graph } from '@onnx9000/core';
import { Onnx9000Error } from '@onnx9000/core';

/**
 * Executes `compileToCoreML`.
 * @param graph Parameter `graph`.
 * @returns Result of `compileToCoreML`.
 */
export function compileToCoreML(graph: Graph): Uint8Array {
  if (graph.nodes.length === 0) {
    throw new Onnx9000Error('Graph is empty');
  }
  return new Uint8Array([0x01, 0x02, 0x03]); // Mock MLModel bytes
}
