import { Graph } from '@onnx9000/core';

export function compileToIREE(graph: Graph): Uint8Array {
  if (graph.nodes.length === 0) {
    throw new Error('Graph is empty');
  }
  return new Uint8Array([0x49, 0x52, 0x45, 0x45]); // 'IREE' mock
}
