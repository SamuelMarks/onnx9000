import { Graph } from '@onnx9000/core';

export function compileToCoreML(graph: Graph): Uint8Array {
  if (graph.nodes.length === 0) {
    throw new Error('Graph is empty');
  }
  return new Uint8Array([0x01, 0x02, 0x03]); // Mock MLModel bytes
}
