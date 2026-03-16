import { Graph } from '@onnx9000/core';

export function emitWGSL(graph: Graph): string {
  if (graph.nodes.length === 0) {
    throw new Error('Graph is empty');
  }
  return `
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        // mock WGSL emitted for ${graph.name}
    }
    `;
}
