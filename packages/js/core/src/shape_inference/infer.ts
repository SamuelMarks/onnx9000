import { Graph } from '../ir/graph.js';
import { Tensor, Shape } from '../ir/tensor.js';

export function inferShapes(graph: Graph): void {
  // A minimal placeholder for static shape inference.
  // It iterates topologically over nodes and assigns default output shapes
  // if they are missing in ValueInfo.

  const shapeMap = new Map<string, Shape>();

  // Initialize with inputs
  for (const input of graph.inputs) {
    shapeMap.set(input.name, input.shape);
  }

  // Initialize with constants
  for (const init of graph.initializers) {
    const t = graph.tensors[init];
    if (t) {
      shapeMap.set(init, t.shape);
    }
  }

  // Very naive forward pass
  for (const node of graph.nodes) {
    // Basic heuristics: if it's an elementwise op, output shape = input shape
    if (node.inputs.length > 0 && node.inputs[0]) {
      const firstInputShape = shapeMap.get(node.inputs[0]);
      if (firstInputShape) {
        for (const output of node.outputs) {
          if (!shapeMap.has(output)) {
            // we assume output shape matches first input shape as a fallback
            shapeMap.set(output, firstInputShape);
          }
        }
      }
    }
  }
}
