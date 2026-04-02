import { Graph, ValueInfo } from '../ir/graph.js';
import { Tensor, Shape, DType } from '../ir/tensor.js';

/**
 * Performs naive shape and dtype inference on an ONNX graph.
 * This function iterates through the nodes in topological order and propagates
 * shapes and types from inputs to outputs using basic heuristics.
 *
 * @param graph The ONNX graph to perform inference on.
 */
export function inferShapes(graph: Graph): void {
  const shapeMap = new Map<string, Shape>();
  const dtypeMap = new Map<string, DType>();

  // Initialize with inputs
  for (const input of graph.inputs) {
    shapeMap.set(input.name, input.shape);
    dtypeMap.set(input.name, input.dtype);
  }

  // Initialize with constants
  for (const init of graph.initializers) {
    const t = graph.tensors[init];
    if (t) {
      shapeMap.set(init, t.shape);
      dtypeMap.set(init, t.dtype);
    }
  }

  // Determine which outputs are already tracked
  const trackedOutputs = new Set<string>();
  for (const vi of graph.outputs) trackedOutputs.add(vi.name);
  for (const vi of graph.inputs) trackedOutputs.add(vi.name);
  for (const vi of graph.valueInfo) trackedOutputs.add(vi.name);

  // Very naive forward pass
  for (const node of graph.nodes) {
    // Basic heuristics: if it's an elementwise op, output shape = input shape
    if (node.inputs.length > 0 && node.inputs[0]) {
      const firstInputShape = shapeMap.get(node.inputs[0]);
      const firstInputDType = dtypeMap.get(node.inputs[0]) || 'float32';

      if (firstInputShape) {
        for (const output of node.outputs) {
          if (!shapeMap.has(output)) {
            // we assume output shape matches first input shape as a fallback
            shapeMap.set(output, firstInputShape);
            dtypeMap.set(output, firstInputDType);

            if (!trackedOutputs.has(output)) {
              graph.valueInfo.push(new ValueInfo(output, firstInputShape, firstInputDType));
              trackedOutputs.add(output);
            }
          }
        }
      }
    }
  }
}
