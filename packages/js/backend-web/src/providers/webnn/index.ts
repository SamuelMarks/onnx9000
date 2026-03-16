import { Graph, Tensor } from '@onnx9000/core';
import { ExecutionProvider } from '../../session.js';

export class WebNNProvider implements ExecutionProvider {
  name = 'WebNN';

  async initialize(): Promise<void> {
    // @ts-ignore
    if (typeof navigator === 'undefined' || !navigator.ml) {
      throw new Error('WebNN is not supported in this environment.');
    }
  }

  async execute(graph: Graph, inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    // Mock WebNN execution
    const results: Record<string, Tensor> = {};
    for (const name of graph.outputs) {
      const outName = typeof name === 'string' ? name : (name as any).name;
      results[outName] = new Tensor(outName, [1], 'float32');
    }
    return results;
  }
}
