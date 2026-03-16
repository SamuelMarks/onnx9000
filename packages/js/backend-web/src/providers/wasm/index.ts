import { Graph, Tensor } from '@onnx9000/core';
import { ExecutionProvider } from '../../session.js';

export class WasmProvider implements ExecutionProvider {
  name = 'Wasm';

  async initialize(): Promise<void> {
    // Mock WASM initialization
  }

  async execute(graph: Graph, inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    // Mock Wasm execution
    const results: Record<string, Tensor> = {};
    for (const name of graph.outputs) {
      const outName = typeof name === 'string' ? name : (name as any).name;
      results[outName] = new Tensor(outName, [1], 'float32');
    }
    return results;
  }
}
