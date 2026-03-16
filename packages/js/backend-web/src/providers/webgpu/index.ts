import { Graph, Tensor } from '@onnx9000/core';
import { ExecutionProvider } from '../../session.js';

export class WebGPUProvider implements ExecutionProvider {
  name = 'WebGPU';

  async initialize(): Promise<void> {
    if (typeof navigator === 'undefined' || !(navigator as any).gpu) {
      throw new Error('WebGPU is not supported in this environment.');
    }
    await (navigator as any).gpu.requestAdapter();
  }

  async execute(graph: Graph, inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    // Mock WebGPU execution
    const results: Record<string, Tensor> = {};
    for (const name of graph.outputs) {
      const outName = typeof name === 'string' ? name : (name as any).name;
      results[outName] = new Tensor(outName, [1], 'float32');
    }
    return results;
  }
}
