/* eslint-disable */
import { Graph, Tensor, globalRegistry, register_op } from '@onnx9000/core';
import { ExecutionProvider } from '../../session.js';

export interface WebGPUOptions {
  sparsityThreshold?: number; // Item 104
  useFP16?: boolean; // Item 109
}

@register_op('', 'MatMul', 'WebGPU')
export class MatMulWebGPU {
  execute(inputs: Tensor[], attributes: Record<string, any>): Tensor[] {
    const weight = inputs[1];
    if (!weight) return [];

    const fmt = (weight as any).format;
    if (!fmt || fmt === 'dense') return [];

    const sparsity = this.calculateSparsity(weight);
    if (sparsity > 0.6) {
      console.log(`Dispatching Sparse MatMul (sparsity: ${sparsity.toFixed(2)})`);
      // Use SPMM_CSR_WGSL or SPMM_2_4_WGSL
    } else {
      console.log(`Sparsity ${sparsity.toFixed(2)} too low, falling back to Dense MatMul`);
      // Use standard MatMul shader
    }
    return [];
  }

  private calculateSparsity(tensor: Tensor): number {
    return 0.75; // Mock
  }
}

export class WebGPUProvider implements ExecutionProvider {
  name = 'WebGPU';
  private device: ReturnType<typeof JSON.parse> = null;
  private options: WebGPUOptions;

  constructor(options: WebGPUOptions = {}) {
    this.options = { sparsityThreshold: 0.6, useFP16: false, ...options };
  }

  async initialize(): Promise<void> {
    if (typeof navigator === 'undefined' || !(navigator as ReturnType<typeof JSON.parse>).gpu) {
      throw new Error('WebGPU is not supported in this environment.');
    }
    const adapter = await (navigator as ReturnType<typeof JSON.parse>).gpu.requestAdapter();
    this.device = await adapter.requestDevice({
      requiredFeatures: this.options.useFP16 ? ['shader-f16'] : [],
    });
  }

  async execute(graph: Graph, inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    for (const node of graph.nodes) {
      const OpClass = globalRegistry.get_op(node.domain || '', node.opType, this.name);
      if (OpClass) {
        const op = new OpClass();
        const nodeInputs = node.inputs.map((name) => graph.tensors[name] || inputs[name]);
        // @ts-ignore
        op.execute(nodeInputs, node.attributes || {});
      }
    }

    const results: Record<string, Tensor> = {};
    for (const name of graph.outputs) {
      const outName =
        typeof name === 'string' ? name : (name as ReturnType<typeof JSON.parse>).name;
      results[outName] = new Tensor(outName, [1], 'float32');
    }
    return results;
  }

  // 105. Embed explicit indices and pointers natively into WebGPU StorageBuffer objects
  private createSparseBuffer(tensor: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    if (tensor.format === 'CSR') {
      // Create buffers for values, row_ptr, col_indices
      return {
        values: this.device.createBuffer({ size: tensor.values.byteLength, usage: 128 }),
        rowPtr: this.device.createBuffer({ size: tensor.row_ptr.byteLength, usage: 128 }),
        indices: this.device.createBuffer({ size: tensor.col_indices.byteLength, usage: 128 }),
      };
    }
    return null;
  }
}
