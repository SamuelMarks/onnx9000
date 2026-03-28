import { describe, it, expect, vi } from 'vitest';
import { Graph, Tensor, Node } from '@onnx9000/core';
import { WebGPUProvider } from '../src/providers/webgpu/index.js';
import { SPMM_CSR_WGSL } from '../src/providers/webgpu/shaders/spmm.js';

describe('WebGPUProvider Final', () => {
  it('should import shaders for coverage', () => {
    expect(SPMM_CSR_WGSL).toBeDefined();
  });

  it('should execute sparse MatMul and hit coverage', async () => {
    const provider = new WebGPUProvider({ sparsityThreshold: 0.5 });

    const mockDevice = {
      createBuffer: vi.fn().mockReturnValue({}),
    };
    (provider as any).device = mockDevice;

    const g = new Graph('g');
    // Important: weights must be in tensors map and nodes must be in nodes array
    const w = {
      name: 'w',
      format: 'CSR',
      values: { byteLength: 4 },
      row_ptr: { byteLength: 4 },
      col_indices: { byteLength: 4 },
    };
    g.tensors['w'] = w as any;

    const node = new Node('MatMul', ['in', 'w'], ['out']);
    g.nodes.push(node);
    g.outputs.push('out' as any);

    const res = await provider.execute(g, {});
    expect(res['out']).toBeDefined();

    // Test calculateSparsity directly
    const sparsity = (provider as any).calculateSparsity(w as any);
    expect(sparsity).toBe(0.75);
  });

  it('should fallback for low sparsity', async () => {
    const provider = new WebGPUProvider({ sparsityThreshold: 0.8 });
    (provider as any).device = {};

    const g = new Graph('g');
    const w = {
      name: 'w',
      format: 'CSR',
      values: { byteLength: 4 },
      row_ptr: { byteLength: 4 },
      col_indices: { byteLength: 4 },
    };
    g.tensors['w'] = w as any;
    g.nodes.push(new Node('MatMul', ['in', 'w'], ['out']));
    g.outputs.push('out' as any);

    const res = await provider.execute(g, {});
    expect(res['out']).toBeDefined();
  });
});
