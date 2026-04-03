import { describe, it, expect, vi } from 'vitest';
import { Graph, Tensor, Node } from '@onnx9000/core';
import { WebGPUProvider } from '../src/providers/webgpu/index.js';
import { SPMM_CSR_WGSL } from '../src/providers/webgpu/shaders/spmm.js';

describe('WebGPUProvider Final', () => {
  it('should import shaders for coverage', () => {
    expect(SPMM_CSR_WGSL).toBeDefined();
  });

  it('should test createSparseBuffer', () => {
    const provider = new WebGPUProvider({} as any);
    (provider as any).device = { createBuffer: () => ({}) };
    const result = (provider as any).createSparseBuffer({
      format: 'CSR',
      values: new Uint8Array(4),
      row_ptr: new Uint8Array(4),
      col_indices: new Uint8Array(4),
    });
    expect(result).not.toBeNull();
    const resultNull = (provider as any).createSparseBuffer({ format: 'COO' });
    expect(resultNull).toBeNull();
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

  it('should request f16 features when useFP16 is true', async () => {
    const provider = new WebGPUProvider({ useFP16: true });
    // Mock navigator.gpu
    const mockRequestDevice = vi.fn().mockResolvedValue('device_fp16');
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue({
            requestDevice: mockRequestDevice,
          }),
        },
      },
      configurable: true,
    });
    await provider.initialize();
    expect(mockRequestDevice).toHaveBeenCalledWith({ requiredFeatures: ['shader-f16'] });
  });

  it('should ignore missing weight inputs for MatMul', async () => {
    const provider = new WebGPUProvider({});
    const g = new Graph('g');
    // MatMul with only 1 input
    g.nodes.push(new Node('MatMul', ['x'], ['y']));
    g.tensors['x'] = new Tensor('x', [2, 2], 'float32');

    // Should not throw and just pass through without dispatching Sparse
    const res = await provider.execute(g, { x: g.tensors['x'] });
    expect(res).toEqual({});
  });

  it('should ignore dense weights and unfound weights', async () => {
    const provider = new WebGPUProvider({});
    const g = new Graph('g');
    g.nodes.push(new Node('MatMul', ['in', 'w'], ['out']));
    g.tensors['w'] = new Tensor('w', [2, 2], 'float32', false, true, new Float32Array([1]));
    (g.tensors['w'] as any).format = 'dense';

    g.nodes.push(new Node('MatMul', ['in2', 'w_missing'], ['out2']));

    g.nodes.push(new Node('MatMul', ['in3', 'w_no_format'], ['out3']));
    g.tensors['w_no_format'] = new Tensor(
      'w_no_format',
      [2, 2],
      'float32',
      false,
      true,
      new Float32Array([1]),
    );

    const res = await provider.execute(g, {});
    expect(res).toEqual({});
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
