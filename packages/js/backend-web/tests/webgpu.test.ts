import { Tensor } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { MatMulWebGPU, WebGPUProvider } from '../src/providers/webgpu/index.js';

describe('WebGPUProvider', () => {
  it('should execute matmul fallback', () => {
    const op = new MatMulWebGPU();
    const t2 = Object.assign(new Tensor('b', [1], 'float32'), { format: 'dense' });
    const res = op.execute([new Tensor('a', [1], 'float32'), t2], {});
    expect(res.length).toBe(0);
  });

  it('should initialize', async () => {
    const prov = new WebGPUProvider();
    (global as typeof globalThis & { navigator: unknown }).navigator = {
      gpu: {
        requestAdapter: async () => ({ requestDevice: async () => ({}) }),
      },
    };
    await prov.initialize();
    expect((prov as never as { device: unknown }).device).toBeDefined();
  });
});
