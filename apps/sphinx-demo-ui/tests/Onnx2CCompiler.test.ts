// @ts-nocheck
import { describe, expect, it, vi } from 'vitest';
import { Onnx2CCompiler } from '../src/core/Onnx2CCompiler.js';

vi.mock('../src/core/WorkerManager.js', () => ({
  WorkerManager: {
    getInstance: vi.fn().mockReturnValue({
      execute: vi.fn().mockResolvedValue('float tensor_a[10]; malloc(40);'),
    }),
  },
}));

describe('Onnx2CCompiler', () => {
  it('should compile and calculate footprint', async () => {
    const comp = new Onnx2CCompiler();
    const code = await comp.compile(new Uint8Array());
    expect(code).toContain('tensor_a');

    const mem = Onnx2CCompiler.calculateMemoryFootprint(code);
    expect(mem).toBe(80); // 10*4 + 40
  });
});
