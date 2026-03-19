import { describe, it, expect } from 'vitest';
import { exportModel, optimize, quantize, Quantizer } from '../src/index';

describe('optimum API', () => {
  it('should export', async () => {
    await expect(exportModel('id', 'dir')).resolves.toBeUndefined();
  });

  it('should optimize', async () => {
    const buffer = new ArrayBuffer(8);
    const optimized = await optimize(buffer);
    expect(optimized).toBe(buffer);
  });

  it('should quantize', async () => {
    const buffer = new ArrayBuffer(8);
    const quantized = await quantize(buffer);
    expect(quantized).toBe(buffer);
  });

  it('should provide Quantizer wrapper', async () => {
    const quantizer = new Quantizer();
    const buffer = new ArrayBuffer(8);
    const res = await quantizer.quantize(buffer, {});
    expect(res).toBe(buffer);
  });
});
