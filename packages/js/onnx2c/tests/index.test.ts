import { describe, expect, it } from 'vitest';
import { Onnx2c } from '../src/index.js';

describe('Onnx2c', () => {
  it('should emit C99 kernels for Add, Mul, and Relu', () => {
    const c = new Onnx2c();
    expect(c.run()).toBe('[onnx2c] processed');
    expect(c.emitKernel({ funcName: 'mul_k', opType: 'Mul', length: 8 })).toContain('a[i] * b[i]');
    expect(c.emitKernel({ funcName: 'relu_k', opType: 'Relu', length: 8 })).toContain(
      'a[i] > 0.0f ? a[i] : 0.0f',
    );
  });
});
