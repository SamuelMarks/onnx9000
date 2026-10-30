import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { MatMul, Relu } from '../src/primitives.js';

describe('primitives', () => {
  it('should call relu', () => {
    const r = new Relu();
    const out = r.call(new Tensor('x', [1], 'float32'));
    expect(out.name).toBe('Relu_out');
  });

  it('should call matmul', () => {
    const m = new MatMul();
    const out = m.call(new Tensor('a', [1], 'float32'), new Tensor('b', [1], 'float32'));
    expect(out.name).toBe('MatMul_out');
  });
});
