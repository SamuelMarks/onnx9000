import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { rwkvV4 } from '../src/models/rwkv.js';

describe('RWKV', () => {
  it('should create and call', () => {
    const model = rwkvV4();
    expect(model).toBeDefined();
    const out = model.call(new Tensor('x', [1], 'int64'));
    expect(out).toBeDefined();
  });
});
