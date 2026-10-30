import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { llama7b } from '../src/models/llama.js';

describe('LLaMA', () => {
  it('should create and call', () => {
    const model = llama7b();
    expect(model).toBeDefined();
    const out = model.call(new Tensor('x', [1], 'int64'), new Tensor('pos', [1], 'int64'));
    expect(out).toBeDefined();
  });
});
