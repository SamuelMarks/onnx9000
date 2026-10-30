import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { swinT } from '../src/models/swin.js';

describe('SwinTransformer', () => {
  it('should create and call', () => {
    const model = swinT();
    expect(model).toBeDefined();
    const out = model.call(new Tensor('x', [1], 'float32'));
    expect(out).toBeDefined();
  });
});
