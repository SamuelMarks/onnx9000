import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { mobilevitS } from '../src/models/mobilevit.js';

describe('MobileViT', () => {
  it('should create and call', () => {
    const model = mobilevitS();
    expect(model).toBeDefined();
    const out = model.call(new Tensor('x', [1], 'float32'));
    expect(out).toBeDefined();
  });
});
