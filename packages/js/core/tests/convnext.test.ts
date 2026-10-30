import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { convnextTiny } from '../src/models/convnext.js';

describe('ConvNeXt', () => {
  it('should create and call', () => {
    const model = convnextTiny();
    expect(model).toBeDefined();
    const out = model.call(new Tensor('x', [1], 'float32'));
    expect(out).toBeDefined();
  });
});
