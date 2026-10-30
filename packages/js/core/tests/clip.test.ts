import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { clipVitBasePatch16 } from '../src/models/clip.js';

describe('CLIP', () => {
  it('should create and call', () => {
    const model = clipVitBasePatch16();
    expect(model).toBeDefined();
    const out = model.call(new Tensor('img', [1], 'float32'), new Tensor('txt', [1], 'int64'));
    expect(out.length).toBe(2);
  });
});
