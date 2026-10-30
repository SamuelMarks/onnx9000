import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { resnet18 } from '../src/models/resnet.js';

describe('ResNet', () => {
  it('should create and call', () => {
    const model = resnet18();
    expect(model).toBeDefined();
    const out = model.call(new Tensor('x', [1], 'float32'));
    expect(out).toBeDefined();
  });
});
