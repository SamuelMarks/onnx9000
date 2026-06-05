import { describe, it, expect } from 'vitest';
import { resnet18 } from '../src/models/resnet.js';

describe('ResNet', () => {
  it('should create and call', () => {
    const model = resnet18();
    expect(model).toBeDefined();
    const out = model.call({} as any);
    expect(out).toBeDefined();
  });
});
