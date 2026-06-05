import { describe, it, expect } from 'vitest';
import { swinT } from '../src/models/swin.js';

describe('SwinTransformer', () => {
  it('should create and call', () => {
    const model = swinT();
    expect(model).toBeDefined();
    const out = model.call({} as any);
    expect(out).toBeDefined();
  });
});
