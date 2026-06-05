import { describe, it, expect } from 'vitest';
import { clipVitBasePatch16 } from '../src/models/clip.js';

describe('CLIP', () => {
  it('should create and call', () => {
    const model = clipVitBasePatch16();
    expect(model).toBeDefined();
    const out = model.call({} as any, {} as any);
    expect(out.length).toBe(2);
  });
});
