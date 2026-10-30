import { describe, expect, it } from 'vitest';
import { Sparse } from '../src/index.js';

describe('Sparse', () => {
  it('should compress and restore COO sparse tensors', () => {
    const s = new Sparse();
    expect(s.run()).toBe('[sparse] processed');

    const dense = new Float32Array([0.1, -3.0, 0.0, 5.0]);
    const coo = s.denseToCoo(dense, 0.5);
    expect(Array.from(coo.indices)).toEqual([1, 3]);
    expect(Array.from(coo.values)).toEqual([-3.0, 5.0]);

    const restored = s.cooToDense(coo);
    expect(Array.from(restored)).toEqual([0, -3.0, 0, 5.0]);
  });
});
