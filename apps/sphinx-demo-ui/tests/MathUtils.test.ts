// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { MathUtils } from '../src/core/MathUtils.js';

describe('MathUtils', () => {
  it('should calculate stats', () => {
    const arr = [1, 2, 3, 4, 5];
    expect(MathUtils.mean(arr)).toBe(3);
    expect(MathUtils.variance(arr)).toBe(2);

    const norm = MathUtils.normalize(arr);
    expect(norm[0]).toBe(0);
    expect(norm[4]).toBe(1);
  });
});
