/* eslint-disable */
// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { MathUtils } from '../../src/core/MathUtils';

describe('MathUtils', () => {
  it('calculates mean', () => {
    expect(MathUtils.mean([1, 2, 3, 4, 5])).toBe(3);
    expect(MathUtils.mean([])).toBe(0);
  });

  it('calculates variance', () => {
    expect(MathUtils.variance([1, 2, 3, 4, 5])).toBe(2);
    expect(MathUtils.variance([2, 2, 2])).toBe(0);
    expect(MathUtils.variance([])).toBe(0);
  });

  it('normalizes data', () => {
    const data = [10, 20, 30, 40, 50];
    const norm = MathUtils.normalize(data);
    expect(norm[0]).toBe(0); // min
    expect(norm[4]).toBe(1); // max
    expect(norm[2]).toBe(0.5); // mid

    expect(MathUtils.normalize([])).toEqual([]);
    expect(MathUtils.normalize([5, 5, 5])).toEqual([0.5, 0.5, 0.5]);
  });
});
