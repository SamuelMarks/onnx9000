/* eslint-disable */
// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { MathUtils } from '../../src/core/MathUtils';

describe('MathUtils normalize edge cases', () => {
  it('should cover flat line case completely', () => {
    // The previous test probably didn't actually hit the 'max - min === 0' branch securely
    // depending on the array data. Let's explicitly hit it.
    const result = MathUtils.normalize([42, 42, 42]);
    expect(result).toEqual([0.5, 0.5, 0.5]);
  });
});
