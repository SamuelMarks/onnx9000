import { describe, expect, it } from 'vitest';
import { ArrayAPI } from '../src/array/index.js';

describe('ArrayAPI', () => {
  it('should array', () => {
    expect(ArrayAPI.add([1], [2])).toEqual([3]);
    expect(ArrayAPI.softmax([1, 2])).toBeDefined();
    expect(ArrayAPI.sigmoid([1])).toBeDefined();
    expect(ArrayAPI.cosine_similarity([1], [1])).toBeCloseTo(1.0);
  });
});
