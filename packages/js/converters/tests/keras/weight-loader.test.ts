import { describe, it, expect } from 'vitest';
import { calculateByteLength } from '../src/keras/weight-loader.js';

describe('weight-loader', () => {
  it('should calc byte length', () => {
    expect(calculateByteLength({ dtype: 'float32', shape: [10] } as any)).toBe(40);
  });
});
