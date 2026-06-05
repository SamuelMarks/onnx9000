import { describe, it, expect } from 'vitest';
import { Keras2ONNXConverter } from '../src/index.js';

describe('Keras2ONNXConverter', () => {
  it('should convert', () => {
    const conv = new Keras2ONNXConverter();
    expect(conv.convert('test')).toBeDefined();
    expect(() => conv.convert('')).toThrow();
  });
});
