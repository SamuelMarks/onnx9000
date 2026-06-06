import { describe, expect, it } from 'vitest';
import { Paddle2ONNXConverter } from '../src/index.js';

describe('Paddle2ONNXConverter', () => {
  it('should convert', () => {
    const c = new Paddle2ONNXConverter();
    expect(c.convert('model')).toContain('[ONNX-IR]');
    expect(() => c.convert('')).toThrow();
  });
});
