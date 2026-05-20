import { describe, it, expect } from 'vitest';
import { Paddle2ONNXConverter } from '../src/index';

describe('Paddle2ONNXConverter', () => {
  it('should convert a paddle model', () => {
    const converter = new Paddle2ONNXConverter();
    expect(converter.convert('paddle_model_data')).toBe('[ONNX-IR] from paddle_model_data');
  });

  it('should throw on empty string', () => {
    const converter = new Paddle2ONNXConverter();
    expect(() => converter.convert('')).toThrow('Invalid model string');
  });
});
