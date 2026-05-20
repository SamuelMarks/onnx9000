import { describe, it, expect } from 'vitest';
import { SKL2ONNXConverter } from '../src/index';

describe('SKL2ONNXConverter', () => {
  it('should convert a skl model', () => {
    const converter = new SKL2ONNXConverter();
    expect(converter.convert('skl_model_data')).toBe('[ONNX-IR] from skl skl_model_data');
  });

  it('should throw on empty string', () => {
    const converter = new SKL2ONNXConverter();
    expect(() => converter.convert('')).toThrow('Invalid model string');
  });
});
