import { describe, it, expect } from 'vitest';
import { Keras2ONNXConverter } from '../src/index';

describe('Keras2ONNXConverter', () => {
  it('should convert a keras model', () => {
    const converter = new Keras2ONNXConverter();
    expect(converter.convert('keras_model_data')).toBe('[ONNX-IR] from keras keras_model_data');
  });

  it('should throw on empty string', () => {
    const converter = new Keras2ONNXConverter();
    expect(() => converter.convert('')).toThrow('Invalid model string');
  });
});
