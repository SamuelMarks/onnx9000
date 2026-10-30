import { describe, expect, it } from 'vitest';
import { Paddle2ONNXConverter } from '../src/index.js';

describe('Paddle2ONNXConverter', () => {
  it('should convert Paddle model strings and map operator names', () => {
    const conv = new Paddle2ONNXConverter();
    expect(conv.convert('pp_ocr_v4')).toBe('[ONNX-IR] from pp_ocr_v4');
    expect(() => conv.convert('')).toThrow('Invalid model string');

    expect(conv.mapOp('conv2d')).toBe('Conv');
    expect(conv.mapOp('custom_paddle_op')).toBe('custom_paddle_op');
  });
});
