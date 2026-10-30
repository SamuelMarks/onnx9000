import { describe, expect, it } from 'vitest';
import { Onnx2tf } from '../src/index.js';

describe('Onnx2tf', () => {
  it('should transpose 4D NCHW shapes to NHWC and preserve non-4D shapes', () => {
    const tf = new Onnx2tf();
    expect(tf.run()).toBe('[onnx2tf] processed');
    expect(tf.nchwToNhwc([1, 3, 224, 224])).toEqual([1, 224, 224, 3]);
    expect(tf.nchwToNhwc([1, 10])).toEqual([1, 10]);
  });
});
