import { describe, expect, it } from 'vitest';
import { Onnx2tf } from '../src/index.js';

describe('Onnx2tf', () => {
  it('should run', () => {
    expect(new Onnx2tf().run()).toBeDefined();
  });
});
