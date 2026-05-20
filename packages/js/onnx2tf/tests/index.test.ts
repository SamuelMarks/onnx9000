import { describe, it, expect } from 'vitest';
import { Onnx2tf } from '../src/index';
describe('onnx2tf', () => {
  it('runs', () => {
    expect(new Onnx2tf().run()).toBe('[onnx2tf] processed');
  });
});
