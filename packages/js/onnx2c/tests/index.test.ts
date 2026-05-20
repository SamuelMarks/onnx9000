import { describe, it, expect } from 'vitest';
import { Onnx2c } from '../src/index';
describe('onnx2c', () => {
  it('runs', () => {
    expect(new Onnx2c().run()).toBe('[onnx2c] processed');
  });
});
