import { describe, it, expect } from 'vitest';
import { Onnxchecker } from '../src/index';
describe('onnx-checker', () => {
  it('runs', () => {
    expect(new Onnxchecker().run()).toBe('[onnx-checker] processed');
  });
});
