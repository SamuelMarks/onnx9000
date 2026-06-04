import { describe, it, expect } from 'vitest';
import { Onnxscript } from '../src/index.js';
describe('onnx-script', () => {
  it('runs', () => {
    expect(new Onnxscript().run()).toBe('[onnx-script] processed');
  });
});
