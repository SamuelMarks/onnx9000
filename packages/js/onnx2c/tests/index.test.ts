import { describe, expect, it } from 'vitest';
import { Onnx2c } from '../src/index.js';

describe('Onnx2c', () => {
  it('should run', () => {
    expect(new Onnx2c().run()).toBeDefined();
  });
});
