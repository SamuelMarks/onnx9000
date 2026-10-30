import { describe, expect, it } from 'vitest';
import { Mlirlowering } from '../src/index.js';

describe('Mlirlowering', () => {
  it('should lower ONNX operators to mhlo, tosa, and linalg dialects', () => {
    const lowering = new Mlirlowering();
    expect(lowering.run()).toBe('[mlir-lowering] processed');

    expect(lowering.lowerOps(['Add', 'CustomOp'], 'tosa')).toEqual([
      { onnxOp: 'Add', mlirOp: 'tosa.add' },
      { onnxOp: 'CustomOp', mlirOp: 'tosa.custom_customop' },
    ]);
    expect(lowering.lowerOps(['MatMul'])).toEqual([
      { onnxOp: 'MatMul', mlirOp: 'mhlo.dot_general' },
    ]);
  });
});
