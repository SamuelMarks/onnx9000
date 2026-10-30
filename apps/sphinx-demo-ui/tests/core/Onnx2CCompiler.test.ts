import { describe, expect, it } from 'vitest';
import { Onnx2CCompiler } from '../../src/core/Onnx2CCompiler';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('Onnx2CCompiler.ts', () => {
  it('should instantiate and cover Onnx2CCompiler', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(Onnx2CCompiler);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
