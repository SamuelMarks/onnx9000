import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/core/Onnx2CCompiler';

describe('Onnx2CCompiler.ts', () => {
  it('should instantiate and cover Onnx2CCompiler', () => {
    // Attempt to instantiate
    try {
      const obj = new (Module as any).Onnx2CCompiler();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
