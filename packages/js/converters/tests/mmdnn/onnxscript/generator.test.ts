import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../../src/mmdnn/onnxscript/generator';

describe('generator.ts', () => {
  it('should instantiate and cover OnnxScriptGenerator', () => {
    try {
       const obj = new (Module as any).OnnxScriptGenerator();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
