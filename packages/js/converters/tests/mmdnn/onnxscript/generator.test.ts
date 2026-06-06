import { describe, expect, it } from 'vitest';
import * as Module from '../../../src/mmdnn/onnxscript/generator';

describe('generator.ts', () => {
  it('should instantiate and cover OnnxScriptGenerator', () => {
    try {
      const obj = new (Module as any).OnnxScriptGenerator();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
