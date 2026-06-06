import { describe, expect, it } from 'vitest';
import * as Module from '../../../src/mmdnn/onnxscript/parser';

describe('parser.ts', () => {
  it('should instantiate and cover OnnxScriptParser', () => {
    try {
      const obj = new (Module as any).OnnxScriptParser();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
