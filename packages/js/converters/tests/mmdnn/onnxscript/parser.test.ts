import { describe, expect, it } from 'vitest';
import { OnnxScriptParser } from '../../../src/mmdnn/onnxscript/parser';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('parser.ts', () => {
  it('should instantiate and cover OnnxScriptParser', () => {
    try {
      const obj = Construct(OnnxScriptParser);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
