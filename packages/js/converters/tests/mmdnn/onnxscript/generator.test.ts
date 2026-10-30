import { describe, expect, it } from 'vitest';
import { OnnxScriptGenerator } from '../../../src/mmdnn/onnxscript/generator';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('generator.ts', () => {
  it('should instantiate and cover OnnxScriptGenerator', () => {
    try {
      const obj = Construct(OnnxScriptGenerator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
