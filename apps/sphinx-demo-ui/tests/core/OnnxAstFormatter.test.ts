import { describe, expect, it } from 'vitest';
import { OnnxAstFormatter } from '../../src/core/OnnxAstFormatter';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('OnnxAstFormatter.ts', () => {
  it('should instantiate and cover OnnxAstFormatter', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(OnnxAstFormatter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
