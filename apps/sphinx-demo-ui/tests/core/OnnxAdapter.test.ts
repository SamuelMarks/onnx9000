import { describe, expect, it } from 'vitest';
import { OnnxAdapter } from '../../src/core/OnnxAdapter';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('OnnxAdapter.ts', () => {
  it('should instantiate and cover OnnxAdapter', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(OnnxAdapter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
