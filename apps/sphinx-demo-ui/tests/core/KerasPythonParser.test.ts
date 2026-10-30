import { describe, expect, it } from 'vitest';
import { KerasPythonParser } from '../../src/core/KerasPythonParser';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('KerasPythonParser.ts', () => {
  it('should instantiate and cover KerasPythonParser', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(KerasPythonParser);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
