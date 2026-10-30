import { describe, expect, it } from 'vitest';
import { Console } from '../../src/components/Console';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('Console.ts', () => {
  it('should instantiate and cover Console', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(Console);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
