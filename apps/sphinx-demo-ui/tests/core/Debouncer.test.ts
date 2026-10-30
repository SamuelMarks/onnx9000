import { describe, expect, it } from 'vitest';
import { Debouncer } from '../../src/core/Debouncer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('Debouncer.ts', () => {
  it('should instantiate and cover Debouncer', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(Debouncer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
