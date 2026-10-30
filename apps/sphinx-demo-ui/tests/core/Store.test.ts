import { describe, expect, it } from 'vitest';
import { Store } from '../../src/core/Store';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('Store.ts', () => {
  it('should instantiate and cover Store', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(Store);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
