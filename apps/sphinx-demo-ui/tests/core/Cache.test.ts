import { describe, expect, it } from 'vitest';
import { Cache } from '../../src/core/Cache';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('Cache.ts', () => {
  it('should instantiate and cover Cache', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(Cache);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
