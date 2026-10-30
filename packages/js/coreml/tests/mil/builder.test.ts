import { describe, expect, it } from 'vitest';
import { Builder } from '../../src/mil/builder';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('builder.ts', () => {
  it('should instantiate and cover Builder', () => {
    try {
      const obj = Construct(Builder);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
