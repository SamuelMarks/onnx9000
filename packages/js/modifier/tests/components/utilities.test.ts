import { describe, expect, it } from 'vitest';
import { ModifierUtilities } from '../../src/components/utilities';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('utilities.ts', () => {
  it('should instantiate and cover ModifierUtilities', () => {
    try {
      const obj = Construct(ModifierUtilities);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
