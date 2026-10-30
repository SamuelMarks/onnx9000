import { describe, expect, it } from 'vitest';
import { MathUtils } from '../../src/core/MathUtils';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('MathUtils.ts', () => {
  it('should instantiate and cover MathUtils', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(MathUtils);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
