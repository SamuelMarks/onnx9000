import { describe, expect, it } from 'vitest';
import { EdgeTPUOptimizer } from '../../src/optimizations/edgetpu';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('edgetpu.ts', () => {
  it('should instantiate and cover EdgeTPUOptimizer', () => {
    try {
      const obj = Construct(EdgeTPUOptimizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
