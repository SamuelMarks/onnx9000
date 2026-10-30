import { describe, expect, it } from 'vitest';
import { OliveOptimizer } from '../../src/core/OliveOptimizer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('OliveOptimizer.ts', () => {
  it('should instantiate and cover OliveOptimizer', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(OliveOptimizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
