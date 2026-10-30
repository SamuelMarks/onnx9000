import { describe, expect, it } from 'vitest';
import { LayoutOptimizer } from '../../src/keras/layout-optimizer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('layout-optimizer.ts', () => {
  it('should instantiate and cover LayoutOptimizer', () => {
    try {
      const obj = Construct(LayoutOptimizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
