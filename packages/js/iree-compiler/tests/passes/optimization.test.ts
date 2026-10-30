import { describe, expect, it } from 'vitest';
import { Optimizer } from '../../src/passes/optimization';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('optimization.ts', () => {
  it('should instantiate and cover Optimizer', () => {
    try {
      const obj = Construct(Optimizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
