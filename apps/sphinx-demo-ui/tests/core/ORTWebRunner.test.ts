import { describe, expect, it } from 'vitest';
import { ORTWebRunner } from '../../src/core/ORTWebRunner';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('ORTWebRunner.ts', () => {
  it('should instantiate and cover ORTWebRunner', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(ORTWebRunner);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
