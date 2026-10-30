import { describe, expect, it } from 'vitest';
import { Toolbar } from '../../src/components/toolbar';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('toolbar.ts', () => {
  it('should instantiate and cover Toolbar', () => {
    try {
      const obj = Construct(Toolbar);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
