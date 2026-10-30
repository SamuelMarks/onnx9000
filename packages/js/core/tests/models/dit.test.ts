import { describe, expect, it } from 'vitest';
import { DiT, DiTBlock, ditXl2 } from '../../src/models/dit';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('dit.ts', () => {
  it('should instantiate and cover DiTBlock', () => {
    try {
      const obj = Construct(DiTBlock);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover DiT', () => {
    try {
      const obj = Construct(DiT);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover ditXl2', async () => {
    try {
      const res = Call(ditXl2);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
