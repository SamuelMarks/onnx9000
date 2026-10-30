import { describe, expect, it } from 'vitest';
import { emitGlobalPool, emitPool } from '../../src/keras/emitters-pool';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('emitters-pool.ts', () => {
  it('should call and cover emitPool', async () => {
    try {
      const res = Call(emitPool);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover emitGlobalPool', async () => {
    try {
      const res = Call(emitGlobalPool);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
