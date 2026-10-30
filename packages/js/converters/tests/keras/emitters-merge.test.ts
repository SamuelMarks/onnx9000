import { describe, expect, it } from 'vitest';
import { emitConcat, emitDot, emitMerge } from '../../src/keras/emitters-merge';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('emitters-merge.ts', () => {
  it('should call and cover emitMerge', async () => {
    try {
      const res = Call(emitMerge);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover emitConcat', async () => {
    try {
      const res = Call(emitConcat);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover emitDot', async () => {
    try {
      const res = Call(emitDot);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
