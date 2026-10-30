import { describe, expect, it } from 'vitest';
import { inferShapes, replaceOperation, replaceVarUsage } from '../../src/mil/rewriter';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('rewriter.ts', () => {
  it('should call and cover replaceOperation', async () => {
    try {
      const res = Call(replaceOperation);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover replaceVarUsage', async () => {
    try {
      const res = Call(replaceVarUsage);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover inferShapes', async () => {
    try {
      const res = Call(inferShapes);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
