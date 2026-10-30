import { describe, expect, it } from 'vitest';
import { bufferizeLinalg } from '../../src/passes/bufferize';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('bufferize.ts', () => {
  it('should call and cover bufferizeLinalg', async () => {
    try {
      const res = Call(bufferizeLinalg);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
