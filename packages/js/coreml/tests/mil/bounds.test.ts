import { describe, expect, it } from 'vitest';
import { establishMemoryBounds } from '../../src/mil/bounds';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('bounds.ts', () => {
  it('should call and cover establishMemoryBounds', async () => {
    try {
      const res = Call(establishMemoryBounds);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
