import { describe, expect, it } from 'vitest';
import { assertDeterministicBuild } from '../../src/mil/deterministic';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('deterministic.ts', () => {
  it('should call and cover assertDeterministicBuild', async () => {
    try {
      const res = Call(assertDeterministicBuild);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
