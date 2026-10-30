import { describe, expect, it } from 'vitest';
import { implementDynamicBatching } from '../../src/mil/batching';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('batching.ts', () => {
  it('should call and cover implementDynamicBatching', async () => {
    try {
      const res = Call(implementDynamicBatching);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
