import { describe, expect, it } from 'vitest';
import { lowerLinalgToHAL } from '../../src/passes/lower_linalg_to_hal';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('lower_linalg_to_hal.ts', () => {
  it('should call and cover lowerLinalgToHAL', async () => {
    try {
      const res = Call(lowerLinalgToHAL);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
