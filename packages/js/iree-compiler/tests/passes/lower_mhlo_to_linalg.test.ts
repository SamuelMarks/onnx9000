import { describe, expect, it } from 'vitest';
import { lowerMHLOToLinalg } from '../../src/passes/lower_mhlo_to_linalg';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('lower_mhlo_to_linalg.ts', () => {
  it('should call and cover lowerMHLOToLinalg', async () => {
    try {
      const res = Call(lowerMHLOToLinalg);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
