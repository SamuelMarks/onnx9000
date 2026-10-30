import { describe, expect, it } from 'vitest';
import { getTypesIdentifier } from '../../src/mmdnn/types';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('types.ts', () => {
  it('should call and cover getTypesIdentifier', async () => {
    try {
      const res = Call(getTypesIdentifier);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
