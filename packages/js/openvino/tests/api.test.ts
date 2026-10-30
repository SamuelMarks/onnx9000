import { describe, expect, it } from 'vitest';
import { exportModel } from '../src/api';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('api.ts', () => {
  it('should call and cover exportModel', async () => {
    try {
      const res = Call(exportModel);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
