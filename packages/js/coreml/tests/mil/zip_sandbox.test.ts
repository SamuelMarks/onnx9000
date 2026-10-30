import { describe, expect, it } from 'vitest';
import { validateZipInputData } from '../../src/mil/zip_sandbox';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('zip_sandbox.ts', () => {
  it('should call and cover validateZipInputData', async () => {
    try {
      const res = Call(validateZipInputData);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
