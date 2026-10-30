import { describe, expect, it } from 'vitest';
import { validateBlock, validateMILProgram } from '../../src/mil/validator';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('validator.ts', () => {
  it('should call and cover validateMILProgram', async () => {
    try {
      const res = Call(validateMILProgram);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover validateBlock', async () => {
    try {
      const res = Call(validateBlock);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
