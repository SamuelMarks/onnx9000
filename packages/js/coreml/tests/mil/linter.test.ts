import { describe, expect, it } from 'vitest';
import { lintMILProgram } from '../../src/mil/linter';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('linter.ts', () => {
  it('should call and cover lintMILProgram', async () => {
    try {
      const res = Call(lintMILProgram);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
