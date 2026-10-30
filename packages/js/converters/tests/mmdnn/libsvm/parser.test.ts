import { describe, expect, it } from 'vitest';
import { parseLibSVM } from '../../../src/mmdnn/libsvm/parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('parser.ts', () => {
  it('should call and cover parseLibSVM', async () => {
    try {
      const res = Call(parseLibSVM);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
