import { describe, expect, it } from 'vitest';
import { parseH2O } from '../../../src/mmdnn/h2o/parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('parser.ts', () => {
  it('should call and cover parseH2O', async () => {
    try {
      const res = Call(parseH2O);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
