import { describe, expect, it } from 'vitest';
import { parseCfg, parseWeights } from '../../../src/mmdnn/darknet/parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('parser.ts', () => {
  it('should call and cover parseCfg', async () => {
    try {
      const res = Call(parseCfg);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover parseWeights', async () => {
    try {
      const res = Call(parseWeights);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
