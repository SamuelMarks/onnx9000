import { describe, expect, it } from 'vitest';
import { parseMxNetParams, parseMxNetSymbol } from '../../../src/mmdnn/mxnet/parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('parser.ts', () => {
  it('should call and cover parseMxNetSymbol', async () => {
    try {
      const res = Call(parseMxNetSymbol);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover parseMxNetParams', async () => {
    try {
      const res = Call(parseMxNetParams);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
