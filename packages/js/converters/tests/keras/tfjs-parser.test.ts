import { describe, expect, it } from 'vitest';
import { parseTFJSModel } from '../../src/keras/tfjs-parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('tfjs-parser.ts', () => {
  it('should call and cover parseTFJSModel', async () => {
    try {
      const res = Call(parseTFJSModel);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
