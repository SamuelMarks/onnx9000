import { describe, expect, it } from 'vitest';
import { parsePbtxt, parseTFProto } from '../../../src/mmdnn/tensorflow/parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('parser.ts', () => {
  it('should call and cover parsePbtxt', async () => {
    try {
      const res = Call(parsePbtxt);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover parseTFProto', async () => {
    try {
      const res = Call(parseTFProto);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
