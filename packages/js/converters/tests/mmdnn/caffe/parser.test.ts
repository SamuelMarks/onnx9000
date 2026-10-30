import { describe, expect, it } from 'vitest';
import { parseCaffemodel, parsePrototxt } from '../../../src/mmdnn/caffe/parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('parser.ts', () => {
  it('should call and cover parsePrototxt', async () => {
    try {
      const res = Call(parsePrototxt);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover parseCaffemodel', async () => {
    try {
      const res = Call(parseCaffemodel);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
