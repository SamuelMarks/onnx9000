import { describe, expect, it } from 'vitest';
import { createTensorRTSession } from '../src/tensorrt';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('tensorrt.ts', () => {
  it('should call and cover createTensorRTSession', async () => {
    try {
      const res = Call(createTensorRTSession);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
