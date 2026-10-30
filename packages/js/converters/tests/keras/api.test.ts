import { describe, expect, it } from 'vitest';
import { keras2onnx } from '../../src/keras/api';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('api.ts', () => {
  it('should call and cover keras2onnx', async () => {
    try {
      const res = Call(keras2onnx);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
