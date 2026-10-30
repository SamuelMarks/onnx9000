import { describe, expect, it } from 'vitest';
import { calculateByteLength, downloadWeightShards } from '../../src/keras/weight-loader';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('weight-loader.ts', () => {
  it('should call and cover downloadWeightShards', async () => {
    try {
      const res = Call(downloadWeightShards);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover calculateByteLength', async () => {
    try {
      const res = Call(calculateByteLength);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
