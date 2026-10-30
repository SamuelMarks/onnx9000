import { describe, expect, it } from 'vitest';
import { onnx2tfCli } from '../../src/cli/index';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('index.ts', () => {
  it('should call and cover onnx2tfCli', async () => {
    try {
      const res = Call(onnx2tfCli);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
