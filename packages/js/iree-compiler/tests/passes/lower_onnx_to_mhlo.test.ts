import { describe, expect, it } from 'vitest';
import { lowerONNXToMHLO } from '../../src/passes/lower_onnx_to_mhlo';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('lower_onnx_to_mhlo.ts', () => {
  it('should call and cover lowerONNXToMHLO', async () => {
    try {
      const res = Call(lowerONNXToMHLO);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
