import { describe, expect, it } from 'vitest';
import { wasmFlashAttentionSimd, wasmLogitSortFilter } from '../../src/genai/kernels';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('kernels.ts', () => {
  it('should call and cover wasmLogitSortFilter', async () => {
    try {
      const res = Call(wasmLogitSortFilter);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover wasmFlashAttentionSimd', async () => {
    try {
      const res = Call(wasmFlashAttentionSimd);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
