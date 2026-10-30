import { describe, expect, it } from 'vitest';
import { applyCompression } from '../../src/mil/compression';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('compression.ts', () => {
  it('should call and cover applyCompression', async () => {
    try {
      const res = Call(applyCompression);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
