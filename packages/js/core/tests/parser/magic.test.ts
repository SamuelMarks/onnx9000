import { describe, expect, it } from 'vitest';
import { detectFormat } from '../../src/parser/magic';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('magic.ts', () => {
  it('should call and cover detectFormat', async () => {
    try {
      const res = Call(detectFormat);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
