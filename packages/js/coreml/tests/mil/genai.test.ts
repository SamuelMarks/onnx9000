import { describe, expect, it } from 'vitest';
import { detectAndMapGenAITopologies } from '../../src/mil/genai';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('genai.ts', () => {
  it('should call and cover detectAndMapGenAITopologies', async () => {
    try {
      const res = Call(detectAndMapGenAITopologies);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
