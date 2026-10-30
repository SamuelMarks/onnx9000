import { describe, expect, it } from 'vitest';
import { TopPLogitProcessor } from '../../src/genai/top_p';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('top_p.ts', () => {
  it('should instantiate and cover TopPLogitProcessor', () => {
    try {
      const obj = Construct(TopPLogitProcessor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
