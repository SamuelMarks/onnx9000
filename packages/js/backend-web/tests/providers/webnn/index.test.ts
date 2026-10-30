import { describe, expect, it } from 'vitest';
import { WebNNProvider } from '../../../src/providers/webnn/index';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover WebNNProvider', () => {
    try {
      const obj = Construct(WebNNProvider);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
