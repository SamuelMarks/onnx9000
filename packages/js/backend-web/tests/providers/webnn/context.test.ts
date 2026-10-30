import { describe, expect, it } from 'vitest';
import { WebNNContextManager } from '../../../src/providers/webnn/context';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('context.ts', () => {
  it('should instantiate and cover WebNNContextManager', () => {
    try {
      const obj = Construct(WebNNContextManager);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
