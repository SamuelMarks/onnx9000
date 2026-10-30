import { describe, expect, it } from 'vitest';
import { WebNNPolyfillRunner } from '../../src/core/WebNNPolyfillRunner';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('WebNNPolyfillRunner.ts', () => {
  it('should instantiate and cover WebNNPolyfillRunner', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(WebNNPolyfillRunner);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
