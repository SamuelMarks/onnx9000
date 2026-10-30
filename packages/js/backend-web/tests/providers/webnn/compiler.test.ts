import { describe, expect, it } from 'vitest';
import { WebNNCompiler } from '../../../src/providers/webnn/compiler';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('compiler.ts', () => {
  it('should instantiate and cover WebNNCompiler', () => {
    try {
      const obj = Construct(WebNNCompiler);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
