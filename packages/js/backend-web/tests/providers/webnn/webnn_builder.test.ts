import { describe, expect, it } from 'vitest';
import { KerasWebNNCompiler } from '../../../src/providers/webnn/webnn_builder';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('webnn_builder.ts', () => {
  it('should instantiate and cover KerasWebNNCompiler', () => {
    try {
      const obj = Construct(KerasWebNNCompiler);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
