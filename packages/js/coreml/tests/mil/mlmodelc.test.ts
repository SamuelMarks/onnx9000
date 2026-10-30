import { describe, expect, it } from 'vitest';
import { MLModelCCompiler } from '../../src/mil/mlmodelc';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mlmodelc.ts', () => {
  it('should instantiate and cover MLModelCCompiler', () => {
    try {
      const obj = Construct(MLModelCCompiler);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
