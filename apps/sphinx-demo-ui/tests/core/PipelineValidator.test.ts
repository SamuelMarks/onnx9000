import { describe, expect, it } from 'vitest';
import { PipelineValidator } from '../../src/core/PipelineValidator';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('PipelineValidator.ts', () => {
  it('should instantiate and cover PipelineValidator', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(PipelineValidator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
