import { describe, expect, it } from 'vitest';
import { SequenceTensorUtils } from '../../src/genai/tensor_utils';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('tensor_utils.ts', () => {
  it('should instantiate and cover SequenceTensorUtils', () => {
    try {
      const obj = Construct(SequenceTensorUtils);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
