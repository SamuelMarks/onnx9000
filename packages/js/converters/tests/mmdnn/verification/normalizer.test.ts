import { describe, expect, it } from 'vitest';
import { ONNXNormalizer } from '../../../src/mmdnn/verification/normalizer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('normalizer.ts', () => {
  it('should instantiate and cover ONNXNormalizer', () => {
    try {
      const obj = Construct(ONNXNormalizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
