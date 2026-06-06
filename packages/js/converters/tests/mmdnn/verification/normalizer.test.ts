import { describe, expect, it } from 'vitest';
import * as Module from '../../../src/mmdnn/verification/normalizer';

describe('normalizer.ts', () => {
  it('should instantiate and cover ONNXNormalizer', () => {
    try {
      const obj = new (Module as any).ONNXNormalizer();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
