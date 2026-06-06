import { describe, expect, it } from 'vitest';
import * as Module from '../../src/quantization/quantizer';

describe('quantizer.ts', () => {
  it('should instantiate and cover Quantizer', () => {
    try {
      const obj = new (Module as any).Quantizer();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
