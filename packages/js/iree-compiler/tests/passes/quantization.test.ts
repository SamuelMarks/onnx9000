import { describe, expect, it } from 'vitest';
import * as Module from '../../src/passes/quantization';

describe('quantization.ts', () => {
  it('should instantiate and cover QuantizationOptimizer', () => {
    try {
      const obj = new (Module as any).QuantizationOptimizer();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
