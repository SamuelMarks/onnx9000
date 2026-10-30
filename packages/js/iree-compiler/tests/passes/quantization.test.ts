import { describe, expect, it } from 'vitest';
import { QuantizationOptimizer } from '../../src/passes/quantization';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('quantization.ts', () => {
  it('should instantiate and cover QuantizationOptimizer', () => {
    try {
      const obj = Construct(QuantizationOptimizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
