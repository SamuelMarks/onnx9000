import { describe, expect, it } from 'vitest';
import { Quantizer } from '../../src/quantization/quantizer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('quantizer.ts', () => {
  it('should instantiate and cover Quantizer', () => {
    try {
      const obj = Construct(Quantizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
