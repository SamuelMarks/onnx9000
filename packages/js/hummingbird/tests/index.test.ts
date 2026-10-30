import { describe, expect, it } from 'vitest';
import { Hummingbird } from '../src/index.js';

describe('Hummingbird', () => {
  it('should compile decision tree nodes into GEMM matrices', () => {
    const hb = new Hummingbird();
    expect(hb.run()).toBe('[hummingbird] processed');

    const weights = hb.compileTreeToGemm(3, [
      { featureIndex: 1, threshold: 2.5 },
      { featureIndex: 2, threshold: -1.0 },
      { featureIndex: -1, threshold: 0.0 },
    ]);
    expect(weights.numFeatures).toBe(3);
    expect(weights.numInternalNodes).toBe(2);
    expect(weights.numLeaves).toBe(3);
    expect(Array.from(weights.thresholds)).toEqual([2.5, -1.0]);

    expect(() => hb.compileTreeToGemm(0, [])).toThrow('numFeatures must be positive');
  });
});
