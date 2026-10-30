import { describe, expect, it } from 'vitest';
import { MobileMemory } from '../src/index.js';

describe('MobileMemory', () => {
  it('should analyze activation memory and process model strings', () => {
    const mm = new MobileMemory();
    expect(mm.process('mobilenet_v3')).toBe('Mobile Memory processed mobilenet_v3');

    const report = mm.analyzeActivations([100, 300, 200, 150]);
    expect(report.unoptimizedPeakBytes).toBe(750);
    expect(report.reusedPeakBytes).toBe(500);
    expect(report.checkpointIndices).toEqual([0, 2]);
  });
});
