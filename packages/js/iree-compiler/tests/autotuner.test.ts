import { describe, it, expect } from 'vitest';
import { Region, Operation, Block } from '../src/ir/core.js';
import { MetaScheduleAutotuner } from '../src/passes/autotuner.js';

describe('Autotuner Passes', () => {
  it('should identify hardware and heuristics', () => {
    const tuner = new MetaScheduleAutotuner();
    expect(tuner.isAppleMSeries('Apple M1 Max')).toBe(true);
    expect(tuner.isNvidia('NVIDIA GeForce RTX 3080')).toBe(true);
    expect(tuner.getHeuristicFallback('wgsl')).toEqual([64, 1, 1]);
  });

  it('should mutate sizes and config', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    const op = new Operation('web.linalg.generic', [], [], {});
    block.pushOperation(op);

    const tuner = new MetaScheduleAutotuner();
    tuner.mutateTilingSizes(op, [32, 32]);
    expect(op.attributes.tiling_sizes).toEqual([32, 32]);

    const configStr = tuner.generateIreeConfig();
    tuner.loadIreeConfig(configStr, region);

    expect(op.attributes.tiling_sizes).toEqual([16, 16]);
  });
});
