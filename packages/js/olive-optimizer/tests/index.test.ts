import { describe, expect, it } from 'vitest';
import { OliveOptimizer } from '../src/index.js';

describe('OliveOptimizer', () => {
  it('should execute optimization workflows and process requests', () => {
    const opt = new OliveOptimizer();
    expect(opt.process('bert-base')).toBe('Olive Optimizer processed bert-base');

    const res1 = opt.runWorkflow({
      passes: ['dce', 'quantization', 'pruning'],
      quantPrecision: 'fp16',
      sparsityRatio: 0.5,
    });
    expect(res1.executedPasses).toEqual(['dce', 'quantization', 'pruning']);
    expect(res1.compressionRatio).toBeGreaterThan(4.0);

    const res2 = opt.runWorkflow({ passes: ['pruning'] });
    expect(res2.compressionRatio).toBeCloseTo(2.0);
  });
});
