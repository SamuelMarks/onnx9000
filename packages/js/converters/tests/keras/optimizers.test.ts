import { describe, it, expect } from 'vitest';
import { optimizeFusedOps, KerasGraphOptimizer } from '../../src/keras/optimizers.js';

describe('optimizers', () => {
  it('should optimize fused ops', () => {
    const ops = optimizeFusedOps([
      { opType: '_FusedConv2D', name: 'c', inputs: [], outputs: [] } as any,
    ]);
    expect(ops.length).toBe(2);
  });

  it('should run keras optimizer', () => {
    const opt = new KerasGraphOptimizer();
    const g: any = { nodes: [] };
    opt.optimize(g);
    expect(g.nodes.length).toBe(0);
  });
});
