import { describe, expect, it } from 'vitest';
import { Graph } from '../src/ir/graph.js';
import { Tensor } from '../src/ir/tensor.js';
import { AutoShardingPass, allReduce } from '../src/sharding.js';

describe('sharding', () => {
  it('should apply sharding pass', () => {
    const pass = new AutoShardingPass();
    const g = new Graph('test');
    pass.apply(g);
    expect(g).toBeDefined();
  });

  it('should record ops', () => {
    const out = allReduce(new Tensor('x', [1], 'float32'));
    expect(out.name).toBe('AllReduce_out');
  });
});
