import { describe, it, expect } from 'vitest';
import { JaxMapper } from '../../src/jax/mapper.js';

describe('JaxMapper', () => {
  it('should map graph', () => {
    const jaxpr: any = {
      invars: ['x'],
      outvars: ['y'],
      constvars: ['w'],
      eqns: [{ primitive: 'add', invars: ['x', 'w'], outvars: ['y'], params: { test: 1 } }],
    };

    const mapper = new JaxMapper(jaxpr, { w: [1, 2, 3] });
    const graph = mapper.map();

    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('Add');
  });
});
