import { describe, it, expect } from 'vitest';
import { JaxMapper } from '../../src/jax/mapper.js';

describe('JAX Mapper', () => {
  it('should map basic math ops', () => {
    const jaxpr = {
      invars: ['x'],
      outvars: ['y'],
      constvars: ['w'],
      eqns: [{ primitive: 'add', invars: ['x', 'w'], outvars: ['y'], params: { alpha: 1 } }],
    };
    const state = { w: [1.0, 2.0] };

    const mapper = new JaxMapper(jaxpr, state);
    const graph = mapper.map();

    expect(graph.inputs.length).toBe(1);
    expect(graph.inputs[0].name).toBe('x');

    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('Add');
    expect(graph.nodes[0].attributes['alpha'].value).toBe(1);

    expect(graph.outputs.length).toBe(1);
    expect(graph.outputs[0].name).toBe('y');

    expect(Object.keys(graph.tensors).includes('w')).toBe(true);
  });
});
