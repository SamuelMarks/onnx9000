import { describe, it, expect } from 'vitest';
import { parseJaxpr } from '../../src/jax/jaxpr_parser.js';

describe('JAX Jaxpr Parser', () => {
  it('should parse basic jaxpr json', () => {
    const jsonStr = JSON.stringify({
      invars: ['x'],
      outvars: ['y'],
      constvars: [],
      eqns: [{ primitive: 'add', invars: ['x', 'x'], outvars: ['y'], params: {} }],
    });

    const jaxpr = parseJaxpr(jsonStr);
    expect(jaxpr.invars).toEqual(['x']);
    expect(jaxpr.outvars).toEqual(['y']);
    expect(jaxpr.eqns.length).toBe(1);
    expect(jaxpr.eqns[0].primitive).toBe('add');
  });
});
