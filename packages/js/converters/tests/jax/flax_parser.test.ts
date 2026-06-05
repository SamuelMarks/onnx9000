import { describe, it, expect } from 'vitest';
import { parseFlaxState } from '../src/jax/flax_parser.js';

describe('flax_parser', () => {
  it('should parse json state', () => {
    const res = parseFlaxState('{"a": 1}');
    expect(res).toEqual({ a: 1 });
  });
});
