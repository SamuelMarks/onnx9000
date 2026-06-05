import { describe, it, expect } from 'vitest';
import { PyTorchFXParser, JAXprParser, XLAHLOParser } from '../src/parsers.js';

describe('parsers', () => {
  it('should parse fx', () => {
    const parser = new PyTorchFXParser();
    const g = parser.parse({
      nodes: [{ target: 'aten.add.Tensor', args: ['a'], kwargs: {}, name: 'out' }],
    });
    expect(g.nodes.length).toBe(1);
    expect(g.nodes[0].opType).toBe('add.Tensor');
  });

  it('should parse jaxpr', () => {
    const parser = new JAXprParser();
    const g = parser.parse({
      invars: [{ name: 'in', type: 'f32', shape: [1] }],
      eqns: [
        { primitive: 'add', invars: [{ name: 'in' }], outvars: [{ name: 'out' }], params: {} },
      ],
      outvars: [{ name: 'out' }],
    });
    expect(g.nodes.length).toBe(1);
    expect(g.nodes[0].opType).toBe('add');
  });

  it('should parse hlo', () => {
    const parser = new XLAHLOParser();
    const g = parser.parse({});
    expect(g.name).toBe('XLA_Exported');
  });
});
