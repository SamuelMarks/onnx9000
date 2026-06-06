import { describe, expect, it } from 'vitest';
import { CGenerator } from '../src/generator.js';

describe('CGenerator', () => {
  it('should generate source', () => {
    const graph: any = {
      name: 'test',
      inputs: [{ name: 'in', shape: [10] }],
      outputs: [{ name: 'out', shape: [10] }],
      tensors: {},
      initializers: [],
      nodes: [{ opType: 'Relu', inputs: ['in'], outputs: ['out'] }],
    };

    const gen = new CGenerator(graph, 'model_', false);
    const header = gen.generateHeader();
    expect(header).toContain('void model_run');

    const src = gen.generateSource();
    expect(src).toContain('Relu -> out');
    expect(src).toContain('> 0 ?');
  });
});
