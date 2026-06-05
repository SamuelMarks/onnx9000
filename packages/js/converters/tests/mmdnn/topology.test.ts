import { describe, it, expect } from 'vitest';
import { topologicalSort } from '../../src/mmdnn/topology.js';

describe('topology sort', () => {
  it('should sort nodes', () => {
    const g: any = {
      initializers: [],
      tensors: {},
      inputs: [{ name: 'in' }],
      outputs: [],
      valueInfo: [],
      opsetImports: [],
      nodes: [
        { name: 'n2', inputs: ['tmp'], outputs: ['out'] },
        { name: 'n1', inputs: ['in'], outputs: ['tmp'] },
      ],
    };

    const sorted = topologicalSort(g, { error: () => {}, warn: () => {}, info: () => {} } as any);
    expect(sorted.nodes[0].name).toBe('n1');
    expect(sorted.nodes[1].name).toBe('n2');
  });
});
