import { describe, expect, it } from 'vitest';
import { Graphsurgeon, type SurgeonGraph } from '../src/index.js';

describe('Graphsurgeon', () => {
  it('should prune dead nodes and locate nodes by opType', () => {
    const gs = new Graphsurgeon();
    expect(gs.run()).toBe('[graphsurgeon] processed');

    const graph: SurgeonGraph = {
      name: 'test_graph',
      inputs: ['in'],
      outputs: ['out'],
      nodes: [
        { name: 'conv', opType: 'Conv', inputs: ['in'], outputs: ['mid'] },
        { name: 'dead', opType: 'Relu', inputs: ['mid'], outputs: ['unused'] },
        { name: 'relu', opType: 'Relu', inputs: ['mid'], outputs: ['out'] },
      ],
    };

    const pruned = gs.pruneDeadNodes(graph);
    expect(pruned.nodes.map((n) => n.name)).toEqual(['conv', 'relu']);
    expect(gs.findByOpType(pruned, 'Relu')).toHaveLength(1);
  });
});
