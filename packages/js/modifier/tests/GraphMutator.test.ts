import { Graph } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { GraphMutator } from '../src/GraphMutator.js';

describe('GraphMutator', () => {
  it('should add and undo', () => {
    const g = new Graph('test');
    const m = new GraphMutator(g);

    m.addNode('Relu', ['in'], ['out']);
    expect(g.nodes.length).toBe(1);

    m.undo();
    expect(g.nodes.length).toBe(0);

    m.redo();
    expect(g.nodes.length).toBe(1);
  });

  it('should extract subgraph', () => {
    const g = new Graph('test');
    const m = new GraphMutator(g);
    m.addNode('Relu', ['in'], ['out']);
    const n = g.nodes[0];

    const sub = m.extractSubgraph([n.id]);
    expect(sub.nodes.length).toBe(1);
  });
});
