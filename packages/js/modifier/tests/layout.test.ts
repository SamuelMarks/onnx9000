import { describe, it, expect } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import { DagreLayoutEngine } from '../src/render/layout.js';

describe('DagreLayoutEngine', () => {
  it('computes TB layout', () => {
    const graph = new Graph('Test');
    const nodeA = new Node('A', [], ['outA']);
    const nodeB = new Node('B', ['outA'], ['outB']);
    const nodeC = new Node('C', ['outA'], ['outC']);
    graph.nodes.push(nodeA, nodeB, nodeC);

    const engine = new DagreLayoutEngine('TB');
    const layout = engine.compute(graph);

    expect(layout.nodes.size).toBe(3);

    // A should be at layer 0
    expect(layout.nodes.get(nodeA.id)!.layer).toBe(0);
    // B and C should be at layer 1
    expect(layout.nodes.get(nodeB.id)!.layer).toBe(1);
    expect(layout.nodes.get(nodeC.id)!.layer).toBe(1);

    expect(layout.edges.length).toBe(2);
    expect(layout.edges[0]!.sourceId).toBe(nodeA.id);
    expect(layout.edges[0]!.targetId).toBe(nodeB.id);
  });

  it('computes LR layout', () => {
    const graph = new Graph('Test');
    const nodeA = new Node('A', [], ['outA']);
    const nodeB = new Node('B', ['outA'], ['outB']);
    graph.nodes.push(nodeA, nodeB);

    const engine = new DagreLayoutEngine('LR');
    const layout = engine.compute(graph);

    expect(layout.nodes.size).toBe(2);
    expect(layout.nodes.get(nodeA.id)!.layer).toBe(0);
    expect(layout.nodes.get(nodeB.id)!.layer).toBe(1);

    expect(layout.edges.length).toBe(1);
    expect(layout.edges[0]!.path.length).toBe(4);
  });

  it('handles isolated nodes and no nodes gracefully', () => {
    const graph = new Graph('Test');
    const engine = new DagreLayoutEngine();

    const layout1 = engine.compute(graph);
    expect(layout1.nodes.size).toBe(0);

    graph.nodes.push(new Node('Isolate', [], []));
    const layout2 = engine.compute(graph);
    expect(layout2.nodes.size).toBe(1);
    expect(layout2.edges.length).toBe(0);
  });

  it('handles bad topological states gracefully', () => {
    const graph = new Graph('Test');
    // B consumes from A, but A doesn't exist
    const nodeB = new Node('B', ['outA'], ['outB']);
    graph.nodes.push(nodeB);

    const engine = new DagreLayoutEngine();
    const layout = engine.compute(graph);
    expect(layout.nodes.size).toBe(1);
    expect(layout.nodes.get(nodeB.id)!.layer).toBe(0);
  });
});
