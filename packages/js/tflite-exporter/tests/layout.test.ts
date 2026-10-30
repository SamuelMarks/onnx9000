import { Graph, Node } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { LayoutOptimizer } from '../src/compiler/layout.js';

describe('LayoutOptimizer', () => {
  it('should optimize', () => {
    const g = new Graph('test');
    g.nodes.push(new Node('Identity', ['a'], ['b']));

    const opt = new LayoutOptimizer(g);
    opt.optimize();
    expect(g.nodes.length).toBe(0);
  });
});
