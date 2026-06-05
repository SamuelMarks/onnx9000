import { describe, it, expect } from 'vitest';
import { PolyfillMLGraph } from '../src/graph.js';
import { Graph } from '@onnx9000/core';

describe('WebNN Graph', () => {
  it('should destroy', () => {
    const g = new PolyfillMLGraph(new Graph('test'));
    expect(g.onnxGraph.name).toBe('test');
    g.destroy();
    expect(g.onnxGraph.name).toBe('destroyed');
  });
});
