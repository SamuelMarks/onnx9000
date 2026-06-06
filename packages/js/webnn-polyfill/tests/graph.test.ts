import { Graph } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { PolyfillMLGraph } from '../src/graph.js';

describe('WebNN Graph', () => {
  it('should destroy', () => {
    const g = new PolyfillMLGraph(new Graph('test'));
    expect(g.onnxGraph.name).toBe('test');
    g.destroy();
    expect(g.onnxGraph.name).toBe('destroyed');
  });
});
