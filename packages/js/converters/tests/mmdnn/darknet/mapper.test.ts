import { describe, it, expect } from 'vitest';
import { DarknetMapper } from '../src/mmdnn/darknet/mapper.js';
import { Graph } from '@onnx9000/core';

describe('DarknetMapper', () => {
  it('should map darknet layers', () => {
    const graph = new Graph('test');
    const weights = new Float32Array([1, 2, 3, 4]);
    const mapper = new DarknetMapper(graph, weights);

    mapper.map([
      { type: 'net', channels: 3 },
      { type: 'convolutional', filters: 16 },
    ]);

    expect(graph.nodes.length).toBeGreaterThan(0);
    expect(graph.nodes[0].opType).toBe('Conv');
  });
});
