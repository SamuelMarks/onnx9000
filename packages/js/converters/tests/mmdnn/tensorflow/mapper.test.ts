import { describe, it, expect } from 'vitest';
import { TFMapper } from '../src/mmdnn/tensorflow/mapper.js';
import { Graph } from '@onnx9000/core';

describe('TFMapper', () => {
  it('should map nodes', () => {
    const mapper = new TFMapper();
    const g = new Graph('test');

    let nodes = mapper.map({ name: 'n1', op: 'Placeholder', input: [], attr: {} }, g);
    expect(nodes[0].opType).toBe('Identity');

    nodes = mapper.map({ name: 'n2', op: 'Const', input: [], attr: {} }, g);
    expect(nodes.length).toBe(0);
    expect(g.initializers).toContain('n2');
  });
});
