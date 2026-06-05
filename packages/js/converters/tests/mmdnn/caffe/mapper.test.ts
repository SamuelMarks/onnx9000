import { describe, it, expect } from 'vitest';
import { CaffeMapper } from '../../../src/mmdnn/caffe/mapper.js';
import { Graph } from '@onnx9000/core';

describe('CaffeMapper', () => {
  it('should map layer', () => {
    const mapper = new CaffeMapper();
    const g = new Graph('test');

    let nodes = mapper.map({ type: 'ReLU', bottom: ['A'], top: ['B'] }, g);
    expect(nodes[0].opType).toBe('Relu');

    nodes = mapper.map({ type: 'Convolution', bottom: ['A'], top: ['B'] }, g);
    expect(nodes[0].opType).toBe('Conv');
  });
});
