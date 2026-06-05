import { describe, it, expect } from 'vitest';
import { MxNetMapper } from '../src/mmdnn/mxnet/mapper.js';
import { Graph } from '@onnx9000/core';

describe('MxNetMapper', () => {
  it('should map mxnet', () => {
    const mapper = new MxNetMapper();
    const g = new Graph('test');
    const nodes = mapper.map({ op: 'Convolution', name: 'c1', attrs: {} }, g);
    expect(nodes[0].opType).toBe('Conv');
  });
});
