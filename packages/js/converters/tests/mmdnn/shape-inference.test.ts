import { describe, it, expect } from 'vitest';
import { ShapeInferenceEngine } from '../../src/mmdnn/shape-inference.js';
import { Graph } from '@onnx9000/core';

describe('ShapeInferenceEngine', () => {
  it('should infer shapes', () => {
    const engine = new ShapeInferenceEngine();
    const g = new Graph('test');
    g.inputs.push({ name: 'in', shape: [1, 2], dtype: 'float32' } as any);
    g.nodes.push({
      opType: 'Relu',
      name: 'r1',
      inputs: ['in'],
      outputs: ['out'],
      attributes: {},
    } as any);

    engine.inferShapes(g, { warn: () => {} } as any);
    expect(engine.getShape('out')).toEqual([1, 2]);
  });
});
