import { describe, it, expect } from 'vitest';
import { TensorFlowGenerator } from '../src/mmdnn/tensorflow/generator.js';

describe('TensorFlowGenerator', () => {
  it('should generate code', () => {
    const gen = new TensorFlowGenerator({
      name: 'test',
      inputs: [],
      outputs: [],
      tensors: {},
      nodes: [],
      valueInfo: [],
    } as any);
    const code = gen.generate();
    expect(code).toContain('import tensorflow');
  });
});
