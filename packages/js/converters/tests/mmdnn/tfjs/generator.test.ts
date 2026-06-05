import { describe, it, expect } from 'vitest';
import { generateTFJSCode, isLinearGraph } from '../src/mmdnn/tfjs/generator.js';

describe('TFJSGenerator', () => {
  it('should generate tfjs code', () => {
    const g: any = {
      inputs: [{ name: 'in', shape: [1] }],
      outputs: [{ name: 'out' }],
      initializers: [],
      tensors: {},
      nodes: [{ opType: 'Relu', inputs: ['in'], outputs: ['out'], attributes: {} }],
    };
    expect(isLinearGraph(g)).toBe(true);

    const code = generateTFJSCode(g);
    expect(code).toContain('import * as tf');
    expect(code).toContain('tf.layers.reLU');
  });
});
