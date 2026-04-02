import { describe, it, expect, vi } from 'vitest';
import { TensorFlowGenerator } from '../../src/mmdnn/tensorflow/generator.js';
import { parsePbtxt } from '../../src/mmdnn/tensorflow/parser.js';
import { Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';

describe('TensorFlow MMDNN Coverage Gaps', () => {
  it('should cover generator branches', () => {
    const graph = new Graph('tf_test');
    graph.inputs.push(new ValueInfo('in', [1], 'float32'));
    graph.nodes.push(new Node('Relu', ['in'], ['out']));

    const gen = new TensorFlowGenerator(graph);
    expect(gen.generate()).toContain('import tensorflow');

    // Test with initializers
    graph.addTensor(new Tensor('w', [1], 'float32', true, false, new Uint8Array(4)));
    expect(gen.generate()).toContain('tf.constant');
  });

  it('should cover parser pbtxt edge cases', () => {
    const pbtxt = `
      node {
        name: "input"
        op: "Placeholder"
        attr {
          key: "dtype"
          value { type: DT_FLOAT }
        }
      }
    `;
    const res = parsePbtxt(pbtxt);
    expect(res.node.length).toBe(1);

    // Error cases
    expect(() => parsePbtxt('invalid {')).toThrow();
  });
});
