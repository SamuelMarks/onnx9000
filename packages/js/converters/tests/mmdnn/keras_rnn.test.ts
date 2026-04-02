import { describe, it, expect } from 'vitest';
import { KerasGenerator } from '../../src/mmdnn/keras/generator.js';
import { Graph, Node } from '@onnx9000/core';

describe('KerasGenerator RNN', () => {
  it('should generate code for LSTM and GRU', () => {
    const graph = new Graph('RNNGraph');
    const lstm = new Node(
      'LSTM',
      ['in', 'w', 'r', 'b'],
      ['out_h', 'out_c'],
      {
        hidden_size: { name: 'hidden_size', value: 32, attrType: 'INT' },
      },
      'lstm_layer',
    );
    graph.nodes.push(lstm);

    const gru = new Node('GRU', ['out_h', 'w2', 'r2'], ['out_gru'], {}, 'gru_layer');
    graph.nodes.push(gru);

    const generator = new KerasGenerator(graph);
    const code = generator.generate();

    expect(code).toContain('LSTM(');
    expect(code).toContain('GRU(');
  });
});
