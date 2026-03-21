import { describe, it, expect } from 'vitest';
import { MxNetMapper } from '../../../../src/mmdnn/mxnet/mapper.js';
import { parseMxNetSymbol } from '../../../../src/mmdnn/mxnet/parser.js';
import { Graph } from '@onnx9000/core';

describe('MXNet Validation - SqueezeNet', () => {
  it('should parse and map SqueezeNet architecture', () => {
    const symbolJson = {
      nodes: [
        { op: 'null', name: 'data', inputs: [] },
        {
          op: 'Convolution',
          name: 'conv1',
          attrs: { kernel: '(7, 7)', num_filter: '96', stride: '(2, 2)' },
          inputs: [[0, 0, 0]],
        },
        { op: 'Activation', name: 'relu1', attrs: { act_type: 'relu' }, inputs: [[1, 0, 0]] },
        {
          op: 'Pooling',
          name: 'pool1',
          attrs: { kernel: '(3, 3)', pool_type: 'max', stride: '(2, 2)' },
          inputs: [[2, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'fire2_squeeze',
          attrs: { kernel: '(1, 1)', num_filter: '16' },
          inputs: [[3, 0, 0]],
        },
        {
          op: 'Activation',
          name: 'fire2_relu_squeeze',
          attrs: { act_type: 'relu' },
          inputs: [[4, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'fire2_expand1x1',
          attrs: { kernel: '(1, 1)', num_filter: '64' },
          inputs: [[5, 0, 0]],
        },
        {
          op: 'Activation',
          name: 'fire2_relu_expand1x1',
          attrs: { act_type: 'relu' },
          inputs: [[6, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'fire2_expand3x3',
          attrs: { kernel: '(3, 3)', num_filter: '64', pad: '(1, 1)' },
          inputs: [[5, 0, 0]],
        },
        {
          op: 'Activation',
          name: 'fire2_relu_expand3x3',
          attrs: { act_type: 'relu' },
          inputs: [[8, 0, 0]],
        },

        {
          op: 'Concat',
          name: 'fire2_concat',
          inputs: [
            [7, 0, 0],
            [9, 0, 0],
          ],
        },

        { op: 'Dropout', name: 'drop1', attrs: { p: '0.5' }, inputs: [[10, 0, 0]] },
        {
          op: 'Convolution',
          name: 'conv10',
          attrs: { kernel: '(1, 1)', num_filter: '1000' },
          inputs: [[11, 0, 0]],
        },
        {
          op: 'Activation',
          name: 'relu_conv10',
          attrs: { act_type: 'relu' },
          inputs: [[12, 0, 0]],
        },
        {
          op: 'Pooling',
          name: 'pool10',
          attrs: { global_pool: 'True', pool_type: 'avg' },
          inputs: [[13, 0, 0]],
        },
        { op: 'Flatten', name: 'flatten0', inputs: [[14, 0, 0]] },
        { op: 'SoftmaxOutput', name: 'softmax', inputs: [[15, 0, 0]] },
      ],
      arg_nodes: [0],
      heads: [[16, 0, 0]],
    };

    const parsed = parseMxNetSymbol(JSON.stringify(symbolJson));
    const mapper = new MxNetMapper();
    const graph = new Graph('squeezenet');

    for (const node of parsed.nodes) {
      const mappedNodes = mapper.map(node, graph);
      for (const mn of mappedNodes) {
        graph.addNode(mn);
      }
    }

    const opTypes = graph.nodes.map((n) => n.opType);
    expect(opTypes).toContain('Conv');
    expect(opTypes).toContain('Relu');
    expect(opTypes).toContain('MaxPool');
    expect(opTypes).toContain('Concat');
    expect(opTypes).toContain('Identity'); // Dropout
    expect(opTypes).toContain('GlobalAveragePool');
    expect(opTypes).toContain('Flatten');
    expect(opTypes).toContain('Softmax');
  });
});
