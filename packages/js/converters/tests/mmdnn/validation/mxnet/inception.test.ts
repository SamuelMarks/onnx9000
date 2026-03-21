import { describe, it, expect } from 'vitest';
import { MxNetMapper } from '../../../../src/mmdnn/mxnet/mapper.js';
import { parseMxNetSymbol } from '../../../../src/mmdnn/mxnet/parser.js';
import { Graph } from '@onnx9000/core';

describe('MXNet Validation - Inception-v3', () => {
  it('should parse and map Inception-v3 architecture', () => {
    const symbolJson = {
      nodes: [
        { op: 'null', name: 'data', inputs: [] },
        {
          op: 'Convolution',
          name: 'conv_1a_3x3',
          attrs: { kernel: '(3, 3)', num_filter: '32', stride: '(2, 2)' },
          inputs: [[0, 0, 0]],
        },
        { op: 'BatchNorm', name: 'conv_1a_3x3_bn', attrs: { eps: '1e-05' }, inputs: [[1, 0, 0]] },
        {
          op: 'Activation',
          name: 'conv_1a_3x3_relu',
          attrs: { act_type: 'relu' },
          inputs: [[2, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'conv_2a_3x3',
          attrs: { kernel: '(3, 3)', num_filter: '32' },
          inputs: [[3, 0, 0]],
        },
        { op: 'BatchNorm', name: 'conv_2a_3x3_bn', attrs: { eps: '1e-05' }, inputs: [[4, 0, 0]] },
        {
          op: 'Activation',
          name: 'conv_2a_3x3_relu',
          attrs: { act_type: 'relu' },
          inputs: [[5, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'conv_2b_3x3',
          attrs: { kernel: '(3, 3)', num_filter: '64', pad: '(1, 1)' },
          inputs: [[6, 0, 0]],
        },
        { op: 'BatchNorm', name: 'conv_2b_3x3_bn', attrs: { eps: '1e-05' }, inputs: [[7, 0, 0]] },
        {
          op: 'Activation',
          name: 'conv_2b_3x3_relu',
          attrs: { act_type: 'relu' },
          inputs: [[8, 0, 0]],
        },

        {
          op: 'Pooling',
          name: 'pool1',
          attrs: { kernel: '(3, 3)', pool_type: 'max', stride: '(2, 2)' },
          inputs: [[9, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'conv_3b_1x1',
          attrs: { kernel: '(1, 1)', num_filter: '80' },
          inputs: [[10, 0, 0]],
        },
        { op: 'BatchNorm', name: 'conv_3b_1x1_bn', attrs: { eps: '1e-05' }, inputs: [[11, 0, 0]] },
        {
          op: 'Activation',
          name: 'conv_3b_1x1_relu',
          attrs: { act_type: 'relu' },
          inputs: [[12, 0, 0]],
        },

        // Simulating inception module
        {
          op: 'Convolution',
          name: 'mixed0_1x1',
          attrs: { kernel: '(1, 1)', num_filter: '64' },
          inputs: [[13, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'mixed0_5x5_reduce',
          attrs: { kernel: '(1, 1)', num_filter: '48' },
          inputs: [[13, 0, 0]],
        },
        {
          op: 'Convolution',
          name: 'mixed0_5x5',
          attrs: { kernel: '(5, 5)', num_filter: '64', pad: '(2, 2)' },
          inputs: [[15, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'mixed0_3x3_reduce',
          attrs: { kernel: '(1, 1)', num_filter: '64' },
          inputs: [[13, 0, 0]],
        },
        {
          op: 'Convolution',
          name: 'mixed0_3x3_1',
          attrs: { kernel: '(3, 3)', num_filter: '96', pad: '(1, 1)' },
          inputs: [[17, 0, 0]],
        },
        {
          op: 'Convolution',
          name: 'mixed0_3x3_2',
          attrs: { kernel: '(3, 3)', num_filter: '96', pad: '(1, 1)' },
          inputs: [[18, 0, 0]],
        },

        {
          op: 'Pooling',
          name: 'mixed0_pool',
          attrs: { kernel: '(3, 3)', pool_type: 'avg', stride: '(1, 1)', pad: '(1, 1)' },
          inputs: [[13, 0, 0]],
        },
        {
          op: 'Convolution',
          name: 'mixed0_pool_reduce',
          attrs: { kernel: '(1, 1)', num_filter: '32' },
          inputs: [[20, 0, 0]],
        },

        {
          op: 'Concat',
          name: 'mixed0_concat',
          inputs: [
            [14, 0, 0],
            [16, 0, 0],
            [19, 0, 0],
            [21, 0, 0],
          ],
        },

        {
          op: 'Pooling',
          name: 'global_pool',
          attrs: { global_pool: 'True', pool_type: 'avg' },
          inputs: [[22, 0, 0]],
        },
        { op: 'Dropout', name: 'dropout', attrs: { p: '0.2' }, inputs: [[23, 0, 0]] },
        { op: 'Flatten', name: 'flatten', inputs: [[24, 0, 0]] },
        { op: 'FullyConnected', name: 'fc1', attrs: { num_hidden: '1000' }, inputs: [[25, 0, 0]] },
        { op: 'SoftmaxOutput', name: 'softmax', inputs: [[26, 0, 0]] },
      ],
      arg_nodes: [0],
      heads: [[27, 0, 0]],
    };

    const parsed = parseMxNetSymbol(JSON.stringify(symbolJson));
    const mapper = new MxNetMapper();
    const graph = new Graph('inception_v3');

    for (const node of parsed.nodes) {
      const mappedNodes = mapper.map(node, graph);
      for (const mn of mappedNodes) {
        graph.addNode(mn);
      }
    }

    const opTypes = graph.nodes.map((n) => n.opType);
    expect(opTypes).toContain('Conv');
    expect(opTypes).toContain('BatchNormalization');
    expect(opTypes).toContain('Relu');
    expect(opTypes).toContain('MaxPool');
    expect(opTypes).toContain('AveragePool');
    expect(opTypes).toContain('Concat');
    expect(opTypes).toContain('Identity'); // Dropout
    expect(opTypes).toContain('GlobalAveragePool');
    expect(opTypes).toContain('Flatten');
    expect(opTypes).toContain('Gemm');
    expect(opTypes).toContain('Softmax');
  });
});
