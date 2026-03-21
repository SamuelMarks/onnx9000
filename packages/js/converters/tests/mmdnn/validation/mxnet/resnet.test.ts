import { describe, it, expect } from 'vitest';
import { MxNetMapper } from '../../../../src/mmdnn/mxnet/mapper.js';
import { parseMxNetSymbol } from '../../../../src/mmdnn/mxnet/parser.js';
import { Graph } from '@onnx9000/core';

describe('MXNet Validation - ResNet', () => {
  it('should parse and map ResNet architecture', () => {
    const symbolJson = {
      nodes: [
        { op: 'null', name: 'data', inputs: [] },
        {
          op: 'Convolution',
          name: 'conv0',
          attrs: { kernel: '(7, 7)', num_filter: '64', pad: '(3, 3)', stride: '(2, 2)' },
          inputs: [[0, 0, 0]],
        },
        {
          op: 'BatchNorm',
          name: 'bn0',
          attrs: { eps: '2e-05', fix_gamma: 'False', momentum: '0.9' },
          inputs: [[1, 0, 0]],
        },
        { op: 'Activation', name: 'relu0', attrs: { act_type: 'relu' }, inputs: [[2, 0, 0]] },
        {
          op: 'Pooling',
          name: 'pool0',
          attrs: { kernel: '(3, 3)', pool_type: 'max', stride: '(2, 2)' },
          inputs: [[3, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'stage1_unit1_conv1',
          attrs: { kernel: '(1, 1)', num_filter: '64' },
          inputs: [[4, 0, 0]],
        },
        { op: 'BatchNorm', name: 'stage1_unit1_bn1', attrs: { eps: '2e-05' }, inputs: [[5, 0, 0]] },
        {
          op: 'Activation',
          name: 'stage1_unit1_relu1',
          attrs: { act_type: 'relu' },
          inputs: [[6, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'stage1_unit1_conv2',
          attrs: { kernel: '(3, 3)', num_filter: '64', pad: '(1, 1)' },
          inputs: [[7, 0, 0]],
        },
        { op: 'BatchNorm', name: 'stage1_unit1_bn2', attrs: { eps: '2e-05' }, inputs: [[8, 0, 0]] },
        {
          op: 'Activation',
          name: 'stage1_unit1_relu2',
          attrs: { act_type: 'relu' },
          inputs: [[9, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'stage1_unit1_conv3',
          attrs: { kernel: '(1, 1)', num_filter: '256' },
          inputs: [[10, 0, 0]],
        },
        {
          op: 'BatchNorm',
          name: 'stage1_unit1_bn3',
          attrs: { eps: '2e-05' },
          inputs: [[11, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'stage1_unit1_sc',
          attrs: { kernel: '(1, 1)', num_filter: '256' },
          inputs: [[4, 0, 0]],
        },
        {
          op: 'BatchNorm',
          name: 'stage1_unit1_sc_bn',
          attrs: { eps: '2e-05' },
          inputs: [[13, 0, 0]],
        },

        {
          op: 'elemwise_add',
          name: 'stage1_unit1_plus',
          inputs: [
            [12, 0, 0],
            [14, 0, 0],
          ],
        },
        {
          op: 'Activation',
          name: 'stage1_unit1_relu3',
          attrs: { act_type: 'relu' },
          inputs: [[15, 0, 0]],
        },

        {
          op: 'Pooling',
          name: 'pool1',
          attrs: { global_pool: 'True', pool_type: 'avg' },
          inputs: [[16, 0, 0]],
        },
        { op: 'Flatten', name: 'flatten0', inputs: [[17, 0, 0]] },
        { op: 'FullyConnected', name: 'fc1', attrs: { num_hidden: '1000' }, inputs: [[18, 0, 0]] },
        { op: 'SoftmaxOutput', name: 'softmax', inputs: [[19, 0, 0]] },
      ],
      arg_nodes: [0],
      heads: [[20, 0, 0]],
    };

    const parsed = parseMxNetSymbol(JSON.stringify(symbolJson));
    const mapper = new MxNetMapper();
    const graph = new Graph('resnet');

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
    expect(opTypes).toContain('Add'); // elemwise_add maps to Add
    expect(opTypes).toContain('GlobalAveragePool');
    expect(opTypes).toContain('Flatten');
    expect(opTypes).toContain('Gemm');
    expect(opTypes).toContain('Softmax');
  });
});
