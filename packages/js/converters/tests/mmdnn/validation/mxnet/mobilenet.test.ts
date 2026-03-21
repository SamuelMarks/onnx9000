import { describe, it, expect } from 'vitest';
import { MxNetMapper } from '../../../../src/mmdnn/mxnet/mapper.js';
import { parseMxNetSymbol } from '../../../../src/mmdnn/mxnet/parser.js';
import { Graph } from '@onnx9000/core';

describe('MXNet Validation - MobileNet', () => {
  it('should parse and map MobileNet architecture', () => {
    const symbolJson = {
      nodes: [
        { op: 'null', name: 'data', inputs: [] },
        {
          op: 'Convolution',
          name: 'conv1',
          attrs: { kernel: '(3, 3)', num_filter: '32', pad: '(1, 1)', stride: '(2, 2)' },
          inputs: [[0, 0, 0]],
        },
        {
          op: 'BatchNorm',
          name: 'conv1_bn',
          attrs: { eps: '1e-05', fix_gamma: 'False', momentum: '0.9' },
          inputs: [[1, 0, 0]],
        },
        { op: 'Activation', name: 'relu1', attrs: { act_type: 'relu' }, inputs: [[2, 0, 0]] },

        {
          op: 'Convolution',
          name: 'conv2_dw',
          attrs: { kernel: '(3, 3)', num_filter: '32', num_group: '32', pad: '(1, 1)' },
          inputs: [[3, 0, 0]],
        },
        { op: 'BatchNorm', name: 'conv2_dw_bn', attrs: { eps: '1e-05' }, inputs: [[4, 0, 0]] },
        {
          op: 'Activation',
          name: 'conv2_dw_relu',
          attrs: { act_type: 'relu' },
          inputs: [[5, 0, 0]],
        },

        {
          op: 'Convolution',
          name: 'conv2_sep',
          attrs: { kernel: '(1, 1)', num_filter: '64' },
          inputs: [[6, 0, 0]],
        },
        { op: 'BatchNorm', name: 'conv2_sep_bn', attrs: { eps: '1e-05' }, inputs: [[7, 0, 0]] },
        {
          op: 'Activation',
          name: 'conv2_sep_relu',
          attrs: { act_type: 'relu' },
          inputs: [[8, 0, 0]],
        },

        {
          op: 'Pooling',
          name: 'pool1',
          attrs: { global_pool: 'True', pool_type: 'avg' },
          inputs: [[9, 0, 0]],
        },
        { op: 'Flatten', name: 'flatten0', inputs: [[10, 0, 0]] },
        { op: 'FullyConnected', name: 'fc1', attrs: { num_hidden: '1000' }, inputs: [[11, 0, 0]] },
        { op: 'SoftmaxOutput', name: 'softmax', inputs: [[12, 0, 0]] },
      ],
      arg_nodes: [0],
      heads: [[13, 0, 0]],
    };

    const parsed = parseMxNetSymbol(JSON.stringify(symbolJson));
    const mapper = new MxNetMapper();
    const graph = new Graph('mobilenet');

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
    expect(opTypes).toContain('GlobalAveragePool');
    expect(opTypes).toContain('Flatten');
    expect(opTypes).toContain('Gemm');
    expect(opTypes).toContain('Softmax');

    // Check if the depthwise conv mapped group property properly
    const dwConvNode = graph.nodes.find((n) => n.name === 'conv2_dw');
    expect(dwConvNode).toBeDefined();
    expect(dwConvNode?.attributes['group'].value).toBe(32);
  });
});
