import { describe, it, expect } from 'vitest';
import { MxNetMapper } from '../../../../src/mmdnn/mxnet/mapper.js';
import { parseMxNetSymbol } from '../../../../src/mmdnn/mxnet/parser.js';
import { Graph } from '@onnx9000/core';

describe('MXNet Validation - VGG', () => {
  it('should parse and map VGG architecture', () => {
    const symbolJson = {
      nodes: [
        { op: 'null', name: 'data', inputs: [] },
        {
          op: 'Convolution',
          name: 'conv1',
          attrs: { kernel: '(3, 3)', num_filter: '64', pad: '(1, 1)' },
          inputs: [[0, 0, 0]],
        },
        { op: 'Activation', name: 'relu1', attrs: { act_type: 'relu' }, inputs: [[1, 0, 0]] },
        {
          op: 'Pooling',
          name: 'pool1',
          attrs: { kernel: '(2, 2)', pool_type: 'max', stride: '(2, 2)' },
          inputs: [[2, 0, 0]],
        },
        { op: 'Flatten', name: 'flatten0', inputs: [[3, 0, 0]] },
        { op: 'FullyConnected', name: 'fc1', attrs: { num_hidden: '4096' }, inputs: [[4, 0, 0]] },
        { op: 'Activation', name: 'relu_fc1', attrs: { act_type: 'relu' }, inputs: [[5, 0, 0]] },
        { op: 'Dropout', name: 'drop1', attrs: { p: '0.5' }, inputs: [[6, 0, 0]] },
        { op: 'FullyConnected', name: 'fc2', attrs: { num_hidden: '1000' }, inputs: [[7, 0, 0]] },
        { op: 'SoftmaxOutput', name: 'softmax', inputs: [[8, 0, 0]] },
      ],
      arg_nodes: [0],
      heads: [[9, 0, 0]],
    };

    const parsed = parseMxNetSymbol(JSON.stringify(symbolJson));
    const mapper = new MxNetMapper();
    const graph = new Graph('vgg');

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
    expect(opTypes).toContain('Flatten');
    expect(opTypes).toContain('Gemm');
    expect(opTypes).toContain('Identity'); // Dropout mapped to Identity
    expect(opTypes).toContain('Softmax');
  });
});
