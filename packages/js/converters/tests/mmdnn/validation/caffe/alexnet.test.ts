import { describe, it, expect } from 'vitest';
import { CaffeMapper } from '../../../../src/mmdnn/caffe/mapper.js';
import { parsePrototxt } from '../../../../src/mmdnn/caffe/parser.js';
import { Graph } from '@onnx9000/core';

describe('Caffe Validation - AlexNet', () => {
  it('should parse and map AlexNet architecture', () => {
    const prototxt = `
name: "AlexNet"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 1 dim: 3 dim: 227 dim: 227 } }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "conv1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "norm1"
  top: "pool1"
  pooling_param {
    pool: 0
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc8"
  top: "prob"
}
`;

    const parsed = parsePrototxt(prototxt);
    const mapper = new CaffeMapper();
    const graph = new Graph('alexnet');

    for (const layer of parsed.layer) {
      const nodes = mapper.map(layer, graph);
      for (const node of nodes) {
        graph.addNode(node);
      }
    }

    const opTypes = graph.nodes.map((n) => n.opType);
    expect(opTypes).toContain('Conv');
    expect(opTypes).toContain('Relu');
    expect(opTypes).toContain('LRN');
    expect(opTypes).toContain('MaxPool');
    expect(opTypes).toContain('Gemm');
    expect(opTypes).toContain('Softmax');
  });
});
