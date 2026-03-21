import { describe, it, expect } from 'vitest';
import { CaffeMapper } from '../../../../src/mmdnn/caffe/mapper.js';
import { parsePrototxt } from '../../../../src/mmdnn/caffe/parser.js';
import { Graph } from '@onnx9000/core';

describe('Caffe Validation - GoogLeNet', () => {
  it('should parse and map GoogLeNet architecture', () => {
    const prototxt = `
name: "GoogLeNet"
layer {
  name: "conv1/7x7_s2"
  type: "Convolution"
  bottom: "data"
  top: "conv1/7x7_s2"
  convolution_param {
    num_output: 64
    pad: 3
    kernel_size: 7
    stride: 2
  }
}
layer {
  name: "conv1/relu_7x7"
  type: "ReLU"
  bottom: "conv1/7x7_s2"
  top: "conv1/7x7_s2"
}
layer {
  name: "pool1/3x3_s2"
  type: "Pooling"
  bottom: "conv1/7x7_s2"
  top: "pool1/3x3_s2"
  pooling_param {
    pool: 0
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "inception_3a/1x1"
  type: "Convolution"
  bottom: "pool1/3x3_s2"
  top: "inception_3a/1x1"
  convolution_param {
    num_output: 64
    kernel_size: 1
  }
}
layer {
  name: "inception_3a/relu_1x1"
  type: "ReLU"
  bottom: "inception_3a/1x1"
  top: "inception_3a/1x1"
}
layer {
  name: "inception_3a/output"
  type: "Concat"
  bottom: "inception_3a/1x1"
  bottom: "inception_3a/3x3"
  bottom: "inception_3a/5x5"
  bottom: "inception_3a/pool_proj"
  top: "inception_3a/output"
}
layer {
  name: "pool5/7x7_s1"
  type: "Pooling"
  bottom: "inception_5b/output"
  top: "pool5/7x7_s1"
  pooling_param {
    pool: 1
    kernel_size: 7
    stride: 1
  }
}
layer {
  name: "loss3/classifier"
  type: "InnerProduct"
  bottom: "pool5/drop_7x7_s1"
  top: "loss3/classifier"
  inner_product_param {
    num_output: 1000
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "loss3/classifier"
  top: "prob"
}
`;

    const parsed = parsePrototxt(prototxt);
    const mapper = new CaffeMapper();
    const graph = new Graph('googlenet');

    for (const layer of parsed.layer) {
      const nodes = mapper.map(layer, graph);
      for (const node of nodes) {
        graph.addNode(node);
      }
    }

    const opTypes = graph.nodes.map((n) => n.opType);
    expect(opTypes).toContain('Conv');
    expect(opTypes).toContain('Relu');
    expect(opTypes).toContain('MaxPool');
    expect(opTypes).toContain('AveragePool');
    expect(opTypes).toContain('Concat');
    expect(opTypes).toContain('Gemm');
    expect(opTypes).toContain('Softmax');
  });
});
